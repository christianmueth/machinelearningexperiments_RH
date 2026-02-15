import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _csv_needs_header(csv_path: str) -> bool:
    """Return True if csv_path is missing or has no non-empty header line.

    This guards against interrupted runs creating an empty/blank file, which would
    otherwise break both resume and final summary generation.
    """
    if (not csv_path) or (not os.path.exists(csv_path)):
        return True
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                # Heuristic: if the first non-empty line contains common columns,
                # assume it's a header.
                if ("seed" in s) and ("anchor_seed" in s) and ("loop_e0" in s):
                    return False
                return True
        return True
    except Exception:
        return True


def find_npz_files(out_root: str) -> List[str]:
    hits: List[str] = []
    for root, _dirs, files in os.walk(out_root):
        for fn in files:
            if fn.endswith(".npz") and fn.startswith("channel_diag"):
                hits.append(os.path.join(root, fn))
    return sorted(hits)


def parse_rel_metadata(path: str, out_root: str) -> Tuple[str, str, int, int, float, float]:
    rel = os.path.relpath(path, out_root).replace("\\", "/")
    parts = rel.split("/")
    backend_label = parts[0].replace("backend_", "") if parts else ""
    seed = -1
    anchor = -1
    wlo = float("nan")
    whi = float("nan")
    for p in parts:
        if p.startswith("seed") and "_anchor" in p:
            a, b = p.split("_anchor")
            seed = int(a.replace("seed", ""))
            anchor = int(b)
        if p.startswith("window_"):
            w = p.replace("window_", "")
            a, b = w.split("_", 1)
            wlo = float(a)
            whi = float(b)
    return rel, backend_label, seed, anchor, wlo, whi


def key_for_pair(seed: int, anchor: int, wlo: float, whi: float) -> Tuple[int, int, float, float]:
    return (int(seed), int(anchor), float(wlo), float(whi))


def make_within_backend_anchor_offset_map(
    base_map: Dict[Tuple[int, int, float, float], str], *, offset: int
) -> Dict[Tuple[int, int, float, float], str]:
    """Create a same-backend pseudo-pairing by shifting anchors within each (seed, window).

    Returns a dict keyed by the *original* keys, whose values point at a different
    artifact path from the same backend.
    """
    off = int(offset)
    if off == 0:
        off = 1

    groups: Dict[Tuple[int, float, float], List[int]] = {}
    path_by: Dict[Tuple[int, float, float, int], str] = {}

    for (seed, anchor, wlo, whi), path in base_map.items():
        gk = (int(seed), float(wlo), float(whi))
        groups.setdefault(gk, []).append(int(anchor))
        path_by[(int(seed), float(wlo), float(whi), int(anchor))] = str(path)

    anchors_sorted: Dict[Tuple[int, float, float], List[int]] = {}
    pos_in_group: Dict[Tuple[int, float, float], Dict[int, int]] = {}
    for gk, anchors in groups.items():
        uniq = sorted(set(int(a) for a in anchors))
        anchors_sorted[gk] = uniq
        pos_in_group[gk] = {int(a): int(i) for i, a in enumerate(uniq)}

    out_map: Dict[Tuple[int, int, float, float], str] = {}
    for (seed, anchor, wlo, whi) in base_map.keys():
        gk = (int(seed), float(wlo), float(whi))
        alist = anchors_sorted.get(gk, [])
        if not alist:
            continue
        if len(alist) == 1:
            target = int(alist[0])
        else:
            pos = pos_in_group[gk].get(int(anchor), 0)
            target = int(alist[(pos + off) % len(alist)])
            if target == int(anchor):
                target = int(alist[(pos + 1) % len(alist)])
        out_map[(int(seed), int(anchor), float(wlo), float(whi))] = path_by[(int(seed), float(wlo), float(whi), int(target))]

    return out_map


def cayley_from_lambda(L: np.ndarray, eta: float) -> np.ndarray:
    L = np.asarray(L, dtype=np.complex128)
    n = int(L.shape[0])
    I = np.eye(n, dtype=np.complex128)
    e = float(eta)
    return (L - 1j * e * I) @ np.linalg.inv(L + 1j * e * I)


def unitary_polar_factor(G: np.ndarray) -> np.ndarray:
    G = np.asarray(G, dtype=np.complex128)
    U, _s, Vh = np.linalg.svd(G)
    return U @ Vh


def principal_angles(Qa: np.ndarray, Qb: np.ndarray) -> np.ndarray:
    if Qa.shape[1] == 0 or Qb.shape[1] == 0:
        return np.asarray([], dtype=np.float64)
    X = np.conjugate(Qa.T) @ Qb
    s = np.linalg.svd(X, compute_uv=False)
    s = np.clip(np.real(s), 0.0, 1.0)
    return np.arccos(s)


def circular_gaps_sorted(sorted_angles: np.ndarray) -> np.ndarray:
    a = np.asarray(sorted_angles, dtype=np.float64)
    if a.size == 0:
        return np.asarray([], dtype=np.float64)
    diffs = np.diff(a)
    wrap = (2.0 * math.pi) - (a[-1] - a[0])
    return np.concatenate([diffs, np.asarray([wrap], dtype=np.float64)])


def holonomy_spectrum_stats(M: np.ndarray) -> Dict[str, Any]:
    eig = np.linalg.eigvals(M)
    phases = np.angle(eig)
    if phases.size == 0:
        return {
            "holonomy_eigphase_meanabs": float("nan"),
            "holonomy_phi_min": float("nan"),
            "holonomy_phi_median": float("nan"),
            "holonomy_phi_max": float("nan"),
            "holonomy_phi_gap_max": float("nan"),
            "holonomy_phi_entropy": float("nan"),
            "holonomy_phi_vec_json": "",
        }

    phi = np.sort(phases)
    gaps = circular_gaps_sorted(np.mod(phi + math.pi, 2.0 * math.pi))
    gap_phi_max = float(np.max(gaps)) if gaps.size else float("nan")

    bins = 24
    hist, _edges = np.histogram(phi, bins=bins, range=(-math.pi, math.pi), density=False)
    p = hist.astype(np.float64)
    p = p / (np.sum(p) + 1e-12)
    p = p[p > 0]
    ent = float(-np.sum(p * np.log(p)))

    vec_json = ""
    if int(phi.size) <= 64:
        vec_json = json.dumps([float(x) for x in phi.tolist()])

    return {
        "holonomy_eigphase_meanabs": float(np.mean(np.abs(phi))),
        "holonomy_phi_min": float(phi[0]),
        "holonomy_phi_median": float(np.median(phi)),
        "holonomy_phi_max": float(phi[-1]),
        "holonomy_phi_gap_max": float(gap_phi_max),
        "holonomy_phi_entropy": float(ent),
        "holonomy_phi_vec_json": vec_json,
    }


def basis_blocks_equalcount(S: np.ndarray, blocks: int) -> Tuple[List[np.ndarray], List[int], Dict[str, Any]]:
    S = np.asarray(S, dtype=np.complex128)
    eigvals, eigvecs = np.linalg.eig(S)
    angles = np.mod(np.angle(eigvals), 2.0 * math.pi)
    n = int(angles.size)
    B = int(max(1, min(int(blocks), n)))

    idx_sorted = np.argsort(angles)
    sizes = [n // B for _ in range(B)]
    for i in range(n - sum(sizes)):
        sizes[i] += 1

    order: List[np.ndarray] = []
    start = 0
    for b in range(B):
        end = start + sizes[b]
        order.append(np.asarray(idx_sorted[start:end], dtype=np.int64))
        start = end

    Qs: List[np.ndarray] = []
    dims: List[int] = []
    centers: List[float] = []
    for idx in order:
        if idx.size == 0:
            Qs.append(np.zeros((n, 0), dtype=np.complex128))
            dims.append(0)
            centers.append(float("nan"))
            continue
        V = eigvecs[:, idx]
        Q, _ = np.linalg.qr(V)
        Q = np.asarray(Q, dtype=np.complex128)
        Qs.append(Q)
        dims.append(int(Q.shape[1]))
        z = np.mean(np.exp(1j * angles[idx]))
        centers.append(float(np.mod(np.angle(z), 2.0 * math.pi)))

    perm = np.argsort(np.nan_to_num(np.asarray(centers), nan=10.0 * math.pi))
    Qs = [Qs[int(i)] for i in perm]
    dims = [dims[int(i)] for i in perm]
    centers = [centers[int(i)] for i in perm]

    extra: Dict[str, Any] = {"n": n, "effective_blocks": B, "block_center_angles": centers}
    return Qs, dims, extra


def fro_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x, ord="fro"))


def safe_inv_unitary(M: np.ndarray) -> np.ndarray:
    return np.conjugate(M.T)


def angle_wrap_pi(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return ((x + np.pi) % (2.0 * np.pi)) - np.pi


def angle_distance(a: float, b: float) -> float:
    d = float(abs(float(a) - float(b)))
    d = float(d % (2.0 * math.pi))
    return float(min(d, 2.0 * math.pi - d))


def imag_fraction(M: np.ndarray) -> float:
    M = np.asarray(M, dtype=np.complex128)
    denom = float(np.linalg.norm(M, ord="fro")) + 1e-12
    return float(np.linalg.norm(np.imag(M), ord="fro") / denom)


def ortho_defect(M: np.ndarray) -> float:
    M = np.asarray(M, dtype=np.complex128)
    n = int(M.shape[0])
    I = np.eye(n, dtype=np.complex128)
    return float(np.linalg.norm(M.T @ M - I, ord="fro") / (np.linalg.norm(I, ord="fro") + 1e-12))


def unit_defect(M: np.ndarray) -> float:
    M = np.asarray(M, dtype=np.complex128)
    n = int(M.shape[0])
    I = np.eye(n, dtype=np.complex128)
    return float(np.linalg.norm(np.conjugate(M.T) @ M - I, ord="fro") / (np.linalg.norm(I, ord="fro") + 1e-12))


def det_imag_frac(M: np.ndarray) -> float:
    d = complex(np.linalg.det(np.asarray(M, dtype=np.complex128)))
    return float(abs(d.imag) / (abs(d) + 1e-18))


def real_polar_resid(M: np.ndarray) -> float:
    """Distance from M to nearest *real* orthogonal matrix.

    Compute MR = Re(M), then Q = polar(MR) in O(n), report ||M - Q||_F / ||I||_F.
    """
    M = np.asarray(M, dtype=np.complex128)
    n = int(M.shape[0])
    if n == 0:
        return float("nan")
    MR = np.asarray(np.real(M), dtype=np.float64)
    U, _s, Vh = np.linalg.svd(MR, full_matrices=False)
    Qr = (U @ Vh).astype(np.float64)
    Q = np.asarray(Qr, dtype=np.complex128)
    I = np.eye(n, dtype=np.complex128)
    return float(np.linalg.norm(M - Q, ord="fro") / (np.linalg.norm(I, ord="fro") + 1e-12))


def minus1_eig_stats(M: np.ndarray, *, tol: float = 1e-3) -> Tuple[int, float]:
    """Return (num eigenvalues within tol of -1, min distance to -1)."""
    M = np.asarray(M, dtype=np.complex128)
    if M.size == 0:
        return 0, float("nan")
    eig = np.linalg.eigvals(M)
    dist = np.abs(eig + 1.0)
    dist = dist[np.isfinite(dist)]
    if dist.size == 0:
        return 0, float("nan")
    return int(np.sum(dist <= float(tol))), float(np.min(dist))


def parse_loop_pairs(s: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split(":", 1)
        out.append((int(a.strip()), int(b.strip())))
    return out


def normalize_index(i: int, n: int) -> int:
    ii = int(i)
    if ii < 0:
        ii = int(n + ii)
    return int(np.clip(ii, 0, n - 1))


def refine_path_indices(i0: int, i1: int, nE: int, refine_steps: int) -> List[int]:
    i0 = int(i0)
    i1 = int(i1)
    if int(refine_steps) <= 0 or i0 == i1:
        return [i0, i1]

    steps = int(refine_steps) + 1
    xs = np.linspace(float(i0), float(i1), num=steps + 1, endpoint=True)
    idx = [int(np.clip(int(round(x)), 0, nE - 1)) for x in xs.tolist()]
    out: List[int] = []
    for t in idx:
        if not out or int(t) != int(out[-1]):
            out.append(int(t))
    if len(out) < 2:
        out = [i0, i1]
    return out


def parse_lambdas(s: str) -> List[float]:
    out: List[float] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def parse_mus(s: str) -> List[float]:
    out: List[float] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def direction_cosine_fro(d1: np.ndarray, d2: np.ndarray) -> float:
    d1 = np.asarray(d1, dtype=np.complex128)
    d2 = np.asarray(d2, dtype=np.complex128)
    n1 = float(np.linalg.norm(d1.ravel()))
    n2 = float(np.linalg.norm(d2.ravel()))
    if (not np.isfinite(n1)) or (not np.isfinite(n2)) or n1 <= 0.0 or n2 <= 0.0:
        return float("nan")
    ip = np.vdot(d1.ravel(), d2.ravel())
    return float(np.real(ip) / (n1 * n2))


def complex_flatten_realimag(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex128).ravel()
    return np.concatenate([np.real(x), np.imag(x)])


def effective_rank2(v1: np.ndarray, v2: np.ndarray, rtol: float = 1e-6) -> int:
    a = complex_flatten_realimag(v1)
    b = complex_flatten_realimag(v2)
    if a.size != b.size or a.size == 0:
        return 0
    M = np.stack([a, b], axis=0).astype(np.float64, copy=False)
    try:
        s = np.linalg.svd(M, compute_uv=False)
    except Exception:
        return 0
    if s.size == 0 or (not np.isfinite(s).all()):
        return 0
    s0 = float(np.max(s))
    if (not np.isfinite(s0)) or s0 <= 0.0:
        return 0
    thr = float(rtol) * s0
    return int(np.sum(s > thr))


@dataclass(frozen=True)
class LoopSpec:
    loop_type: str  # rectangle|degenerate
    direction: str  # fwd|rev|degenerate
    double_loop: bool = False
    degenerate_axis: str = ""  # E|lambda


def loop_points_elambda(i_path: Sequence[int], j0: int, j1: int, spec: LoopSpec) -> List[Tuple[int, int]]:
    i_path = [int(x) for x in list(i_path)]
    if len(i_path) < 2:
        raise ValueError("i_path must have length >= 2")

    start = int(i_path[0])
    end = int(i_path[-1])
    j0 = int(j0)
    j1 = int(j1)

    P0 = (start, j0)
    P3 = (start, j1)

    edge0 = [(int(i), j0) for i in i_path]
    edge1 = [(int(i), j1) for i in i_path]

    if spec.loop_type == "rectangle" and spec.direction == "fwd" and not spec.double_loop:
        pts: List[Tuple[int, int]] = []
        pts.extend(edge0)
        pts.append((end, j1))
        pts.extend(list(reversed(edge1))[1:])
        pts.append(P0)
        return pts

    if spec.loop_type == "rectangle" and spec.direction == "rev" and not spec.double_loop:
        pts = []
        pts.append(P0)
        pts.append(P3)
        pts.extend(edge1[1:])
        pts.append((end, j0))
        pts.extend(list(reversed(edge0))[1:])
        return pts

    if spec.loop_type == "rectangle" and spec.direction == "fwd" and spec.double_loop:
        pts1 = loop_points_elambda(i_path, j0, j1, LoopSpec(loop_type="rectangle", direction="fwd"))
        return pts1 + pts1[1:]

    if spec.loop_type == "degenerate" and spec.degenerate_axis == "E":
        return [(start, j0), (end, j0), (start, j0)]

    if spec.loop_type == "degenerate" and spec.degenerate_axis == "lambda":
        return [(start, j0), (start, j1), (start, j0)]

    raise ValueError(f"unknown loop spec: {spec}")


def compute_monodromy_elambda(
    lamA: np.ndarray,
    lamB: np.ndarray,
    lambdas: Sequence[float],
    *,
    pts: Sequence[Tuple[int, int]],
    eta: float,
    blocks: int,
    basis_cache: Optional[Dict[Tuple[Any, ...], Tuple[List[np.ndarray], List[int], Dict[str, Any]]]] = None,
    lamB2: Optional[np.ndarray] = None,
    mu: float = 0.0,
) -> Tuple[List[Optional[np.ndarray]], List[Dict[str, Any]]]:
    bases: List[List[np.ndarray]] = []
    if basis_cache is None:
        basis_cache = {}

    # Track deformation magnitude along steps
    step_dLambda: List[float] = []

    mu_key: Optional[float] = None
    if lamB2 is not None:
        mu_key = float(np.round(float(mu), 12))

    def _eval_L(ei: int, lj: int) -> np.ndarray:
        lam = float(lambdas[int(lj)])
        A = np.asarray(lamA[int(ei)], dtype=np.complex128)
        B1 = np.asarray(lamB[int(ei)], dtype=np.complex128)
        if lamB2 is None:
            B = B1
        else:
            B2 = np.asarray(lamB2[int(ei)], dtype=np.complex128)
            mm = float(mu)
            B = (1.0 - mm) * B1 + mm * B2
        return (1.0 - lam) * A + lam * B

    for k, (ei, lj) in enumerate(pts):
        if mu_key is None:
            key: Tuple[Any, ...] = (int(ei), int(lj))
        else:
            key = (int(ei), int(lj), float(mu_key))
        if key in basis_cache:
            Qs, dims, extra = basis_cache[key]
        else:
            L = _eval_L(int(ei), int(lj))
            S = cayley_from_lambda(L, eta=float(eta))
            Qs, dims, extra = basis_blocks_equalcount(S, blocks=int(blocks))
            basis_cache[key] = (Qs, dims, extra)
        bases.append(Qs)

        if k > 0:
            ei0, lj0 = pts[k - 1]
            L0 = _eval_L(int(ei0), int(lj0))
            L1 = _eval_L(int(ei), int(lj))
            step_dLambda.append(float(fro_norm(L1 - L0)))

    B = int(len(bases[0]))
    Ms: List[Optional[np.ndarray]] = [None for _ in range(B)]
    diags: List[Dict[str, Any]] = []

    for b in range(B):
        valid = True
        M: Optional[np.ndarray] = None
        step_angles: List[float] = []
        step_overlaps: List[float] = []
        step_smin: List[float] = []
        step_W_imag: List[float] = []
        step_W_orth: List[float] = []
        step_W_unit: List[float] = []
        step_W_det_imag: List[float] = []

        for k in range(len(pts) - 1):
            U = np.asarray(bases[k][b], dtype=np.complex128)
            V = np.asarray(bases[k + 1][b], dtype=np.complex128)
            if U.shape[1] != V.shape[1]:
                valid = False
                break
            if U.shape[1] == 0:
                continue

            G = np.conjugate(U.T) @ V
            s = np.linalg.svd(G, compute_uv=False)
            smin = float(np.min(np.real(s))) if s.size else float("nan")
            step_smin.append(smin)

            W = unitary_polar_factor(G)
            if M is None:
                M = np.eye(W.shape[0], dtype=np.complex128)
            M = M @ W

            step_W_imag.append(float(imag_fraction(W)))
            step_W_orth.append(float(ortho_defect(W)))
            step_W_unit.append(float(unit_defect(W)))
            step_W_det_imag.append(float(det_imag_frac(W)))

            ang = principal_angles(U, V)
            step_angles.append(float(np.max(ang)) if ang.size else float("nan"))
            ov = float((np.linalg.norm(G, ord="fro") ** 2) / (float(U.shape[1]) + 1e-12))
            step_overlaps.append(ov)

        dim = int(bases[0][b].shape[1])
        if (not valid) or (dim == 0) or (M is None):
            Ms[b] = None
        else:
            Ms[b] = M

        diags.append(
            {
                "valid": bool(valid and (dim > 0) and (M is not None)),
                "dim": int(dim),
                "step_angle_median": float(np.nanmedian(step_angles)) if step_angles else float("nan"),
                "step_angle_worst": float(np.nanmax(step_angles)) if step_angles else float("nan"),
                "step_overlap_median": float(np.nanmedian(step_overlaps)) if step_overlaps else float("nan"),
                "step_smin_median": float(np.nanmedian(step_smin)) if step_smin else float("nan"),
                "step_smin_min": float(np.nanmin(step_smin)) if step_smin else float("nan"),
                "step_smin_p10": float(np.nanquantile(np.asarray(step_smin, dtype=np.float64), 0.10)) if step_smin else float("nan"),
                "step_dLambda_median": float(np.nanmedian(step_dLambda)) if step_dLambda else float("nan"),
                "step_dLambda_max": float(np.nanmax(step_dLambda)) if step_dLambda else float("nan"),
                "step_transport_imag_median": float(np.nanmedian(step_W_imag)) if step_W_imag else float("nan"),
                "step_transport_orth_defect_median": float(np.nanmedian(step_W_orth)) if step_W_orth else float("nan"),
                "step_transport_unit_defect_median": float(np.nanmedian(step_W_unit)) if step_W_unit else float("nan"),
                "step_transport_det_imag_frac_median": float(np.nanmedian(step_W_det_imag)) if step_W_det_imag else float("nan"),
                "step_transport_imag_worst": float(np.nanmax(step_W_imag)) if step_W_imag else float("nan"),
                "step_transport_orth_defect_worst": float(np.nanmax(step_W_orth)) if step_W_orth else float("nan"),
                "step_transport_unit_defect_worst": float(np.nanmax(step_W_unit)) if step_W_unit else float("nan"),
                "step_transport_det_imag_frac_worst": float(np.nanmax(step_W_det_imag)) if step_W_det_imag else float("nan"),
            }
        )

    return Ms, diags


def dump_holonomy_npz(
    out_dir: str,
    *,
    dump_id: int,
    M: np.ndarray,
    meta: Dict[str, Any],
) -> str:
    os.makedirs(out_dir, exist_ok=True)

    M = np.asarray(M, dtype=np.complex128)
    eigvals, eigvecs = np.linalg.eig(M)
    d = np.abs(eigvals + 1.0)
    idx = int(np.argmin(d)) if d.size else -1
    v = eigvecs[:, idx] if (idx >= 0 and eigvecs.size) else np.zeros((M.shape[0],), dtype=np.complex128)
    vn = float(np.linalg.norm(v))
    if vn > 0:
        v = v / vn

    path = os.path.join(out_dir, f"holonomy_{int(dump_id):08d}.npz")
    np.savez_compressed(
        path,
        M=M,
        eigvals=eigvals,
        eigvecs=eigvecs,
        minus1_idx=np.asarray([idx], dtype=np.int64),
        v_minus1=v,
        dist_to_minus1=np.asarray([float(d[idx]) if (idx >= 0 and d.size) else float("nan")], dtype=np.float64),
        meta_json=np.asarray([json.dumps(meta, sort_keys=True)], dtype=object),
    )
    return path


def _progress_key(d: Dict[str, Any]) -> str:
    return json.dumps(d, sort_keys=True)


def _load_done_progress(progress_path: str) -> set:
    done: set = set()
    if not os.path.exists(progress_path):
        return done
    try:
        with open(progress_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                done.add(s)
    except Exception:
        return done
    return done


def _reconstruct_progress_from_rows_csv(rows_csv: str) -> set:
    """Best-effort reconstruction of loop-task completion keys from an existing rows CSV."""
    done: set = set()
    if (not rows_csv) or (not os.path.exists(rows_csv)):
        return done
    try:
        df = pd.read_csv(rows_csv)
    except Exception:
        return done

    needed = [
        "seed",
        "anchor_seed",
        "wlo",
        "whi",
        "loop_e0",
        "loop_e1",
        "loop_refine_steps",
        "mu",
        "lambda_j0",
        "lambda_j1",
        "deform_alpha",
        "blocks",
        "within_backend",
        "within_backend_anchor_offset",
    ]
    for c in needed:
        if c not in df.columns:
            # Back-compat with older outputs that did not record mu.
            if c == "mu":
                continue
            return done

    cols = needed
    for _idx, r in df[cols].drop_duplicates().iterrows():
        key = {
            "seed": int(r["seed"]),
            "anchor_seed": int(r["anchor_seed"]),
            "wlo": float(r["wlo"]),
            "whi": float(r["whi"]),
            "loop_e0": int(r["loop_e0"]),
            "loop_e1": int(r["loop_e1"]),
            "loop_refine_steps": int(r["loop_refine_steps"]),
            "mu": float(r["mu"]) if ("mu" in df.columns) else 0.0,
            "lambda_j0": int(r["lambda_j0"]),
            "lambda_j1": int(r["lambda_j1"]),
            "deform_alpha": float(r["deform_alpha"]),
            "blocks": int(r["blocks"]),
            "within_backend": str(r["within_backend"]),
            "within_backend_anchor_offset": int(r["within_backend_anchor_offset"]),
        }
        done.add(_progress_key(key))

    return done


def _next_dump_id(dump_dir: str) -> int:
    if not os.path.isdir(dump_dir):
        return 0
    mx = -1
    try:
        for fn in os.listdir(dump_dir):
            if not (fn.startswith("holonomy_") and fn.endswith(".npz")):
                continue
            mid = fn.replace("holonomy_", "").replace(".npz", "")
            try:
                mx = max(mx, int(mid))
            except Exception:
                continue
    except Exception:
        return 0
    return int(mx + 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root_a", required=True)
    ap.add_argument("--out_root_b", required=True)
    ap.add_argument(
        "--out_root_b2",
        default="",
        help="Optional third root providing an independent deformation direction B2. When set, the family is built as B(mu)=(1-mu)B1+mu*B2 and then interpolated with A via lambda.",
    )
    ap.add_argument("--out_root", required=True, help="where to write suite outputs")
    ap.add_argument(
        "--within_backend",
        default="off",
        choices=["off", "A", "B"],
        help="If set to A or B, build a pseudo-geometry by pairing two different anchors within that same backend (anchor-offset mapping).",
    )
    ap.add_argument(
        "--within_backend_anchor_offset",
        type=int,
        default=1,
        help="Anchor offset used for within-backend pseudo-pairing (ignored unless --within_backend is A or B).",
    )
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--loop_pairs", default="0:-1", help="comma list like '0:-1,0:-2,1:-1'")
    ap.add_argument("--refine_steps", type=int, default=0, help="number of intermediate E indices along E-edges")
    ap.add_argument("--lambda_grid", type=int, default=5, help="number of lambda samples in [0,1]")
    ap.add_argument("--lambdas", default="", help="explicit comma list overriding --lambda_grid")
    ap.add_argument(
        "--suite_mode",
        default="full",
        choices=["full", "fwd_rect"],
        help="full: run rectangle/rev/double/degenerate sanity suite (default). fwd_rect: only the canonical forward rectangle (fast, for scanning walls).",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing outputs under out_root by skipping already-completed loop tasks and appending to the rows CSV.",
    )
    ap.add_argument(
        "--flush_every_loops",
        type=int,
        default=1,
        help="Checkpoint frequency (in completed loop-tasks). Smaller is safer for long runs.",
    )
    ap.add_argument(
        "--deform_alpha",
        type=float,
        default=1.0,
        help="Scale deformation strength by using lambda_eff = deform_alpha * lambda when mixing A/B (default 1.0).",
    )

    ap.add_argument("--mus", default="", help="Optional comma list of mu samples in [0,1]. Only used when --out_root_b2 is set.")
    ap.add_argument(
        "--mu_grid",
        type=int,
        default=0,
        help="If >1 and --mus is empty: number of mu samples in [0,1]. Only used when --out_root_b2 is set.",
    )
    ap.add_argument(
        "--direction_cosine_check",
        default="auto",
        choices=["auto", "off", "on"],
        help="Check that (B1-A) and (B2-A) are not nearly colinear. auto: only when --out_root_b2 is set.",
    )
    ap.add_argument(
        "--direction_cosine_tol",
        type=float,
        default=0.95,
        help="Fail loudly if |cos| exceeds this threshold (default 0.95).",
    )
    ap.add_argument(
        "--direction_cosine_samples",
        type=int,
        default=3,
        help="Number of E-snapshots to sample for the cosine check (default 3).",
    )

    ap.add_argument(
        "--dump_holonomy",
        default="off",
        choices=["off", "fwd_rect", "pi_only", "all"],
        help="Optional Phase-3F support: dump small-dim holonomy blocks to NPZ files under out_root for flip-mode identity/composition analysis.",
    )
    ap.add_argument(
        "--dump_holonomy_max_dim",
        type=int,
        default=16,
        help="Only dump holonomy blocks with dim <= this (default 16).",
    )
    ap.add_argument(
        "--dump_holonomy_tol_pi",
        type=float,
        default=0.25,
        help="For --dump_holonomy=pi_only: | |wrap(det_phase)| - pi | <= tol (default 0.25 rad).",
    )
    ap.add_argument(
        "--dump_holonomy_subdir",
        default="phase3f_holonomy_dumps",
        help="Subdirectory under out_root where NPZ dumps are written (default phase3f_holonomy_dumps).",
    )
    ap.add_argument("--eta", type=float, default=-1.0, help="if >0 use fixed eta; else use min(cayley_eta_a,cayley_eta_b)")
    ap.add_argument("--plots", action="store_true")

    ap.add_argument("--only_seed", type=int, default=-1, help="If >=0: restrict to this seed")
    ap.add_argument("--only_anchor", type=int, default=-1, help="If >=0: restrict to this anchor_seed")
    ap.add_argument(
        "--only_window",
        default="",
        help="Optional window filter 'wlo:whi' (floats). If set, restrict to this window.",
    )
    args = ap.parse_args()

    outA = str(args.out_root_a)
    outB = str(args.out_root_b)
    outB2 = str(args.out_root_b2).strip()
    out = str(args.out_root)

    os.makedirs(out, exist_ok=True)
    os.makedirs(out, exist_ok=True)

    within_backend = str(args.within_backend).strip()
    within_anchor_offset = int(args.within_backend_anchor_offset)

    if outB2 and within_backend != "off":
        raise SystemExit("--out_root_b2 is not compatible with --within_backend controls (needs distinct A,B1,B2)")

    filesA = find_npz_files(outA)
    filesB = find_npz_files(outB)
    filesB2: List[str] = []
    if outB2:
        filesB2 = find_npz_files(outB2)
    if within_backend == "A":
        if not filesA:
            raise SystemExit("No channel_diag*.npz found under out_root_a")
    elif within_backend == "B":
        if not filesB:
            raise SystemExit("No channel_diag*.npz found under out_root_b")
    else:
        if not filesA or not filesB:
            raise SystemExit("No channel_diag*.npz found under one of the roots")
        if outB2 and (not filesB2):
            raise SystemExit("No channel_diag*.npz found under out_root_b2")

    mapA: Dict[Tuple[int, int, float, float], str] = {}
    mapB: Dict[Tuple[int, int, float, float], str] = {}
    mapB2: Dict[Tuple[int, int, float, float], str] = {}

    for p in filesA:
        _rel, _backend, seed, anchor, wlo, whi = parse_rel_metadata(p, outA)
        mapA[key_for_pair(seed, anchor, wlo, whi)] = p

    for p in filesB:
        _rel, _backend, seed, anchor, wlo, whi = parse_rel_metadata(p, outB)
        mapB[key_for_pair(seed, anchor, wlo, whi)] = p

    if outB2:
        for p in filesB2:
            _rel, _backend, seed, anchor, wlo, whi = parse_rel_metadata(p, outB2)
            mapB2[key_for_pair(seed, anchor, wlo, whi)] = p

    # Optional targeting filters (useful for refinement experiments)
    only_seed = int(args.only_seed)
    only_anchor = int(args.only_anchor)
    only_window = str(args.only_window).strip()
    only_wlo = float("nan")
    only_whi = float("nan")
    if only_window:
        try:
            a, b = only_window.split(":", 1)
            only_wlo = float(a)
            only_whi = float(b)
        except Exception as e:
            raise SystemExit(f"Invalid --only_window='{only_window}' (expected wlo:whi)") from e

    def _keep_key(k: Tuple[int, int, float, float]) -> bool:
        s, a, wlo, whi = k
        if only_seed >= 0 and int(s) != only_seed:
            return False
        if only_anchor >= 0 and int(a) != only_anchor:
            return False
        if only_window and (float(wlo) != float(only_wlo) or float(whi) != float(only_whi)):
            return False
        return True

    if (only_seed >= 0) or (only_anchor >= 0) or bool(only_window):
        mapA = {k: v for k, v in mapA.items() if _keep_key(k)}
        mapB = {k: v for k, v in mapB.items() if _keep_key(k)}
        if outB2:
            mapB2 = {k: v for k, v in mapB2.items() if _keep_key(k)}

    if within_backend == "A":
        outB = outA
        mapB = make_within_backend_anchor_offset_map(mapA, offset=within_anchor_offset)
        keys = sorted(mapA.keys())
    elif within_backend == "B":
        outA = outB
        mapA = dict(mapB)
        mapB = make_within_backend_anchor_offset_map(mapA, offset=within_anchor_offset)
        keys = sorted(mapA.keys())
    else:
        keys = sorted(set(mapA.keys()).intersection(set(mapB.keys())))
        if outB2:
            keys = sorted(set(keys).intersection(set(mapB2.keys())))
        if not keys:
            raise SystemExit("No matching (seed,anchor,window) artifacts between roots")

    if within_backend == "A":
        base_control_tag = f"withinA_anchor_offset{within_anchor_offset}"
    elif within_backend == "B":
        base_control_tag = f"withinB_anchor_offset{within_anchor_offset}"
    else:
        base_control_tag = "matched"

    loop_pairs = parse_loop_pairs(str(args.loop_pairs).strip())

    deform_alpha = float(args.deform_alpha)
    if (not np.isfinite(deform_alpha)) or deform_alpha < 0:
        raise SystemExit("--deform_alpha must be finite and >= 0")

    if str(args.lambdas).strip():
        lambdas = parse_lambdas(str(args.lambdas).strip())
    else:
        nlam = int(max(2, args.lambda_grid))
        lambdas = [float(x) for x in np.linspace(0.0, 1.0, num=nlam, endpoint=True).tolist()]

    lambdas_eff = [float(deform_alpha * float(l)) for l in lambdas]

    # Optional second deformation axis (mu): we run the same (E,lambda) loop suite for each fixed mu slice.
    if outB2:
        if str(args.mus).strip():
            mus = parse_mus(str(args.mus).strip())
        else:
            nmu = int(args.mu_grid)
            if nmu > 1:
                mus = [float(x) for x in np.linspace(0.0, 1.0, num=nmu, endpoint=True).tolist()]
            else:
                mus = [0.0, 0.5, 1.0]
        mus = [float(m) for m in mus]
        for m in mus:
            if (not np.isfinite(m)) or m < 0.0 or m > 1.0:
                raise SystemExit("--mus values must be finite and in [0,1]")
    else:
        mus = [0.0]

    # Adjacent lambda rectangles (small rectangles)
    lambda_pairs: List[Tuple[int, int]] = [(j, j + 1) for j in range(len(lambdas) - 1)]

    suite_mode = str(args.suite_mode).strip()
    if suite_mode == "fwd_rect":
        suite_specs = [LoopSpec(loop_type="rectangle", direction="fwd")]
    else:
        suite_specs = [
            LoopSpec(loop_type="rectangle", direction="fwd"),
            LoopSpec(loop_type="rectangle", direction="rev"),
            LoopSpec(loop_type="rectangle", direction="fwd", double_loop=True),
            LoopSpec(loop_type="degenerate", direction="degenerate", degenerate_axis="E"),
            LoopSpec(loop_type="degenerate", direction="degenerate", degenerate_axis="lambda"),
        ]

    out_csv = os.path.join(out, "phase3e_elambda_suite_rows.csv")
    progress_path = os.path.join(out, "phase3e_elambda_suite_progress.jsonl")

    if (not bool(args.resume)) and os.path.exists(out_csv):
        raise SystemExit(f"Refusing to overwrite existing rows CSV (use --resume or delete): {out_csv}")

    done = _load_done_progress(progress_path) if bool(args.resume) else set()
    if bool(args.resume) and _csv_needs_header(out_csv):
        # Inconsistent state: progress exists but rows CSV is empty/blank (common if interrupted
        # before the first flush). Prefer recomputing rather than skipping everything.
        if done:
            print(
                f"[phase3e_elambda_loop_suite] warning: resume requested but rows CSV is empty/blank; ignoring progress and recomputing tasks: {out_csv}"
            )
        done = _reconstruct_progress_from_rows_csv(out_csv)
    elif bool(args.resume) and (not done) and os.path.exists(out_csv):
        done = _reconstruct_progress_from_rows_csv(out_csv)

    dump_mode = str(args.dump_holonomy)
    dump_dir = os.path.join(out, str(args.dump_holonomy_subdir))
    dump_id_global = _next_dump_id(dump_dir) if bool(args.resume) else 0

    pending_rows: List[Dict[str, Any]] = []
    pending_progress: List[str] = []
    pending_loops = 0

    failed = 0
    loop_id = 0

    # µ-direction non-degeneracy KPIs: computed from endpoint snapshots.
    # Compare the true µ direction (B2-B1) to the λ direction (B1-A).
    mu_cos_lam_med_per_key: List[float] = []
    mu_norm_ratio_med_per_key: List[float] = []
    mu_rank_med_per_key: List[float] = []

    mu_cos_S_med_per_key: List[float] = []
    mu_norm_ratio_S_med_per_key: List[float] = []
    mu_rank_S_med_per_key: List[float] = []

    mu_cos_trace_med_per_key: List[float] = []
    mu_norm_ratio_trace_med_per_key: List[float] = []
    mu_rank_trace_med_per_key: List[float] = []

    mu_cos_logdet_med_per_key: List[float] = []
    mu_norm_ratio_logdet_med_per_key: List[float] = []
    mu_rank_logdet_med_per_key: List[float] = []

    mu_cos_series_med_per_key: List[float] = []
    mu_norm_ratio_series_med_per_key: List[float] = []
    mu_rank_series_med_per_key: List[float] = []

    mu_cos_resid_med_per_key: List[float] = []
    mu_norm_ratio_resid_med_per_key: List[float] = []
    mu_rank_resid_med_per_key: List[float] = []

    for k in keys:
        pA = mapA[k]
        pB = mapB[k]
        pB2 = mapB2[k] if outB2 else ""

        _relA, _backendA, seedA_meta, anchorA_meta, _wloA, _whiA = parse_rel_metadata(pA, outA)
        _relB, _backendB, seedB_meta, anchorB_meta, _wloB, _whiB = parse_rel_metadata(pB, outB)

        dA = np.load(pA)
        dB = np.load(pB)
        dB2 = np.load(pB2) if outB2 else None
        if "lambda_snap" not in dA.files or "lambda_snap" not in dB.files:
            failed += 1
            continue
        if outB2 and (dB2 is not None) and ("lambda_snap" not in dB2.files):
            failed += 1
            continue

        lamA = np.asarray(dA["lambda_snap"], dtype=np.complex128)
        lamB = np.asarray(dB["lambda_snap"], dtype=np.complex128)
        lamB2 = np.asarray(dB2["lambda_snap"], dtype=np.complex128) if (outB2 and (dB2 is not None)) else None
        EA = np.asarray(dA["energies_snap"], dtype=np.float64)
        EB = np.asarray(dB["energies_snap"], dtype=np.float64)

        if lamA.shape != lamB.shape:
            failed += 1
            continue
        if (lamB2 is not None) and (lamA.shape != lamB2.shape):
            failed += 1
            continue
        if EA.shape != EB.shape:
            failed += 1
            continue

        nE = int(lamA.shape[0])
        if nE < 2:
            continue

        # If B2 is provided, validate that (B1-A) and (B2-A) are not nearly colinear.
        do_cos_check = (str(args.direction_cosine_check) == "on") or (str(args.direction_cosine_check) == "auto" and (lamB2 is not None))
        if do_cos_check and (lamB2 is not None):
            nsamp = int(max(2, args.direction_cosine_samples))
            idx = [0, nE - 1]
            if nsamp >= 3:
                idx.append(int(nE // 2))
            if nsamp >= 4:
                idx.append(int(nE // 3))
            if nsamp >= 5:
                idx.append(int((2 * nE) // 3))
            idx = [int(np.clip(int(i), 0, nE - 1)) for i in idx]
            idx = list(dict.fromkeys(idx))[:nsamp]

            cos_vals: List[float] = []
            for ei in idx:
                d1 = np.asarray(lamB[int(ei)], dtype=np.complex128) - np.asarray(lamA[int(ei)], dtype=np.complex128)
                d2 = np.asarray(lamB2[int(ei)], dtype=np.complex128) - np.asarray(lamA[int(ei)], dtype=np.complex128)
                cos_vals.append(direction_cosine_fro(d1, d2))
            cos_arr = np.asarray(cos_vals, dtype=np.float64)
            cos_arr = cos_arr[np.isfinite(cos_arr)]
            if cos_arr.size:
                cos_abs_med = float(np.nanmedian(np.abs(cos_arr)))
                if cos_abs_med > float(args.direction_cosine_tol):
                    raise SystemExit(
                        f"Direction-cosine check failed for key={k}: median |cos|={cos_abs_med:.6g} exceeds tol={float(args.direction_cosine_tol):.6g}"
                    )

        # µ-direction non-degeneracy KPIs: compare true µ direction (B2-B1) against λ direction (B1-A).
        if lamB2 is not None:
            nsamp = int(max(2, args.direction_cosine_samples))
            idx = [0, nE - 1]
            if nsamp >= 3:
                idx.append(int(nE // 2))
            if nsamp >= 4:
                idx.append(int(nE // 3))
            if nsamp >= 5:
                idx.append(int((2 * nE) // 3))
            idx = [int(np.clip(int(i), 0, nE - 1)) for i in idx]
            idx = list(dict.fromkeys(idx))[:nsamp]

            cos_mu_lam: List[float] = []
            norm_ratio: List[float] = []
            ranks: List[int] = []

            has_S = (dB2 is not None) and ("S_snap" in dA.files) and ("S_snap" in dB.files) and ("S_snap" in dB2.files)
            SA = np.asarray(dA["S_snap"], dtype=np.complex128) if has_S else None
            SB = np.asarray(dB["S_snap"], dtype=np.complex128) if has_S else None
            SB2 = np.asarray(dB2["S_snap"], dtype=np.complex128) if has_S else None

            has_trace = (dB2 is not None) and ("trace_powers" in dA.files) and ("trace_powers" in dB.files) and ("trace_powers" in dB2.files)
            trA = np.asarray(dA["trace_powers"], dtype=np.complex128) if has_trace else None
            trB = np.asarray(dB["trace_powers"], dtype=np.complex128) if has_trace else None
            trB2 = np.asarray(dB2["trace_powers"], dtype=np.complex128) if has_trace else None

            has_logdet = (dB2 is not None) and ("logdet_IminusS" in dA.files) and ("logdet_IminusS" in dB.files) and ("logdet_IminusS" in dB2.files)
            ldA = np.asarray(dA["logdet_IminusS"], dtype=np.complex128) if has_logdet else None
            ldB = np.asarray(dB["logdet_IminusS"], dtype=np.complex128) if has_logdet else None
            ldB2 = np.asarray(dB2["logdet_IminusS"], dtype=np.complex128) if has_logdet else None

            has_series = (dB2 is not None) and ("series_partial" in dA.files) and ("series_partial" in dB.files) and ("series_partial" in dB2.files)
            spA = np.asarray(dA["series_partial"], dtype=np.complex128) if has_series else None
            spB = np.asarray(dB["series_partial"], dtype=np.complex128) if has_series else None
            spB2 = np.asarray(dB2["series_partial"], dtype=np.complex128) if has_series else None

            has_resid = (dB2 is not None) and ("resid_partial" in dA.files) and ("resid_partial" in dB.files) and ("resid_partial" in dB2.files)
            rpA = np.asarray(dA["resid_partial"], dtype=np.complex128) if has_resid else None
            rpB = np.asarray(dB["resid_partial"], dtype=np.complex128) if has_resid else None
            rpB2 = np.asarray(dB2["resid_partial"], dtype=np.complex128) if has_resid else None

            cos_mu_lam_S: List[float] = []
            norm_ratio_S: List[float] = []
            ranks_S: List[int] = []

            cos_mu_lam_trace: List[float] = []
            norm_ratio_trace: List[float] = []
            ranks_trace: List[int] = []

            cos_mu_lam_logdet: List[float] = []
            norm_ratio_logdet: List[float] = []
            ranks_logdet: List[int] = []

            cos_mu_lam_series: List[float] = []
            norm_ratio_series: List[float] = []
            ranks_series: List[int] = []

            cos_mu_lam_resid: List[float] = []
            norm_ratio_resid: List[float] = []
            ranks_resid: List[int] = []

            for ei in idx:
                dl = np.asarray(lamB[int(ei)], dtype=np.complex128) - np.asarray(lamA[int(ei)], dtype=np.complex128)
                dm = np.asarray(lamB2[int(ei)], dtype=np.complex128) - np.asarray(lamB[int(ei)], dtype=np.complex128)
                cos_mu_lam.append(direction_cosine_fro(dl, dm))
                nl = float(np.linalg.norm(dl.ravel()))
                nm = float(np.linalg.norm(dm.ravel()))
                if np.isfinite(nl) and nl > 0 and np.isfinite(nm):
                    norm_ratio.append(float(nm / nl))
                ranks.append(effective_rank2(dl, dm))

                if has_S and SA is not None and SB is not None and SB2 is not None:
                    if (SA.shape == SB.shape) and (SA.shape == SB2.shape) and (int(ei) < int(SA.shape[0])):
                        dlS = np.asarray(SB[int(ei)], dtype=np.complex128) - np.asarray(SA[int(ei)], dtype=np.complex128)
                        dmS = np.asarray(SB2[int(ei)], dtype=np.complex128) - np.asarray(SB[int(ei)], dtype=np.complex128)
                        cos_mu_lam_S.append(direction_cosine_fro(dlS, dmS))
                        nlS = float(np.linalg.norm(dlS.ravel()))
                        nmS = float(np.linalg.norm(dmS.ravel()))
                        if np.isfinite(nlS) and nlS > 0 and np.isfinite(nmS):
                            norm_ratio_S.append(float(nmS / nlS))
                        ranks_S.append(effective_rank2(dlS, dmS))

                if has_trace and trA is not None and trB is not None and trB2 is not None:
                    if (trA.shape == trB.shape) and (trA.shape == trB2.shape) and (int(ei) < int(trA.shape[0])):
                        dlT = np.asarray(trB[int(ei)], dtype=np.complex128) - np.asarray(trA[int(ei)], dtype=np.complex128)
                        dmT = np.asarray(trB2[int(ei)], dtype=np.complex128) - np.asarray(trB[int(ei)], dtype=np.complex128)
                        cos_mu_lam_trace.append(direction_cosine_fro(dlT, dmT))
                        nlT = float(np.linalg.norm(dlT.ravel()))
                        nmT = float(np.linalg.norm(dmT.ravel()))
                        if np.isfinite(nlT) and nlT > 0 and np.isfinite(nmT):
                            norm_ratio_trace.append(float(nmT / nlT))
                        ranks_trace.append(effective_rank2(dlT, dmT))

                if has_logdet and ldA is not None and ldB is not None and ldB2 is not None:
                    if (ldA.shape == ldB.shape) and (ldA.shape == ldB2.shape) and (int(ei) < int(ldA.shape[0])):
                        dlL = np.asarray(ldB[int(ei)], dtype=np.complex128) - np.asarray(ldA[int(ei)], dtype=np.complex128)
                        dmL = np.asarray(ldB2[int(ei)], dtype=np.complex128) - np.asarray(ldB[int(ei)], dtype=np.complex128)
                        cos_mu_lam_logdet.append(direction_cosine_fro(dlL, dmL))
                        nlL = float(np.linalg.norm(np.asarray(dlL).ravel()))
                        nmL = float(np.linalg.norm(np.asarray(dmL).ravel()))
                        if np.isfinite(nlL) and nlL > 0 and np.isfinite(nmL):
                            norm_ratio_logdet.append(float(nmL / nlL))
                        ranks_logdet.append(effective_rank2(dlL, dmL))

                if has_series and spA is not None and spB is not None and spB2 is not None:
                    if (spA.shape == spB.shape) and (spA.shape == spB2.shape) and (int(ei) < int(spA.shape[0])):
                        dlP = np.asarray(spB[int(ei)], dtype=np.complex128) - np.asarray(spA[int(ei)], dtype=np.complex128)
                        dmP = np.asarray(spB2[int(ei)], dtype=np.complex128) - np.asarray(spB[int(ei)], dtype=np.complex128)
                        cos_mu_lam_series.append(direction_cosine_fro(dlP, dmP))
                        nlP = float(np.linalg.norm(dlP.ravel()))
                        nmP = float(np.linalg.norm(dmP.ravel()))
                        if np.isfinite(nlP) and nlP > 0 and np.isfinite(nmP):
                            norm_ratio_series.append(float(nmP / nlP))
                        ranks_series.append(effective_rank2(dlP, dmP))

                if has_resid and rpA is not None and rpB is not None and rpB2 is not None:
                    if (rpA.shape == rpB.shape) and (rpA.shape == rpB2.shape) and (int(ei) < int(rpA.shape[0])):
                        dlR = np.asarray(rpB[int(ei)], dtype=np.complex128) - np.asarray(rpA[int(ei)], dtype=np.complex128)
                        dmR = np.asarray(rpB2[int(ei)], dtype=np.complex128) - np.asarray(rpB[int(ei)], dtype=np.complex128)
                        cos_mu_lam_resid.append(direction_cosine_fro(dlR, dmR))
                        nlR = float(np.linalg.norm(dlR.ravel()))
                        nmR = float(np.linalg.norm(dmR.ravel()))
                        if np.isfinite(nlR) and nlR > 0 and np.isfinite(nmR):
                            norm_ratio_resid.append(float(nmR / nlR))
                        ranks_resid.append(effective_rank2(dlR, dmR))

            cos_arr = np.asarray(cos_mu_lam, dtype=np.float64)
            cos_arr = cos_arr[np.isfinite(cos_arr)]
            if cos_arr.size:
                mu_cos_lam_med_per_key.append(float(np.nanmedian(np.abs(cos_arr))))

            ratio_arr = np.asarray(norm_ratio, dtype=np.float64)
            ratio_arr = ratio_arr[np.isfinite(ratio_arr)]
            if ratio_arr.size:
                mu_norm_ratio_med_per_key.append(float(np.nanmedian(ratio_arr)))

            rk_arr = np.asarray(ranks, dtype=np.float64)
            rk_arr = rk_arr[np.isfinite(rk_arr)]
            if rk_arr.size:
                mu_rank_med_per_key.append(float(np.nanmedian(rk_arr)))

            if cos_mu_lam_S:
                cosS = np.asarray(cos_mu_lam_S, dtype=np.float64)
                cosS = cosS[np.isfinite(cosS)]
                if cosS.size:
                    mu_cos_S_med_per_key.append(float(np.nanmedian(np.abs(cosS))))

            if norm_ratio_S:
                rS = np.asarray(norm_ratio_S, dtype=np.float64)
                rS = rS[np.isfinite(rS)]
                if rS.size:
                    mu_norm_ratio_S_med_per_key.append(float(np.nanmedian(rS)))

            if ranks_S:
                rkS = np.asarray(ranks_S, dtype=np.float64)
                rkS = rkS[np.isfinite(rkS)]
                if rkS.size:
                    mu_rank_S_med_per_key.append(float(np.nanmedian(rkS)))

            if cos_mu_lam_trace:
                cT = np.asarray(cos_mu_lam_trace, dtype=np.float64)
                cT = cT[np.isfinite(cT)]
                if cT.size:
                    mu_cos_trace_med_per_key.append(float(np.nanmedian(np.abs(cT))))
            if norm_ratio_trace:
                rT = np.asarray(norm_ratio_trace, dtype=np.float64)
                rT = rT[np.isfinite(rT)]
                if rT.size:
                    mu_norm_ratio_trace_med_per_key.append(float(np.nanmedian(rT)))
            if ranks_trace:
                rkT = np.asarray(ranks_trace, dtype=np.float64)
                rkT = rkT[np.isfinite(rkT)]
                if rkT.size:
                    mu_rank_trace_med_per_key.append(float(np.nanmedian(rkT)))

            if cos_mu_lam_logdet:
                cL = np.asarray(cos_mu_lam_logdet, dtype=np.float64)
                cL = cL[np.isfinite(cL)]
                if cL.size:
                    mu_cos_logdet_med_per_key.append(float(np.nanmedian(np.abs(cL))))
            if norm_ratio_logdet:
                rL = np.asarray(norm_ratio_logdet, dtype=np.float64)
                rL = rL[np.isfinite(rL)]
                if rL.size:
                    mu_norm_ratio_logdet_med_per_key.append(float(np.nanmedian(rL)))
            if ranks_logdet:
                rkL = np.asarray(ranks_logdet, dtype=np.float64)
                rkL = rkL[np.isfinite(rkL)]
                if rkL.size:
                    mu_rank_logdet_med_per_key.append(float(np.nanmedian(rkL)))

            if cos_mu_lam_series:
                cP = np.asarray(cos_mu_lam_series, dtype=np.float64)
                cP = cP[np.isfinite(cP)]
                if cP.size:
                    mu_cos_series_med_per_key.append(float(np.nanmedian(np.abs(cP))))
            if norm_ratio_series:
                rP = np.asarray(norm_ratio_series, dtype=np.float64)
                rP = rP[np.isfinite(rP)]
                if rP.size:
                    mu_norm_ratio_series_med_per_key.append(float(np.nanmedian(rP)))
            if ranks_series:
                rkP = np.asarray(ranks_series, dtype=np.float64)
                rkP = rkP[np.isfinite(rkP)]
                if rkP.size:
                    mu_rank_series_med_per_key.append(float(np.nanmedian(rkP)))

            if cos_mu_lam_resid:
                cR = np.asarray(cos_mu_lam_resid, dtype=np.float64)
                cR = cR[np.isfinite(cR)]
                if cR.size:
                    mu_cos_resid_med_per_key.append(float(np.nanmedian(np.abs(cR))))
            if norm_ratio_resid:
                rR = np.asarray(norm_ratio_resid, dtype=np.float64)
                rR = rR[np.isfinite(rR)]
                if rR.size:
                    mu_norm_ratio_resid_med_per_key.append(float(np.nanmedian(rR)))
            if ranks_resid:
                rkR = np.asarray(ranks_resid, dtype=np.float64)
                rkR = rkR[np.isfinite(rkR)]
                if rkR.size:
                    mu_rank_resid_med_per_key.append(float(np.nanmedian(rkR)))

        etaA = float(dA["cayley_eta"]) if ("cayley_eta" in dA.files) else float("nan")
        etaB = float(dB["cayley_eta"]) if ("cayley_eta" in dB.files) else float("nan")
        if float(args.eta) > 0:
            eta = float(args.eta)
        else:
            eta = float(min(etaA, etaB))
        if (not np.isfinite(eta)) or eta <= 0:
            failed += 1
            continue

        seed, anchor, wlo, whi = k

        for (e0_raw, e1_raw) in loop_pairs:
            i0 = normalize_index(int(e0_raw), nE)
            i1 = normalize_index(int(e1_raw), nE)
            if i0 == i1:
                continue

            i_path = refine_path_indices(i0, i1, nE, refine_steps=int(args.refine_steps))

            for mu in mus:
                control_tag = str(base_control_tag)
                if lamB2 is not None:
                    control_tag = f"{control_tag}|mu={float(mu):.6g}"

                for (j0, j1) in lambda_pairs:
                    prog = {
                        "seed": int(seed),
                        "anchor_seed": int(anchor),
                        "wlo": float(wlo),
                        "whi": float(whi),
                        "loop_e0": int(i0),
                        "loop_e1": int(i1),
                        "loop_refine_steps": int(args.refine_steps),
                        "mu": float(mu),
                        "lambda_j0": int(j0),
                        "lambda_j1": int(j1),
                        "deform_alpha": float(deform_alpha),
                        "blocks": int(args.blocks),
                        "within_backend": str(within_backend),
                        "within_backend_anchor_offset": int(within_anchor_offset),
                    }
                    pk = _progress_key(prog)
                    if bool(args.resume) and (pk in done):
                        loop_id += 1
                        continue

                    # For each rectangle spec, compute and then attach sanity metrics to fwd baseline.
                    fwd_row_index_by_key: Dict[Tuple[int, int], int] = {}
                    fwd_M_by_key: Dict[Tuple[int, int], np.ndarray] = {}
                    fwd_norm_by_key: Dict[Tuple[int, int], float] = {}

                    # refinement consistency baseline (only for 1D blocks): compare coarse E-edge vs refined.
                    Ms_coarse: Optional[List[Optional[np.ndarray]]] = None
                    Ms_refined: Optional[List[Optional[np.ndarray]]] = None
                    if (suite_mode != "fwd_rect") and int(args.refine_steps) > 0:
                        coarse_pts = loop_points_elambda([i0, i1], j0, j1, LoopSpec(loop_type="rectangle", direction="fwd"))
                        refined_pts = loop_points_elambda(i_path, j0, j1, LoopSpec(loop_type="rectangle", direction="fwd"))
                        Ms_coarse, _ = compute_monodromy_elambda(
                            lamA,
                            lamB,
                            lambdas_eff,
                            pts=coarse_pts,
                            eta=float(eta),
                            blocks=int(args.blocks),
                            lamB2=lamB2,
                            mu=float(mu),
                        )
                        Ms_refined, _ = compute_monodromy_elambda(
                            lamA,
                            lamB,
                            lambdas_eff,
                            pts=refined_pts,
                            eta=float(eta),
                            blocks=int(args.blocks),
                            lamB2=lamB2,
                            mu=float(mu),
                        )

                    dump_max_dim = int(args.dump_holonomy_max_dim)
                    dump_tol_pi = float(args.dump_holonomy_tol_pi)

                    for spec in suite_specs:
                        pts = loop_points_elambda(i_path, j0, j1, spec)
                        Ms, diags = compute_monodromy_elambda(
                            lamA,
                            lamB,
                            lambdas_eff,
                            pts=pts,
                            eta=float(eta),
                            blocks=int(args.blocks),
                            lamB2=lamB2,
                            mu=float(mu),
                        )

                        for b, M in enumerate(Ms):
                            diag = diags[b]
                            dim = int(diag["dim"])
                            row: Dict[str, Any] = {
                                "seed": int(seed),
                                "anchor_seed": int(anchor),
                                "seed_a": int(seedA_meta),
                                "anchor_a": int(anchorA_meta),
                                "seed_b": int(seedB_meta),
                                "anchor_b": int(anchorB_meta),
                                "wlo": float(wlo),
                                "whi": float(whi),
                                "artifact_a": os.path.relpath(pA, outA).replace("\\", "/"),
                                "artifact_b": os.path.relpath(pB, outB).replace("\\", "/"),
                                "artifact_b1": os.path.relpath(pB, outB).replace("\\", "/"),
                                "artifact_b2": os.path.relpath(pB2, outB2).replace("\\", "/") if outB2 else "",
                                "within_backend": str(within_backend),
                                "within_backend_anchor_offset": int(within_anchor_offset),
                                "control_tag": str(control_tag),
                                "loop_id": int(loop_id),
                                "loop_e0": int(i0),
                                "loop_e1": int(i1),
                                "loop_refine_steps": int(args.refine_steps),
                                "mu": float(mu),
                                "lambda0": float(lambdas[int(j0)]),
                                "lambda1": float(lambdas[int(j1)]),
                                "lambda_eff0": float(lambdas_eff[int(j0)]),
                                "lambda_eff1": float(lambdas_eff[int(j1)]),
                                "lambda_j0": int(j0),
                                "lambda_j1": int(j1),
                                "deform_alpha": float(deform_alpha),
                                "blocks": int(args.blocks),
                                "block": int(b),
                                "dim": int(dim),
                                "loop_type": str(spec.loop_type),
                                "loop_direction": str(spec.direction),
                                "loop_double": bool(spec.double_loop),
                                "loop_degenerate_axis": str(spec.degenerate_axis),
                                "loop_E0": float(EA[i0]),
                                "loop_E1": float(EA[i1]),
                                "eta": float(eta),
                                "valid": bool(diag["valid"] and (M is not None)),
                                "holonomy_det_phase": float("nan"),
                                "holonomy_norm_IminusM": float("nan"),
                                "holonomy_real_polar_resid": float("nan"),
                                "holonomy_num_eigs_near_minus1": float("nan"),
                                "holonomy_min_dist_to_minus1": float("nan"),
                                "holonomy_unit_defect": float("nan"),
                                "holonomy_imag_fraction": float("nan"),
                                "holonomy_ortho_defect": float("nan"),
                                "holonomy_det_imag_frac": float("nan"),
                                "holonomy_eigphase_meanabs": float("nan"),
                                "holonomy_phi_min": float("nan"),
                                "holonomy_phi_median": float("nan"),
                                "holonomy_phi_max": float("nan"),
                                "holonomy_phi_gap_max": float("nan"),
                                "holonomy_phi_entropy": float("nan"),
                                "holonomy_phi_vec_json": "",
                                "holonomy_dump": "",
                                "step_angle_median": float(diag["step_angle_median"]),
                                "step_angle_worst": float(diag["step_angle_worst"]),
                                "step_overlap_median": float(diag["step_overlap_median"]),
                                "step_smin_median": float(diag["step_smin_median"]),
                                "step_smin_min": float(diag.get("step_smin_min", float("nan"))),
                                "step_smin_p10": float(diag.get("step_smin_p10", float("nan"))),
                                "step_dLambda_median": float(diag["step_dLambda_median"]),
                                "step_dLambda_max": float(diag["step_dLambda_max"]),
                                "step_transport_imag_median": float(diag.get("step_transport_imag_median", float("nan"))),
                                "step_transport_orth_defect_median": float(diag.get("step_transport_orth_defect_median", float("nan"))),
                                "step_transport_unit_defect_median": float(diag.get("step_transport_unit_defect_median", float("nan"))),
                                "step_transport_det_imag_frac_median": float(diag.get("step_transport_det_imag_frac_median", float("nan"))),
                                "step_transport_imag_worst": float(diag.get("step_transport_imag_worst", float("nan"))),
                                "step_transport_orth_defect_worst": float(diag.get("step_transport_orth_defect_worst", float("nan"))),
                                "step_transport_unit_defect_worst": float(diag.get("step_transport_unit_defect_worst", float("nan"))),
                                "step_transport_det_imag_frac_worst": float(diag.get("step_transport_det_imag_frac_worst", float("nan"))),
                                "reverse_consistency": float("nan"),
                                "double_loop_defect": float("nan"),
                                "degenerate_loop_ratio_E": float("nan"),
                                "degenerate_loop_ratio_lambda": float("nan"),
                                "refine_det_phase_defect": float("nan"),
                            }

                        if row["valid"] and (M is not None):
                            det_phase = float(np.angle(np.linalg.det(M)))
                            I = np.eye(M.shape[0], dtype=np.complex128)
                            hol_norm = float(fro_norm(I - M))
                            row["holonomy_det_phase"] = float(det_phase)
                            row["holonomy_norm_IminusM"] = float(hol_norm)
                            row["holonomy_real_polar_resid"] = float(real_polar_resid(M))
                            nminus1, mind = minus1_eig_stats(M)
                            row["holonomy_num_eigs_near_minus1"] = float(nminus1)
                            row["holonomy_min_dist_to_minus1"] = float(mind)
                            row["holonomy_unit_defect"] = float(unit_defect(M))
                            row["holonomy_imag_fraction"] = float(imag_fraction(M))
                            row["holonomy_ortho_defect"] = float(ortho_defect(M))
                            row["holonomy_det_imag_frac"] = float(det_imag_frac(M))
                            row.update(holonomy_spectrum_stats(M))

                            should_dump = False
                            if dump_mode != "off" and int(dim) <= int(dump_max_dim):
                                if dump_mode == "all":
                                    should_dump = True
                                elif dump_mode == "fwd_rect":
                                    should_dump = (
                                        row["loop_type"] == "rectangle"
                                        and row["loop_direction"] == "fwd"
                                        and (row["loop_double"] is False)
                                    )
                                elif dump_mode == "pi_only":
                                    ph = float(angle_wrap_pi(np.asarray([row["holonomy_det_phase"]], dtype=np.float64))[0])
                                    aph = float(abs(ph))
                                    should_dump = bool(abs(aph - math.pi) <= float(dump_tol_pi))

                            if should_dump:
                                meta = {
                                    "seed": int(row["seed"]),
                                    "anchor_seed": int(row["anchor_seed"]),
                                    "wlo": float(row["wlo"]),
                                    "whi": float(row["whi"]),
                                    "loop_id": int(row["loop_id"]),
                                    "loop_e0": int(row["loop_e0"]),
                                    "loop_e1": int(row["loop_e1"]),
                                    "mu": float(row.get("mu", 0.0)),
                                    "lambda_eff0": float(row["lambda_eff0"]),
                                    "lambda_eff1": float(row["lambda_eff1"]),
                                    "lambda_j0": int(row["lambda_j0"]),
                                    "lambda_j1": int(row["lambda_j1"]),
                                    "blocks": int(row["blocks"]),
                                    "block": int(row["block"]),
                                    "dim": int(row["dim"]),
                                    "loop_type": str(row["loop_type"]),
                                    "loop_direction": str(row["loop_direction"]),
                                    "loop_double": bool(row["loop_double"]),
                                    "loop_degenerate_axis": str(row["loop_degenerate_axis"]),
                                    "deform_alpha": float(row["deform_alpha"]),
                                    "eta": float(row["eta"]),
                                    "holonomy_det_phase": float(row["holonomy_det_phase"]),
                                    "holonomy_min_dist_to_minus1": float(row["holonomy_min_dist_to_minus1"]),
                                    "step_smin_min": float(row.get("step_smin_min", float("nan"))),
                                }
                                dump_path = dump_holonomy_npz(
                                    dump_dir,
                                    dump_id=int(dump_id_global),
                                    M=np.asarray(M, dtype=np.complex128),
                                    meta=meta,
                                )
                                row["holonomy_dump"] = os.path.relpath(dump_path, out).replace("\\", "/")
                                dump_id_global += 1

                        # baseline fwd rectangle
                        if (
                            row["valid"]
                            and row["loop_type"] == "rectangle"
                            and row["loop_direction"] == "fwd"
                            and (row["loop_double"] is False)
                            and (M is not None)
                        ):
                            kk = (int(row["block"]), int(row["dim"]))
                            fwd_row_index_by_key[kk] = int(len(pending_rows))
                            fwd_M_by_key[kk] = np.asarray(M, dtype=np.complex128)
                            fwd_norm_by_key[kk] = float(row["holonomy_norm_IminusM"])

                            if (Ms_coarse is not None) and (Ms_refined is not None) and int(dim) == 1:
                                Mc = Ms_coarse[int(b)]
                                Mr = Ms_refined[int(b)]
                                if (Mc is not None) and (Mr is not None):
                                    phc = float(np.angle(np.linalg.det(np.asarray(Mc, dtype=np.complex128))))
                                    phr = float(np.angle(np.linalg.det(np.asarray(Mr, dtype=np.complex128))))
                                    row["refine_det_phase_defect"] = float(angle_distance(phc, phr))

                        # reverse consistency
                        if (
                            row["valid"]
                            and row["loop_type"] == "rectangle"
                            and row["loop_direction"] == "rev"
                            and (row["loop_double"] is False)
                            and (M is not None)
                        ):
                            kk = (int(row["block"]), int(row["dim"]))
                            if kk in fwd_row_index_by_key and kk in fwd_M_by_key:
                                defect = fro_norm(np.asarray(M, dtype=np.complex128) - safe_inv_unitary(fwd_M_by_key[kk]))
                                pending_rows[fwd_row_index_by_key[kk]]["reverse_consistency"] = float(defect)

                        # double loop defect
                        if (
                            row["valid"]
                            and row["loop_type"] == "rectangle"
                            and row["loop_direction"] == "fwd"
                            and (row["loop_double"] is True)
                            and (M is not None)
                        ):
                            kk = (int(row["block"]), int(row["dim"]))
                            if kk in fwd_row_index_by_key and kk in fwd_M_by_key:
                                defect = fro_norm(np.asarray(M, dtype=np.complex128) - (fwd_M_by_key[kk] @ fwd_M_by_key[kk]))
                                pending_rows[fwd_row_index_by_key[kk]]["double_loop_defect"] = float(defect)

                        # degenerate ratios
                        if row["valid"] and row["loop_type"] == "degenerate" and (M is not None):
                            kk = (int(row["block"]), int(row["dim"]))
                            if kk in fwd_row_index_by_key and kk in fwd_norm_by_key:
                                ratio = float(row["holonomy_norm_IminusM"] / (fwd_norm_by_key[kk] + 1e-12))
                                if str(row["loop_degenerate_axis"]) == "E":
                                    pending_rows[fwd_row_index_by_key[kk]]["degenerate_loop_ratio_E"] = float(ratio)
                                if str(row["loop_degenerate_axis"]) == "lambda":
                                    pending_rows[fwd_row_index_by_key[kk]]["degenerate_loop_ratio_lambda"] = float(ratio)

                        pending_rows.append(row)

                    pending_progress.append(pk)
                    pending_loops += 1
                    loop_id += 1

                    flush_every = int(max(1, int(args.flush_every_loops)))
                    if pending_loops >= flush_every:
                        df_chunk = pd.DataFrame(pending_rows)
                        write_header = _csv_needs_header(out_csv)
                        df_chunk.to_csv(out_csv, index=False, mode="a", header=write_header)
                        with open(progress_path, "a", encoding="utf-8") as f:
                            for s in pending_progress:
                                f.write(s + "\n")
                        done.update(pending_progress)
                        pending_rows = []
                        pending_progress = []
                        pending_loops = 0

    if pending_rows:
        df_chunk = pd.DataFrame(pending_rows)
        write_header = _csv_needs_header(out_csv)
        df_chunk.to_csv(out_csv, index=False, mode="a", header=write_header)
        with open(progress_path, "a", encoding="utf-8") as f:
            for s in pending_progress:
                f.write(s + "\n")
        done.update(pending_progress)
        pending_rows = []
        pending_progress = []
        pending_loops = 0

    if not os.path.exists(out_csv):
        raise SystemExit("No outputs were written (no valid rows produced)")

    try:
        df = pd.read_csv(out_csv)
    except pd.errors.EmptyDataError:
        raise SystemExit(
            f"Rows CSV exists but is empty/unparseable (likely interrupted before first flush). Delete it or rerun with --resume after this fix: {out_csv}"
        )

    summary: Dict[str, Any] = {
        "out_root": out,
        "out_root_a": outA,
        "out_root_b": outB,
        "out_root_b2": outB2,
        "within_backend": str(within_backend),
        "within_backend_anchor_offset": int(within_anchor_offset),
        "deform_alpha": float(deform_alpha),
        "mus": [float(m) for m in mus],
        "direction_cosine_check": str(args.direction_cosine_check),
        "direction_cosine_tol": float(args.direction_cosine_tol),
        "direction_cosine_samples": int(args.direction_cosine_samples),
        "n_pairs": int(len(keys)),
        "n_rows": int(len(df)),
        "n_pairs_failed": int(failed),
        "blocks": int(args.blocks),
        "loop_pairs": str(args.loop_pairs),
        "refine_steps": int(args.refine_steps),
        "eta": float(args.eta),
        "suite_mode": str(suite_mode),
        "resume": bool(args.resume),
        "flush_every_loops": int(args.flush_every_loops),
        "lambdas": [float(x) for x in lambdas],
        "lambdas_eff": [float(x) for x in lambdas_eff],
        "lambda_pairs": [(int(a), int(b)) for (a, b) in lambda_pairs],
    }

    summary["mu_direction_kpis"] = {
        "lambda_snap": {
            "n_keys": int(len(mu_cos_lam_med_per_key)),
            "mu_cos_to_lambda_abs_median": float(np.nanmedian(np.asarray(mu_cos_lam_med_per_key, dtype=np.float64)))
            if mu_cos_lam_med_per_key
            else float("nan"),
            "mu_norm_ratio_median": float(np.nanmedian(np.asarray(mu_norm_ratio_med_per_key, dtype=np.float64)))
            if mu_norm_ratio_med_per_key
            else float("nan"),
            "mu_effective_rank_median": float(np.nanmedian(np.asarray(mu_rank_med_per_key, dtype=np.float64)))
            if mu_rank_med_per_key
            else float("nan"),
        },
        "S_snap": {
            "n_keys": int(len(mu_cos_S_med_per_key)),
            "mu_cos_to_lambda_abs_median": float(np.nanmedian(np.asarray(mu_cos_S_med_per_key, dtype=np.float64)))
            if mu_cos_S_med_per_key
            else float("nan"),
            "mu_norm_ratio_median": float(np.nanmedian(np.asarray(mu_norm_ratio_S_med_per_key, dtype=np.float64)))
            if mu_norm_ratio_S_med_per_key
            else float("nan"),
            "mu_effective_rank_median": float(np.nanmedian(np.asarray(mu_rank_S_med_per_key, dtype=np.float64)))
            if mu_rank_S_med_per_key
            else float("nan"),
        },
        "trace_powers": {
            "n_keys": int(len(mu_cos_trace_med_per_key)),
            "mu_cos_to_lambda_abs_median": float(np.nanmedian(np.asarray(mu_cos_trace_med_per_key, dtype=np.float64)))
            if mu_cos_trace_med_per_key
            else float("nan"),
            "mu_norm_ratio_median": float(np.nanmedian(np.asarray(mu_norm_ratio_trace_med_per_key, dtype=np.float64)))
            if mu_norm_ratio_trace_med_per_key
            else float("nan"),
            "mu_effective_rank_median": float(np.nanmedian(np.asarray(mu_rank_trace_med_per_key, dtype=np.float64)))
            if mu_rank_trace_med_per_key
            else float("nan"),
        },
        "logdet_IminusS": {
            "n_keys": int(len(mu_cos_logdet_med_per_key)),
            "mu_cos_to_lambda_abs_median": float(np.nanmedian(np.asarray(mu_cos_logdet_med_per_key, dtype=np.float64)))
            if mu_cos_logdet_med_per_key
            else float("nan"),
            "mu_norm_ratio_median": float(np.nanmedian(np.asarray(mu_norm_ratio_logdet_med_per_key, dtype=np.float64)))
            if mu_norm_ratio_logdet_med_per_key
            else float("nan"),
            "mu_effective_rank_median": float(np.nanmedian(np.asarray(mu_rank_logdet_med_per_key, dtype=np.float64)))
            if mu_rank_logdet_med_per_key
            else float("nan"),
        },
        "series_partial": {
            "n_keys": int(len(mu_cos_series_med_per_key)),
            "mu_cos_to_lambda_abs_median": float(np.nanmedian(np.asarray(mu_cos_series_med_per_key, dtype=np.float64)))
            if mu_cos_series_med_per_key
            else float("nan"),
            "mu_norm_ratio_median": float(np.nanmedian(np.asarray(mu_norm_ratio_series_med_per_key, dtype=np.float64)))
            if mu_norm_ratio_series_med_per_key
            else float("nan"),
            "mu_effective_rank_median": float(np.nanmedian(np.asarray(mu_rank_series_med_per_key, dtype=np.float64)))
            if mu_rank_series_med_per_key
            else float("nan"),
        },
        "resid_partial": {
            "n_keys": int(len(mu_cos_resid_med_per_key)),
            "mu_cos_to_lambda_abs_median": float(np.nanmedian(np.asarray(mu_cos_resid_med_per_key, dtype=np.float64)))
            if mu_cos_resid_med_per_key
            else float("nan"),
            "mu_norm_ratio_median": float(np.nanmedian(np.asarray(mu_norm_ratio_resid_med_per_key, dtype=np.float64)))
            if mu_norm_ratio_resid_med_per_key
            else float("nan"),
            "mu_effective_rank_median": float(np.nanmedian(np.asarray(mu_rank_resid_med_per_key, dtype=np.float64)))
            if mu_rank_resid_med_per_key
            else float("nan"),
        },
    }

    if len(df):
        valid = df[df["valid"] == True]
        summary["valid_fraction"] = float(len(valid) / max(1, len(df)))

        fwd_rect = df[(df["loop_type"] == "rectangle") & (df["loop_direction"] == "fwd") & (df["loop_double"] == False) & (df["valid"] == True)]
        if len(fwd_rect):
            summary["reverse_consistency_median"] = float(np.nanmedian(fwd_rect["reverse_consistency"].values))
            summary["double_loop_defect_median"] = float(np.nanmedian(fwd_rect["double_loop_defect"].values))
            summary["degenerate_ratio_E_median"] = float(np.nanmedian(fwd_rect["degenerate_loop_ratio_E"].values))
            summary["degenerate_ratio_lambda_median"] = float(np.nanmedian(fwd_rect["degenerate_loop_ratio_lambda"].values))
            summary["holonomy_imag_fraction_median"] = float(np.nanmedian(fwd_rect["holonomy_imag_fraction"].values))
            summary["holonomy_ortho_defect_median"] = float(np.nanmedian(fwd_rect["holonomy_ortho_defect"].values))
            summary["holonomy_unit_defect_median"] = float(np.nanmedian(fwd_rect["holonomy_unit_defect"].values))
            summary["holonomy_det_imag_frac_median"] = float(np.nanmedian(fwd_rect["holonomy_det_imag_frac"].values))
            summary["refine_det_phase_defect_median"] = float(np.nanmedian(fwd_rect["refine_det_phase_defect"].values))
            summary["step_smin_median_median"] = float(np.nanmedian(fwd_rect["step_smin_median"].values))
            summary["step_dLambda_max_median"] = float(np.nanmedian(fwd_rect["step_dLambda_max"].values))

            ph = angle_wrap_pi(fwd_rect["holonomy_det_phase"].values.astype(np.float64))
            absph = np.abs(ph)
            tol = 1e-6
            summary["det_phase_frac_near_0"] = float(np.mean(absph < tol))
            summary["det_phase_frac_near_pi"] = float(np.mean(np.abs(absph - np.pi) < tol))

            # Group-ID stratification by det-phase class
            m0 = absph < tol
            mpi = np.abs(absph - np.pi) < tol
            mother = ~(m0 | mpi)

            def summarize(vals: np.ndarray) -> Dict[str, float]:
                v = np.asarray(vals, dtype=np.float64)
                v = v[np.isfinite(v)]
                if v.size == 0:
                    return {"median": float("nan"), "p90": float("nan"), "p95": float("nan"), "max": float("nan")}
                return {
                    "median": float(np.nanmedian(v)),
                    "p90": float(np.nanquantile(v, 0.90)),
                    "p95": float(np.nanquantile(v, 0.95)),
                    "max": float(np.nanmax(v)),
                }

            metrics = {
                "holonomy_imag_fraction": fwd_rect["holonomy_imag_fraction"].values,
                "holonomy_ortho_defect": fwd_rect["holonomy_ortho_defect"].values,
                "holonomy_unit_defect": fwd_rect["holonomy_unit_defect"].values,
                "holonomy_det_imag_frac": fwd_rect["holonomy_det_imag_frac"].values,
                "holonomy_real_polar_resid": fwd_rect["holonomy_real_polar_resid"].values,
                "holonomy_num_eigs_near_minus1": fwd_rect["holonomy_num_eigs_near_minus1"].values,
                "holonomy_min_dist_to_minus1": fwd_rect["holonomy_min_dist_to_minus1"].values,
                "step_transport_imag_median": fwd_rect["step_transport_imag_median"].values,
                "step_transport_orth_defect_median": fwd_rect["step_transport_orth_defect_median"].values,
                "step_transport_unit_defect_median": fwd_rect["step_transport_unit_defect_median"].values,
                "step_transport_det_imag_frac_median": fwd_rect["step_transport_det_imag_frac_median"].values,
                "step_smin_median": fwd_rect["step_smin_median"].values,
                "step_smin_min": fwd_rect["step_smin_min"].values,
                "step_smin_p10": fwd_rect["step_smin_p10"].values,
                "step_dLambda_max": fwd_rect["step_dLambda_max"].values,
            }

            group_id: Dict[str, Any] = {"overall": {}, "near0": {}, "nearpi": {}, "other": {}}
            for name, arr in metrics.items():
                group_id["overall"][name] = summarize(arr)
                group_id["near0"][name] = summarize(np.asarray(arr)[m0])
                group_id["nearpi"][name] = summarize(np.asarray(arr)[mpi])
                group_id["other"][name] = summarize(np.asarray(arr)[mother])
            summary["group_id_kpis"] = group_id

            # Conservative regime label
            med_im = group_id["overall"]["holonomy_imag_fraction"]["median"]
            med_detim = group_id["overall"]["holonomy_det_imag_frac"]["median"]
            med_u = group_id["overall"]["holonomy_unit_defect"]["median"]
            med_o = group_id["overall"]["holonomy_ortho_defect"]["median"]
            if np.isfinite(med_im) and np.isfinite(med_detim) and (med_im <= 1e-12) and (med_detim <= 1e-12) and np.isfinite(med_u) and np.isfinite(med_o):
                if (med_u <= 1e-10) and (med_o <= 1e-10):
                    summary["regime_label"] = "effectively_real"
                else:
                    summary["regime_label"] = "mixed_realish"
            else:
                summary["regime_label"] = "genuinely_complex"

            # Conditional health correlations
            mask_pi = np.abs(absph - np.pi) < tol
            mask_0 = absph < tol
            if np.any(mask_pi):
                summary["pi_step_smin_median"] = float(np.nanmedian(fwd_rect.loc[mask_pi, "step_smin_median"].values))
                summary["pi_step_dLambda_max_median"] = float(np.nanmedian(fwd_rect.loc[mask_pi, "step_dLambda_max"].values))
            if np.any(mask_0):
                summary["zero_step_smin_median"] = float(np.nanmedian(fwd_rect.loc[mask_0, "step_smin_median"].values))
                summary["zero_step_dLambda_max_median"] = float(np.nanmedian(fwd_rect.loc[mask_0, "step_dLambda_max"].values))

            # π-hotspot exports + matched near-0 controls
            try:
                hotspots = fwd_rect.loc[mask_pi].copy().reset_index(drop=True)

                def keep_cols(df_in: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
                    cols2 = [c for c in cols if c in df_in.columns]
                    return df_in.loc[:, cols2].copy()

                base_cols = [
                    "control_tag",
                    "seed",
                    "anchor_seed",
                    "wlo",
                    "whi",
                    "mu",
                    "loop_id",
                    "loop_e0",
                    "loop_e1",
                    "loop_refine_steps",
                    "lambda_j0",
                    "lambda_j1",
                    "lambda0",
                    "lambda1",
                    "blocks",
                    "block",
                    "dim",
                    "loop_E0",
                    "loop_E1",
                    "eta",
                    "holonomy_det_phase",
                    "holonomy_norm_IminusM",
                    "holonomy_num_eigs_near_minus1",
                    "holonomy_min_dist_to_minus1",
                    "holonomy_real_polar_resid",
                    "holonomy_imag_fraction",
                    "holonomy_ortho_defect",
                    "holonomy_unit_defect",
                    "holonomy_det_imag_frac",
                    "step_smin_min",
                    "step_smin_p10",
                    "step_smin_median",
                    "step_dLambda_max",
                    "step_dLambda_median",
                    "step_angle_worst",
                    "step_overlap_median",
                    "step_transport_imag_median",
                    "step_transport_imag_worst",
                    "step_transport_orth_defect_median",
                    "step_transport_orth_defect_worst",
                    "step_transport_unit_defect_median",
                    "step_transport_unit_defect_worst",
                    "reverse_consistency",
                    "double_loop_defect",
                    "degenerate_loop_ratio_E",
                    "degenerate_loop_ratio_lambda",
                    "refine_det_phase_defect",
                    "artifact_a",
                    "artifact_b",
                    "artifact_b1",
                    "artifact_b2",
                ]

                keep_cols(hotspots, base_cols).to_csv(os.path.join(out, "phase3e_elambda_pi_hotspots.csv"), index=False)

                key_cols = [
                    "control_tag",
                    "seed",
                    "anchor_seed",
                    "wlo",
                    "whi",
                    "loop_e0",
                    "loop_e1",
                    "lambda_j0",
                    "lambda_j1",
                    "block",
                    "dim",
                ]
                if "mu" in fwd_rect.columns:
                    key_cols.insert(5, "mu")
                zeros = fwd_rect.loc[mask_0].copy()
                zeros_first = zeros.sort_values(key_cols).drop_duplicates(key_cols, keep="first")

                pi_keys = hotspots[key_cols].copy()
                pi_keys["pi_row_id"] = np.arange(len(pi_keys), dtype=np.int64)

                matched0 = pi_keys.merge(zeros_first, on=key_cols, how="left")
                matched0 = keep_cols(matched0, ["pi_row_id"] + base_cols)
                matched0.to_csv(os.path.join(out, "phase3e_elambda_near0_matched_controls.csv"), index=False)
            except Exception:
                pass

    with open(os.path.join(out, "phase3e_elambda_suite_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Per-loop localization table for (E-pair, lambda-interval)
    if len(df):
        fwd_rect = df[(df["loop_type"] == "rectangle") & (df["loop_direction"] == "fwd") & (df["loop_double"] == False) & (df["valid"] == True)].copy()
        if len(fwd_rect):
            ph = angle_wrap_pi(fwd_rect["holonomy_det_phase"].values.astype(np.float64))
            fwd_rect["det_phase_abs"] = np.abs(ph)
            tol = 1e-6
            by_cols = ["loop_id", "loop_e0", "loop_e1", "lambda0", "lambda1"]
            if "mu" in fwd_rect.columns:
                by_cols.insert(1, "mu")
            by = fwd_rect.groupby(by_cols, dropna=False)
            loop_rows: List[Dict[str, Any]] = []
            for keys0, g in by:
                # keys0 is either (loop_id, loop_e0, loop_e1, lambda0, lambda1)
                # or (loop_id, mu, loop_e0, loop_e1, lambda0, lambda1)
                if isinstance(keys0, tuple) and (len(keys0) == 6):
                    lid, mu0, e0, e1, l0, l1 = keys0
                else:
                    lid, e0, e1, l0, l1 = keys0
                    mu0 = float("nan")
                absph = g["det_phase_abs"].values.astype(np.float64)
                loop_rows.append(
                    {
                        "loop_id": int(lid),
                        "mu": float(mu0),
                        "loop_e0": int(e0),
                        "loop_e1": int(e1),
                        "lambda0": float(l0),
                        "lambda1": float(l1),
                        "n": int(len(g)),
                        "frac_near_0": float(np.mean(absph < tol)) if absph.size else float("nan"),
                        "frac_near_pi": float(np.mean(np.abs(absph - np.pi) < tol)) if absph.size else float("nan"),
                        "step_smin_median": float(np.nanmedian(g["step_smin_median"].values.astype(np.float64))),
                        "step_dLambda_max_median": float(np.nanmedian(g["step_dLambda_max"].values.astype(np.float64))),
                    }
                )
            loop_df = pd.DataFrame(loop_rows).sort_values(["loop_id"]).reset_index(drop=True)
            loop_df.to_csv(os.path.join(out, "phase3e_elambda_loop_quantization.csv"), index=False)

            # Aggregate grid to make (E,lambda) localization scanable.
            gb_cols = ["loop_e0", "loop_e1", "lambda0", "lambda1"]
            if "mu" in loop_df.columns:
                gb_cols.insert(0, "mu")
            agg = (
                loop_df.groupby(gb_cols, dropna=False)
                .agg(
                    mean_frac_near_pi=("frac_near_pi", "mean"),
                    mean_frac_near_0=("frac_near_0", "mean"),
                    median_step_smin=("step_smin_median", "median"),
                    median_step_dLambda_max=("step_dLambda_max_median", "median"),
                    n_loops=("frac_near_pi", "size"),
                )
                .reset_index()
            )
            agg.to_csv(os.path.join(out, "phase3e_elambda_quantization_grid.csv"), index=False)

            if bool(args.plots):
                try:
                    import matplotlib.pyplot as plt

                    # det-phase histogram
                    plt.figure(figsize=(8, 4))
                    plt.hist(angle_wrap_pi(fwd_rect["holonomy_det_phase"].values.astype(np.float64)), bins=80, range=(-np.pi, np.pi), alpha=0.9)
                    plt.xlabel("det phase (rad), principal branch")
                    plt.ylabel("count")
                    plt.title("(E,lambda) interpolation loop det-phase histogram")
                    plt.tight_layout()
                    plt.savefig(os.path.join(out, "phase3e_elambda_det_phase_hist.png"), dpi=160)
                    plt.close()

                    # Scatter: conditioning vs det-phase abs
                    absph = fwd_rect["det_phase_abs"].values.astype(np.float64)
                    smin = fwd_rect["step_smin_median"].values.astype(np.float64)
                    plt.figure(figsize=(7, 5))
                    plt.scatter(smin, absph, s=6, alpha=0.35)
                    plt.axhline(np.pi, color="k", linewidth=1, alpha=0.5)
                    plt.xlabel("step_smin_median")
                    plt.ylabel("|det phase| (rad)")
                    plt.title("(E,lambda) interpolation: |det phase| vs step conditioning")
                    plt.tight_layout()
                    plt.savefig(os.path.join(out, "phase3e_elambda_absphase_vs_smin.png"), dpi=160)
                    plt.close()

                except Exception as e:
                    print(f"[phase3e_elambda_loop_suite] plots skipped due to error: {e}")

    print("[phase3e_elambda_loop_suite] wrote phase3e_elambda_suite_rows.csv and phase3e_elambda_suite_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
