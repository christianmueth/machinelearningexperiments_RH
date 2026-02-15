import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def find_npz_files(out_root: str) -> List[str]:
    hits: List[str] = []
    for root, _dirs, files in os.walk(out_root):
        for fn in files:
            if fn.endswith(".npz") and fn.startswith("channel_diag"):
                hits.append(os.path.join(root, fn))
    return sorted(hits)


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


@dataclass(frozen=True)
class LoopSpec:
    loop_type: str  # rectangle|degenerate
    direction: str  # fwd|rev|degenerate
    double_loop: bool = False
    degenerate_axis: str = ""  # E|geom


def loop_points(i0: int, i1: int, spec: LoopSpec) -> List[Tuple[int, int]]:
    # geom index: 0=backend A, 1=backend B
    P0 = (i0, 0)
    P1 = (i1, 0)
    P2 = (i1, 1)
    P3 = (i0, 1)

    if spec.loop_type == "rectangle" and spec.direction == "fwd" and not spec.double_loop:
        return [P0, P1, P2, P3, P0]
    if spec.loop_type == "rectangle" and spec.direction == "rev" and not spec.double_loop:
        return [P0, P3, P2, P1, P0]
    if spec.loop_type == "rectangle" and spec.direction == "fwd" and spec.double_loop:
        return [P0, P1, P2, P3, P0, P1, P2, P3, P0]
    if spec.loop_type == "degenerate" and spec.degenerate_axis == "E":
        return [P0, P1, P0]
    if spec.loop_type == "degenerate" and spec.degenerate_axis == "geom":
        return [P0, P3, P0]

    raise ValueError(f"unknown loop spec: {spec}")


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
    """How close M is to a real orthogonal matrix (not just unitary).

    Orthogonal condition is M^T M = I (no conjugation).
    """
    M = np.asarray(M, dtype=np.complex128)
    n = int(M.shape[0])
    I = np.eye(n, dtype=np.complex128)
    return float(np.linalg.norm(M.T @ M - I, ord="fro") / (np.linalg.norm(I, ord="fro") + 1e-12))


def unit_defect(M: np.ndarray) -> float:
    """How close M is to unitary: ||M* M - I||_F / ||I||_F."""
    M = np.asarray(M, dtype=np.complex128)
    n = int(M.shape[0])
    I = np.eye(n, dtype=np.complex128)
    return float(np.linalg.norm(np.conjugate(M.T) @ M - I, ord="fro") / (np.linalg.norm(I, ord="fro") + 1e-12))


def det_imag_frac(M: np.ndarray) -> float:
    """|Im(det M)| / |det M|."""
    M = np.asarray(M, dtype=np.complex128)
    d = complex(np.linalg.det(M))
    denom = abs(d) + 1e-18
    return float(abs(d.imag) / denom)


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
    """Parse loop pairs like '0:-1,0:1,2:5' (indices may be negative)."""
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
    """Return E index path from i0 to i1 with optional refinement.

    refine_steps=0 => [i0, i1]
    refine_steps>=1 => insert that many intermediate indices roughly evenly.
    """
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


def loop_points_path(i_path: Sequence[int], spec: LoopSpec) -> List[Tuple[int, int]]:
    """Loop points for a possibly refined E-path.

    i_path: sequence of E indices from start to end (len>=2)
    geom index: 0=backend A, 1=backend B
    """
    i_path = [int(x) for x in list(i_path)]
    if len(i_path) < 2:
        raise ValueError("i_path must have length >= 2")

    start = int(i_path[0])
    end = int(i_path[-1])
    P0 = (start, 0)
    P3 = (start, 1)

    edge0 = [(int(i), 0) for i in i_path]
    edge1 = [(int(i), 1) for i in i_path]

    if spec.loop_type == "rectangle" and spec.direction == "fwd" and not spec.double_loop:
        pts: List[Tuple[int, int]] = []
        pts.extend(edge0)
        pts.append((end, 1))
        pts.extend(list(reversed(edge1))[1:])
        pts.append(P0)
        return pts

    if spec.loop_type == "rectangle" and spec.direction == "rev" and not spec.double_loop:
        pts = []
        pts.append(P0)
        pts.append(P3)
        pts.extend(edge1[1:])
        pts.append((end, 0))
        pts.extend(list(reversed(edge0))[1:])
        return pts

    if spec.loop_type == "rectangle" and spec.direction == "fwd" and spec.double_loop:
        pts1 = loop_points_path(i_path, LoopSpec(loop_type="rectangle", direction="fwd"))
        return pts1 + pts1[1:]

    if spec.loop_type == "degenerate" and spec.degenerate_axis == "E":
        return [(start, 0), (end, 0), (start, 0)]
    if spec.loop_type == "degenerate" and spec.degenerate_axis == "geom":
        return [(start, 0), (start, 1), (start, 0)]

    raise ValueError(f"unknown loop spec: {spec}")


def compute_monodromy_geomloop(
    lamA: np.ndarray,
    lamB: np.ndarray,
    *,
    pts: Sequence[Tuple[int, int]],
    eta: float,
    blocks: int,
) -> Tuple[List[Optional[np.ndarray]], List[Dict[str, Any]]]:
    """Compute per-block holonomy using a loop in (E-index, geom-index)."""

    bases: List[List[np.ndarray]] = []

    # cache bases for repeated points
    basis_cache: Dict[Tuple[int, int], Tuple[List[np.ndarray], List[int], Dict[str, Any]]] = {}

    for (ei, gi) in pts:
        key = (int(ei), int(gi))
        if key in basis_cache:
            Qs, dims, extra = basis_cache[key]
        else:
            L = lamA[int(ei)] if int(gi) == 0 else lamB[int(ei)]
            S = cayley_from_lambda(L, eta=float(eta))
            Qs, dims, extra = basis_blocks_equalcount(S, blocks=int(blocks))
            basis_cache[key] = (Qs, dims, extra)
        bases.append(Qs)

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

            G = np.conjugate(U.T) @ V  # r x r
            s = np.linalg.svd(G, compute_uv=False)
            smin = float(np.min(np.real(s))) if s.size else float("nan")
            step_smin.append(smin)

            W = unitary_polar_factor(G)
            if M is None:
                M = np.eye(W.shape[0], dtype=np.complex128)
            M = M @ W

            # per-edge transport diagnostics (group-ID KPIs)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root_a", required=True)
    ap.add_argument("--out_root_b", required=True)
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
    ap.add_argument("--loop_e0", type=int, default=0, help="legacy single-loop start index (used if --loop_pairs not set)")
    ap.add_argument("--loop_e1", type=int, default=-1, help="legacy single-loop end index (used if --loop_pairs not set)")
    ap.add_argument("--loop_pairs", default="", help="comma list like '0:-1,0:1,2:5' (overrides --loop_e0/--loop_e1)")
    ap.add_argument("--refine_steps", type=int, default=0, help="number of intermediate E indices along E-edges")
    ap.add_argument("--eta", type=float, default=-1.0, help="if >0 use fixed eta; else use min(cayley_eta_a,cayley_eta_b)")
    ap.add_argument(
        "--shuffle_controls",
        type=int,
        default=0,
        help="If >0, run this many randomized pairing controls (permute B within each (anchor,window) group).",
    )
    ap.add_argument("--shuffle_seed", type=int, default=0)
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()

    outA = str(args.out_root_a)
    outB = str(args.out_root_b)
    out = str(args.out_root)
    os.makedirs(out, exist_ok=True)

    within_backend = str(args.within_backend).strip()
    within_anchor_offset = int(args.within_backend_anchor_offset)

    filesA = find_npz_files(outA)
    filesB = find_npz_files(outB)
    if within_backend == "A":
        if not filesA:
            raise SystemExit("No channel_diag*.npz found under out_root_a")
    elif within_backend == "B":
        if not filesB:
            raise SystemExit("No channel_diag*.npz found under out_root_b")
    else:
        if not filesA or not filesB:
            raise SystemExit("No channel_diag*.npz found under one of the roots")

    mapA: Dict[Tuple[int, int, float, float], str] = {}
    mapB: Dict[Tuple[int, int, float, float], str] = {}

    for p in filesA:
        _rel, _backend, seed, anchor, wlo, whi = parse_rel_metadata(p, outA)
        mapA[key_for_pair(seed, anchor, wlo, whi)] = p

    for p in filesB:
        _rel, _backend, seed, anchor, wlo, whi = parse_rel_metadata(p, outB)
        mapB[key_for_pair(seed, anchor, wlo, whi)] = p

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
        if not keys:
            raise SystemExit("No matching (seed,anchor,window) artifacts between roots")

    def group_key_for_shuffle(k: Tuple[int, int, float, float]) -> Tuple[int, float, float]:
        seed, anchor, wlo, whi = k
        return (int(anchor), float(wlo), float(whi))

    def make_shuffled_mapB(rng: np.random.Generator) -> Dict[Tuple[int, int, float, float], str]:
        """Return a permuted version of mapB keyed by the original keys.

        For each group (anchor, wlo, whi), we permute the mapping across seeds.
        This breaks the legacy↔geomv9 correspondence while preserving marginal
        distributions per anchor+window.
        """
        out_map: Dict[Tuple[int, int, float, float], str] = {}
        groups: Dict[Tuple[int, float, float], List[Tuple[int, int, float, float]]] = {}
        for kk in keys:
            groups.setdefault(group_key_for_shuffle(kk), []).append(kk)

        for gk, gkeys in groups.items():
            gkeys_sorted = sorted(gkeys)
            idx = np.arange(len(gkeys_sorted))
            perm = rng.permutation(idx)
            for i, j in zip(idx.tolist(), perm.tolist()):
                out_map[gkeys_sorted[i]] = mapB[gkeys_sorted[j]]

        return out_map

    suite_specs: List[LoopSpec] = [
        LoopSpec(loop_type="rectangle", direction="fwd"),
        LoopSpec(loop_type="rectangle", direction="rev"),
        LoopSpec(loop_type="rectangle", direction="fwd", double_loop=True),
        LoopSpec(loop_type="degenerate", direction="degenerate", degenerate_axis="E"),
        LoopSpec(loop_type="degenerate", direction="degenerate", degenerate_axis="geom"),
    ]

    def run_suite(mappedB: Dict[Tuple[int, int, float, float], str], *, control_tag: str) -> Tuple[pd.DataFrame, int]:
        rows: List[Dict[str, Any]] = []
        failed_local = 0

        loop_pairs: List[Tuple[int, int]]
        if str(args.loop_pairs).strip():
            loop_pairs = parse_loop_pairs(str(args.loop_pairs).strip())
        else:
            loop_pairs = [(int(args.loop_e0), int(args.loop_e1))]

        for k in keys:
            pA = mapA[k]
            pB = mappedB[k]

            _relA, _backendA, seedA_meta, anchorA_meta, _wloA, _whiA = parse_rel_metadata(pA, outA)
            _relB, _backendB, seedB_meta, anchorB_meta, _wloB, _whiB = parse_rel_metadata(pB, outB)

            dA = np.load(pA)
            dB = np.load(pB)
            if "lambda_snap" not in dA.files or "lambda_snap" not in dB.files:
                failed_local += 1
                continue

            lamA = np.asarray(dA["lambda_snap"], dtype=np.complex128)
            lamB = np.asarray(dB["lambda_snap"], dtype=np.complex128)
            EA = np.asarray(dA["energies_snap"], dtype=np.float64)
            EB = np.asarray(dB["energies_snap"], dtype=np.float64)

            if lamA.shape != lamB.shape:
                failed_local += 1
                continue
            if EA.shape != EB.shape:
                failed_local += 1
                continue

            nE = int(lamA.shape[0])
            if nE < 2:
                continue

            etaA = float(dA["cayley_eta"]) if ("cayley_eta" in dA.files) else float("nan")
            etaB = float(dB["cayley_eta"]) if ("cayley_eta" in dB.files) else float("nan")
            if float(args.eta) > 0:
                eta = float(args.eta)
            else:
                eta = float(min(etaA, etaB))
            if (not np.isfinite(eta)) or eta <= 0:
                failed_local += 1
                continue

            seed, anchor, wlo, whi = k

            for loop_id, (e0_raw, e1_raw) in enumerate(loop_pairs):
                i0 = normalize_index(int(e0_raw), nE)
                i1 = normalize_index(int(e1_raw), nE)
                if i0 == i1:
                    continue

                i_path = refine_path_indices(i0, i1, nE, refine_steps=int(args.refine_steps))

                # Precompute optional refinement-consistency baseline for the fwd rectangle.
                Ms_coarse: Optional[List[Optional[np.ndarray]]] = None
                Ms_refined: Optional[List[Optional[np.ndarray]]] = None
                if int(args.refine_steps) > 0:
                    coarse_pts = loop_points_path([i0, i1], LoopSpec(loop_type="rectangle", direction="fwd"))
                    refined_pts = loop_points_path(i_path, LoopSpec(loop_type="rectangle", direction="fwd"))
                    Ms_coarse, _ = compute_monodromy_geomloop(lamA, lamB, pts=coarse_pts, eta=float(eta), blocks=int(args.blocks))
                    Ms_refined, _ = compute_monodromy_geomloop(lamA, lamB, pts=refined_pts, eta=float(eta), blocks=int(args.blocks))

                fwd_row_index_by_key: Dict[Tuple[int, int], int] = {}
                fwd_M_by_key: Dict[Tuple[int, int], np.ndarray] = {}
                fwd_norm_by_key: Dict[Tuple[int, int], float] = {}

                for spec in suite_specs:
                    pts = loop_points_path(i_path, spec)
                    Ms, diags = compute_monodromy_geomloop(
                        lamA,
                        lamB,
                        pts=pts,
                        eta=float(eta),
                        blocks=int(args.blocks),
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
                        "within_backend": str(within_backend),
                        "within_backend_anchor_offset": int(within_anchor_offset),
                        "control_tag": str(control_tag),
                        "loop_id": int(loop_id),
                        "loop_e0": int(i0),
                        "loop_e1": int(i1),
                        "loop_refine_steps": int(args.refine_steps),
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
                        "holonomy_eigphase_meanabs": float("nan"),
                        "holonomy_phi_min": float("nan"),
                        "holonomy_phi_median": float("nan"),
                        "holonomy_phi_max": float("nan"),
                        "holonomy_phi_gap_max": float("nan"),
                        "holonomy_phi_entropy": float("nan"),
                        "holonomy_phi_vec_json": "",
                        "step_angle_median": float(diag["step_angle_median"]),
                        "step_angle_worst": float(diag["step_angle_worst"]),
                        "step_overlap_median": float(diag["step_overlap_median"]),
                        "step_smin_median": float(diag["step_smin_median"]),
                        "step_smin_min": float(diag.get("step_smin_min", float("nan"))),
                        "step_smin_p10": float(diag.get("step_smin_p10", float("nan"))),
                        "holonomy_imag_fraction": float("nan"),
                        "holonomy_ortho_defect": float("nan"),
                        "holonomy_unit_defect": float("nan"),
                        "holonomy_det_imag_frac": float("nan"),
                        "reverse_consistency": float("nan"),
                        "double_loop_defect": float("nan"),
                        "degenerate_loop_ratio_E": float("nan"),
                        "degenerate_loop_ratio_geom": float("nan"),
                        "refine_det_phase_defect": float("nan"),
                        "step_transport_imag_median": float(diag.get("step_transport_imag_median", float("nan"))),
                        "step_transport_orth_defect_median": float(diag.get("step_transport_orth_defect_median", float("nan"))),
                        "step_transport_unit_defect_median": float(diag.get("step_transport_unit_defect_median", float("nan"))),
                        "step_transport_det_imag_frac_median": float(diag.get("step_transport_det_imag_frac_median", float("nan"))),
                        "step_transport_imag_worst": float(diag.get("step_transport_imag_worst", float("nan"))),
                        "step_transport_orth_defect_worst": float(diag.get("step_transport_orth_defect_worst", float("nan"))),
                        "step_transport_unit_defect_worst": float(diag.get("step_transport_unit_defect_worst", float("nan"))),
                        "step_transport_det_imag_frac_worst": float(diag.get("step_transport_det_imag_frac_worst", float("nan"))),
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
                            row["holonomy_imag_fraction"] = float(imag_fraction(M))
                            row["holonomy_ortho_defect"] = float(ortho_defect(M))
                            row["holonomy_unit_defect"] = float(unit_defect(M))
                            row["holonomy_det_imag_frac"] = float(det_imag_frac(M))
                            row.update(holonomy_spectrum_stats(M))

                        # baseline fwd rectangle
                        if (
                            row["valid"]
                            and row["loop_type"] == "rectangle"
                            and row["loop_direction"] == "fwd"
                            and (row["loop_double"] is False)
                            and (M is not None)
                        ):
                            kk = (int(row["block"]), int(row["dim"]))
                            fwd_row_index_by_key[kk] = int(len(rows))
                            fwd_M_by_key[kk] = np.asarray(M, dtype=np.complex128)
                            fwd_norm_by_key[kk] = float(row["holonomy_norm_IminusM"])

                            # refinement consistency (only defined for 1D blocks)
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
                                rows[fwd_row_index_by_key[kk]]["reverse_consistency"] = float(defect)

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
                                rows[fwd_row_index_by_key[kk]]["double_loop_defect"] = float(defect)

                        # degenerate ratios
                        if row["valid"] and row["loop_type"] == "degenerate" and (M is not None):
                            kk = (int(row["block"]), int(row["dim"]))
                            if kk in fwd_row_index_by_key and kk in fwd_norm_by_key:
                                ratio = float(row["holonomy_norm_IminusM"] / (fwd_norm_by_key[kk] + 1e-12))
                                if str(row["loop_degenerate_axis"]) == "E":
                                    rows[fwd_row_index_by_key[kk]]["degenerate_loop_ratio_E"] = float(ratio)
                                if str(row["loop_degenerate_axis"]) == "geom":
                                    rows[fwd_row_index_by_key[kk]]["degenerate_loop_ratio_geom"] = float(ratio)

                        rows.append(row)

        return pd.DataFrame(rows), int(failed_local)

    if within_backend == "A":
        base_control_tag = f"withinA_anchor_offset{within_anchor_offset}"
    elif within_backend == "B":
        base_control_tag = f"withinB_anchor_offset{within_anchor_offset}"
    else:
        base_control_tag = "matched"

    df, failed = run_suite(mapB, control_tag=base_control_tag)
    out_csv = os.path.join(out, "phase3e_geomloop_suite_rows.csv")
    df.to_csv(out_csv, index=False)

    summary: Dict[str, Any] = {
        "out_root": out,
        "out_root_a": outA,
        "out_root_b": outB,
        "within_backend": str(within_backend),
        "within_backend_anchor_offset": int(within_anchor_offset),
        "n_pairs": int(len(keys)),
        "n_rows": int(len(df)),
        "n_pairs_failed": int(failed),
        "blocks": int(args.blocks),
        "loop_e0": int(args.loop_e0),
        "loop_e1": int(args.loop_e1),
        "loop_pairs": str(args.loop_pairs),
        "refine_steps": int(args.refine_steps),
        "eta": float(args.eta),
    }

    if len(df):
        valid = df[df["valid"] == True]
        summary["valid_fraction"] = float(len(valid) / max(1, len(df)))

        fwd_rect = df[(df["loop_type"] == "rectangle") & (df["loop_direction"] == "fwd") & (df["loop_double"] == False) & (df["valid"] == True)]
        if len(fwd_rect):
            summary["holonomy_norm_IminusM_median"] = float(np.nanmedian(fwd_rect["holonomy_norm_IminusM"].values))
            summary["reverse_consistency_median"] = float(np.nanmedian(fwd_rect["reverse_consistency"].values))
            summary["double_loop_defect_median"] = float(np.nanmedian(fwd_rect["double_loop_defect"].values))
            summary["degenerate_ratio_E_median"] = float(np.nanmedian(fwd_rect["degenerate_loop_ratio_E"].values))
            summary["degenerate_ratio_geom_median"] = float(np.nanmedian(fwd_rect["degenerate_loop_ratio_geom"].values))
            summary["holonomy_imag_fraction_median"] = float(np.nanmedian(fwd_rect["holonomy_imag_fraction"].values))
            summary["holonomy_ortho_defect_median"] = float(np.nanmedian(fwd_rect["holonomy_ortho_defect"].values))
            summary["holonomy_unit_defect_median"] = float(np.nanmedian(fwd_rect["holonomy_unit_defect"].values))
            summary["holonomy_det_imag_frac_median"] = float(np.nanmedian(fwd_rect["holonomy_det_imag_frac"].values))
            summary["refine_det_phase_defect_median"] = float(np.nanmedian(fwd_rect["refine_det_phase_defect"].values))

            # Group-ID stratification by det-phase class
            ph = angle_wrap_pi(fwd_rect["holonomy_det_phase"].values.astype(np.float64))
            absph = np.abs(ph)
            tol = 1e-6
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

            # Quantization diagnostics for det-phase (in 1D blocks, det phase is the Berry phase).
            # (ph/absph/tol already computed above)
            summary["det_phase_abs_median"] = float(np.nanmedian(absph))
            summary["det_phase_abs_p90"] = float(np.nanquantile(absph, 0.90))
            summary["det_phase_abs_p95"] = float(np.nanquantile(absph, 0.95))
            summary["det_phase_abs_max"] = float(np.nanmax(absph))
            summary["det_phase_frac_near_0"] = float(np.mean(absph < tol))
            summary["det_phase_frac_near_pi"] = float(np.mean(np.abs(absph - np.pi) < tol))
            summary["det_phase_quantization_tol"] = float(tol)

            # π-hotspot exports + matched near-0 controls
            try:
                mask_pi = np.abs(absph - np.pi) < tol
                mask_0 = absph < tol

                hotspots = fwd_rect.loc[mask_pi].copy()
                hotspots = hotspots.reset_index(drop=True)

                def keep_cols(df_in: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
                    cols2 = [c for c in cols if c in df_in.columns]
                    return df_in.loc[:, cols2].copy()

                base_cols = [
                    "control_tag",
                    "seed",
                    "anchor_seed",
                    "wlo",
                    "whi",
                    "loop_id",
                    "loop_e0",
                    "loop_e1",
                    "loop_refine_steps",
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
                    "degenerate_loop_ratio_geom",
                    "refine_det_phase_defect",
                    "artifact_a",
                    "artifact_b",
                ]

                keep_cols(hotspots, base_cols).to_csv(os.path.join(out, "phase3e_geomloop_pi_hotspots.csv"), index=False)

                key_cols = ["control_tag", "seed", "anchor_seed", "wlo", "whi", "loop_e0", "loop_e1", "block", "dim"]
                zeros = fwd_rect.loc[mask_0].copy()
                zeros_first = zeros.sort_values(key_cols).drop_duplicates(key_cols, keep="first")

                pi_keys = hotspots[key_cols].copy()
                pi_keys["pi_row_id"] = np.arange(len(pi_keys), dtype=np.int64)

                matched0 = pi_keys.merge(zeros_first, on=key_cols, how="left", suffixes=("", ""))
                matched0 = keep_cols(matched0, ["pi_row_id"] + base_cols)
                matched0.to_csv(os.path.join(out, "phase3e_geomloop_near0_matched_controls.csv"), index=False)
            except Exception:
                pass

    with open(os.path.join(out, "phase3e_geomloop_suite_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[phase3e_geomloop_suite] wrote phase3e_geomloop_suite_rows.csv and phase3e_geomloop_suite_summary.json")
    print(json.dumps(summary, indent=2))

    if bool(args.plots):
        try:
            import matplotlib.pyplot as plt

            fwd_rect = df[(df["loop_type"] == "rectangle") & (df["loop_direction"] == "fwd") & (df["loop_double"] == False) & (df["valid"] == True)].copy()
            if len(fwd_rect):
                plt.figure(figsize=(8, 5))
                plt.hist(fwd_rect["holonomy_norm_IminusM"].values, bins=50, alpha=0.9)
                plt.xlabel("||I - M||_F")
                plt.ylabel("count")
                plt.title("Geom-loop holonomy strength (legacy vs geom_v9 rectangle)")
                plt.tight_layout()
                plt.savefig(os.path.join(out, "phase3e_geomloop_strength_hist.png"), dpi=160)
                plt.close()

                vals = fwd_rect["reverse_consistency"].values
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    plt.figure(figsize=(7, 4))
                    plt.hist(vals, bins=50, alpha=0.9)
                    plt.xlabel("reverse_consistency = ||M_rev - inv(M_fwd)||_F")
                    plt.ylabel("count")
                    plt.title("Geom-loop reverse consistency")
                    plt.tight_layout()
                    plt.savefig(os.path.join(out, "phase3e_geomloop_reverse_consistency_hist.png"), dpi=160)
                    plt.close()

                dfa = fwd_rect.copy()
                dfa = dfa[dfa["holonomy_phi_vec_json"].astype(str).str.len() > 0]
                if len(dfa):
                    plt.figure(figsize=(8, 5))
                    anchors = sorted(dfa["anchor_seed"].unique().tolist())
                    anchors = anchors[:8]
                    for a in anchors:
                        g = dfa[dfa["anchor_seed"] == a]
                        phis: List[float] = []
                        for s in g["holonomy_phi_vec_json"].tolist():
                            try:
                                v = json.loads(s)
                                phis.extend([abs(float(x)) for x in v])
                            except Exception:
                                pass
                        if len(phis) < 5:
                            continue
                        x = np.sort(np.asarray(phis, dtype=np.float64))
                        y = np.linspace(0.0, 1.0, x.size, endpoint=True)
                        plt.plot(x, y, linewidth=1.2, alpha=0.8, label=f"anchor {a}")
                    plt.xlabel("|holonomy eigenphase| (rad)")
                    plt.ylabel("empirical CDF")
                    plt.title("Geom-loop holonomy eigenphase magnitude CDF (subset anchors)")
                    plt.legend(fontsize=8)
                    plt.tight_layout()
                    plt.savefig(os.path.join(out, "phase3e_geomloop_phi_cdf_by_anchor.png"), dpi=160)
                    plt.close()

                # det-phase histogram (highly diagnostic for 1D blocks)
                ph = fwd_rect["holonomy_det_phase"].values.astype(np.float64)
                ph = angle_wrap_pi(ph)
                plt.figure(figsize=(8, 4))
                plt.hist(ph, bins=80, range=(-np.pi, np.pi), alpha=0.9)
                plt.xlabel("det phase (rad), principal branch")
                plt.ylabel("count")
                plt.title("Geom-loop holonomy det-phase histogram")
                plt.tight_layout()
                plt.savefig(os.path.join(out, "phase3e_geomloop_det_phase_hist.png"), dpi=160)
                plt.close()

                # Per-loop quantization summary (A1 loop randomization)
                tol = 1e-6
                g = fwd_rect.copy()
                ph_all = angle_wrap_pi(g["holonomy_det_phase"].values.astype(np.float64))
                g["det_phase_abs"] = np.abs(ph_all)
                by = g.groupby(["loop_id", "loop_e0", "loop_e1"], dropna=False)
                loop_rows: List[Dict[str, Any]] = []
                for (lid, le0, le1), gg in by:
                    absph = gg["det_phase_abs"].values.astype(np.float64)
                    loop_rows.append(
                        {
                            "loop_id": int(lid),
                            "loop_e0": int(le0),
                            "loop_e1": int(le1),
                            "n": int(len(gg)),
                            "frac_near_0": float(np.mean(absph < tol)) if absph.size else float("nan"),
                            "frac_near_pi": float(np.mean(np.abs(absph - np.pi) < tol)) if absph.size else float("nan"),
                            "abs_median": float(np.nanmedian(absph)) if absph.size else float("nan"),
                            "refine_defect_median": float(np.nanmedian(gg["refine_det_phase_defect"].values.astype(np.float64))),
                        }
                    )
                loop_df = pd.DataFrame(loop_rows).sort_values(["loop_id"]).reset_index(drop=True)
                loop_df.to_csv(os.path.join(out, "phase3e_geomloop_loop_quantization.csv"), index=False)

                if len(loop_df):
                    plt.figure(figsize=(max(7, 0.45 * len(loop_df)), 4))
                    plt.bar(np.arange(len(loop_df)), loop_df["frac_near_pi"].values.astype(np.float64), alpha=0.9)
                    plt.xticks(
                        np.arange(len(loop_df)),
                        [f"{int(a)}:{int(b)}" for a, b in zip(loop_df["loop_e0"].tolist(), loop_df["loop_e1"].tolist())],
                        rotation=45,
                        ha="right",
                        fontsize=8,
                    )
                    plt.ylabel("frac det-phase near pi")
                    plt.title("Geom-loop Z2 rate by (E0,E1) loop")
                    plt.tight_layout()
                    plt.savefig(os.path.join(out, "phase3e_geomloop_frac_near_pi_by_loop.png"), dpi=160)
                    plt.close()

        except Exception as e:
            print(f"[phase3e_geomloop_suite] plots skipped due to error: {e}")

    # Optional randomized pairing controls
    if int(args.shuffle_controls) > 0:
        rng = np.random.default_rng(int(args.shuffle_seed))
        ctrl_summ: List[Dict[str, Any]] = []
        ctrl_rows: List[pd.DataFrame] = []

        for r in range(int(args.shuffle_controls)):
            mappedB = make_shuffled_mapB(rng)
            dfc, failed_c = run_suite(mappedB, control_tag=f"shuffle{r}")
            ctrl_rows.append(dfc)

            fwd = dfc[(dfc["loop_type"] == "rectangle") & (dfc["loop_direction"] == "fwd") & (dfc["loop_double"] == False) & (dfc["valid"] == True)].copy()
            ph = fwd["holonomy_det_phase"].values.astype(np.float64)
            ph = angle_wrap_pi(ph)
            absph = np.abs(ph)
            tol = 1e-6
            ctrl_summ.append(
                {
                    "control_tag": f"shuffle{r}",
                    "n_rows": int(len(dfc)),
                    "n_failed_pairs": int(failed_c),
                    "det_phase_frac_near_0": float(np.mean(absph < tol)) if absph.size else float("nan"),
                    "det_phase_frac_near_pi": float(np.mean(np.abs(absph - np.pi) < tol)) if absph.size else float("nan"),
                    "det_phase_abs_median": float(np.nanmedian(absph)) if absph.size else float("nan"),
                }
            )

        df_ctrl = pd.concat(ctrl_rows, axis=0, ignore_index=True)
        df_ctrl.to_csv(os.path.join(out, "phase3e_geomloop_suite_rows_shuffle_controls.csv"), index=False)

        ctrl_df = pd.DataFrame(ctrl_summ)
        ctrl_df.to_csv(os.path.join(out, "phase3e_geomloop_shuffle_controls_summary.csv"), index=False)

        # aggregate summary
        agg = {
            "shuffle_controls": int(args.shuffle_controls),
            "shuffle_seed": int(args.shuffle_seed),
            "det_phase_frac_near_pi_mean": float(np.nanmean(ctrl_df["det_phase_frac_near_pi"].values)) if len(ctrl_df) else float("nan"),
            "det_phase_frac_near_pi_p95": float(np.nanquantile(ctrl_df["det_phase_frac_near_pi"].values, 0.95)) if len(ctrl_df) else float("nan"),
            "det_phase_frac_near_0_mean": float(np.nanmean(ctrl_df["det_phase_frac_near_0"].values)) if len(ctrl_df) else float("nan"),
        }
        with open(os.path.join(out, "phase3e_geomloop_shuffle_controls_summary.json"), "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2)

        if bool(args.plots):
            try:
                import matplotlib.pyplot as plt

                vals = ctrl_df["det_phase_frac_near_pi"].values.astype(np.float64)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    plt.figure(figsize=(7, 4))
                    plt.hist(vals, bins=30, alpha=0.9)
                    plt.xlabel("shuffle control: frac det-phase near pi")
                    plt.ylabel("count")
                    plt.title("Randomized pairing control distribution")
                    plt.tight_layout()
                    plt.savefig(os.path.join(out, "phase3e_geomloop_shuffle_frac_near_pi_hist.png"), dpi=160)
                    plt.close()
            except Exception as e:
                print(f"[phase3e_geomloop_suite] shuffle control plot skipped: {e}")


if __name__ == "__main__":
    main()
