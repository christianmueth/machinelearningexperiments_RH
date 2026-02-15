import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


def unitary_procrustes(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Find unitary U minimizing ||A - U B||_F (complex Procrustes)."""
    A = np.asarray(A, dtype=np.complex128)
    B = np.asarray(B, dtype=np.complex128)
    M = A @ np.conjugate(B.T)
    U, _s, Vh = np.linalg.svd(M)
    return U @ Vh


def unitary_polar_factor(G: np.ndarray) -> np.ndarray:
    """Return unitary polar factor of a square matrix G via SVD.

    For overlap matrix G = U^* V (r x r), the unitary W = polar(G) gives the
    closest unitary in Frobenius norm and is the standard discrete connection
    used in Kato/Procrustes-style eigenbundle transport.
    """
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
    """Gaps between consecutive sorted angles in [0,2pi), including wrap gap."""
    a = np.asarray(sorted_angles, dtype=np.float64)
    if a.size == 0:
        return np.asarray([], dtype=np.float64)
    diffs = np.diff(a)
    wrap = (2.0 * math.pi) - (a[-1] - a[0])
    return np.concatenate([diffs, np.asarray([wrap], dtype=np.float64)])


def gapcuts_partition(
    angles: np.ndarray,
    *,
    max_blocks: int,
    q: float,
    min_gap_rad: float,
    rng: Optional[np.random.Generator] = None,
    jitter_sigma: float = 0.0,
) -> List[np.ndarray]:
    """Partition indices on circle by cutting at 'large' phase gaps.

    Returns a list of index arrays, each contiguous along the sorted circle.
    """
    ang = np.asarray(angles, dtype=np.float64)
    n = int(ang.size)
    if n == 0:
        return []

    if jitter_sigma > 0.0:
        if rng is None:
            rng = np.random.default_rng(0)
        ang = np.mod(ang + rng.normal(loc=0.0, scale=float(jitter_sigma), size=n), 2.0 * math.pi)

    idx_sorted = np.argsort(ang)
    a_sorted = ang[idx_sorted]
    gaps = circular_gaps_sorted(a_sorted)
    if gaps.size == 0:
        return [idx_sorted]

    thresh = float(np.quantile(gaps, float(q)))
    thresh = max(thresh, float(min_gap_rad))
    cut_positions = np.where(gaps >= thresh)[0]

    # Each cut at position k means we cut between a_sorted[k] and a_sorted[k+1] (mod n)
    # cap number of blocks
    max_cuts = int(max(0, min(int(max_blocks) - 1, int(cut_positions.size))))
    if max_cuts == 0:
        return [idx_sorted]

    # keep the largest gaps if too many
    if int(cut_positions.size) > max_cuts:
        order = np.argsort(-gaps[cut_positions])
        cut_positions = cut_positions[order[:max_cuts]]

    # sort cut positions along the circle
    cut_positions = np.sort(cut_positions)

    # Build segments; treat wrap by starting after the last cut
    start = int((cut_positions[-1] + 1) % n)
    segments: List[np.ndarray] = []
    cur = start
    for cp in list(cut_positions) + [cut_positions[-1]]:
        end = int((cp + 1) % n)
        if cur == end:
            continue
        if cur < end:
            seg = idx_sorted[cur:end]
        else:
            seg = np.concatenate([idx_sorted[cur:], idx_sorted[:end]])
        segments.append(seg)
        cur = end

    # segments count should equal number of cuts, but above loop makes it = cuts+?; fix
    # Rebuild correctly: traverse from start, stopping at each cut.
    segments = []
    cur = start
    for cp in cut_positions:
        end = int((cp + 1) % n)
        if cur < end:
            seg = idx_sorted[cur:end]
        else:
            seg = np.concatenate([idx_sorted[cur:], idx_sorted[:end]])
        segments.append(seg)
        cur = end
    # final segment from last end back to start
    if cur < start:
        seg = idx_sorted[cur:start]
    elif cur > start:
        seg = np.concatenate([idx_sorted[cur:], idx_sorted[:start]])
    else:
        seg = np.asarray([], dtype=np.int64)
    if seg.size:
        segments.append(seg)

    # filter empties
    segments = [np.asarray(s, dtype=np.int64) for s in segments if int(np.asarray(s).size) > 0]

    # Ensure at least 2 blocks
    if len(segments) < 2:
        return [idx_sorted]

    return segments


def basis_blocks_from_S(
    S: np.ndarray,
    *,
    mode: str,
    blocks: int,
    gapcuts_max_blocks: int,
    gapcuts_q: float,
    gapcuts_min_gap_rad: float,
    rng: Optional[np.random.Generator] = None,
    gapcuts_jitter_sigma: float = 0.0,
) -> Tuple[List[np.ndarray], List[int], Dict[str, Any]]:
    """Return list of orthonormal bases Q[b] that partition eigenvectors."""
    S = np.asarray(S, dtype=np.complex128)
    eigvals, eigvecs = np.linalg.eig(S)
    angles = np.mod(np.angle(eigvals), 2.0 * math.pi)
    n = int(angles.size)

    extra: Dict[str, Any] = {"n": n}

    if str(mode) == "phasebands":
        B = int(max(1, min(int(blocks), n)))
        band_w = 2.0 * math.pi / B
        labels = np.floor(angles / band_w).astype(int)
        labels = np.clip(labels, 0, B - 1)
        order = [np.where(labels == b)[0] for b in range(B)]
        extra.update({"effective_blocks": B})
    elif str(mode) == "gapcuts":
        segs = gapcuts_partition(
            angles,
            max_blocks=int(gapcuts_max_blocks),
            q=float(gapcuts_q),
            min_gap_rad=float(gapcuts_min_gap_rad),
            rng=rng,
            jitter_sigma=float(gapcuts_jitter_sigma),
        )
        order = segs
        extra.update({"effective_blocks": int(len(order))})
    else:
        # equalcount
        B = int(max(1, min(int(blocks), n)))
        idx_sorted = np.argsort(angles)
        sizes = [n // B for _ in range(B)]
        for i in range(n - sum(sizes)):
            sizes[i] += 1
        order = []
        start = 0
        for b in range(B):
            end = start + sizes[b]
            order.append(idx_sorted[start:end])
            start = end
        extra.update({"effective_blocks": B})

    Qs: List[np.ndarray] = []
    dims: List[int] = []
    for idx in order:
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            Qs.append(np.zeros((n, 0), dtype=np.complex128))
            dims.append(0)
            continue
        V = eigvecs[:, idx]
        Q, _ = np.linalg.qr(V)
        Q = np.asarray(Q, dtype=np.complex128)
        Qs.append(Q)
        dims.append(int(Q.shape[1]))

    # canonicalize by mean angle of eigenvalues inside the block (for stable labeling)
    # (this does not change subspaces, just ordering of blocks)
    block_centers: List[float] = []
    for idx in order:
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            block_centers.append(float("nan"))
            continue
        z = np.mean(np.exp(1j * angles[idx]))
        block_centers.append(float(np.mod(np.angle(z), 2.0 * math.pi)))
    perm = np.argsort(np.nan_to_num(np.asarray(block_centers), nan=10.0 * math.pi))
    Qs = [Qs[int(i)] for i in perm]
    dims = [dims[int(i)] for i in perm]
    extra["block_center_angles"] = [block_centers[int(i)] for i in perm]

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


@dataclass(frozen=True)
class LoopSpec:
    loop_type: str
    eta_scale: float
    direction: str  # 'fwd' | 'rev' | 'degenerate'
    double_loop: bool = False
    degenerate_axis: str = ""  # 'E' | 'eta'


def loop_points(i0: int, i1: int, eta1: float, eta2: float, spec: LoopSpec) -> List[Tuple[int, float]]:
    P0 = (i0, float(eta1))
    P1 = (i1, float(eta1))
    P2 = (i1, float(eta2))
    P3 = (i0, float(eta2))

    if spec.loop_type == "rectangle" and spec.direction == "fwd":
        pts = [P0, P1, P2, P3, P0]
    elif spec.loop_type == "rectangle" and spec.direction == "rev":
        pts = [P0, P3, P2, P1, P0]
    elif spec.loop_type == "rectangle" and spec.double_loop:
        pts = [P0, P1, P2, P3, P0, P1, P2, P3, P0]
    elif spec.loop_type == "degenerate" and spec.degenerate_axis == "E":
        pts = [P0, P1, P0]
    elif spec.loop_type == "degenerate" and spec.degenerate_axis == "eta":
        pts = [P0, P3, P0]
    else:
        raise ValueError(f"unknown loop spec: {spec}")

    return pts


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
    gaps = circular_gaps_sorted(np.mod(phi + math.pi, 2.0 * math.pi))  # map to [0,2pi)
    gap_phi_max = float(np.max(gaps)) if gaps.size else float("nan")

    # entropy via histogram
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


def compute_monodromy(
    lam_snap: np.ndarray,
    *,
    pts: Sequence[Tuple[int, float]],
    block_mode: str,
    blocks: int,
    gapcuts_max_blocks: int,
    gapcuts_q: float,
    gapcuts_min_gap_rad: float,
    rng: Optional[np.random.Generator] = None,
    gapcuts_jitter_sigma: float = 0.0,
) -> Tuple[List[Optional[np.ndarray]], List[Dict[str, Any]]]:
    """Return per-block holonomy matrices M[b] (or None if invalid) and per-block step diagnostics."""
    bases: List[List[np.ndarray]] = []
    blockmeta: List[Dict[str, Any]] = []

    # Critical loop-sanity detail: if a point repeats in the loop (e.g. closure P4=P0,
    # or degenerate out-and-back), reuse the same basis instead of recomputing it.
    # Recomputing injects an arbitrary gauge choice and can create spurious holonomy.
    basis_cache: Dict[Tuple[int, float], Tuple[List[np.ndarray], List[int], Dict[str, Any]]] = {}

    for (ei, eta) in pts:
        ei_i = int(ei)
        eta_f = float(eta)
        key = (ei_i, float(np.round(eta_f, 12)))
        if key in basis_cache:
            Qs, dims, extra = basis_cache[key]
        else:
            S = cayley_from_lambda(lam_snap[ei_i], eta=float(eta_f))
            Qs, dims, extra = basis_blocks_from_S(
                S,
                mode=str(block_mode),
                blocks=int(blocks),
                gapcuts_max_blocks=int(gapcuts_max_blocks),
                gapcuts_q=float(gapcuts_q),
                gapcuts_min_gap_rad=float(gapcuts_min_gap_rad),
                rng=rng,
                gapcuts_jitter_sigma=float(gapcuts_jitter_sigma),
            )
            basis_cache[key] = (Qs, dims, extra)
        bases.append(Qs)
        blockmeta.append({"dims": dims, **extra})

    B = int(len(bases[0]))
    Ms: List[Optional[np.ndarray]] = [None for _ in range(B)]
    diag_per_block: List[Dict[str, Any]] = []

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

            # Discrete connection in block coordinates: W = polar(U^* V) (r x r)
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

        diag_per_block.append(
            {
                "valid": bool(valid and (dim > 0) and (M is not None)),
                "dim": int(dim),
                "step_angle_median": float(np.nanmedian(step_angles)) if step_angles else float("nan"),
                "step_angle_worst": float(np.nanmax(step_angles)) if step_angles else float("nan"),
                "step_overlap_median": float(np.nanmedian(step_overlaps)) if step_overlaps else float("nan"),
                "step_smin_median": float(np.nanmedian(step_smin)) if step_smin else float("nan"),
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

    return Ms, diag_per_block


def fro_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x, ord="fro"))


def imag_fraction(M: np.ndarray) -> float:
    M = np.asarray(M, dtype=np.complex128)
    denom = float(np.linalg.norm(M, ord="fro")) + 1e-12
    return float(np.linalg.norm(np.imag(M), ord="fro") / denom)


def ortho_defect(M: np.ndarray) -> float:
    """||M^T M - I||_F / ||I||_F (orthogonality defect)."""
    M = np.asarray(M, dtype=np.complex128)
    n = int(M.shape[0])
    I = np.eye(n, dtype=np.complex128)
    return float(np.linalg.norm(M.T @ M - I, ord="fro") / (np.linalg.norm(I, ord="fro") + 1e-12))


def unit_defect(M: np.ndarray) -> float:
    """||M* M - I||_F / ||I||_F (unitarity defect)."""
    M = np.asarray(M, dtype=np.complex128)
    n = int(M.shape[0])
    I = np.eye(n, dtype=np.complex128)
    return float(np.linalg.norm(np.conjugate(M.T) @ M - I, ord="fro") / (np.linalg.norm(I, ord="fro") + 1e-12))


def det_imag_frac(M: np.ndarray) -> float:
    d = complex(np.linalg.det(np.asarray(M, dtype=np.complex128)))
    return float(abs(d.imag) / (abs(d) + 1e-18))


def angle_wrap_pi(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return ((x + np.pi) % (2.0 * np.pi)) - np.pi


def safe_inv_unitary(M: np.ndarray) -> np.ndarray:
    # for unitary-ish matrices, inv ~ adjoint
    return np.conjugate(M.T)


def area_proxy(E0: float, E1: float, eta1: float, eta2: float) -> float:
    return float(abs(float(E1) - float(E0)) * abs(float(eta2) - float(eta1)))


def parse_floats_csv(s: str) -> List[float]:
    out: List[float] = []
    for p in str(s).split(","):
        p = p.strip()
        if not p:
            continue
        out.append(float(p))
    return out


def parse_str_csv(s: str) -> List[str]:
    out: List[str] = []
    for p in str(s).split(","):
        p = p.strip()
        if not p:
            continue
        out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--block_modes", default="equalcount,gapcuts", help="comma list: equalcount,phasebands,gapcuts")
    ap.add_argument("--blocks", type=int, default=4, help="for equalcount/phasebands")
    ap.add_argument("--gapcuts_max_blocks", type=int, default=6)
    ap.add_argument("--gapcuts_q", type=float, default=0.90)
    ap.add_argument("--gapcuts_min_gap_rad", type=float, default=0.20)
    ap.add_argument("--gapcuts_jitter_runs", type=int, default=0)
    ap.add_argument("--gapcuts_jitter_sigma", type=float, default=0.03)
    ap.add_argument("--eta_scales", default="1.25,1.5,2.0")
    ap.add_argument("--loop_e0", type=int, default=0)
    ap.add_argument("--loop_e1", type=int, default=-1)
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()

    out_root = str(args.out_root)
    npz_files = find_npz_files(out_root)
    if not npz_files:
        raise SystemExit(f"No channel_diag*.npz found under {out_root}")

    block_modes = parse_str_csv(args.block_modes)
    eta_scales = parse_floats_csv(args.eta_scales)

    suite_specs: List[LoopSpec] = []
    for s in eta_scales:
        suite_specs.append(LoopSpec(loop_type="rectangle", eta_scale=float(s), direction="fwd"))
        suite_specs.append(LoopSpec(loop_type="rectangle", eta_scale=float(s), direction="rev"))
        suite_specs.append(LoopSpec(loop_type="rectangle", eta_scale=float(s), direction="fwd", double_loop=True))
        suite_specs.append(LoopSpec(loop_type="degenerate", eta_scale=float(s), direction="degenerate", degenerate_axis="E"))
        suite_specs.append(LoopSpec(loop_type="degenerate", eta_scale=float(s), direction="degenerate", degenerate_axis="eta"))

    all_rows: List[Dict[str, Any]] = []
    failed = 0

    for p in npz_files:
        rel, backend_label, seed, anchor, wlo, whi = parse_rel_metadata(p, out_root)
        d = np.load(p)

        if "lambda_snap" not in d.files:
            failed += 1
            continue
        lam_snap = np.asarray(d["lambda_snap"], dtype=np.complex128)
        E_snap = np.asarray(d["energies_snap"], dtype=np.float64)
        eta0 = float(d["cayley_eta"]) if ("cayley_eta" in d.files) else float("nan")
        if (not np.isfinite(eta0)) or (eta0 <= 0):
            failed += 1
            continue

        nE = int(lam_snap.shape[0])
        if nE < 2:
            continue

        i0 = int(args.loop_e0) if int(args.loop_e0) >= 0 else int(nE + int(args.loop_e0))
        i1 = int(args.loop_e1) if int(args.loop_e1) >= 0 else int(nE + int(args.loop_e1))
        i0 = int(np.clip(i0, 0, nE - 1))
        i1 = int(np.clip(i1, 0, nE - 1))
        if i0 == i1:
            i1 = 0 if i0 != 0 else nE - 1

        for mode in block_modes:
            rng = np.random.default_rng(0)

            jitter_runs = 1
            jitter_sigma = 0.0
            if str(mode) == "gapcuts" and int(args.gapcuts_jitter_runs) > 0:
                jitter_runs = int(args.gapcuts_jitter_runs)
                jitter_sigma = float(args.gapcuts_jitter_sigma)

            for jr in range(jitter_runs):
                jitter_tag = ""
                js = 0.0
                if str(mode) == "gapcuts" and jitter_runs > 1:
                    jitter_tag = f"jitter{jr}"
                    js = float(jitter_sigma)

                # Keep indices for updating suite KPIs on the fwd-rectangle rows.
                fwd_row_index_by_key: Dict[Tuple[float, int, int], int] = {}
                fwd_M_by_key: Dict[Tuple[float, int, int], np.ndarray] = {}
                fwd_norm_by_key: Dict[Tuple[float, int, int], float] = {}

                for spec in suite_specs:
                    eta1 = float(eta0)
                    eta2 = float(spec.eta_scale) * float(eta0)
                    pts = loop_points(i0, i1, eta1, eta2, spec)

                    Ms, diags = compute_monodromy(
                        lam_snap,
                        pts=pts,
                        block_mode=str(mode),
                        blocks=int(args.blocks),
                        gapcuts_max_blocks=int(args.gapcuts_max_blocks),
                        gapcuts_q=float(args.gapcuts_q),
                        gapcuts_min_gap_rad=float(args.gapcuts_min_gap_rad),
                        rng=rng,
                        gapcuts_jitter_sigma=float(js),
                    )

                    for b, M in enumerate(Ms):
                        diag = diags[b]
                        dim = int(diag["dim"])
                        E0 = float(E_snap[i0])
                        E1 = float(E_snap[i1])
                        apx = 0.0
                        if spec.loop_type == "rectangle":
                            apx = area_proxy(E0, E1, eta1, eta2)
                        elif spec.loop_type == "degenerate":
                            apx = 0.0

                        row: Dict[str, Any] = {
                            "backend_label": str(backend_label),
                            "seed": int(seed),
                            "anchor_seed": int(anchor),
                            "wlo": float(wlo),
                            "whi": float(whi),
                            "artifact": rel,
                            "block_mode": str(mode),
                            "blocks": int(args.blocks),
                            "gapcuts_max_blocks": int(args.gapcuts_max_blocks),
                            "gapcuts_q": float(args.gapcuts_q),
                            "gapcuts_min_gap_rad": float(args.gapcuts_min_gap_rad),
                            "gapcuts_jitter_tag": jitter_tag,
                            "loop_type": str(spec.loop_type),
                            "loop_direction": str(spec.direction),
                            "loop_double": bool(spec.double_loop),
                            "loop_degenerate_axis": str(spec.degenerate_axis),
                            "loop_E0": float(E0),
                            "loop_E1": float(E1),
                            "eta1": float(eta1),
                            "eta2": float(eta2),
                            "eta_scale": float(spec.eta_scale),
                            "loop_area_proxy": float(apx),
                            "block": int(b),
                            "dim": int(dim),
                            "valid": bool(diag["valid"] and (M is not None)),
                            "holonomy_det_phase": float("nan"),
                            "holonomy_trace": float("nan"),
                            "holonomy_norm_IminusM": float("nan"),
                            "holonomy_imag_fraction": float("nan"),
                            "holonomy_ortho_defect": float("nan"),
                            "holonomy_unit_defect": float("nan"),
                            "holonomy_det_imag_frac": float("nan"),
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
                            "step_transport_imag_median": float(diag.get("step_transport_imag_median", float("nan"))),
                            "step_transport_orth_defect_median": float(diag.get("step_transport_orth_defect_median", float("nan"))),
                            "step_transport_unit_defect_median": float(diag.get("step_transport_unit_defect_median", float("nan"))),
                            "step_transport_det_imag_frac_median": float(diag.get("step_transport_det_imag_frac_median", float("nan"))),
                            "step_transport_imag_worst": float(diag.get("step_transport_imag_worst", float("nan"))),
                            "step_transport_orth_defect_worst": float(diag.get("step_transport_orth_defect_worst", float("nan"))),
                            "step_transport_unit_defect_worst": float(diag.get("step_transport_unit_defect_worst", float("nan"))),
                            "step_transport_det_imag_frac_worst": float(diag.get("step_transport_det_imag_frac_worst", float("nan"))),
                            # suite KPIs (filled later for fwd rectangle rows)
                            "reverse_consistency": float("nan"),
                            "double_loop_defect": float("nan"),
                            "degenerate_loop_ratio_E": float("nan"),
                            "degenerate_loop_ratio_eta": float("nan"),
                            "area_monotone_ok": float("nan"),
                            "lattice_fit_alpha": float("nan"),
                            "lattice_fit_rmse": float("nan"),
                            "lattice_fit_frac_within_tol": float("nan"),
                        }

                        if row["valid"] and (M is not None):
                            det_phase = float(np.angle(np.linalg.det(M)))
                            tr = np.trace(M)
                            I = np.eye(M.shape[0], dtype=np.complex128)
                            hol_norm = float(fro_norm(I - M))
                            row["holonomy_det_phase"] = float(det_phase)
                            row["holonomy_trace"] = float(complex(tr).real)
                            row["holonomy_norm_IminusM"] = float(hol_norm)

                            row["holonomy_imag_fraction"] = float(imag_fraction(M))
                            row["holonomy_ortho_defect"] = float(ortho_defect(M))
                            row["holonomy_unit_defect"] = float(unit_defect(M))
                            row["holonomy_det_imag_frac"] = float(det_imag_frac(M))

                            row.update(holonomy_spectrum_stats(M))

                            # quick lattice fit (optional-but-cheap): phi_k ≈ alpha*m_k + beta
                            phi_json = row.get("holonomy_phi_vec_json", "")
                            if phi_json:
                                phi = np.asarray(json.loads(phi_json), dtype=np.float64)
                                beta = float(np.mean(phi))
                                centered = phi - beta
                                # grid search alpha
                                alphas = np.linspace(0.15, 2.5, 240)
                                best = (float("inf"), float("nan"), 0.0)
                                tol = 0.05
                                for a in alphas:
                                    m = np.rint(centered / float(a))
                                    fit = float(a) * m
                                    rmse = float(np.sqrt(np.mean((centered - fit) ** 2)))
                                    frac = float(np.mean(np.abs(centered - fit) <= tol))
                                    if rmse < best[0]:
                                        best = (rmse, float(a), frac)
                                row["lattice_fit_rmse"] = float(best[0])
                                row["lattice_fit_alpha"] = float(best[1])
                                row["lattice_fit_frac_within_tol"] = float(best[2])

                        # If this is the baseline fwd rectangle single loop, record for KPIs.
                        if (
                            row["valid"]
                            and row["loop_type"] == "rectangle"
                            and row["loop_direction"] == "fwd"
                            and (row["loop_double"] is False)
                            and (M is not None)
                        ):
                            k = (float(row["eta_scale"]), int(row["block"]), int(row["dim"]))
                            fwd_row_index_by_key[k] = int(len(all_rows))
                            fwd_M_by_key[k] = np.asarray(M, dtype=np.complex128)
                            fwd_norm_by_key[k] = float(row["holonomy_norm_IminusM"])

                        # If this is reverse loop, attach reverse_consistency to the corresponding fwd row.
                        if (
                            row["valid"]
                            and row["loop_type"] == "rectangle"
                            and row["loop_direction"] == "rev"
                            and (row["loop_double"] is False)
                            and (M is not None)
                        ):
                            k = (float(row["eta_scale"]), int(row["block"]), int(row["dim"]))
                            if k in fwd_row_index_by_key and k in fwd_M_by_key:
                                M_fwd = fwd_M_by_key[k]
                                M_rev = np.asarray(M, dtype=np.complex128)
                                defect = fro_norm(M_rev - safe_inv_unitary(M_fwd))
                                all_rows[fwd_row_index_by_key[k]]["reverse_consistency"] = float(defect)

                        # If this is double loop, attach double_loop_defect to the corresponding fwd row.
                        if (
                            row["valid"]
                            and row["loop_type"] == "rectangle"
                            and row["loop_direction"] == "fwd"
                            and (row["loop_double"] is True)
                            and (M is not None)
                        ):
                            k = (float(row["eta_scale"]), int(row["block"]), int(row["dim"]))
                            if k in fwd_row_index_by_key and k in fwd_M_by_key:
                                M_fwd = fwd_M_by_key[k]
                                M_double = np.asarray(M, dtype=np.complex128)
                                defect = fro_norm(M_double - (M_fwd @ M_fwd))
                                all_rows[fwd_row_index_by_key[k]]["double_loop_defect"] = float(defect)

                        # Degenerate loops: attach degenerate ratios (relative to fwd rectangle) to fwd row.
                        if (
                            row["valid"]
                            and row["loop_type"] == "degenerate"
                            and (row["loop_double"] is False)
                            and (M is not None)
                        ):
                            k = (float(row["eta_scale"]), int(row["block"]), int(row["dim"]))
                            if k in fwd_row_index_by_key and k in fwd_norm_by_key:
                                deg_norm = float(row["holonomy_norm_IminusM"])
                                rect_norm = float(fwd_norm_by_key[k])
                                ratio = float(deg_norm / (rect_norm + 1e-12))
                                if str(row["loop_degenerate_axis"]) == "E":
                                    all_rows[fwd_row_index_by_key[k]]["degenerate_loop_ratio_E"] = float(ratio)
                                if str(row["loop_degenerate_axis"]) == "eta":
                                    all_rows[fwd_row_index_by_key[k]]["degenerate_loop_ratio_eta"] = float(ratio)

                        all_rows.append(row)

    df = pd.DataFrame(all_rows)

    # area monotone check: for each artifact/block, ||I-M|| should increase with loop area proxy
    if len(df):
        rect = df[(df["loop_type"] == "rectangle") & (df["loop_direction"] == "fwd") & (df["loop_double"] == False) & (df["valid"] == True)].copy()
        if len(rect):
            grp_cols = ["artifact", "block_mode", "gapcuts_jitter_tag", "block", "dim"]
            ok_flags = []
            for _k, g in rect.groupby(grp_cols):
                g2 = g.sort_values("loop_area_proxy")
                y = g2["holonomy_norm_IminusM"].values
                if len(y) < 2:
                    ok = float("nan")
                else:
                    ok = float(np.all(np.diff(y) >= -1e-6))
                ok_flags.append((*_k, ok))
            ok_df = pd.DataFrame(ok_flags, columns=grp_cols + ["area_monotone_ok"])
            df = df.merge(ok_df, on=grp_cols, how="left")

    out_csv = os.path.join(out_root, "phase3e_monodromy_suite_rows.csv")
    df.to_csv(out_csv, index=False)

    summary: Dict[str, Any] = {
        "out_root": out_root,
        "n_artifacts": int(len(npz_files)),
        "n_rows": int(len(df)),
        "n_artifacts_failed": int(failed),
        "block_modes": block_modes,
        "blocks": int(args.blocks),
        "gapcuts_max_blocks": int(args.gapcuts_max_blocks),
        "gapcuts_q": float(args.gapcuts_q),
        "gapcuts_min_gap_rad": float(args.gapcuts_min_gap_rad),
        "eta_scales": eta_scales,
    }

    if len(df):
        valid = df[df["valid"] == True]
        summary["valid_fraction"] = float(len(valid) / max(1, len(df)))
        if len(valid):
            summary["holonomy_norm_IminusM_median"] = float(np.nanmedian(valid["holonomy_norm_IminusM"].values))

            # Suite KPIs are attached to the forward-rectangle rows.
            fwd_rect = df[
                (df["loop_type"] == "rectangle")
                & (df["loop_direction"] == "fwd")
                & (df["loop_double"] == False)
                & (df["valid"] == True)
            ].copy()
            if len(fwd_rect):
                summary["reverse_consistency_median"] = float(np.nanmedian(fwd_rect["reverse_consistency"].values))
                summary["double_loop_defect_median"] = float(np.nanmedian(fwd_rect["double_loop_defect"].values))
                summary["degenerate_ratio_E_median"] = float(np.nanmedian(fwd_rect["degenerate_loop_ratio_E"].values))
                summary["degenerate_ratio_eta_median"] = float(np.nanmedian(fwd_rect["degenerate_loop_ratio_eta"].values))
                if "area_monotone_ok" in fwd_rect.columns:
                    summary["area_monotone_ok_fraction"] = float(np.nanmean(fwd_rect["area_monotone_ok"].values))
                else:
                    summary["area_monotone_ok_fraction"] = float("nan")
                summary["lattice_fit_rmse_median"] = float(np.nanmedian(fwd_rect["lattice_fit_rmse"].values))
                summary["lattice_fit_frac_within_tol_median"] = float(
                    np.nanmedian(fwd_rect["lattice_fit_frac_within_tol"].values)
                )

                # Group-ID KPIs (overall + stratified by det-phase class)
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
                    "step_transport_imag_median": fwd_rect["step_transport_imag_median"].values,
                    "step_transport_orth_defect_median": fwd_rect["step_transport_orth_defect_median"].values,
                    "step_transport_unit_defect_median": fwd_rect["step_transport_unit_defect_median"].values,
                    "step_transport_det_imag_frac_median": fwd_rect["step_transport_det_imag_frac_median"].values,
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
            else:
                summary["reverse_consistency_median"] = float("nan")
                summary["double_loop_defect_median"] = float("nan")
                summary["degenerate_ratio_E_median"] = float("nan")
                summary["degenerate_ratio_eta_median"] = float("nan")
                summary["area_monotone_ok_fraction"] = float("nan")
                summary["lattice_fit_rmse_median"] = float("nan")
                summary["lattice_fit_frac_within_tol_median"] = float("nan")

    with open(os.path.join(out_root, "phase3e_monodromy_suite_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[phase3e_monodromy_suite] wrote phase3e_monodromy_suite_rows.csv and phase3e_monodromy_suite_summary.json")
    print(json.dumps(summary, indent=2))

    if bool(args.plots):
        try:
            import matplotlib.pyplot as plt

            dfp = df.copy()
            dfp = dfp[(dfp["loop_type"] == "rectangle") & (dfp["loop_direction"] == "fwd") & (dfp["loop_double"] == False) & (dfp["valid"] == True)]
            if len(dfp):
                plt.figure(figsize=(8, 5))
                for mode in sorted(dfp["block_mode"].unique().tolist()):
                    m = dfp[dfp["block_mode"] == mode]
                    plt.scatter(m["loop_area_proxy"], m["holonomy_norm_IminusM"], s=8, alpha=0.35, label=str(mode))
                plt.xlabel("loop_area_proxy = |ΔE|·|Δη|")
                plt.ylabel("||I - M||_F")
                plt.title("Phase-3E holonomy strength vs loop area")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(out_root, "phase3e_suite_strength_vs_area.png"), dpi=160)
                plt.close()

            # Reverse consistency proxy histogram
            dfr = df.copy()
            dfr = dfr[(dfr["loop_type"] == "rectangle") & (dfr["loop_direction"] == "fwd") & (dfr["loop_double"] == False) & (dfr["valid"] == True)]
            vals = dfr["reverse_consistency"].values
            vals = vals[np.isfinite(vals)]
            if vals.size:
                plt.figure(figsize=(7, 4))
                plt.hist(vals, bins=40, alpha=0.9)
                plt.xlabel("reverse_consistency (proxy)")
                plt.ylabel("count")
                plt.title("Reverse-loop consistency proxy")
                plt.tight_layout()
                plt.savefig(os.path.join(out_root, "phase3e_suite_reverse_consistency_hist.png"), dpi=160)
                plt.close()

            # Eigenphase overlay by anchor (aggregate |phi| CDF)
            dfa = df.copy()
            dfa = dfa[(dfa["loop_type"] == "rectangle") & (dfa["loop_direction"] == "fwd") & (dfa["loop_double"] == False) & (dfa["valid"] == True)]
            dfa = dfa[dfa["holonomy_phi_vec_json"].astype(str).str.len() > 0]
            if len(dfa):
                plt.figure(figsize=(8, 5))
                anchors = sorted(dfa["anchor_seed"].unique().tolist())
                anchors = anchors[:8]  # cap to avoid unreadable plot
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
                plt.title("Holonomy eigenphase magnitude CDF by anchor (subset)")
                plt.legend(fontsize=8)
                plt.tight_layout()
                plt.savefig(os.path.join(out_root, "phase3e_suite_phi_cdf_by_anchor.png"), dpi=160)
                plt.close()

        except Exception as e:
            print(f"[phase3e_monodromy_suite] plots skipped due to error: {e}")


if __name__ == "__main__":
    main()
