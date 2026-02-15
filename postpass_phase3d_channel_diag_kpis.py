import os
import json
import math
import argparse
import itertools

import numpy as np
import pandas as pd


def find_npz_files(out_root: str):
    hits = []
    for root, _dirs, files in os.walk(out_root):
        for fn in files:
            if fn.endswith(".npz") and fn.startswith("channel_diag"):
                hits.append(os.path.join(root, fn))
    return sorted(hits)


def fro_norm(x):
    return float(np.linalg.norm(x, ord="fro"))


def unitarity_defect(S):
    S = np.asarray(S, dtype=np.complex128)
    n = int(S.shape[0])
    I = np.eye(n, dtype=np.complex128)
    SS = np.conjugate(S.T) @ S
    return float(fro_norm(SS - I) / (fro_norm(I) + 1e-12))


def normality_defect(S):
    S = np.asarray(S, dtype=np.complex128)
    SS = np.conjugate(S.T) @ S
    SSt = S @ np.conjugate(S.T)
    denom = float(fro_norm(S) ** 2) + 1e-12
    return float(fro_norm(SS - SSt) / denom)


def orbit_residuals_from_trace(logdet_IminusS, trace_powers):
    """Return partial residuals for k=1..K: ld - (-sum_{j<=k} Tr(S^j)/j)."""
    ld = np.asarray(logdet_IminusS, dtype=np.complex128)
    tks = np.asarray(trace_powers, dtype=np.complex128)
    K = int(tks.shape[0])
    series = np.zeros(K, dtype=np.complex128)
    s = np.complex128(0.0 + 0.0j)
    for k in range(1, K + 1):
        s += -(tks[k - 1] / np.complex128(float(k)))
        series[k - 1] = s
    resid = ld - series
    return series, resid


def orbit_residuals_from_trace_damped(logdet_IminuszS, trace_powers, orbit_z: float):
    """Return partial residuals for k=1..K at fixed damping z in (0,1)."""
    z = np.complex128(float(orbit_z))
    ld = np.asarray(logdet_IminuszS, dtype=np.complex128)
    tks = np.asarray(trace_powers, dtype=np.complex128)
    K = int(tks.shape[0])
    series = np.zeros(K, dtype=np.complex128)
    s = np.complex128(0.0 + 0.0j)
    for k in range(1, K + 1):
        s += -((z ** np.complex128(float(k))) * tks[k - 1] / np.complex128(float(k)))
        series[k - 1] = s
    resid = ld - series
    return series, resid


def circular_kmeans_angles(angles, M, iters=20, *, init_mode="quantile", rng=None):
    """K-means on the unit circle angles in [0, 2pi).

    init_mode:
      - 'quantile': deterministic init via angle quantiles
      - 'random': deterministic given rng
    """
    angles = np.asarray(angles, dtype=np.float64)
    n = int(angles.size)
    if n == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float64)
    M = int(max(1, min(int(M), n)))

    if str(init_mode) == "random":
        if rng is None:
            rng = np.random.default_rng(0)
        # choose random sample (deterministic given rng)
        pick = rng.choice(n, size=M, replace=False)
        centers = np.asarray(angles[pick], dtype=np.float64)
    else:
        # init centers via quantiles (deterministic)
        qs = np.linspace(0.0, 1.0, M, endpoint=False) + 0.5 / M
        init = np.quantile(angles, qs)
        centers = np.asarray(init, dtype=np.float64)

    def circ_dist(a, c):
        # shortest angular distance
        d = np.abs(a - c)
        return np.minimum(d, 2.0 * math.pi - d)

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(int(iters)):
        # assign
        D = np.stack([circ_dist(angles, c) for c in centers], axis=1)
        new_labels = np.argmin(D, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # update: mean direction
        for j in range(M):
            mask = labels == j
            if not np.any(mask):
                continue
            z = np.exp(1j * angles[mask])
            mu = np.mean(z)
            centers[j] = float(np.mod(np.angle(mu), 2.0 * math.pi))

    return labels, centers


def cluster_projectors_from_S(S, M, *, init_mode="quantile", anchor_seed=0, salt=777):
    """Compute cluster subspaces/projectors from eigendecomposition.

        Returns list of dicts with keys:
            - Q: orthonormal basis (n x r)
            - dim: r
            - center_angle
            - idx: eigenvector indices in this cluster
    """
    S = np.asarray(S, dtype=np.complex128)
    eigvals, eigvecs = np.linalg.eig(S)
    angles = np.mod(np.angle(eigvals), 2.0 * math.pi)
    rng = np.random.default_rng(int(anchor_seed) + int(salt))
    labels, centers = circular_kmeans_angles(angles, M=int(M), init_mode=str(init_mode), rng=rng)

    clusters = []
    for j in range(int(centers.size)):
        idx = np.where(labels == j)[0]
        if idx.size == 0:
            clusters.append(
                {
                    "Q": np.zeros((S.shape[0], 0), dtype=np.complex128),
                    "dim": 0,
                    "center_angle": float(centers[j]),
                    "idx": np.zeros((0,), dtype=np.int64),
                }
            )
            continue
        V = eigvecs[:, idx]
        # Orthonormalize with QR for projector stability
        Q, _ = np.linalg.qr(V)
        clusters.append(
            {
                "Q": np.asarray(Q, dtype=np.complex128),
                "dim": int(Q.shape[1]),
                "center_angle": float(centers[j]),
                "idx": np.asarray(idx, dtype=np.int64),
                "_eig_angles": angles,
            }
        )
    return clusters


def canonicalize_clusters(clusters):
    """Sort clusters by center_angle for gauge-invariant labeling."""
    return sorted(list(clusters), key=lambda c: float(c.get("center_angle", 0.0)))


def commutator_norm(S, Q):
    """||[S,P]||_F / ||S||_F with P=Q Q*."""
    S = np.asarray(S, dtype=np.complex128)
    Q = np.asarray(Q, dtype=np.complex128)
    denom = float(np.linalg.norm(S, ord="fro")) + 1e-12
    if Q.shape[1] == 0:
        return float("nan")
    P = Q @ np.conjugate(Q.T)
    C = S @ P - P @ S
    return float(np.linalg.norm(C, ord="fro") / denom)


def overlap_score(Qa, Qb):
    """Normalized overlap score in [0,1] based on Frobenius of Qa*Qb."""
    if Qa.shape[1] == 0 or Qb.shape[1] == 0:
        return 0.0
    X = np.conjugate(Qa.T) @ Qb
    # Normalize by dim of Qa
    return float((np.linalg.norm(X, ord="fro") ** 2) / (float(Qa.shape[1]) + 1e-12))


def principal_angles(Qa, Qb):
    if Qa.shape[1] == 0 or Qb.shape[1] == 0:
        return np.asarray([], dtype=np.float64)
    X = np.conjugate(Qa.T) @ Qb
    s = np.linalg.svd(X, compute_uv=False)
    s = np.clip(np.real(s), 0.0, 1.0)
    return np.arccos(s)


def circular_distance_rad(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = np.abs(a - b)
    return np.minimum(d, 2.0 * math.pi - d)


def cluster_gap_proxies(clusters):
    """Return per-cluster gap proxies (angle units, radians).

    Two proxies:
      - gap_min_rad: min distance between any in-cluster eigenangle and any out-of-cluster eigenangle
      - gap_center_rad: min distance between cluster center_angle and any out-of-cluster eigenangle
    """
    if not clusters:
        return []
    eig_angles = clusters[0].get("_eig_angles", None)
    if eig_angles is None:
        return [{"gap_min_rad": float("nan"), "gap_center_rad": float("nan")} for _ in clusters]
    eig_angles = np.asarray(eig_angles, dtype=np.float64)
    n = int(eig_angles.size)
    out = []
    for c in clusters:
        idx = np.asarray(c.get("idx", []), dtype=np.int64)
        if idx.size == 0 or idx.size == n:
            out.append({"gap_min_rad": float("nan"), "gap_center_rad": float("nan")})
            continue
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        a_in = eig_angles[idx]
        a_out = eig_angles[mask]

        D = circular_distance_rad(a_in.reshape(-1, 1), a_out.reshape(1, -1))
        gap_min = float(np.min(D))

        center = float(c.get("center_angle", float("nan")))
        if np.isfinite(center) and a_out.size:
            gap_center = float(np.min(circular_distance_rad(a_out, center)))
        else:
            gap_center = float("nan")

        out.append({"gap_min_rad": gap_min, "gap_center_rad": gap_center})
    return out


def match_clusters(A, B):
    """Match clusters A->B by maximum overlap with optional bruteforce for M<=8."""
    M = min(len(A), len(B))
    O = np.zeros((M, M), dtype=np.float64)
    for ii in range(M):
        for jj in range(M):
            O[ii, jj] = overlap_score(A[ii]["Q"], B[jj]["Q"])
    if M <= 8:
        perm, score = best_perm_bruteforce(O)
    else:
        perm = list(np.argmax(O, axis=1))
        score = float(np.sum(O[np.arange(M), perm]))
    return perm, score, O


def normalize_columns(V):
    V = np.asarray(V, dtype=np.complex128)
    if V.ndim != 2:
        raise ValueError("normalize_columns expects 2D array")
    norms = np.linalg.norm(V, axis=0)
    norms = np.where(norms > 0, norms, 1.0)
    return V / norms


def tracked_transport_pairs(S_snap, *, M: int, init_mode: str, anchor_seed: int):
    """Continuity-based channel tracking across energy.

    Initialization at E0 uses the same clustering-on-eigenangle to define a starting
    partition (dims r_m). Thereafter, for each adjacent step, we assign eigenvectors
    at E_{j+1} to channels by maximizing overlap with the previous channel subspaces,
    enforcing a disjoint partition matching the previous dims.

    Returns:
      pairs: list of dict rows for adjacent steps and each channel
      summary: dict with medians/worsts over adjacent steps
    """
    S_snap = np.asarray(S_snap, dtype=np.complex128)
    nE = int(S_snap.shape[0])
    if nE < 2:
        return [], {"tracked_adjacent_angle_median": float("nan"), "tracked_adjacent_angle_worst": float("nan"), "tracked_adjacent_overlap_median": float("nan")}

    # Init at E0 via existing clustering
    clusters0 = canonicalize_clusters(
        cluster_projectors_from_S(S_snap[0], M=int(M), init_mode=str(init_mode), anchor_seed=int(anchor_seed), salt=777)
    )
    Meff = int(len(clusters0))
    dims = [int(c.get("dim", 0)) for c in clusters0]
    Q_prev_list = [np.asarray(c.get("Q"), dtype=np.complex128) for c in clusters0]

    pairs = []
    angs_all = []
    ovs_all = []

    for j in range(0, nE - 1):
        # Eigendecomposition at next step
        eigvals, eigvecs = np.linalg.eig(np.asarray(S_snap[j + 1], dtype=np.complex128))
        V = normalize_columns(eigvecs)
        n = int(V.shape[1])

        # Score matrix: score[m, i] = ||Q_prev[m]^* v_i||^2
        scores = np.zeros((Meff, n), dtype=np.float64)
        for m in range(Meff):
            Qp = Q_prev_list[m]
            if Qp.size == 0 or Qp.shape[1] == 0:
                continue
            proj = np.conjugate(Qp.T) @ V  # (r, n)
            scores[m, :] = np.sum(np.abs(proj) ** 2, axis=0)

        # Greedy disjoint assignment: channels in descending dim pick best remaining eigenvectors.
        remaining = np.ones(n, dtype=bool)
        order = list(sorted(range(Meff), key=lambda k: int(dims[k]), reverse=True))
        picks = [np.zeros((0,), dtype=np.int64) for _ in range(Meff)]
        for m in order:
            r = int(dims[m])
            if r <= 0:
                continue
            s = np.asarray(scores[m, :], dtype=np.float64)
            s = np.where(remaining, s, -np.inf)
            # pick top-r
            idx = np.argsort(s)[-r:]
            idx = np.asarray(idx, dtype=np.int64)
            picks[m] = idx
            remaining[idx] = False

        # Build next channel subspaces and record transport metrics
        Q_next_list = []
        for m in range(Meff):
            idx = picks[m]
            if idx.size == 0:
                Q_next = np.zeros((V.shape[0], 0), dtype=np.complex128)
            else:
                Q_next, _ = np.linalg.qr(V[:, idx])
                Q_next = np.asarray(Q_next, dtype=np.complex128)
            Q_next_list.append(Q_next)

            Qp = Q_prev_list[m]
            ang = principal_angles(Qp, Q_next)
            ang_max = float(np.max(ang)) if ang.size else float("nan")
            ov = float(overlap_score(Qp, Q_next))
            angs_all.append(ang_max)
            ovs_all.append(ov)

            pairs.append(
                {
                    "E_idx": int(j),
                    "channel": int(m),
                    "angle_max_rad": float(ang_max),
                    "overlap": float(ov),
                    "dim": int(dims[m]),
                }
            )

        Q_prev_list = Q_next_list

    angs = np.asarray([a for a in angs_all if np.isfinite(a)], dtype=np.float64)
    ovs = np.asarray([o for o in ovs_all if np.isfinite(o)], dtype=np.float64)
    summary = {
        "tracked_adjacent_angle_median": float(np.median(angs)) if angs.size else float("nan"),
        "tracked_adjacent_angle_worst": float(np.max(angs)) if angs.size else float("nan"),
        "tracked_adjacent_overlap_median": float(np.median(ovs)) if ovs.size else float("nan"),
    }
    return pairs, summary


def best_perm_bruteforce(overlap_mat):
    """Maximize sum overlap_mat[i, perm[i]] (bruteforce; ok for M<=8)."""
    M = int(overlap_mat.shape[0])
    best_p = None
    best_val = -1.0
    for perm in itertools.permutations(range(M)):
        v = 0.0
        for i in range(M):
            v += float(overlap_mat[i, perm[i]])
        if v > best_val:
            best_val = v
            best_p = perm
    return list(best_p), float(best_val)


def unitary_procrustes(A, B):
    """Find unitary/orthogonal U minimizing ||A - U B||_F.

    Works for complex matrices too.
    """
    A = np.asarray(A, dtype=np.complex128)
    B = np.asarray(B, dtype=np.complex128)
    if A.shape != B.shape:
        raise ValueError("unitary_procrustes: shape mismatch")
    M = A @ np.conjugate(B.T)
    U, _s, Vh = np.linalg.svd(M)
    return U @ Vh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--clusters", type=int, default=8)
    ap.add_argument("--cluster_init_mode", default="random", choices=["random", "quantile"])

    # Gates (defaults are prereg-friendly and overrideable)
    ap.add_argument("--gate_unitarity_median", type=float, default=1e-2)
    ap.add_argument("--gate_comm_median", type=float, default=1e-1)
    ap.add_argument("--gate_angle_median_rad", type=float, default=0.3)
    ap.add_argument("--gate_energy_transport_median_rad", type=float, default=0.8)
    ap.add_argument("--gate_energy_transport_worst_rad", type=float, default=1.2)
    ap.add_argument("--gate_orbit_improves", action="store_true", help="Require median R_K <= median R_4")

    # Transport diagnostics upgrades
    ap.add_argument(
        "--transport_is_gate",
        action="store_true",
        help="If set, include energy-transport gates in overall_pass. Otherwise transport is diagnostic-only.",
    )
    ap.add_argument("--gap_large_rad", type=float, default=0.4, help="Gap threshold (radians) for 'large-gap' stratification")
    ap.add_argument(
        "--extraction_noise_repeats",
        type=int,
        default=2,
        help="How many independent extractions to run at same-E for baseline (>=2 to enable)",
    )

    ap.add_argument(
        "--transport_mode",
        default="both",
        choices=["clustering", "tracked", "both"],
        help="Compute clustering-based transport, continuity-tracked transport, or both.",
    )

    args = ap.parse_args()

    out_root = str(args.out_root)
    npz_files = find_npz_files(out_root)
    if not npz_files:
        raise SystemExit(f"No channel_diag*.npz found under {out_root}")

    rows = []
    transport_rows = []
    transport_pair_rows = []
    extraction_noise_rows = []
    tracked_transport_rows = []
    tracked_pair_rows = []

    # First pass: per-artifact KPIs
    for path in npz_files:
        rel = os.path.relpath(path, out_root).replace("\\", "/")

        # Parse metadata from folder structure: backend_*/seed*_anchor*/window_*/
        parts = rel.split("/")
        backend_label = parts[0].replace("backend_", "") if parts else ""
        seed = None
        anchor = None
        window = None
        for p in parts:
            if p.startswith("seed") and "_anchor" in p:
                a, b = p.split("_anchor")
                seed = int(a.replace("seed", ""))
                anchor = int(b)
            if p.startswith("window_"):
                w = p.replace("window_", "")
                a, b = w.split("_", 1)
                window = (float(a), float(b))

        d = np.load(path)
        S_snap = np.asarray(d["S_snap"], dtype=np.complex128)  # (nE, n, n)
        E_snap = np.asarray(d["energies_snap"], dtype=np.float64)
        logdet_IminusS = np.asarray(d["logdet_IminusS"], dtype=np.complex128)
        trace_powers = np.asarray(d["trace_powers"], dtype=np.complex128)

        lam_def = np.asarray(d["lambda_hermitian_defect"], dtype=np.float64) if ("lambda_hermitian_defect" in d.files) else None

        orbit_z = float(d["orbit_z"]) if ("orbit_z" in d.files) else None
        logdet_IminuszS = np.asarray(d["logdet_IminuszS"], dtype=np.complex128) if ("logdet_IminuszS" in d.files) else None

        nE = int(S_snap.shape[0])
        K = int(trace_powers.shape[1])

        # Energy-transport diagnostics:
        #  (a) legacy metric: match clusters across energies (E0 -> Ej)
        #  (b) adjacent metric: match clusters between consecutive energies (Ej -> Ej+1), report raw + |dE|-normalized
        #  (c) gap-stratified summary and overlap diagnostic
        #  (d) same-E control: repeat extraction at same S(E) to measure clustering/extraction noise floor

        clusters_by_E = []
        gaps_by_E = []
        if str(args.transport_mode) in ("clustering", "both"):
            for i in range(nE):
                clusters_i = canonicalize_clusters(
                    cluster_projectors_from_S(
                        S_snap[i],
                        M=int(args.clusters),
                        init_mode=str(args.cluster_init_mode),
                        anchor_seed=int(anchor) if anchor is not None else 0,
                        salt=777,
                    )
                )
                clusters_by_E.append(clusters_i)
                gaps_by_E.append(cluster_gap_proxies(clusters_i))

        angle_max_vs_E0 = []
        adjacent_angle_max = []
        adjacent_angle_per_dE = []
        adjacent_overlap = []
        adjacent_angle_max_largegap = []
        adjacent_overlap_largegap = []

        if str(args.transport_mode) in ("clustering", "both"):
            # (a) E0 -> Ej
            base = clusters_by_E[0]
            for j in range(1, nE):
                cur = clusters_by_E[j]
                perm, _score, _O = match_clusters(base, cur)
                M = min(len(base), len(cur))
                for ii in range(M):
                    jj = perm[ii]
                    ang = principal_angles(base[ii]["Q"], cur[jj]["Q"])
                    if ang.size:
                        angle_max_vs_E0.append(float(np.max(ang)))

        if str(args.transport_mode) in ("clustering", "both"):
            # (b,c) adjacent transport with gap + overlap
            for j in range(0, nE - 1):
                A = clusters_by_E[j]
                B = clusters_by_E[j + 1]
                gaps_A = gaps_by_E[j]
                dE = float(abs(E_snap[j + 1] - E_snap[j]))
                perm, score, _O = match_clusters(A, B)
                M = min(len(A), len(B))

                for ii in range(M):
                    jj = perm[ii]
                    ang = principal_angles(A[ii]["Q"], B[jj]["Q"])
                    if not ang.size:
                        continue
                    ang_max = float(np.max(ang))
                    ov = float(overlap_score(A[ii]["Q"], B[jj]["Q"]))
                    gap_min = float(gaps_A[ii].get("gap_min_rad", float("nan"))) if ii < len(gaps_A) else float("nan")
                    gap_center = float(gaps_A[ii].get("gap_center_rad", float("nan"))) if ii < len(gaps_A) else float("nan")

                    adjacent_angle_max.append(ang_max)
                    adjacent_overlap.append(ov)
                    if dE > 0:
                        adjacent_angle_per_dE.append(float(ang_max / dE))

                    if np.isfinite(gap_center) and gap_center >= float(args.gap_large_rad):
                        adjacent_angle_max_largegap.append(ang_max)
                        adjacent_overlap_largegap.append(ov)

                    transport_pair_rows.append(
                        {
                            "backend_label": str(backend_label),
                            "seed": int(seed) if seed is not None else -1,
                            "anchor_seed": int(anchor) if anchor is not None else -1,
                            "wlo": float(window[0]) if window else float("nan"),
                            "whi": float(window[1]) if window else float("nan"),
                            "E_idx": int(j),
                            "E": float(E_snap[j]),
                            "E_next": float(E_snap[j + 1]),
                            "dE": float(dE),
                            "cluster_i": int(ii),
                            "cluster_j": int(jj),
                            "perm_score": float(score),
                            "gap_min_rad": float(gap_min),
                            "gap_center_rad": float(gap_center),
                            "angle_max_rad": float(ang_max),
                            "angle_per_dE": float(ang_max / dE) if dE > 0 else float("nan"),
                            "overlap": float(ov),
                            "artifact": rel,
                        }
                    )

        # Continuity-tracked transport (Fork B)
        if str(args.transport_mode) in ("tracked", "both"):
            pairs_tr, summ_tr = tracked_transport_pairs(
                S_snap,
                M=int(args.clusters),
                init_mode=str(args.cluster_init_mode),
                anchor_seed=int(anchor) if anchor is not None else 0,
            )
            tracked_transport_rows.append(
                {
                    "backend_label": str(backend_label),
                    "seed": int(seed) if seed is not None else -1,
                    "anchor_seed": int(anchor) if anchor is not None else -1,
                    "wlo": float(window[0]) if window else float("nan"),
                    "whi": float(window[1]) if window else float("nan"),
                    "nE_snap": int(nE),
                    "tracked_adjacent_angle_median": float(summ_tr.get("tracked_adjacent_angle_median", float("nan"))),
                    "tracked_adjacent_angle_worst": float(summ_tr.get("tracked_adjacent_angle_worst", float("nan"))),
                    "tracked_adjacent_overlap_median": float(summ_tr.get("tracked_adjacent_overlap_median", float("nan"))),
                    "artifact": rel,
                }
            )
            for r in pairs_tr:
                tracked_pair_rows.append(
                    {
                        "backend_label": str(backend_label),
                        "seed": int(seed) if seed is not None else -1,
                        "anchor_seed": int(anchor) if anchor is not None else -1,
                        "wlo": float(window[0]) if window else float("nan"),
                        "whi": float(window[1]) if window else float("nan"),
                        "E_idx": int(r["E_idx"]),
                        "channel": int(r["channel"]),
                        "angle_max_rad": float(r["angle_max_rad"]),
                        "overlap": float(r["overlap"]),
                        "dim": int(r["dim"]),
                        "artifact": rel,
                    }
                )

        # (d) same-E extraction noise control
        if int(args.extraction_noise_repeats) >= 2:
            for i in range(nE):
                ex = []
                # Create two (or more) extractions by varying salt
                for r in range(int(args.extraction_noise_repeats)):
                    ex.append(
                        canonicalize_clusters(
                            cluster_projectors_from_S(
                                S_snap[i],
                                M=int(args.clusters),
                                init_mode=str(args.cluster_init_mode),
                                anchor_seed=int(anchor) if anchor is not None else 0,
                                salt=10_000 + 997 * r,
                            )
                        )
                    )
                # Compare repeat 0 vs repeat 1
                A = ex[0]
                B = ex[1]
                perm, score, _O = match_clusters(A, B)
                M = min(len(A), len(B))
                ang_list = []
                ov_list = []
                for ii in range(M):
                    jj = perm[ii]
                    ang = principal_angles(A[ii]["Q"], B[jj]["Q"])
                    if ang.size:
                        ang_list.append(float(np.max(ang)))
                        ov_list.append(float(overlap_score(A[ii]["Q"], B[jj]["Q"])))
                extraction_noise_rows.append(
                    {
                        "backend_label": str(backend_label),
                        "seed": int(seed) if seed is not None else -1,
                        "anchor_seed": int(anchor) if anchor is not None else -1,
                        "wlo": float(window[0]) if window else float("nan"),
                        "whi": float(window[1]) if window else float("nan"),
                        "E_idx": int(i),
                        "E": float(E_snap[i]),
                        "perm_score": float(score),
                        "sameE_angle_median": float(np.median(np.asarray(ang_list, dtype=np.float64))) if ang_list else float("nan"),
                        "sameE_angle_worst": float(np.max(np.asarray(ang_list, dtype=np.float64))) if ang_list else float("nan"),
                        "sameE_overlap_median": float(np.median(np.asarray(ov_list, dtype=np.float64))) if ov_list else float("nan"),
                        "artifact": rel,
                    }
                )

        transport_rows.append(
            {
                "backend_label": str(backend_label),
                "seed": int(seed) if seed is not None else -1,
                "anchor_seed": int(anchor) if anchor is not None else -1,
                "wlo": float(window[0]) if window else float("nan"),
                "whi": float(window[1]) if window else float("nan"),
                "nE_snap": int(nE),
                "E_span": float(np.max(E_snap) - np.min(E_snap)) if nE else float("nan"),
                "energy_transport_angle_median": float(np.median(np.asarray(angle_max_vs_E0, dtype=np.float64))) if angle_max_vs_E0 else float("nan"),
                "energy_transport_angle_worst": float(np.max(np.asarray(angle_max_vs_E0, dtype=np.float64))) if angle_max_vs_E0 else float("nan"),
                "energy_transport_adjacent_angle_median": float(np.median(np.asarray(adjacent_angle_max, dtype=np.float64))) if adjacent_angle_max else float("nan"),
                "energy_transport_adjacent_angle_worst": float(np.max(np.asarray(adjacent_angle_max, dtype=np.float64))) if adjacent_angle_max else float("nan"),
                "energy_transport_adjacent_angle_per_dE_median": float(np.median(np.asarray(adjacent_angle_per_dE, dtype=np.float64))) if adjacent_angle_per_dE else float("nan"),
                "energy_transport_adjacent_angle_per_dE_worst": float(np.max(np.asarray(adjacent_angle_per_dE, dtype=np.float64))) if adjacent_angle_per_dE else float("nan"),
                "energy_transport_adjacent_overlap_median": float(np.median(np.asarray(adjacent_overlap, dtype=np.float64))) if adjacent_overlap else float("nan"),
                "energy_transport_adjacent_largegap_angle_median": float(np.median(np.asarray(adjacent_angle_max_largegap, dtype=np.float64))) if adjacent_angle_max_largegap else float("nan"),
                "energy_transport_adjacent_largegap_overlap_median": float(np.median(np.asarray(adjacent_overlap_largegap, dtype=np.float64))) if adjacent_overlap_largegap else float("nan"),
                "gap_large_rad": float(args.gap_large_rad),
                "artifact": rel,
            }
        )

        for i in range(nE):
            S = S_snap[i]
            u = unitarity_defect(S)
            ndef = normality_defect(S)

            # orbit residuals
            if logdet_IminuszS is not None and orbit_z is not None:
                series, resid = orbit_residuals_from_trace_damped(logdet_IminuszS[i], trace_powers[i], orbit_z=float(orbit_z))
                denom_ld = float(abs(logdet_IminuszS[i]))
            else:
                series, resid = orbit_residuals_from_trace(logdet_IminusS[i], trace_powers[i])
                denom_ld = float(abs(logdet_IminusS[i]))
            # normalized residuals
            denom = max(1.0, denom_ld)
            R = np.abs(resid) / denom

            # cluster commutators
            clusters = cluster_projectors_from_S(
                S,
                M=int(args.clusters),
                init_mode=str(args.cluster_init_mode),
                anchor_seed=int(anchor) if anchor is not None else 0,
            )
            comms = []
            dims = []
            for c in clusters:
                dims.append(int(c["dim"]))
                comms.append(commutator_norm(S, c["Q"]))

            rows.append(
                {
                    "backend_label": str(backend_label),
                    "seed": int(seed) if seed is not None else -1,
                    "anchor_seed": int(anchor) if anchor is not None else -1,
                    "wlo": float(window[0]) if window else float("nan"),
                    "whi": float(window[1]) if window else float("nan"),
                    "E_idx": int(i),
                    "E": float(E_snap[i]),
                    "K": int(K),
                    "unitarity_defect": float(u),
                    "normality_defect": float(ndef),
                    "orbit_R4": float(R[3]) if K >= 4 else float("nan"),
                    "orbit_RK": float(R[-1]),
                    "orbit_z": float(orbit_z) if orbit_z is not None else float("nan"),
                    "lambda_hermitian_defect": float(lam_def[i]) if lam_def is not None else float("nan"),
                    "comm_median": float(np.nanmedian(np.asarray(comms, dtype=np.float64))) if len(comms) else float("nan"),
                    "comm_max": float(np.nanmax(np.asarray(comms, dtype=np.float64))) if len(comms) else float("nan"),
                    "cluster_dims": json.dumps(dims),
                    "artifact": rel,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_root, "kpi_rows.csv"), index=False)

    df_transport = pd.DataFrame(transport_rows)
    df_transport.to_csv(os.path.join(out_root, "kpi_energy_transport.csv"), index=False)

    df_pairs = pd.DataFrame(transport_pair_rows)
    if len(df_pairs):
        df_pairs.to_csv(os.path.join(out_root, "kpi_energy_transport_pairs.csv"), index=False)

    df_tr = pd.DataFrame(tracked_transport_rows)
    if len(df_tr):
        df_tr.to_csv(os.path.join(out_root, "kpi_energy_transport_tracked.csv"), index=False)

    df_pairs_tr = pd.DataFrame(tracked_pair_rows)
    if len(df_pairs_tr):
        df_pairs_tr.to_csv(os.path.join(out_root, "kpi_energy_transport_pairs_tracked.csv"), index=False)
    df_noise = pd.DataFrame(extraction_noise_rows)
    df_noise.to_csv(os.path.join(out_root, "kpi_extraction_noise.csv"), index=False)

    # Second pass: anchor stability of the *extracted channels* across different anchor-seeded
    # clustering initializations, on the same operator snapshots.
    anchor_ref = 2
    stab_rows = []

    # Index npz files by (backend, seed, anchor, window)
    by_key = {}
    for path in npz_files:
        rel = os.path.relpath(path, out_root).replace("\\", "/")
        parts = rel.split("/")
        backend_label = parts[0].replace("backend_", "") if parts else ""
        seed = None
        anchor = None
        window = None
        for p in parts:
            if p.startswith("seed") and "_anchor" in p:
                a, b = p.split("_anchor")
                seed = int(a.replace("seed", ""))
                anchor = int(b)
            if p.startswith("window_"):
                w = p.replace("window_", "")
                a, b = w.split("_", 1)
                window = (float(a), float(b))
        if seed is None or anchor is None or window is None:
            continue
        by_key[(backend_label, int(seed), int(anchor), float(window[0]), float(window[1]))] = path

    # For each seed/window, compare each anchor to anchor_ref
    seeds = sorted(set(df["seed"].tolist()))
    wins = sorted(set(zip(df["wlo"].tolist(), df["whi"].tolist())))

    for backend_label in sorted(set(df["backend_label"].tolist())):
        for seed in seeds:
            for (wlo, whi) in wins:
                kref = (backend_label, int(seed), int(anchor_ref), float(wlo), float(whi))
                if kref not in by_key:
                    continue
                dref = np.load(by_key[kref])
                Sref = np.asarray(dref["S_snap"], dtype=np.complex128)
                nE = int(Sref.shape[0])

                ref_clusters_by_E = [
                    canonicalize_clusters(
                        cluster_projectors_from_S(
                            Sref[i],
                            M=int(args.clusters),
                            init_mode=str(args.cluster_init_mode),
                            anchor_seed=int(anchor_ref),
                        )
                    )
                    for i in range(nE)
                ]

                for anchor in sorted(set(df["anchor_seed"].tolist())):
                    if int(anchor) == int(anchor_ref):
                        continue
                    kcmp = (backend_label, int(seed), int(anchor), float(wlo), float(whi))
                    if kcmp not in by_key:
                        continue
                    dcmp = np.load(by_key[kcmp])
                    Scmp = np.asarray(dcmp["S_snap"], dtype=np.complex128)

                    for i in range(nE):
                        A = ref_clusters_by_E[i]
                        B = canonicalize_clusters(
                            cluster_projectors_from_S(
                                Scmp[i],
                                M=int(args.clusters),
                                init_mode=str(args.cluster_init_mode),
                                anchor_seed=int(anchor),
                            )
                        )
                        M = min(len(A), len(B))

                        # Match clusters by maximum overlap (permutation) in the SAME ambient space.
                        perm, score, _O = match_clusters(A, B)

                        ang_max_list = []
                        for ii in range(M):
                            jj = perm[ii]
                            ang = principal_angles(A[ii]["Q"], B[jj]["Q"])
                            if ang.size:
                                ang_max_list.append(float(np.max(ang)))

                        stab_rows.append(
                            {
                                "backend_label": str(backend_label),
                                "seed": int(seed),
                                "wlo": float(wlo),
                                "whi": float(whi),
                                "E_idx": int(i),
                                "anchor_ref": int(anchor_ref),
                                "anchor_cmp": int(anchor),
                                "perm_score": float(score),
                                "angle_max_median": float(np.median(np.asarray(ang_max_list, dtype=np.float64))) if ang_max_list else float("nan"),
                                "angle_max_worst": float(np.max(np.asarray(ang_max_list, dtype=np.float64))) if ang_max_list else float("nan"),
                            }
                        )

    df_stab = pd.DataFrame(stab_rows)
    df_stab.to_csv(os.path.join(out_root, "kpi_anchor_stability.csv"), index=False)

    # Summary + gates
    summary = {
        "unitarity_defect_median": float(np.median(df["unitarity_defect"].values)),
        "normality_defect_median": float(np.median(df["normality_defect"].values)),
        "comm_median_median": float(np.median(df["comm_median"].values)),
        "orbit_RK_median": float(np.median(df["orbit_RK"].values)),
        "lambda_hermitian_defect_median": float(np.nanmedian(df["lambda_hermitian_defect"].values)),
        "lambda_hermitian_defect_max": float(np.nanmax(df["lambda_hermitian_defect"].values)),
    }

    if len(df_transport):
        summary.update(
            {
                "energy_transport_angle_median_median": float(np.nanmedian(df_transport["energy_transport_angle_median"].values)),
                "energy_transport_angle_worst_max": float(np.nanmax(df_transport["energy_transport_angle_worst"].values)),
                "energy_transport_adjacent_angle_median_median": float(np.nanmedian(df_transport["energy_transport_adjacent_angle_median"].values)),
                "energy_transport_adjacent_angle_worst_max": float(np.nanmax(df_transport["energy_transport_adjacent_angle_worst"].values)),
                "energy_transport_adjacent_angle_per_dE_median_median": float(
                    np.nanmedian(df_transport["energy_transport_adjacent_angle_per_dE_median"].values)
                ),
                "energy_transport_adjacent_overlap_median_median": float(np.nanmedian(df_transport["energy_transport_adjacent_overlap_median"].values)),
                "energy_transport_adjacent_largegap_angle_median_median": float(
                    np.nanmedian(df_transport["energy_transport_adjacent_largegap_angle_median"].values)
                ),
                "energy_transport_adjacent_largegap_overlap_median_median": float(
                    np.nanmedian(df_transport["energy_transport_adjacent_largegap_overlap_median"].values)
                ),
            }
        )
    else:
        summary.update(
            {
                "energy_transport_angle_median_median": float("nan"),
                "energy_transport_angle_worst_max": float("nan"),
                "energy_transport_adjacent_angle_median_median": float("nan"),
                "energy_transport_adjacent_angle_worst_max": float("nan"),
                "energy_transport_adjacent_angle_per_dE_median_median": float("nan"),
                "energy_transport_adjacent_overlap_median_median": float("nan"),
                "energy_transport_adjacent_largegap_angle_median_median": float("nan"),
                "energy_transport_adjacent_largegap_overlap_median_median": float("nan"),
            }
        )

    if len(df_noise):
        summary.update(
            {
                "sameE_angle_median_median": float(np.nanmedian(df_noise["sameE_angle_median"].values)),
                "sameE_angle_worst_max": float(np.nanmax(df_noise["sameE_angle_worst"].values)),
                "sameE_overlap_median_median": float(np.nanmedian(df_noise["sameE_overlap_median"].values)),
            }
        )
    else:
        summary.update(
            {
                "sameE_angle_median_median": float("nan"),
                "sameE_angle_worst_max": float("nan"),
                "sameE_overlap_median_median": float("nan"),
            }
        )

    # Gap-stratified global summaries (uses centroid-gap proxy)
    if len(df_pairs) and ("gap_center_rad" in df_pairs.columns):
        g = np.asarray(df_pairs["gap_center_rad"].values, dtype=np.float64)
        g = g[np.isfinite(g)]
        if g.size:
            thr_cov = float(args.gap_large_rad)
            summary["gap_center_ge_gap_large_fraction"] = float(np.mean(g >= thr_cov))

            q75 = float(np.quantile(g, 0.75))
            q90 = float(np.quantile(g, 0.90))
            for label, thr in [("q75", q75), ("q90", q90)]:
                mask = np.isfinite(df_pairs["gap_center_rad"].values) & (df_pairs["gap_center_rad"].values >= thr)
                sub = df_pairs.loc[mask]
                summary[f"gap_center_{label}_rad"] = float(thr)
                summary[f"adjacent_angle_median_gapcenter_ge_{label}"] = float(np.nanmedian(sub["angle_max_rad"].values)) if len(sub) else float("nan")
                summary[f"adjacent_overlap_median_gapcenter_ge_{label}"] = float(np.nanmedian(sub["overlap"].values)) if len(sub) else float("nan")
        else:
            summary["gap_center_ge_gap_large_fraction"] = float("nan")
            summary["gap_center_q75_rad"] = float("nan")
            summary["gap_center_q90_rad"] = float("nan")
            summary["adjacent_angle_median_gapcenter_ge_q75"] = float("nan")
            summary["adjacent_angle_median_gapcenter_ge_q90"] = float("nan")
            summary["adjacent_overlap_median_gapcenter_ge_q75"] = float("nan")
            summary["adjacent_overlap_median_gapcenter_ge_q90"] = float("nan")
    else:
        summary["gap_center_ge_gap_large_fraction"] = float("nan")
        summary["gap_center_q75_rad"] = float("nan")
        summary["gap_center_q90_rad"] = float("nan")
        summary["adjacent_angle_median_gapcenter_ge_q75"] = float("nan")
        summary["adjacent_angle_median_gapcenter_ge_q90"] = float("nan")
        summary["adjacent_overlap_median_gapcenter_ge_q75"] = float("nan")
        summary["adjacent_overlap_median_gapcenter_ge_q90"] = float("nan")

    # Tracked-transport global summaries
    if len(df_tr):
        summary["tracked_adjacent_angle_median_median"] = float(np.nanmedian(df_tr["tracked_adjacent_angle_median"].values))
        summary["tracked_adjacent_angle_worst_max"] = float(np.nanmax(df_tr["tracked_adjacent_angle_worst"].values))
        summary["tracked_adjacent_overlap_median_median"] = float(np.nanmedian(df_tr["tracked_adjacent_overlap_median"].values))
    else:
        summary["tracked_adjacent_angle_median_median"] = float("nan")
        summary["tracked_adjacent_angle_worst_max"] = float("nan")
        summary["tracked_adjacent_overlap_median_median"] = float("nan")

    if len(df_stab):
        summary.update(
            {
                "angle_max_median_median": float(np.nanmedian(df_stab["angle_max_median"].values)),
                "angle_max_worst_max": float(np.nanmax(df_stab["angle_max_worst"].values)),
                "perm_score_median": float(np.nanmedian(df_stab["perm_score"].values)),
            }
        )
    else:
        summary.update(
            {
                "angle_max_median_median": float("nan"),
                "angle_max_worst_max": float("nan"),
                "perm_score_median": float("nan"),
            }
        )

    # Optional orbit-improves check
    orbit_improves = None
    if bool(args.gate_orbit_improves):
        if np.isfinite(df["orbit_R4"].values).all():
            orbit_improves = bool(np.median(df["orbit_RK"].values) <= np.median(df["orbit_R4"].values))
        else:
            orbit_improves = False
    summary["orbit_improves_gate_used"] = bool(args.gate_orbit_improves)
    summary["orbit_improves"] = orbit_improves

    gates = {
        "gate_A_unitarity": bool(summary["unitarity_defect_median"] <= float(args.gate_unitarity_median)),
        "gate_1_commutator": bool(summary["comm_median_median"] <= float(args.gate_comm_median)),
        "gate_2_angles": bool(summary["angle_max_median_median"] <= float(args.gate_angle_median_rad)) if np.isfinite(summary["angle_max_median_median"]) else False,
        "gate_E_transport_median": bool(summary["energy_transport_angle_median_median"] <= float(args.gate_energy_transport_median_rad)) if np.isfinite(summary["energy_transport_angle_median_median"]) else False,
        "gate_E_transport_worst": bool(summary["energy_transport_angle_worst_max"] <= float(args.gate_energy_transport_worst_rad)) if np.isfinite(summary["energy_transport_angle_worst_max"]) else False,
    }
    if bool(args.gate_orbit_improves):
        gates["gate_4_orbit_improves"] = bool(orbit_improves)

    basic_gate_keys = ["gate_A_unitarity", "gate_1_commutator", "gate_2_angles"]
    transport_gate_keys = ["gate_E_transport_median", "gate_E_transport_worst"]
    if bool(args.gate_orbit_improves):
        basic_gate_keys.append("gate_4_orbit_improves")

    overall_pass_basic = bool(all(bool(gates[k]) for k in basic_gate_keys))
    overall_pass_with_transport = bool(all(bool(gates[k]) for k in (basic_gate_keys + transport_gate_keys)))
    overall_pass = overall_pass_with_transport if bool(args.transport_is_gate) else overall_pass_basic

    with open(os.path.join(out_root, "kpi_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "gates": gates,
                "overall_pass": overall_pass,
                "overall_pass_basic": overall_pass_basic,
                "overall_pass_with_transport": overall_pass_with_transport,
                "transport_is_gate": bool(args.transport_is_gate),
            },
            f,
            indent=2,
        )

    # Also write a flat CSV summary
    pd.DataFrame(
        [
            {
                **summary,
                **gates,
                "overall_pass": overall_pass,
                "overall_pass_basic": overall_pass_basic,
                "overall_pass_with_transport": overall_pass_with_transport,
                "transport_is_gate": bool(args.transport_is_gate),
            }
        ]
    ).to_csv(
        os.path.join(out_root, "kpi_summary.csv"), index=False
    )

    print(
        "[postpass_phase3d_channel_diag_kpis] wrote kpi_rows.csv, kpi_anchor_stability.csv, kpi_summary.json"
    )
    print(
        json.dumps(
            {
                "summary": summary,
                "gates": gates,
                "overall_pass": overall_pass,
                "overall_pass_basic": overall_pass_basic,
                "overall_pass_with_transport": overall_pass_with_transport,
                "transport_is_gate": bool(args.transport_is_gate),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
