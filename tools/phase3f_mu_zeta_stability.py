"""Mu-slice zeta stability runner.

Purpose
- Given a Phase-3F run directory that contains multiple mu slices (e.g. validate_mugrid_*),
  compute two small, finite diagnostics per mu:
  1) A scalar Euler-product surrogate built from per-generator weights.
  2) A finite noncommutative word-zeta probe (short words up to length L).

This is meant to support the "true 2D" narrative by showing that the zeta-like
aggregates change smoothly or predictably with mu, and by quantifying cancellation.

Inputs expected in run_dir
- phase3f_event_cells_with_gen.csv
- phase3f_holonomy_dumps/holonomy_*.npz (with key "M")

Outputs written to run_dir (by default)
- mu_zeta_scalar_summary.csv
- mu_word_zeta_summary.csv

Notes
- The scalar surrogate here is not the primitive-class Euler product; it is a mu-slice
  diagnostic built from available generator matrices in that slice.
"""

from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GenRep:
    gen_id: str
    gen_local_id: int
    mu: float
    M: np.ndarray


def _parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _load_M(npz_path: Path) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as z:
        if "M" not in z:
            raise KeyError(f"Missing key 'M' in {npz_path}")
        M = z["M"]
    M = np.asarray(M)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"Bad M shape {M.shape} in {npz_path}")
    return M.astype(np.complex128)


def _fro_norm_I_minus_M(M: np.ndarray) -> float:
    d = M.shape[0]
    return float(np.linalg.norm(np.eye(d, dtype=np.complex128) - M, ord="fro"))


def _theta_from_trace(M: np.ndarray) -> float:
    d = M.shape[0]
    tr = complex(np.trace(M))
    x = float(np.real(tr)) / float(d)
    x = max(-1.0, min(1.0, x))
    return float(math.acos(x))


def _min_dist_to_minus1(M: np.ndarray) -> float:
    eig = np.linalg.eigvals(M)
    return float(np.min(np.abs(eig + 1.0)))


def _gen_feature_vec(g: GenRep, eps: float) -> np.ndarray:
    """Low-dimensional feature vector for cross-mu matching.

    Features are chosen to be cheap and fairly stable across discretizations:
    - theta from trace (bounded)
    - log1p(defect) where defect = ||I-M||_F
    - log1p(1/dist_to_-1)
    """

    theta = _theta_from_trace(g.M)
    defect = _fro_norm_I_minus_M(g.M)
    distm1 = max(_min_dist_to_minus1(g.M), float(eps))
    return np.array([theta, math.log1p(defect), math.log1p(1.0 / distm1)], dtype=float)


def _standardize_features(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Xz, mean, std) with std floored to avoid division by zero."""

    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0)
    sd = np.where(sd <= 1e-12, 1.0, sd)
    return (X - mu) / sd, mu, sd


def _greedy_min_cost_matching(cost: np.ndarray, max_cost: float) -> dict[int, int]:
    """Greedy one-to-one matching from rows->cols.

    Not Hungarian-optimal, but robust enough for small n and serves as a
    falsification/robustness check.
    """

    if cost.ndim != 2:
        raise ValueError("cost must be 2D")
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return {}

    remaining_rows = set(range(n_rows))
    remaining_cols = set(range(n_cols))
    pairs: list[tuple[float, int, int]] = []
    for i in range(n_rows):
        for j in range(n_cols):
            pairs.append((float(cost[i, j]), int(i), int(j)))
    pairs.sort(key=lambda t: t[0])

    match: dict[int, int] = {}
    for c, i, j in pairs:
        if c > float(max_cost):
            break
        if i in remaining_rows and j in remaining_cols:
            match[i] = j
            remaining_rows.remove(i)
            remaining_cols.remove(j)
    return match


def _match_invariants_across_mu(
    mu_to_gens: Dict[float, List[GenRep]],
    ref_mu: float | None,
    eps: float,
    max_cost: float,
) -> tuple[Dict[float, List[GenRep]], dict]:
    """Build a consistent alphabet across mu via invariant-based matching.

    Returns (mu_to_selected_gens, meta).

    - Choose a reference mu (explicit or smallest).
    - Standardize features based on reference mu.
    - Greedily match each other mu's generators to the reference.
    - Keep only reference generators that have matches for all mu.
    """

    mus = sorted(mu_to_gens.keys())
    if not mus:
        return {}, {"ref_mu": "", "n_ref": 0, "n_matched_all": 0}

    ref = float(ref_mu) if ref_mu is not None else float(mus[0])
    if ref not in mu_to_gens:
        raise ValueError(f"--match_ref_mu {ref:g} not present in selected mu values")

    ref_gens = list(mu_to_gens[ref])
    Xref = np.vstack([_gen_feature_vec(g, eps=eps) for g in ref_gens]) if ref_gens else np.zeros((0, 3), float)
    Xref_z, feat_mu, feat_sd = _standardize_features(Xref) if Xref.size else (Xref, np.zeros(3), np.ones(3))

    # For each ref index, store matched GenRep per mu.
    matches: dict[float, dict[int, GenRep]] = {ref: {i: ref_gens[i] for i in range(len(ref_gens))}}

    for mu in mus:
        if mu == ref:
            continue
        gens = list(mu_to_gens[mu])
        X = np.vstack([_gen_feature_vec(g, eps=eps) for g in gens]) if gens else np.zeros((0, 3), float)
        Xz = (X - feat_mu) / feat_sd if X.size else X
        if len(ref_gens) == 0 or len(gens) == 0:
            matches[mu] = {}
            continue

        # cost[i,j] = ||ref_i - mu_j||_2
        cost = np.linalg.norm(Xref_z[:, None, :] - Xz[None, :, :], axis=2)
        idx_map = _greedy_min_cost_matching(cost, max_cost=float(max_cost))
        matches[mu] = {i: gens[j] for i, j in idx_map.items()}

    # Keep only ref indices that exist for all mu slices.
    keep_ref_idx: list[int] = []
    for i in range(len(ref_gens)):
        ok = True
        for mu in mus:
            if i not in matches.get(mu, {}):
                ok = False
                break
        if ok:
            keep_ref_idx.append(i)

    out: Dict[float, List[GenRep]] = {}
    for mu in mus:
        out[mu] = [matches[mu][i] for i in keep_ref_idx]

    meta = {
        "ref_mu": float(ref),
        "n_ref": int(len(ref_gens)),
        "n_matched_all": int(len(keep_ref_idx)),
    }
    return out, meta


def _enumerate_words(n_gens: int, max_len: int) -> Iterable[Tuple[int, ...]]:
    for L in range(1, max_len + 1):
        for w in itertools.product(range(n_gens), repeat=L):
            yield tuple(int(x) for x in w)


def _word_is_power(w: Tuple[int, ...]) -> bool:
    L = len(w)
    for k in range(1, L):
        if L % k != 0:
            continue
        block = w[:k]
        if block * (L // k) == w:
            return True
    return False


def _build_mu_generators(run_dir: Path) -> Dict[float, List[GenRep]]:
    events = pd.read_csv(run_dir / "phase3f_event_cells_with_gen.csv")
    if "mu" not in events.columns or "gen_id" not in events.columns or "rep_holonomy_dump" not in events.columns:
        raise ValueError("phase3f_event_cells_with_gen.csv missing required columns")

    mu_to_gens: Dict[float, List[GenRep]] = {}

    # Use first encountered representative dump for each (mu, gen_id).
    seen: set[tuple[float, str]] = set()
    for _, r in events.iterrows():
        mu = float(r["mu"])
        gen_id = str(r["gen_id"])
        key = (mu, gen_id)
        if key in seen:
            continue
        seen.add(key)

        gen_local_id = int(r.get("gen_local_id", -1))
        dump_rel = Path(str(r["rep_holonomy_dump"]))
        npz_path = (run_dir / dump_rel).resolve()
        M = _load_M(npz_path)

        mu_to_gens.setdefault(mu, []).append(GenRep(gen_id=gen_id, gen_local_id=gen_local_id, mu=mu, M=M))

    # Sort gens by local id then gen_id for stable ordering.
    for mu, gens in list(mu_to_gens.items()):
        mu_to_gens[mu] = sorted(gens, key=lambda g: (g.gen_local_id, g.gen_id))

    return dict(sorted(mu_to_gens.items(), key=lambda kv: kv[0]))


def _persistent_intersection_gen_ids(mu_to_gens: Dict[float, List[GenRep]]) -> set[str]:
    """Return gen_id values that appear in every mu slice.

    This supports a stronger notion of stability: comparing the *same* generator alphabet
    across mu, rather than a potentially different first-k subset per mu.
    """

    mus = list(mu_to_gens.keys())
    if not mus:
        return set()

    common: set[str] | None = None
    for mu in mus:
        ids = {g.gen_id for g in mu_to_gens[mu]}
        common = ids if common is None else (common & ids)
        if not common:
            return set()
    return set(common)


def _scalar_zeta_for_mu(gens: Sequence[GenRep], s_values: Sequence[float]) -> List[dict]:
    mats = [g.M for g in gens]
    d = mats[0].shape[0]

    theta_g = np.array([_theta_from_trace(M) for M in mats], dtype=float)
    defect_g = np.array([_fro_norm_I_minus_M(M) for M in mats], dtype=float)

    rows: List[dict] = []
    for s in s_values:
        # Per-generator weight; intentionally bounded by tanh(defect).
        w = np.exp(-float(s) * theta_g) * np.tanh(defect_g)
        w = np.clip(w, 0.0, 1.0 - 1e-15)
        logZ = float(np.sum(-np.log1p(-w)))
        Z = float(math.exp(logZ))
        rows.append(
            {
                "s": float(s),
                "n_gens": int(len(gens)),
                "dim": int(d),
                "theta_g_min": float(np.min(theta_g)),
                "theta_g_median": float(np.median(theta_g)),
                "theta_g_max": float(np.max(theta_g)),
                "defect_g_min": float(np.min(defect_g)),
                "defect_g_median": float(np.median(defect_g)),
                "defect_g_max": float(np.max(defect_g)),
                "w_min": float(np.min(w)),
                "w_median": float(np.median(w)),
                "w_max": float(np.max(w)),
                "logZ": float(logZ),
                "Z": float(Z),
            }
        )
    return rows


def _word_zeta_for_mu(
    gens: Sequence[GenRep],
    s_values: Sequence[float],
    max_word_len: int,
    length_mode: str,
    term_mode: str,
    eps: float,
    length_scale: float,
) -> List[dict]:
    mats = [g.M for g in gens]
    d = mats[0].shape[0]

    # Precompute primitive words.
    primitive_words: List[Tuple[int, ...]] = []
    ell_list: List[float] = []
    term_list: List[complex] = []

    theta_g = np.array([_theta_from_trace(M) for M in mats], dtype=float)
    defect_g = np.array([_fro_norm_I_minus_M(M) for M in mats], dtype=float)
    distm1_g = np.array([_min_dist_to_minus1(M) for M in mats], dtype=float)

    for w in _enumerate_words(len(gens), max_word_len):
        if _word_is_power(w):
            continue

        M = np.eye(d, dtype=np.complex128)
        for idx in w:
            M = M @ mats[int(idx)]

        if length_mode == "theta_sum":
            ell = float(np.sum(theta_g[list(w)]))
        elif length_mode == "gen_defect_sum":
            ell = float(np.sum(np.log1p(defect_g[list(w)])))
        elif length_mode == "gen_minus1_log10_sum":
            ell = float(np.sum(-np.log10(np.maximum(distm1_g[list(w)], eps))))
            ell = max(0.0, ell)
        elif length_mode == "word_defect":
            ell = float(np.log1p(_fro_norm_I_minus_M(M)))
        elif length_mode == "word_minus1_log10":
            ell = float(-math.log10(max(_min_dist_to_minus1(M), eps)))
            ell = max(0.0, ell)
        elif length_mode == "word_minus1_log1p_inv":
            dist = max(_min_dist_to_minus1(M), eps)
            ell = float(math.log1p(1.0 / dist))
        else:
            raise ValueError(f"Unknown length_mode: {length_mode}")

        ell = float(length_scale) * ell

        tr = complex(np.trace(M))
        if term_mode == "trace":
            term = tr
        elif term_mode == "trace_real":
            term = complex(float(np.real(tr)), 0.0)
        elif term_mode == "abs_trace":
            term = complex(float(abs(tr)), 0.0)
        else:
            raise ValueError(f"Unknown term_mode: {term_mode}")

        primitive_words.append(w)
        ell_list.append(float(ell))
        term_list.append(complex(term))

    ell_arr = np.asarray(ell_list, dtype=float)
    term_arr = np.asarray(term_list, dtype=np.complex128)

    # s-independent distribution
    ell_min = float(np.min(ell_arr)) if ell_arr.size else float("nan")
    ell_med = float(np.median(ell_arr)) if ell_arr.size else float("nan")
    ell_max = float(np.max(ell_arr)) if ell_arr.size else float("nan")
    ell_p10 = float(np.quantile(ell_arr, 0.10)) if ell_arr.size else float("nan")
    ell_p90 = float(np.quantile(ell_arr, 0.90)) if ell_arr.size else float("nan")
    ell_zero_frac = float(np.mean(ell_arr <= 0.0)) if ell_arr.size else float("nan")

    rows: List[dict] = []
    for s in s_values:
        weight = np.exp(-float(s) * ell_arr)
        contrib = weight * term_arr
        cr = np.real(contrib)
        ca = np.abs(contrib)

        neg = cr < 0
        pos = cr > 0
        abs_sum = float(np.sum(ca))
        abs_neg = float(np.sum(ca[neg])) if ca.size else 0.0
        abs_pos = float(np.sum(ca[pos])) if ca.size else 0.0

        denom = float(np.sum(np.abs(cr)))
        rows.append(
            {
                "s": float(s),
                "n_gens": int(len(gens)),
                "dim": int(d),
                "n_words_primitive": int(len(primitive_words)),
                "sum_contrib_real": float(np.sum(cr)),
                "sum_contrib_abs": float(abs_sum),
                "frac_neg": float(np.mean(neg)) if neg.size else float("nan"),
                "abs_neg_share": float(abs_neg / abs_sum) if abs_sum != 0.0 else float("nan"),
                "cancellation_ratio": float(float(np.sum(cr)) / denom) if denom != 0.0 else float("nan"),
                "ell_min": ell_min,
                "ell_median": ell_med,
                "ell_max": ell_max,
                "ell_p10": ell_p10,
                "ell_p90": ell_p90,
                "ell_zero_frac": ell_zero_frac,
                "weight_median": float(np.median(weight)) if weight.size else float("nan"),
                "weight_max": float(np.max(weight)) if weight.size else float("nan"),
                "theta_g_median": float(np.median(theta_g)),
                "defect_g_median": float(np.median(defect_g)),
                "distm1_g_median": float(np.median(distm1_g)),
            }
        )

    return rows


def _select_fixed_alphabet(
    gens: Sequence[GenRep],
    restrict_k_gens: int | None,
    insufficient_mode: str,
) -> tuple[List[GenRep], int]:
    """Return (selected_gens, n_available).

    Selection is stable: takes the first k by (gen_local_id, gen_id).
    """

    available = sorted(gens, key=lambda g: (g.gen_local_id, g.gen_id))
    n_avail = len(available)
    if restrict_k_gens is None:
        return list(available), n_avail

    k = int(restrict_k_gens)
    if k <= 0:
        raise ValueError("--restrict_k_gens must be positive")

    if n_avail < k:
        if insufficient_mode == "skip":
            return [], n_avail
        if insufficient_mode == "use_all":
            return list(available), n_avail
        raise ValueError(f"Unknown insufficient_mode: {insufficient_mode}")

    return list(available[:k]), n_avail


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute scalar + word zeta diagnostics across mu slices")
    ap.add_argument("--run_dir", required=True, help="Run directory containing phase3f_event_cells_with_gen.csv")
    ap.add_argument("--out_dir", default=None, help="Where to write outputs (default: run_dir)")
    ap.add_argument("--mu_values", default=None, help="Comma-separated mu values to include (default: all present)")

    ap.add_argument(
        "--alphabet_strategy",
        default="slice",
        choices=["slice", "persistent_intersection", "match_invariants"],
        help=(
            "How to choose the generator alphabet per mu. "
            "'slice' uses each mu slice's available generators. "
            "'persistent_intersection' restricts to gen_id that appear in every included mu slice. "
            "'match_invariants' builds a consistent alphabet by matching generators across mu by invariants."
        ),
    )

    ap.add_argument(
        "--match_ref_mu",
        type=float,
        default=None,
        help=(
            "Reference mu for --alphabet_strategy match_invariants (default: smallest included mu). "
            "Must be one of the included mu values."
        ),
    )
    ap.add_argument(
        "--match_max_cost",
        type=float,
        default=1.0,
        help=(
            "Maximum standardized feature distance to accept a match for --alphabet_strategy match_invariants. "
            "Smaller is stricter. Default: 1.0."
        ),
    )

    ap.add_argument(
        "--restrict_k_gens",
        type=int,
        default=None,
        help=(
            "Optionally restrict each mu-slice to the first k generators (by gen_local_id/gen_id) "
            "to keep a constant alphabet size across mu."
        ),
    )
    ap.add_argument(
        "--insufficient_gens_mode",
        default="skip",
        choices=["skip", "use_all"],
        help=(
            "What to do if a mu slice has fewer than k generators when --restrict_k_gens is set: "
            "'skip' omits that mu; 'use_all' uses whatever is available. Default: skip."
        ),
    )

    ap.add_argument("--s_values", default="0.5,1,2,3", help="Comma-separated s values")
    ap.add_argument("--max_word_len", type=int, default=4, help="Max word length for word-zeta (default: 4)")
    ap.add_argument(
        "--length_mode",
        default="word_minus1_log1p_inv",
        choices=[
            "theta_sum",
            "gen_defect_sum",
            "word_defect",
            "word_minus1_log10",
            "word_minus1_log1p_inv",
            "gen_minus1_log10_sum",
        ],
        help="Word length mode for word-zeta",
    )
    ap.add_argument(
        "--term_mode",
        default="trace_real",
        choices=["trace", "trace_real", "abs_trace"],
        help="Term mode for word-zeta",
    )
    ap.add_argument("--eps", type=float, default=1e-16, help="Epsilon for minus1 distance")
    ap.add_argument("--length_scale", type=float, default=1.0, help="Scale factor for ell(w)")

    ap.add_argument("--out_prefix", default="mu_zeta_stability", help="Output filename prefix")

    args = ap.parse_args()
    run_dir = Path(str(args.run_dir)).resolve()
    out_dir = Path(str(args.out_dir)).resolve() if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    s_values = _parse_csv_floats(str(args.s_values))

    mu_to_gens = _build_mu_generators(run_dir)
    if args.mu_values:
        keep = set(_parse_csv_floats(str(args.mu_values)))
        mu_to_gens = {mu: gens for mu, gens in mu_to_gens.items() if mu in keep}

    persistent_ids: set[str] | None = None
    match_meta: dict | None = None
    if str(args.alphabet_strategy) == "persistent_intersection":
        persistent_ids = _persistent_intersection_gen_ids(mu_to_gens)
    elif str(args.alphabet_strategy) == "match_invariants":
        mu_to_gens, match_meta = _match_invariants_across_mu(
            mu_to_gens,
            ref_mu=args.match_ref_mu,
            eps=float(args.eps),
            max_cost=float(args.match_max_cost),
        )

    scalar_rows: List[dict] = []
    word_rows: List[dict] = []

    for mu, gens in mu_to_gens.items():
        gens_total = list(gens)
        gens_in = gens_total
        if persistent_ids is not None:
            gens_in = [g for g in gens_total if g.gen_id in persistent_ids]
            gens_in = sorted(gens_in, key=lambda g: (g.gen_local_id, g.gen_id))

        selected_gens, n_avail = _select_fixed_alphabet(
            gens_in,
            restrict_k_gens=args.restrict_k_gens,
            insufficient_mode=str(args.insufficient_gens_mode),
        )
        if not selected_gens:
            continue

        persistent_mus_s = ",".join([f"{x:g}" for x in sorted(mu_to_gens.keys())])
        persistent_size = ""
        if persistent_ids is not None:
            persistent_size = int(len(persistent_ids))
        elif match_meta is not None:
            persistent_size = int(match_meta.get("n_matched_all", 0))

        for r in _scalar_zeta_for_mu(selected_gens, s_values):
            r.update(
                {
                    "mu": float(mu),
                    "alphabet_strategy": str(args.alphabet_strategy),
                    "persistent_intersection_mus": persistent_mus_s,
                    "persistent_intersection_size": persistent_size,
                    "match_ref_mu": float(match_meta.get("ref_mu")) if match_meta is not None and match_meta.get("ref_mu", "") != "" else "",
                    "match_max_cost": float(args.match_max_cost) if str(args.alphabet_strategy) == "match_invariants" else "",
                    "n_gens_available_total": int(len(gens_total)),
                    "n_gens_available": int(n_avail),
                    "n_gens_used": int(len(selected_gens)),
                    "restrict_k_gens": int(args.restrict_k_gens) if args.restrict_k_gens is not None else "",
                    "insufficient_gens_mode": str(args.insufficient_gens_mode) if args.restrict_k_gens is not None else "",
                }
            )
            scalar_rows.append(r)

        for r in _word_zeta_for_mu(
            selected_gens,
            s_values,
            max_word_len=int(args.max_word_len),
            length_mode=str(args.length_mode),
            term_mode=str(args.term_mode),
            eps=float(args.eps),
            length_scale=float(args.length_scale),
        ):
            r.update(
                {
                    "mu": float(mu),
                    "alphabet_strategy": str(args.alphabet_strategy),
                    "persistent_intersection_mus": persistent_mus_s,
                    "persistent_intersection_size": persistent_size,
                    "match_ref_mu": float(match_meta.get("ref_mu")) if match_meta is not None and match_meta.get("ref_mu", "") != "" else "",
                    "match_max_cost": float(args.match_max_cost) if str(args.alphabet_strategy) == "match_invariants" else "",
                    "n_gens_available_total": int(len(gens_total)),
                    "n_gens_available": int(n_avail),
                    "n_gens_used": int(len(selected_gens)),
                    "restrict_k_gens": int(args.restrict_k_gens) if args.restrict_k_gens is not None else "",
                    "insufficient_gens_mode": str(args.insufficient_gens_mode) if args.restrict_k_gens is not None else "",
                }
            )
            word_rows.append(r)

    scalar_out = out_dir / f"{args.out_prefix}_scalar_summary.csv"
    word_out = out_dir / f"{args.out_prefix}_word_summary.csv"

    if not scalar_rows:
        print("WARNING: no scalar rows produced (no mu slices selected). Writing empty CSV.")
        pd.DataFrame(columns=["mu", "s"]).to_csv(scalar_out, index=False)
    else:
        pd.DataFrame(scalar_rows).sort_values(["mu", "s"]).to_csv(scalar_out, index=False)

    if not word_rows:
        print("WARNING: no word rows produced (no mu slices selected). Writing empty CSV.")
        pd.DataFrame(columns=["mu", "s"]).to_csv(word_out, index=False)
    else:
        pd.DataFrame(word_rows).sort_values(["mu", "s"]).to_csv(word_out, index=False)

    print(f"Wrote: {scalar_out}")
    print(f"Wrote: {word_out}")


if __name__ == "__main__":
    main()
