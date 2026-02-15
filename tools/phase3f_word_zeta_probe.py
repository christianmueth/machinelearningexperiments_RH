import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _theta_from_trace(M: np.ndarray) -> float:
    d = M.shape[0]
    tr = np.trace(M)
    x = float(np.real(tr) / float(d))
    x = _clamp(x, -1.0, 1.0)
    return float(math.acos(x))


def _fro_norm_I_minus_M(M: np.ndarray) -> float:
    d = M.shape[0]
    return float(np.linalg.norm(np.eye(d, dtype=M.dtype) - M, ord="fro"))


def _min_dist_to_minus1(M: np.ndarray) -> float:
    # Compute min |eig + 1|; for d<=8 this is fine.
    eig = np.linalg.eigvals(M)
    return float(np.min(np.abs(eig + 1.0)))


def _load_M(npz_path: Path) -> np.ndarray:
    z = np.load(str(npz_path), allow_pickle=True)
    if "M" not in z.files:
        raise KeyError(f"NPZ missing 'M': {npz_path}")
    M = z["M"]
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"Bad M shape {getattr(M, 'shape', None)} in {npz_path}")
    return M


def _parse_csv_floats(s: str) -> List[float]:
    out: List[float] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("Expected at least one float")
    return out


def _nearly_equal(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(a - b) <= tol


def _pick_mu(df: pd.DataFrame, mu: Optional[float]) -> float:
    mus = sorted({float(x) for x in pd.to_numeric(df["mu"], errors="coerce").dropna().unique().tolist()})
    if not mus:
        raise ValueError("No mu values present")
    if mu is None:
        return mus[-1]
    # match by closeness
    best = min(mus, key=lambda x: abs(x - float(mu)))
    return best


def _word_is_power(word: Sequence[int]) -> bool:
    # word is represented as sequence of integer symbols (already includes sign if desired)
    n = len(word)
    if n <= 1:
        return False
    for k in range(1, n):
        if n % k != 0:
            continue
        base = word[:k]
        if list(base) * (n // k) == list(word):
            return True
    return False


def _word_to_str(word: Sequence[int], names: Sequence[str]) -> str:
    return "".join(names[i] for i in word)


@dataclass(frozen=True)
class Gen:
    name: str
    M: np.ndarray
    theta: float
    defect: float
    dist_minus1: float


def _build_generators(
    run_dir: Path,
    mu: Optional[float],
    use_word_test_pair: bool,
) -> List[Gen]:
    events_path = run_dir / "phase3f_event_cells_with_gen.csv"
    if not events_path.exists():
        raise SystemExit(f"Missing: {events_path}")
    df = pd.read_csv(events_path)
    if df.empty:
        raise SystemExit("No rows in phase3f_event_cells_with_gen.csv")

    mu_pick = _pick_mu(df, mu)
    df = df[np.isfinite(pd.to_numeric(df["mu"], errors="coerce"))].copy()
    df = df[df["mu"].apply(lambda x: _nearly_equal(float(x), mu_pick, 1e-10))].copy()
    if df.empty:
        raise SystemExit(f"No event rows for mu={mu_pick}")

    # optionally use the (genA, genB) pair from phase3f_word_test.json
    wanted: Optional[set[str]] = None
    if use_word_test_pair:
        wp = run_dir / "phase3f_word_test.json"
        if not wp.exists():
            raise SystemExit(f"Missing: {wp} (needed for --use_word_test_pair)")
        import json

        with open(wp, "r", encoding="utf-8") as f:
            d = json.load(f)
        genA = str(d.get("genA", "")).strip()
        genB = str(d.get("genB", "")).strip()
        if not genA or not genB:
            raise SystemExit("phase3f_word_test.json missing genA/genB")
        wanted = {genA, genB}

    # choose one representative dump per gen_id: minimal dist_to_minus1
    if "rep_holonomy_dump_abs" in df.columns:
        dump_col = "rep_holonomy_dump_abs"
    else:
        dump_col = "rep_holonomy_dump"

    df["rep_holonomy_min_dist_to_minus1"] = pd.to_numeric(df.get("rep_holonomy_min_dist_to_minus1"), errors="coerce")
    rows = []
    for gen_id, g in df.groupby("gen_id", sort=True):
        gen_id = str(gen_id)
        if wanted is not None and gen_id not in wanted:
            continue
        g2 = g.sort_values(["rep_holonomy_min_dist_to_minus1"], ascending=[True]).head(1)
        row = g2.iloc[0]
        dump_path = str(row.get(dump_col, "")).strip()
        if not dump_path:
            continue
        p = Path(dump_path)
        if not p.is_absolute():
            p = (run_dir / dump_path).resolve()
        if not p.exists():
            continue
        rows.append((gen_id, p))

    if not rows:
        raise SystemExit("Could not resolve any generator holonomy dumps")

    # stable ordering: by gen_local_id from generators.csv if present, else by gen_id
    gen_order: Dict[str, int] = {}
    gens_csv = run_dir / "phase3f_generators.csv"
    if gens_csv.exists():
        gdf = pd.read_csv(gens_csv)
        if not gdf.empty and "gen_id" in gdf.columns and "gen_local_id" in gdf.columns and "mu" in gdf.columns:
            gdf = gdf[gdf["mu"].apply(lambda x: _nearly_equal(float(x), mu_pick, 1e-10))].copy()
            for _, r in gdf.iterrows():
                gid = str(r.get("gen_id", "")).strip()
                try:
                    gen_order[gid] = int(r.get("gen_local_id"))
                except Exception:
                    continue

    rows_sorted = sorted(rows, key=lambda t: (gen_order.get(t[0], 10**9), t[0]))

    gens: List[Gen] = []
    for i, (gen_id, dump_path) in enumerate(rows_sorted):
        M = _load_M(dump_path)
        theta = _theta_from_trace(M)
        defect = _fro_norm_I_minus_M(M)
        dist_minus1 = _min_dist_to_minus1(M)
        gens.append(Gen(name=f"g{i}", M=M, theta=theta, defect=defect, dist_minus1=dist_minus1))

    if len(gens) < 2:
        raise SystemExit(f"Need at least 2 generators at mu={mu_pick}; got {len(gens)}")

    return gens


def _enumerate_words(n_symbols: int, max_len: int) -> Iterable[Tuple[int, ...]]:
    # full enumeration of words over n_symbols letters
    for L in range(1, max_len + 1):
        # iterative product to avoid recursion
        words = [()]  # type: ignore[var-annotated]
        for _ in range(L):
            new_words = []
            for w in words:
                for j in range(n_symbols):
                    new_words.append(w + (j,))
            words = new_words
        for w in words:
            yield w


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Noncommutative word-zeta probe: enumerate short words in the active generator set, "
            "compute trace-like contributions of M_word with an exp(-s*ell(w)) weight, and summarize stability."
        )
    )
    ap.add_argument("--run_dir", required=True, help="Refine/refine2 run directory containing phase3f_event_cells_with_gen.csv")
    ap.add_argument("--mu", type=float, default=None, help="Which mu slice to use (default: max mu present in run)")
    ap.add_argument(
        "--use_word_test_pair",
        action="store_true",
        help="Use (genA, genB) from phase3f_word_test.json rather than all generators",
    )
    ap.add_argument("--max_word_len", type=int, default=4, help="Max word length to enumerate (default: 4)")
    ap.add_argument("--s_values", default="0.5,1,2,3", help="Comma-separated s values to evaluate")
    ap.add_argument(
        "--length_mode",
        default="theta_sum",
        choices=[
            "theta_sum",
            "gen_defect_sum",
            "word_defect",
            "word_minus1_log10",
            "word_minus1_log1p_inv",
            "gen_minus1_log10_sum",
        ],
        help=(
            "How to define word length ell(w). 'theta_sum' uses sum acos(Re(tr)/d) per letter. "
            "'gen_defect_sum' uses sum log1p(||I-M_g||_F) per letter. "
            "'word_defect' uses log1p(||I-M_w||_F) for the full word (captures cancellations). "
            "'word_minus1_log10' uses max(0, -log10(min|eig(M_w)+1|)). "
            "'word_minus1_log1p_inv' uses log1p(1/min|eig(M_w)+1|) (no mass at 0 unless dist is infinite). "
            "'gen_minus1_log10_sum' uses sum max(0, -log10(min|eig(M_g)+1|)) per letter."
        ),
    )
    ap.add_argument(
        "--term_mode",
        default="trace",
        choices=["trace", "trace_real", "abs_trace"],
        help="Which scalar term to use from M_w before applying exp(-s*ell): trace, real(trace), or |trace|",
    )
    ap.add_argument(
        "--eps",
        type=float,
        default=1e-16,
        help="Small epsilon used in log10 for minus1-distance length modes (default: 1e-16)",
    )
    ap.add_argument(
        "--length_scale",
        type=float,
        default=1.0,
        help="Optional multiplicative scale applied to ell(w) before exp(-s*ell) (default: 1.0)",
    )
    ap.add_argument("--out_prefix", default="word_zeta_probe", help="Output filename prefix")
    ap.add_argument(
        "--dump_terms",
        action="store_true",
        help="Also write per-word term table (can be large)",
    )
    ap.add_argument(
        "--write_cancellation_summary",
        action="store_true",
        help=(
            "Write an additional per-s cancellation summary CSV (<out_prefix>_cancellation.csv) that "
            "quantifies sign-cancellation in contrib_real and where the absolute mass sits."
        ),
    )
    args = ap.parse_args()

    run_dir = Path(str(args.run_dir)).resolve()
    gens = _build_generators(run_dir=run_dir, mu=args.mu, use_word_test_pair=bool(args.use_word_test_pair))

    names = [g.name for g in gens]
    mats = [g.M for g in gens]
    thetas = np.array([g.theta for g in gens], dtype=float)
    defects = np.array([g.defect for g in gens], dtype=float)
    distm1 = np.array([g.dist_minus1 for g in gens], dtype=float)

    s_values = _parse_csv_floats(str(args.s_values))
    eps = float(args.eps)
    length_scale = float(args.length_scale)

    d = mats[0].shape[0]
    n_total = int(sum(len(gens) ** L for L in range(1, int(args.max_word_len) + 1)))

    # Precompute primitive words and their (ell, tr, term) once.
    primitive_words: List[Tuple[int, ...]] = []
    primitive_ell: List[float] = []
    primitive_tr: List[complex] = []
    primitive_term: List[complex] = []

    for w in _enumerate_words(len(gens), int(args.max_word_len)):
        if _word_is_power(w):
            continue

        M = np.eye(d, dtype=np.complex128)
        for idx in w:
            M = M @ mats[int(idx)]

        if args.length_mode == "theta_sum":
            ell = float(np.sum(thetas[list(w)]))
        elif args.length_mode == "gen_defect_sum":
            ell = float(np.sum(np.log1p(defects[list(w)])))
        elif args.length_mode == "gen_minus1_log10_sum":
            ell = float(np.sum(-np.log10(np.maximum(distm1[list(w)], eps))))
            ell = max(0.0, ell)
        elif args.length_mode == "word_defect":
            ell = float(np.log1p(_fro_norm_I_minus_M(M)))
        elif args.length_mode == "word_minus1_log10":
            ell = float(-math.log10(max(_min_dist_to_minus1(M), eps)))
            ell = max(0.0, ell)
        elif args.length_mode == "word_minus1_log1p_inv":
            dist = max(_min_dist_to_minus1(M), eps)
            ell = float(math.log1p(1.0 / dist))
        else:
            raise ValueError(f"Unknown length_mode: {args.length_mode}")

        ell = length_scale * ell

        tr = complex(np.trace(M))
        if args.term_mode == "trace":
            term = tr
        elif args.term_mode == "trace_real":
            term = complex(float(np.real(tr)), 0.0)
        elif args.term_mode == "abs_trace":
            term = complex(float(abs(tr)), 0.0)
        else:
            raise ValueError(f"Unknown term_mode: {args.term_mode}")

        primitive_words.append(w)
        primitive_ell.append(ell)
        primitive_tr.append(tr)
        primitive_term.append(term)

    ell_arr = np.asarray(primitive_ell, dtype=float)
    tr_arr = np.asarray(primitive_tr, dtype=np.complex128)
    term_arr = np.asarray(primitive_term, dtype=np.complex128)

    term_rows = []
    agg_rows = []
    cancel_rows = []

    # These are independent of s.
    ell_min = float(np.min(ell_arr)) if ell_arr.size else float("nan")
    ell_med = float(np.median(ell_arr)) if ell_arr.size else float("nan")
    ell_max = float(np.max(ell_arr)) if ell_arr.size else float("nan")
    ell_p10 = float(np.quantile(ell_arr, 0.10)) if ell_arr.size else float("nan")
    ell_p90 = float(np.quantile(ell_arr, 0.90)) if ell_arr.size else float("nan")
    ell_zero_frac = float(np.mean(ell_arr <= 0.0)) if ell_arr.size else float("nan")

    for s in s_values:
        weight_arr = np.exp(-float(s) * ell_arr)
        contrib_arr = weight_arr * term_arr

        contrib_real_arr = np.real(contrib_arr)
        contrib_abs_arr = np.abs(contrib_arr)
        sum_real = float(np.sum(contrib_real_arr))
        sum_abs = float(np.sum(contrib_abs_arr))

        w_min = float(np.min(weight_arr)) if weight_arr.size else float("nan")
        w_med = float(np.median(weight_arr)) if weight_arr.size else float("nan")
        w_max = float(np.max(weight_arr)) if weight_arr.size else float("nan")
        w_p10 = float(np.quantile(weight_arr, 0.10)) if weight_arr.size else float("nan")
        w_p90 = float(np.quantile(weight_arr, 0.90)) if weight_arr.size else float("nan")

        if args.dump_terms:
            for w, ell, tr, contrib in zip(primitive_words, ell_arr.tolist(), tr_arr.tolist(), contrib_arr.tolist()):
                term_rows.append(
                    {
                        "s": float(s),
                        "word": _word_to_str(w, names),
                        "len": int(len(w)),
                        "ell": float(ell),
                        "tr_real": float(np.real(tr)),
                        "tr_imag": float(np.imag(tr)),
                        "contrib_real": float(np.real(contrib)),
                        "contrib_abs": float(abs(contrib)),
                    }
                )

        if args.write_cancellation_summary:
            neg = contrib_real_arr < 0
            pos = contrib_real_arr > 0
            abs_sum = float(np.sum(contrib_abs_arr))
            abs_neg = float(np.sum(contrib_abs_arr[neg])) if contrib_abs_arr.size else 0.0
            abs_pos = float(np.sum(contrib_abs_arr[pos])) if contrib_abs_arr.size else 0.0
            cancel_rows.append(
                {
                    "s": float(s),
                    "n_words_primitive": int(len(primitive_words)),
                    "frac_neg": float(np.mean(neg)) if neg.size else float("nan"),
                    "frac_pos": float(np.mean(pos)) if pos.size else float("nan"),
                    "sum_contrib_real": float(sum_real),
                    "sum_pos_real": float(np.sum(contrib_real_arr[pos])) if pos.size else 0.0,
                    "sum_neg_real": float(np.sum(contrib_real_arr[neg])) if neg.size else 0.0,
                    "sum_contrib_abs": float(abs_sum),
                    "abs_neg_share": float(abs_neg / abs_sum) if abs_sum != 0.0 else float("nan"),
                    "abs_pos_share": float(abs_pos / abs_sum) if abs_sum != 0.0 else float("nan"),
                    "cancellation_ratio": float(sum_real / float(np.sum(np.abs(contrib_real_arr))))
                    if float(np.sum(np.abs(contrib_real_arr))) != 0.0
                    else float("nan"),
                    "contrib_real_p10": float(np.quantile(contrib_real_arr, 0.10)) if contrib_real_arr.size else float("nan"),
                    "contrib_real_p50": float(np.quantile(contrib_real_arr, 0.50)) if contrib_real_arr.size else float("nan"),
                    "contrib_real_p90": float(np.quantile(contrib_real_arr, 0.90)) if contrib_real_arr.size else float("nan"),
                }
            )

        agg_rows.append(
            {
                "s": float(s),
                "length_mode": str(args.length_mode),
                "term_mode": str(args.term_mode),
                "max_word_len": int(args.max_word_len),
                "n_gens": int(len(gens)),
                "n_words_total": int(n_total),
                "n_words_primitive": int(len(primitive_words)),
                "sum_contrib_real": float(sum_real),
                "sum_contrib_abs": float(sum_abs),
                "ell_min": ell_min,
                "ell_median": ell_med,
                "ell_max": ell_max,
                "ell_p10": ell_p10,
                "ell_p90": ell_p90,
                "ell_zero_frac": ell_zero_frac,
                "weight_min": w_min,
                "weight_median": w_med,
                "weight_max": w_max,
                "weight_p10": w_p10,
                "weight_p90": w_p90,
                "theta_g_min": float(np.min(thetas)),
                "theta_g_median": float(np.median(thetas)),
                "theta_g_max": float(np.max(thetas)),
                "defect_g_min": float(np.min(defects)),
                "defect_g_median": float(np.median(defects)),
                "defect_g_max": float(np.max(defects)),
                "distm1_g_min": float(np.min(distm1)),
                "distm1_g_median": float(np.median(distm1)),
                "distm1_g_max": float(np.max(distm1)),
            }
        )

    agg_out = run_dir / f"{args.out_prefix}_summary.csv"
    pd.DataFrame(agg_rows).to_csv(agg_out, index=False)
    print(f"Wrote: {agg_out}")

    if args.write_cancellation_summary:
        cancel_out = run_dir / f"{args.out_prefix}_cancellation.csv"
        pd.DataFrame(cancel_rows).to_csv(cancel_out, index=False)
        print(f"Wrote: {cancel_out}")

    if args.dump_terms:
        terms_out = run_dir / f"{args.out_prefix}_terms.csv"
        pd.DataFrame(term_rows).to_csv(terms_out, index=False)
        print(f"Wrote: {terms_out}")


if __name__ == "__main__":
    main()
