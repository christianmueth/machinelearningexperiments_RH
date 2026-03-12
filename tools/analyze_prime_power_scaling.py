import argparse
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PrimeSummary:
    p: int
    n_k: int
    slope_log_abs: float
    intercept_log_abs: float
    r2_log_abs: float
    mean_abs_log_add_err: float
    max_abs_log_add_err: float


def _safe_log_abs(x: float, *, floor: float = 1e-300) -> float:
    return float(math.log(max(float(floor), abs(float(x)))))


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return (slope, intercept, R^2) for y ~ slope*x + intercept."""

    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size or x.size < 2:
        return float("nan"), float("nan"), float("nan")

    X = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    slope = float(coef[0])
    intercept = float(coef[1])

    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / max(1e-300, ss_tot)
    return slope, intercept, float(r2)


def _prime_power_additivity_errors(log_abs_by_k: dict[int, float], k_max: int) -> tuple[float, float]:
    """Return (mean_abs_err, max_abs_err) for log|P(k1+k2)| - log|P(k1)| - log|P(k2)|."""

    errs: list[float] = []
    for k1 in range(1, k_max + 1):
        for k2 in range(1, k_max + 1 - k1):
            k3 = k1 + k2
            if k1 not in log_abs_by_k or k2 not in log_abs_by_k or k3 not in log_abs_by_k:
                continue
            errs.append(float(log_abs_by_k[k3] - log_abs_by_k[k1] - log_abs_by_k[k2]))

    if not errs:
        return float("nan"), float("nan")
    a = np.asarray(errs, dtype=float)
    return float(np.mean(np.abs(a))), float(np.max(np.abs(a)))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Analyze prime-power scaling on per-prime k-observable tables emitted by tools/fe_defect_perturbation_u0.py. "
            "Focuses on semigroup-like laws within prime powers: P_p(k+l) ~ P_p(k)P_p(l), and P_p(k) ~ (P_p(1))^k (in log|.|)."
        )
    )
    ap.add_argument("--kobs_csv", required=True, help="CSV produced by --k_obs_out_csv with --k_obs_emit_per_prime 1")
    ap.add_argument("--label", default="Fu", help="Which label row to analyze (e.g. Fu)")
    ap.add_argument("--u", type=float, default=0.2, help="Target u value (used with a tolerance)")
    ap.add_argument("--u_tol", type=float, default=1e-9, help="Tolerance for matching u")
    ap.add_argument(
        "--value_col",
        choices=["F_k_abs", "P_k_abs", "F_k_re", "P_k_re", "F_k_im", "P_k_im"],
        default="P_k_abs",
        help="Which per-k column to analyze; abs columns are usually the most stable.",
    )
    ap.add_argument(
        "--diff_primitive",
        choices=["0", "1"],
        default="0",
        help=(
            "If 1, analyze the arithmetic prime-power primitive along the p-chain by differencing: X_k := V_k - V_{k-1} "
            "(with V_0 := 0). This corresponds to Möbius inversion over divisors of n restricted to prime powers (mu(p)=-1, mu(p^2)=0, ...)."
        ),
    )
    ap.add_argument("--k_max", type=int, default=7, help="Max k to use in scaling tests")
    ap.add_argument("--out_csv", default="", help="Optional output CSV for per-prime summaries")

    args = ap.parse_args()

    df = pd.read_csv(str(args.kobs_csv))
    if "k_obs_scope" not in df.columns:
        raise SystemExit("kobs_csv missing k_obs_scope column; re-run FE tool with updated version")

    df = df[df["k_obs_scope"] == "per_prime"].copy()
    if df.empty:
        raise SystemExit("No per_prime rows found; ensure --k_obs_emit_per_prime 1 was used")

    df = df[df["label"].astype(str) == str(args.label)].copy()
    df = df[np.abs(df["u"].astype(float) - float(args.u)) <= float(args.u_tol)].copy()
    if df.empty:
        raise SystemExit("No matching rows for given --label/--u")

    if str(args.value_col) not in df.columns:
        raise SystemExit(f"Missing column {args.value_col} in kobs_csv")

    k_max = int(args.k_max)
    if k_max <= 1:
        raise SystemExit("--k_max must be >= 2")

    summaries: list[PrimeSummary] = []

    for p, g in df.groupby("p"):
        p_int = int(p)
        g = g.copy()
        g["k"] = g["k"].astype(int)
        g = g[(g["k"] >= 1) & (g["k"] <= k_max)]
        if g.empty:
            continue

        # Build dict k -> value.
        val_by_k = {int(r.k): float(getattr(r, str(args.value_col))) for r in g.itertuples(index=False)}

        if str(args.diff_primitive).strip() == "1":
            diff_by_k: dict[int, float] = {}
            for k in range(1, k_max + 1):
                v_k = float(val_by_k.get(k, 0.0))
                v_prev = float(val_by_k.get(k - 1, 0.0)) if (k - 1) >= 1 else 0.0
                diff_by_k[k] = float(v_k - v_prev)
            val_by_k = diff_by_k

        log_abs_by_k = {k: _safe_log_abs(v) for k, v in val_by_k.items()}

        ks = np.asarray(sorted(log_abs_by_k.keys()), dtype=float)
        ys = np.asarray([log_abs_by_k[int(k)] for k in ks], dtype=float)
        slope, intercept, r2 = _fit_line(ks, ys)

        mean_abs_err, max_abs_err = _prime_power_additivity_errors(log_abs_by_k, k_max=k_max)

        summaries.append(
            PrimeSummary(
                p=p_int,
                n_k=int(len(ks)),
                slope_log_abs=float(slope),
                intercept_log_abs=float(intercept),
                r2_log_abs=float(r2),
                mean_abs_log_add_err=float(mean_abs_err),
                max_abs_log_add_err=float(max_abs_err),
            )
        )

    if not summaries:
        raise SystemExit("No per-prime summaries produced (check k range / filters)")

    out = pd.DataFrame([s.__dict__ for s in summaries]).sort_values(["p"])

    # Print a compact global view.
    def fmt(x: float) -> str:
        if not math.isfinite(float(x)):
            return "nan"
        return f"{float(x):.4g}"

    print("prime-power scaling report")
    print(f"  rows: {len(out)} primes")
    print(f"  value_col: {args.value_col}")
    print(f"  label/u: {args.label} @ u={float(args.u):g}")
    print(f"  k_max: {k_max}")
    print("  aggregate stats:")
    print("    median R2(log|.| vs k):", fmt(float(out["r2_log_abs"].median())))
    print("    median mean_abs_log_add_err:", fmt(float(out["mean_abs_log_add_err"].median())))
    print("    max max_abs_log_add_err:", fmt(float(out["max_abs_log_add_err"].max())))

    if str(args.out_csv).strip():
        out.to_csv(str(args.out_csv), index=False)
        print(f"wrote {args.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
