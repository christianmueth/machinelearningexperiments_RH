import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "nan"
    ax = abs(float(x))
    if ax != 0 and (ax < 1e-3 or ax >= 1e3):
        return f"{float(x):.3g}"
    return f"{float(x):.4g}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize tracked Swap± eigenvalue branches from Experiment E output.")
    ap.add_argument(
        "--run",
        required=True,
        help="Run directory under runs/, e.g. runs/20260210_040420__expE_doc_validation_track",
    )
    ap.add_argument("--sigma", type=float, default=0.5, help="Sigma slice to print per-t details for")
    ap.add_argument("--basis", choices=["rel", "incoming"], default="rel", help="Which tracked branch columns to use")
    args = ap.parse_args()

    run_dir = Path(args.run)
    fn = run_dir / "E_spine_checks.csv"
    df = pd.read_csv(fn)

    suf = "" if args.basis == "rel" else "_in"
    cols_num = [
        f"lam_plus_track{suf}_relerr_mod",
        f"lam_minus_track{suf}_relerr_mod",
        f"lam_plus_track{suf}_gap",
        f"lam_minus_track{suf}_gap",
        f"lam_plus_track{suf}_jump",
        f"lam_minus_track{suf}_jump",
    ]
    for c in cols_num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    rows = []
    for (N, sigma), g in df.groupby(["N", "sigma"]):
        def stat(col: str, how: str) -> float:
            if col not in g.columns:
                return float("nan")
            x = pd.to_numeric(g[col], errors="coerce").to_numpy()
            if how == "median":
                return float(np.nanmedian(x))
            if how == "max":
                return float(np.nanmax(x))
            if how == "min":
                return float(np.nanmin(x))
            raise ValueError(how)

        rows.append(
            {
                "N": int(N),
                "sigma": float(sigma),
                "plus_relerr_med": stat(f"lam_plus_track{suf}_relerr_mod", "median"),
                "plus_relerr_max": stat(f"lam_plus_track{suf}_relerr_mod", "max"),
                "plus_gap_min": stat(f"lam_plus_track{suf}_gap", "min"),
                "plus_jump_max": stat(f"lam_plus_track{suf}_jump", "max"),
                "minus_relerr_med": stat(f"lam_minus_track{suf}_relerr_mod", "median"),
                "minus_relerr_max": stat(f"lam_minus_track{suf}_relerr_mod", "max"),
                "minus_gap_min": stat(f"lam_minus_track{suf}_gap", "min"),
                "minus_jump_max": stat(f"lam_minus_track{suf}_jump", "max"),
            }
        )

    out = pd.DataFrame(rows).sort_values(["N", "sigma"], ascending=True)

    print("RUN:", run_dir)
    print(f"\nBranch tracking summary (Swap±, basis={args.basis}): relerr_mod, gap, jump")
    if out.empty:
        print("(no rows)")
    else:
        print(
            out.to_string(
                index=False,
                formatters={c: _fmt for c in out.columns if c not in ["N", "sigma"]},
            )
        )

    # label counts at a sigma slice
    sigma = float(args.sigma)
    for N in sorted(df["N"].unique()):
        g = df[(df["sigma"] == sigma) & (df["N"] == N)].copy()
        if g.empty:
            continue
        print(f"\nLabel counts (tracked branch) N={int(N)} sigma={sigma}:")
        for side in ["plus", "minus"]:
            col = f"lam_{side}_track{suf}_best"
            if col not in g.columns:
                continue
            vc = g[col].fillna("(na)").value_counts()
            print(" ", side, {k: int(v) for k, v in vc.items()})

    # per-t details
    for N in sorted(df["N"].unique()):
        g = df[(df["sigma"] == sigma) & (df["N"] == N)].sort_values("t")
        if g.empty:
            continue
        cols = [
            "t",
            f"lam_plus_track{suf}_relerr_mod",
            f"lam_plus_track{suf}_gap",
            f"lam_plus_track{suf}_jump",
            f"lam_minus_track{suf}_relerr_mod",
            f"lam_minus_track{suf}_gap",
            f"lam_minus_track{suf}_jump",
        ]
        cols = [c for c in cols if c in g.columns]
        print(f"\nPer-t (N={int(N)}, sigma={sigma}):")
        print(g[cols].to_string(index=False, float_format=lambda v: f"{float(v):.4g}"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
