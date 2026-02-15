import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "csv",
        nargs="?",
        default=r"runs/20260210_031717__expE_doc_validation_track/E_spine_checks.csv",
        help="Path to E_spine_checks.csv",
    )
    ap.add_argument("--sigma", type=float, default=0.5)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)

    cols = [
        "swap_plus_best",
        "swap_minus_best",
        "chan_sym_best",
        "chan_anti_best",
    ]

    for N in sorted(df["N"].unique()):
        sub = df[(df["N"] == N) & (df["sigma"] == args.sigma)].copy()
        print(f"\nN={int(N)} sigma={args.sigma} rows={len(sub)}")
        for col in cols:
            if col not in sub.columns:
                continue
            vc = sub[col].value_counts(dropna=False)
            print(f"{col}: {vc.to_dict()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
