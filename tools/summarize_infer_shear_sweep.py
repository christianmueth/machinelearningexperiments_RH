from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize infer_satake_from_sixby6 shear sweep CSVs")
    ap.add_argument("csvs", nargs="+", help="One or more CSV paths")
    ap.add_argument("--top", type=int, default=5, help="Show top-N worst primes for the last CSV")
    args = ap.parse_args()

    rows = []
    last_df: pd.DataFrame | None = None
    last_path: Path | None = None

    for csv_path_str in args.csvs:
        path = Path(csv_path_str)
        df = pd.read_csv(path)
        last_df = df
        last_path = path

        max_eig = df[["abs_eig1_minus1", "abs_eig2_minus1"]].max(axis=1)
        rows.append(
            {
                "csv": str(path).replace("\\", "/"),
                "X_shear": float(df["X_shear"].iloc[0]) if "X_shear" in df.columns and len(df) else float("nan"),
                "X_lower": float(df["X_lower"].iloc[0]) if "X_lower" in df.columns and len(df) else float("nan"),
                "gamma": float(df["gamma"].iloc[0]) if "gamma" in df.columns and len(df) else float("nan"),
                "n": int(len(df)),
                "median_max|eig-1|": float(max_eig.median()),
                "max_max|eig-1|": float(max_eig.max()),
                "median_|tr-2|": float(df["abs_trace_minus2"].median()),
                "median_unit_def": float(df["unitarity_defect"].median()),
                "max_unit_def": float(df["unitarity_defect"].max()),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["X_shear", "X_lower", "gamma"], na_position="last")
    pd.set_option("display.width", 220)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3e}"))

    if last_df is not None and last_path is not None:
        max_eig = last_df[["abs_eig1_minus1", "abs_eig2_minus1"]].max(axis=1)
        worst = (
            last_df.assign(max_eig=max_eig)
            .sort_values("max_eig", ascending=False)
            .head(int(args.top))
        )
        last_path_disp = str(last_path).replace("\\\\", "/")
        print(f"\nTop-{int(args.top)} primes by max|eig-1| in {last_path_disp}:")
        cols = [
            c
            for c in [
                "p",
                "X_shear",
                "X_lower",
                "gamma",
                "max_eig",
                "unitarity_defect",
                "eig1_re",
                "eig1_im",
                "eig2_re",
                "eig2_im",
            ]
            if c in worst.columns
        ]
        print(worst[cols].to_string(index=False, float_format=lambda x: f"{x:.3e}"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
