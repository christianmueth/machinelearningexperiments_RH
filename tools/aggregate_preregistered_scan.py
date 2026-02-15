"""Aggregate emitted per-job rows.csv into raw + KPI tables.

This is safe to run while a scan is still in progress; it only aggregates
existing outputs.

Usage (PowerShell):
  & .\.venv\Scripts\python.exe tools\aggregate_preregistered_scan.py out_root

Outputs (written under out_root):
  - raw_agg.csv
  - kpi_by_seed_agg.csv
  - kpi_fraction_agg.csv

Notes:
- This does not change any statistics or rerun any measurement.
- If the scan driver already writes raw.csv/kpi_by_seed.csv/etc at the end,
  these *_agg.csv files can be used as a mid-scan snapshot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python tools/aggregate_preregistered_scan.py <out_root>")
        return 2

    out_root = Path(sys.argv[1])
    if not out_root.exists():
        print(f"ERROR: not found: {out_root}")
        return 2

    rows_csvs = list(out_root.rglob("rows.csv"))
    if not rows_csvs:
        print("No rows.csv found.")
        return 0

    frames = []
    for p in rows_csvs:
        try:
            frames.append(pd.read_csv(p))
        except Exception as e:
            print(f"WARN: failed to read {p}: {e}")

    if not frames:
        print("No readable rows.csv files.")
        return 1

    big = pd.concat(frames, ignore_index=True)
    raw_path = out_root / "raw_agg.csv"
    big.to_csv(raw_path, index=False)
    print(f"Wrote: {raw_path}")

    # KPI: per-seed and fraction (mirrors the scan driver's logic)
    if "reject" not in big.columns:
        print("No 'reject' column; skipping KPI tables.")
        return 0

    if "dn_identity_rank" in big.columns:
        median_rank_agg = ("dn_identity_rank", "median")
    else:
        median_rank_agg = ("p_family", lambda x: np.nan)

    by_seed = (
        big.groupby(["backend_label", "wlo", "whi", "seed"], dropna=False)
        .agg(
            any_reject=("reject", "max"),
            best_p=("p_family", "min"),
            worst_p=("p_family", "max"),
            median_p=("p_family", "median"),
            median_dn_identity_rank=median_rank_agg,
            rows=("reject", "size"),
        )
        .reset_index()
    )
    by_seed["any_reject"] = by_seed["any_reject"].astype(int)

    by_seed_path = out_root / "kpi_by_seed_agg.csv"
    by_seed.to_csv(by_seed_path, index=False)
    print(f"Wrote: {by_seed_path}")

    frac = (
        by_seed.groupby(["backend_label", "wlo", "whi"], dropna=False)
        .agg(
            n_seeds=("seed", "nunique"),
            n_seeds_any_reject=("any_reject", "sum"),
            frac_seeds_any_reject=("any_reject", "mean"),
            best_p_overall=("best_p", "min"),
            median_best_p=("best_p", "median"),
            median_rank=("median_dn_identity_rank", "median"),
        )
        .reset_index()
    )

    frac_path = out_root / "kpi_fraction_agg.csv"
    frac.to_csv(frac_path, index=False)
    print(f"Wrote: {frac_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
