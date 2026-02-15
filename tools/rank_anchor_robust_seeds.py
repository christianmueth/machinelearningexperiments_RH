"""Rank scan seeds by anchor robustness.

This is Phase-3B-legal: it only post-processes scan outputs.

It tries to read <scan_root>/kpi_by_seed.csv (preferred). If that file doesn't
contain anchor robustness columns (older runs), it falls back to computing them
from <scan_root>/raw.csv.

Prereg-safe defaults:
- If <scan_root>/preregistration.json exists, the script uses its first declared
    window as the "primary" window, unless you explicitly override --wlo/--whi.
- The script is read-only by default; use --write_rankings to write an auditable
    seed_rankings.csv artifact.

PowerShell examples:
  & .\.venv\Scripts\python.exe tools\rank_anchor_robust_seeds.py --scan_root out_phase3B_preregistered_uniqueness_64_127_v5
  & .\.venv\Scripts\python.exe tools\rank_anchor_robust_seeds.py --scan_root out_phase3B_preregistered_uniqueness_64_127_v6_phase --backend_label geom_v6_theta0p125_l0.05_c2.0_autotune

Exit code:
- 0 on success
- 2 if required files are missing
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _compute_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"backend_label", "seed", "anchor_seed", "wlo", "whi", "reject", "p_family"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"raw.csv missing columns: {missing}")

    by_seed = (
        raw.groupby(["backend_label", "wlo", "whi", "seed"], dropna=False)
        .agg(
            any_reject=("reject", "max"),
            anchor_reject_fraction=("reject", "mean"),
            anchor_reject_count=("reject", "sum"),
            anchor_min_p=("p_family", "min"),
            anchor_median_p=("p_family", "median"),
            n_anchors=("anchor_seed", "nunique"),
            rows=("reject", "size"),
        )
        .reset_index()
    )
    by_seed["any_reject"] = by_seed["any_reject"].astype(int)
    return by_seed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_root", type=str, required=True)
    ap.add_argument("--wlo", type=float, default=None, help="Override primary window low edge")
    ap.add_argument("--whi", type=float, default=None, help="Override primary window high edge")
    ap.add_argument("--backend_label", type=str, default=None)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--min_anchor_reject_fraction", type=float, default=0.0)
    ap.add_argument(
        "--write_rankings",
        action="store_true",
        help="Write scan_root/seed_rankings.csv (full ranking, not just top-K) for audit.",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Optional explicit output path (overrides --write_rankings default location).",
    )
    args = ap.parse_args()

    root = Path(args.scan_root)
    prereg_path = root / "preregistration.json"
    kpi_by_seed_path = root / "kpi_by_seed.csv"
    raw_path = root / "raw.csv"

    # Determine primary window (prereg-safe): first prereg window unless overridden.
    wlo = None if args.wlo is None else float(args.wlo)
    whi = None if args.whi is None else float(args.whi)
    if (wlo is None or whi is None) and prereg_path.exists():
        try:
            import json

            with open(prereg_path, "r", encoding="utf-8") as f:
                prereg = json.load(f)
            wins = prereg.get("windows", None)
            if isinstance(wins, list) and len(wins) >= 1:
                pw = wins[0]
                if isinstance(pw, (list, tuple)) and len(pw) == 2:
                    if wlo is None:
                        wlo = float(pw[0])
                    if whi is None:
                        whi = float(pw[1])
        except Exception:
            # fall back to defaults below
            pass
    if wlo is None:
        wlo = 2.0
    if whi is None:
        whi = 5.0

    if kpi_by_seed_path.exists():
        by_seed = pd.read_csv(kpi_by_seed_path)
    elif raw_path.exists():
        by_seed = _compute_from_raw(pd.read_csv(raw_path))
    else:
        print(f"Missing both {kpi_by_seed_path} and {raw_path}")
        return 2

    # If kpi_by_seed.csv exists but is old, compute from raw.csv.
    needed_cols = {"anchor_reject_fraction", "anchor_min_p", "anchor_median_p", "anchor_reject_count"}
    if not needed_cols.issubset(set(by_seed.columns)):
        if not raw_path.exists():
            raise ValueError(
                "kpi_by_seed.csv does not include anchor robustness columns, and raw.csv is missing; "
                "cannot compute robust ranking."
            )
        by_seed = _compute_from_raw(pd.read_csv(raw_path))

    df = by_seed[(by_seed["wlo"] == float(wlo)) & (by_seed["whi"] == float(whi))].copy()
    if args.backend_label is not None:
        df = df[df["backend_label"] == str(args.backend_label)].copy()

    min_frac = float(args.min_anchor_reject_fraction)
    if np.isfinite(min_frac) and (min_frac > 0):
        df = df[df["anchor_reject_fraction"] >= min_frac].copy()

    # Prefer high anchor_reject_fraction, then low min p.
    df = df.sort_values(
        ["anchor_reject_fraction", "anchor_min_p", "anchor_median_p", "seed"],
        ascending=[False, True, True, True],
    )

    show_cols = [
        c
        for c in [
            "backend_label",
            "seed",
            "anchor_reject_fraction",
            "anchor_reject_count",
            "anchor_min_p",
            "anchor_median_p",
            "rows",
        ]
        if c in df.columns
    ]

    topn = int(max(1, args.top))
    out = df[show_cols].head(topn)

    print(f"scan_root: {root}")
    print(f"window: ({float(wlo)}, {float(whi)})")
    if args.backend_label is not None:
        print(f"backend_label: {args.backend_label}")
    if min_frac > 0:
        print(f"min_anchor_reject_fraction: {min_frac}")
    print(f"rows considered: {len(df)}")
    print("\nTop seeds by anchor robustness:")
    if len(out):
        print(out.to_string(index=False))
    else:
        print("(none)")

    out_path = None
    if args.out_csv:
        out_path = Path(args.out_csv)
    elif args.write_rankings:
        out_path = root / "seed_rankings.csv"
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df[show_cols].to_csv(out_path, index=False)
        print(f"\nWrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
