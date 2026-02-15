"""Compute per-seed anchor robustness KPIs for both windows (narrow vs wide).

This is Phase-3B-legal: it only post-processes scan outputs.

Inputs:
- scan_root/raw.csv (required)
- scan_root/preregistration.json (optional; used only to label primary/secondary windows)

Outputs:
- scan_root/kpi_by_seed_windowpair.csv
- scan_root/kpi_fraction_windowpair.csv

The output makes window-specific effects explicit by putting both windows on one row:
  anchor_reject_fraction_primary / secondary
  anchor_min_p_primary / secondary

PowerShell:
  & .\.venv\Scripts\python.exe tools\windowpair_anchor_kpis.py --scan_root out_phase3B_preregistered_uniqueness_64_127_v6_phase
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _pick_windows(scan_root: Path) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    prereg_path = scan_root / "preregistration.json"
    if not prereg_path.exists():
        return None, None
    try:
        with open(prereg_path, "r", encoding="utf-8") as f:
            prereg = json.load(f)
        wins = prereg.get("windows", None)
        if not (isinstance(wins, list) and len(wins) >= 1):
            return None, None
        primary = wins[0]
        secondary = wins[1] if len(wins) >= 2 else None
        p = (float(primary[0]), float(primary[1])) if isinstance(primary, (list, tuple)) and len(primary) == 2 else None
        s = (float(secondary[0]), float(secondary[1])) if isinstance(secondary, (list, tuple)) and len(secondary) == 2 else None
        return p, s
    except Exception:
        return None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_root", type=str, required=True)
    args = ap.parse_args()

    root = Path(args.scan_root)
    raw_path = root / "raw.csv"
    if not raw_path.exists():
        print(f"Missing: {raw_path}")
        return 2

    raw = pd.read_csv(raw_path)
    required = {"backend_label", "seed", "anchor_seed", "wlo", "whi", "reject", "p_family"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"raw.csv missing columns: {missing}")

    # Per window per seed
    by_seed = (
        raw.groupby(["backend_label", "wlo", "whi", "seed"], dropna=False)
        .agg(
            anchor_reject_fraction=("reject", "mean"),
            anchor_reject_count=("reject", "sum"),
            anchor_min_p=("p_family", "min"),
            anchor_median_p=("p_family", "median"),
            n_anchors=("anchor_seed", "nunique"),
        )
        .reset_index()
    )

    # Determine which windows are primary/secondary (if prereg available)
    primary, secondary = _pick_windows(root)
    if primary is None:
        # fallback: treat (2,5) as primary if present, else first window in data
        if ((by_seed["wlo"] == 2.0) & (by_seed["whi"] == 5.0)).any():
            primary = (2.0, 5.0)
        else:
            first = by_seed[["wlo", "whi"]].drop_duplicates().sort_values(["wlo", "whi"]).head(1)
            if len(first):
                primary = (float(first.iloc[0]["wlo"]), float(first.iloc[0]["whi"]))

    if secondary is None:
        # fallback: pick the other window if exactly two exist
        wins = by_seed[["wlo", "whi"]].drop_duplicates().sort_values(["wlo", "whi"]).to_records(index=False)
        wins = [(float(a), float(b)) for (a, b) in wins]
        if primary is not None:
            others = [w for w in wins if w != primary]
            if others:
                secondary = others[0]

    def _slice(win: tuple[float, float] | None) -> pd.DataFrame:
        if win is None:
            return pd.DataFrame(columns=["backend_label", "seed"])
        wlo, whi = win
        return by_seed[(by_seed["wlo"] == float(wlo)) & (by_seed["whi"] == float(whi))].copy()

    prim = _slice(primary)
    sec = _slice(secondary)

    def _rename(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        keep = [
            "backend_label",
            "seed",
            "anchor_reject_fraction",
            "anchor_reject_count",
            "anchor_min_p",
            "anchor_median_p",
            "n_anchors",
        ]
        df = df[[c for c in keep if c in df.columns]].copy()
        ren = {c: f"{c}_{suffix}" for c in df.columns if c not in {"backend_label", "seed"}}
        return df.rename(columns=ren)

    out = prim[["backend_label", "seed"]].drop_duplicates().copy() if len(prim) else sec[["backend_label", "seed"]].drop_duplicates().copy()
    out = out.merge(_rename(prim, "primary"), on=["backend_label", "seed"], how="left")
    out = out.merge(_rename(sec, "secondary"), on=["backend_label", "seed"], how="left")

    # Add explicit labels for the chosen windows
    if primary is not None:
        out["primary_window"] = f"{primary[0]},{primary[1]}"
    if secondary is not None:
        out["secondary_window"] = f"{secondary[0]},{secondary[1]}"

    out_path = root / "kpi_by_seed_windowpair.csv"
    out.to_csv(out_path, index=False)

    # Backend-level summary
    frac = (
        out.groupby(["backend_label"], dropna=False)
        .agg(
            n_seeds=("seed", "nunique"),
            median_anchor_reject_fraction_primary=("anchor_reject_fraction_primary", "median"),
            median_anchor_reject_fraction_secondary=("anchor_reject_fraction_secondary", "median"),
            best_anchor_min_p_primary=("anchor_min_p_primary", "min"),
            best_anchor_min_p_secondary=("anchor_min_p_secondary", "min"),
        )
        .reset_index()
    )
    frac_path = root / "kpi_fraction_windowpair.csv"
    frac.to_csv(frac_path, index=False)

    print(f"Wrote: {out_path}")
    print(f"Wrote: {frac_path}")
    if primary is not None:
        print(f"primary_window: {primary}")
    if secondary is not None:
        print(f"secondary_window: {secondary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
