"""Summarize stability of raw vs Möbius-extracted k-observables.

Input is the summary CSV emitted by tools/sweep_fe_kobs_stability.py.

Outputs a compact interpretation table answering:
- How much does each metric change as k_max increases?
- How much does each metric vary across completion cases?
- How much does each metric vary across p_modes?

This is intentionally descriptive (hygiene / audit), not a proof of primes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DeltaStats:
    mean_abs_delta: float
    max_abs_delta: float


def _safe_float(x: object) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _delta_stats_over_kmax(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """For each (case,p_mode), compute mean/max |Δ| over successive k_max."""

    rows: list[dict] = []

    for (case, p_mode), g in df.groupby(["case", "p_mode"], dropna=False):
        g2 = g.sort_values("k_max")
        vals = g2[value_col].to_numpy(dtype=float)
        kms = g2["k_max"].to_numpy(dtype=int)

        m = np.isfinite(vals)
        vals = vals[m]
        kms = kms[m]
        if vals.size < 2:
            rows.append(
                {
                    "case": str(case),
                    "p_mode": str(p_mode),
                    "value_col": str(value_col),
                    "k_max_min": int(np.min(kms)) if kms.size else np.nan,
                    "k_max_max": int(np.max(kms)) if kms.size else np.nan,
                    "mean_abs_delta": float("nan"),
                    "max_abs_delta": float("nan"),
                }
            )
            continue

        d = np.abs(np.diff(vals))
        rows.append(
            {
                "case": str(case),
                "p_mode": str(p_mode),
                "value_col": str(value_col),
                "k_max_min": int(np.min(kms)),
                "k_max_max": int(np.max(kms)),
                "mean_abs_delta": float(np.mean(d)) if d.size else float("nan"),
                "max_abs_delta": float(np.max(d)) if d.size else float("nan"),
            }
        )

    return pd.DataFrame(rows)


def _spread_across_group(df: pd.DataFrame, *, group_cols: list[str], value_col: str, spread_label: str) -> pd.DataFrame:
    """Compute spread across the complement dimension inside each group.

    For each group in group_cols, compute:
      - min, max, range (=max-min)
      - mean, std
    """

    rows: list[dict] = []
    for keys, g in df.groupby(group_cols, dropna=False):
        vals = g[value_col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            vmin = vmax = vmean = vstd = vrange = float("nan")
        else:
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
            vrange = float(vmax - vmin)
            vmean = float(np.mean(vals))
            vstd = float(np.std(vals))

        row: dict[str, object] = {"value_col": str(value_col), "spread": str(spread_label)}
        if isinstance(keys, tuple):
            for c, k in zip(group_cols, keys):
                row[str(c)] = k
        else:
            row[str(group_cols[0])] = keys
        row.update({"min": vmin, "max": vmax, "range": vrange, "mean": vmean, "std": vstd, "n": int(vals.size)})
        rows.append(row)

    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Summarize stability/spread for FE k-obs sweep CSV")
    ap.add_argument("--in_csv", required=True, help="Input sweep CSV (from sweep_fe_kobs_stability.py)")
    ap.add_argument("--out_prefix", default="", help="Optional output prefix; writes CSVs if provided")

    args = ap.parse_args(argv)

    df = pd.read_csv(str(args.in_csv))
    if df.empty:
        raise SystemExit("empty input CSV")

    need = {
        "case",
        "p_mode",
        "k_max",
        "kobs_F_l2",
        "kobs_P_l2",
        "kobs_F_maxabs",
        "kobs_P_maxabs",
    }
    missing = sorted([c for c in need if c not in df.columns])
    if missing:
        raise SystemExit(f"missing columns: {missing}")

    # Normalize types
    df = df.copy()
    df["k_max"] = df["k_max"].apply(lambda x: int(_safe_float(x)) if np.isfinite(_safe_float(x)) else np.nan)

    value_cols = ["kobs_F_l2", "kobs_P_l2", "kobs_F_maxabs", "kobs_P_maxabs"]

    deltas = pd.concat([_delta_stats_over_kmax(df, c) for c in value_cols], ignore_index=True)

    # Spread across completions: group by (p_mode,k_max) and look across cases.
    spread_comp = pd.concat(
        [_spread_across_group(df, group_cols=["p_mode", "k_max"], value_col=c, spread_label="across_completion") for c in value_cols],
        ignore_index=True,
    )

    # Spread across p_modes: group by (case,k_max) and look across p_mode.
    spread_pmode = pd.concat(
        [_spread_across_group(df, group_cols=["case", "k_max"], value_col=c, spread_label="across_p_mode") for c in value_cols],
        ignore_index=True,
    )

    # Print compact aggregates.
    def _compact_deltas(table: pd.DataFrame) -> None:
        print("\nΔ over k_max (mean/max |Δ| over successive k_max)")
        agg = (
            table.groupby(["value_col"], dropna=False)
            .agg({"mean_abs_delta": ["mean", "max"], "max_abs_delta": ["mean", "max"]})
            .reset_index()
        )
        agg.columns = [
            "value_col",
            "mean_abs_delta_mean",
            "mean_abs_delta_max",
            "max_abs_delta_mean",
            "max_abs_delta_max",
        ]
        print(agg.to_string(index=False))

    def _compact_spread(name: str, table: pd.DataFrame) -> None:
        print(f"\n{name} (range/std)")
        agg = (
            table.groupby(["value_col"], dropna=False)
            .agg({"range": ["mean", "max"], "std": ["mean", "max"], "n": ["mean", "max"]})
            .reset_index()
        )
        agg.columns = ["value_col", "range_mean", "range_max", "std_mean", "std_max", "n_mean", "n_max"]
        print(agg.to_string(index=False))

    _compact_deltas(deltas)
    _compact_spread("Spread across completion", spread_comp)
    _compact_spread("Spread across p_mode", spread_pmode)

    out_prefix = str(args.out_prefix).strip()
    if out_prefix:
        pd.DataFrame(df).to_csv(f"{out_prefix}_raw.csv", index=False)
        deltas.to_csv(f"{out_prefix}_delta_over_kmax.csv", index=False)
        spread_comp.to_csv(f"{out_prefix}_spread_across_completion.csv", index=False)
        spread_pmode.to_csv(f"{out_prefix}_spread_across_pmode.csv", index=False)
        print(f"\nwrote {out_prefix}_*.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
