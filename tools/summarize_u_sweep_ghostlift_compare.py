import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Pivot/compare the u-sweep ghostlift rows into a single per-u table with p vs invp columns. "
            "Designed for quick plotting/inspection."
        )
    )
    ap.add_argument("--in_csv", default="out/u_sweep_ghostlift_rows_window016_024.csv")
    ap.add_argument("--out_csv", default="out/u_sweep_ghostlift_compare_window016_024.csv")
    args = ap.parse_args()

    inp = Path(str(args.in_csv))
    df = pd.read_csv(str(inp))
    if df.empty:
        raise SystemExit("input CSV is empty")

    # Keep a small set of columns.
    keep = [
        "u",
        "p_mode",
        "gated_out",
        "A1",
        "A1_abs",
        "invp_A1_near0",
        "med_abs_logratio_squarefree_pairs",
        "med_abs_logratio_primepowers",
        "med_abs_logratio_mixed",
    ]
    for c in keep:
        if c not in df.columns:
            raise SystemExit(f"missing required column: {c}")

    df = df[keep].copy()

    # Pivot to columns by mode.
    out = (
        df.pivot(index="u", columns="p_mode")
        .sort_index(axis=1, level=1)
        .reset_index()
    )

    # Flatten MultiIndex columns.
    out.columns = [
        (c[0] if c[1] == "" else f"{c[0]}_{c[1]}") if isinstance(c, tuple) else str(c)
        for c in out.columns
    ]

    # Add simple winner flags for pq-fit.
    pq_p = pd.to_numeric(out.get("med_abs_logratio_squarefree_pairs_p"), errors="coerce")
    pq_i = pd.to_numeric(out.get("med_abs_logratio_squarefree_pairs_invp"), errors="coerce")
    out["pq_fit_margin_p_minus_invp"] = pq_p - pq_i
    out["pq_winner"] = np.where(pq_p < pq_i, "p", np.where(pq_i < pq_p, "invp", "tie"))

    out_path = Path(str(args.out_csv))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(str(out_path), index=False)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
