import argparse
import math
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def angle_wrap_pi(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return ((x + np.pi) % (2.0 * np.pi)) - np.pi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows_csv", required=True)
    ap.add_argument("--out_csv", default="")
    ap.add_argument("--tol_pi", type=float, default=0.25)
    ap.add_argument("--min_cells", type=int, default=2)
    ap.add_argument(
        "--fixed_e",
        type=int,
        default=0,
        help="If 1: require multiple distinct lambda-cells for the same fixed (loop_e0,loop_e1) per pair+block. If 0: count distinct full cells (loop_e0,loop_e1,lam0,lam1).",
    )
    args = ap.parse_args()

    rows_csv = str(args.rows_csv)
    if not os.path.exists(rows_csv):
        raise SystemExit(f"Missing rows CSV: {rows_csv}")

    df = pd.read_csv(rows_csv)
    fwd = df[
        (df.get("loop_type") == "rectangle")
        & (df.get("loop_direction") == "fwd")
        & (df.get("loop_double") == False)
        & (df.get("valid", True) == True)
    ].copy()

    if not len(fwd):
        raise SystemExit("No canonical fwd-rectangle valid rows found")

    lam0_col = "lambda_eff0" if "lambda_eff0" in fwd.columns else "lambda0"
    lam1_col = "lambda_eff1" if "lambda_eff1" in fwd.columns else "lambda1"

    ph = angle_wrap_pi(fwd["holonomy_det_phase"].values.astype(np.float64))
    aph = np.abs(ph)
    is_pi = np.abs(aph - math.pi) <= float(args.tol_pi)
    pi = fwd[is_pi].copy()

    if not len(pi):
        raise SystemExit("No pi rows at this tol")

    fixed_e = bool(int(args.fixed_e) != 0)

    if fixed_e:
        cell_cols = [lam0_col, lam1_col]
        key_cols = ["seed", "anchor_seed", "wlo", "whi", "block", "dim", "loop_e0", "loop_e1"]
    else:
        cell_cols = ["loop_e0", "loop_e1", lam0_col, lam1_col]
        key_cols = ["seed", "anchor_seed", "wlo", "whi", "block", "dim"]

    # Count distinct lambda-cells per pair+block
    counts = (
        pi.groupby(key_cols, dropna=False)
        .apply(lambda g: g[cell_cols].drop_duplicates().shape[0])
        .reset_index(name="n_pi_cells")
    )

    counts = counts[counts["n_pi_cells"] >= int(args.min_cells)].copy()
    counts = counts.sort_values(["n_pi_cells", "seed", "anchor_seed"], ascending=[False, True, True]).reset_index(drop=True)

    # For each candidate, also export the actual cells (as a compact string)
    cell_lists: List[str] = []
    for _idx, r in counts.iterrows():
        mask = (
            (pi["seed"].astype(int) == int(r["seed"]))
            & (pi["anchor_seed"].astype(int) == int(r["anchor_seed"]))
            & (np.abs(pi["wlo"].astype(float) - float(r["wlo"])) < 1e-12)
            & (np.abs(pi["whi"].astype(float) - float(r["whi"])) < 1e-12)
            & (pi["block"].astype(int) == int(r["block"]))
            & (pi["dim"].astype(int) == int(r["dim"]))
        )
        g = pi[mask]
        cells = (
            g[cell_cols]
            .drop_duplicates()
            .sort_values(cell_cols, ascending=True)
            .values.tolist()
        )
        if fixed_e:
            # stringified list of lambda-cells, each as lam0,lam1
            cell_lists.append(";".join([f"{float(a):.10g},{float(b):.10g}" for (a, b) in cells]))
        else:
            # stringified list of cells, each as e0,e1,lam0,lam1
            cell_lists.append(";".join([f"{int(a)},{int(b)},{float(c):.10g},{float(d):.10g}" for (a, b, c, d) in cells]))

    counts["pi_cells"] = cell_lists

    out_csv = str(args.out_csv).strip() or os.path.join(os.path.dirname(rows_csv), "phase3f_multipi_pairs.csv")
    counts.to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
