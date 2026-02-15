import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def angle_wrap_pi(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return ((x + np.pi) % (2.0 * np.pi)) - np.pi


def _maybe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _fmt_cell_list(cells: List[Tuple[Any, Any]], *, max_items: int = 12) -> str:
    if not cells:
        return ""
    s = ";".join([f"{a}:{b}" for a, b in cells[: max_items]])
    if len(cells) > max_items:
        s += f";...(+{len(cells) - max_items})"
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=".",
        help="Workspace root containing out_phase3E_elambda* folders (default: .)",
    )
    ap.add_argument(
        "--glob",
        default="out_phase3E_elambda*",
        help="Glob for Phase-3E run folders (default: out_phase3E_elambda*)",
    )
    ap.add_argument(
        "--out_csv",
        default="phase3f_multilambda_candidates.csv",
        help="Output CSV path (default: phase3f_multilambda_candidates.csv in --root)",
    )
    ap.add_argument("--tol_pi", type=float, default=0.25, help="| |wrap(det_phase)| - pi | <= tol (radians)")

    ap.add_argument("--only_blocks", type=int, default=-1, help="If >=0: filter blocks == this")
    ap.add_argument("--only_dim", type=int, default=-1, help="If >=0: filter dim == this")

    ap.add_argument("--max_runs", type=int, default=200, help="Safety cap: max run folders to scan")
    ap.add_argument("--max_rows", type=int, default=-1, help="Safety cap: if >0 only read first N rows of each rows.csv")

    args = ap.parse_args()

    root = Path(str(args.root)).resolve()
    run_dirs = [p for p in sorted(root.glob(str(args.glob))) if p.is_dir()]
    run_dirs = run_dirs[: int(args.max_runs)]

    out_rows: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        rows_csv = run_dir / "phase3e_elambda_suite_rows.csv"
        if not rows_csv.exists():
            continue

        try:
            df = pd.read_csv(rows_csv, nrows=(None if int(args.max_rows) <= 0 else int(args.max_rows)))
        except Exception:
            continue

        req = {"loop_type", "loop_direction", "loop_double", "valid", "holonomy_det_phase"}
        if not req.issubset(df.columns):
            continue

        fwd = df[
            (df["loop_type"] == "rectangle")
            & (df["loop_direction"] == "fwd")
            & (df["loop_double"] == False)
            & (df["valid"] == True)
        ].copy()

        if int(args.only_blocks) >= 0 and "blocks" in fwd.columns:
            fwd = fwd[fwd["blocks"].astype(int) == int(args.only_blocks)]

        if int(args.only_dim) >= 0 and "dim" in fwd.columns:
            fwd = fwd[fwd["dim"].astype(int) == int(args.only_dim)]

        if len(fwd) == 0:
            continue

        ph = angle_wrap_pi(fwd["holonomy_det_phase"].values.astype(np.float64))
        is_pi = np.abs(np.abs(ph) - math.pi) <= float(args.tol_pi)
        pi = fwd[is_pi].copy()
        if len(pi) == 0:
            continue

        # Determine lambda-cell key
        if "lambda_j0" in pi.columns and "lambda_j1" in pi.columns:
            lam_cols = ["lambda_j0", "lambda_j1"]
            lam_kind = "j"
        else:
            lam0 = "lambda_eff0" if "lambda_eff0" in pi.columns else ("lambda0" if "lambda0" in pi.columns else "")
            lam1 = "lambda_eff1" if "lambda_eff1" in pi.columns else ("lambda1" if "lambda1" in pi.columns else "")
            if not lam0 or not lam1:
                continue
            lam_cols = [lam0, lam1]
            lam_kind = "eff"

        key_cols = [
            "seed",
            "anchor_seed",
            "wlo",
            "whi",
            "loop_e0",
            "loop_e1",
            "block",
            "dim",
        ]
        if "mu" in pi.columns:
            key_cols.insert(4, "mu")
        if not set(key_cols).issubset(pi.columns):
            continue

        # Add some stress / conditioning signals if present
        stress_cols = [c for c in ["step_smin_min", "step_cond_max", "holonomy_min_dist_to_minus1"] if c in pi.columns]

        # Distinct lambda cells per (pair,rectangle,block,dim)
        cells_unique = pi[key_cols + lam_cols].drop_duplicates()
        per = (
            cells_unique.groupby(key_cols, dropna=False)
            .size()
            .reset_index(name="n_distinct_lambda_cells")
        )

        # Overall stats for the run
        out_rows.append(
            {
                "run": run_dir.name,
                "run_path": str(run_dir),
                "pi_rows": int(len(pi)),
                "n_groups": int(len(per)),
                "pairs_with_multi_lambda": int((per["n_distinct_lambda_cells"] > 1).sum()),
                "max_lambda_cells_per_group": int(per["n_distinct_lambda_cells"].max()),
                "lambda_cell_key_kind": lam_kind,
            }
        )

        # Emit details for the multi-lambda groups
        multi = per[per["n_distinct_lambda_cells"] > 1].copy()
        if len(multi) == 0:
            continue

        # Build cell list per group
        grp = cells_unique.groupby(key_cols, dropna=False)
        cell_list_map: Dict[Tuple[Any, ...], List[Tuple[Any, Any]]] = {}
        for k, gk in grp:
            lst = [(gk.iloc[i][lam_cols[0]], gk.iloc[i][lam_cols[1]]) for i in range(len(gk))]
            cell_list_map[k] = lst

        # Also count pi rows per lambda-cell within each group for ranking
        pi_counts = (
            pi[key_cols + lam_cols]
            .groupby(key_cols + lam_cols, dropna=False)
            .size()
            .reset_index(name="pi_rows_in_cell")
        )

        for row in multi.itertuples(index=False):
            k = tuple(getattr(row, c) for c in key_cols)
            # gather per-cell counts
            sub = pi_counts
            for i, c in enumerate(key_cols):
                sub = sub[sub[c] == k[i]]
            sub = sub.sort_values("pi_rows_in_cell", ascending=False)
            top_cells = [(sub.iloc[i][lam_cols[0]], sub.iloc[i][lam_cols[1]]) for i in range(min(12, len(sub)))]
            top_counts = [int(sub.iloc[i]["pi_rows_in_cell"]) for i in range(min(12, len(sub)))]

            # stress stats on all pi rows in this group
            sub_pi = pi
            for i, c in enumerate(key_cols):
                sub_pi = sub_pi[sub_pi[c] == k[i]]

            stress_summary: Dict[str, float] = {}
            for sc in stress_cols:
                v = pd.to_numeric(sub_pi[sc], errors="coerce").values
                v = v[np.isfinite(v)]
                stress_summary[f"{sc}_min"] = float(np.min(v)) if v.size else float("nan")
                stress_summary[f"{sc}_p10"] = float(np.quantile(v, 0.10)) if v.size else float("nan")
                stress_summary[f"{sc}_median"] = float(np.median(v)) if v.size else float("nan")

            detail = {
                "run": run_dir.name,
                "run_path": str(run_dir),
                **{c: getattr(row, c) for c in key_cols},
                "n_distinct_lambda_cells": int(getattr(row, "n_distinct_lambda_cells")),
                "lambda_cells_list": _fmt_cell_list(cell_list_map.get(k, []), max_items=24),
                "top_lambda_cells": _fmt_cell_list(top_cells, max_items=12),
                "top_lambda_cells_pi_counts": ";".join([str(x) for x in top_counts]),
                "pi_rows_in_group": int(len(sub_pi)),
                "lambda_cell_key_kind": lam_kind,
            }
            detail.update(stress_summary)

            out_rows.append(detail)

    out_df = pd.DataFrame(out_rows)
    out_path = Path(str(args.out_csv))
    if not out_path.is_absolute():
        out_path = root / out_path

    if len(out_df) == 0:
        out_df = pd.DataFrame(
            [
                {
                    "run": "",
                    "run_path": "",
                    "pi_rows": 0,
                    "n_groups": 0,
                    "pairs_with_multi_lambda": 0,
                    "max_lambda_cells_per_group": 0,
                    "lambda_cell_key_kind": "",
                }
            ]
        ).iloc[0:0]

    # Sort: multi-lambda detail rows first if present
    sort_cols = [
        "pairs_with_multi_lambda",
        "max_lambda_cells_per_group",
        "pi_rows",
    ]
    for c in sort_cols:
        if c not in out_df.columns:
            out_df[c] = np.nan

    out_df = out_df.sort_values(sort_cols, ascending=[False, False, False], na_position="last").reset_index(drop=True)
    out_df.to_csv(out_path, index=False)

    print(f"Wrote: {out_path}")
    # Quick headline
    if "pairs_with_multi_lambda" in out_df.columns and out_df["pairs_with_multi_lambda"].notna().any():
        best = out_df[out_df["pairs_with_multi_lambda"].fillna(0).astype(int) > 0]
        if len(best) == 0:
            print("No multi-lambda (>=2 distinct pi lambda-cells per group) found in scanned runs.")
        else:
            top = best.head(10)
            print("Top multi-lambda candidates:")
            cols = [
                "run",
                "seed",
                "anchor_seed",
                "wlo",
                "whi",
                "loop_e0",
                "loop_e1",
                "block",
                "dim",
                "n_distinct_lambda_cells",
                "pi_rows_in_group",
                "top_lambda_cells",
                "top_lambda_cells_pi_counts",
            ]
            cols = [c for c in cols if c in top.columns]
            print(top[cols].to_string(index=False))


if __name__ == "__main__":
    main()
