import argparse
import json
import math
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def angle_wrap_pi(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return ((x + np.pi) % (2.0 * np.pi)) - np.pi


def _event_id(e0: int, e1: int, l0: float, l1: float, block: int, dim: int) -> str:
    return f"E{int(e0)}_{int(e1)}_lam{float(l0):.10g}_{float(l1):.10g}_b{int(block)}_d{int(dim)}"


def load_v_minus1(npz_path: str) -> np.ndarray:
    z = np.load(npz_path, allow_pickle=True)
    v = np.asarray(z["v_minus1"], dtype=np.complex128)
    vn = float(np.linalg.norm(v))
    if vn > 0:
        v = v / vn
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True, help="Run folder containing phase3e_elambda_suite_rows.csv")
    ap.add_argument(
        "--dump_index_csv",
        default="",
        help="Optional CSV with holonomy dumps (e.g. phase3f_pi_dump_index.csv) to merge onto rows when holonomy_dump is absent.",
    )
    ap.add_argument(
        "--dump_root",
        default="",
        help="Base directory for resolving relative holonomy_dump paths (defaults to --out_root). Useful when dumps live in a separate folder.",
    )
    ap.add_argument("--tol_pi", type=float, default=0.20, help="| |wrap(det_phase)| - pi | <= tol (radians)")
    ap.add_argument("--min_n", type=int, default=10, help="Minimum rows per event cell")
    ap.add_argument("--max_dim_for_flip", type=int, default=32, help="Only attempt flip-mode overlap stats for dim <= this")
    args = ap.parse_args()

    out_root = str(args.out_root)
    dump_root = str(args.dump_root).strip() or out_root
    rows_csv = os.path.join(out_root, "phase3e_elambda_suite_rows.csv")
    if not os.path.exists(rows_csv):
        raise SystemExit(f"Missing rows CSV: {rows_csv}")

    df = pd.read_csv(rows_csv)

    # Optional: merge in holonomy_dump paths from an external index.
    if str(args.dump_index_csv).strip():
        dump_csv = str(args.dump_index_csv)
        if not os.path.isabs(dump_csv):
            dump_csv = os.path.join(out_root, dump_csv)
        if os.path.exists(dump_csv):
            di = pd.read_csv(dump_csv)
            merge_cols = [
                "seed",
                "anchor_seed",
                "wlo",
                "whi",
                "loop_e0",
                "loop_e1",
                "lambda_j0",
                "lambda_j1",
                "block",
                "dim",
            ]
            keep = [c for c in merge_cols if c in df.columns and c in di.columns]
            if keep and ("holonomy_dump" in di.columns):
                df = df.merge(di[keep + ["holonomy_dump"]], on=keep, how="left", suffixes=("", "_dump"))
                if "holonomy_dump_dump" in df.columns:
                    # Prefer explicit dump index where present
                    df["holonomy_dump"] = df["holonomy_dump_dump"].where(df["holonomy_dump_dump"].notna(), df.get("holonomy_dump", ""))
                    df.drop(columns=["holonomy_dump_dump"], inplace=True)

    # Canonical fwd rectangles are the natural wall-event cells.
    fwd = df[
        (df.get("loop_type") == "rectangle")
        & (df.get("loop_direction") == "fwd")
        & (df.get("loop_double") == False)
        & (df.get("valid", True) == True)
    ].copy()

    if not len(fwd):
        raise SystemExit("No canonical fwd-rectangle valid rows found")

    if "holonomy_det_phase" not in fwd.columns:
        raise SystemExit("rows CSV missing holonomy_det_phase")

    lam0_col = "lambda_eff0" if "lambda_eff0" in fwd.columns else "lambda0"
    lam1_col = "lambda_eff1" if "lambda_eff1" in fwd.columns else "lambda1"

    ph = angle_wrap_pi(fwd["holonomy_det_phase"].values.astype(np.float64))
    aph = np.abs(ph)
    is_pi = np.abs(aph - math.pi) <= float(args.tol_pi)
    fwd["det_phase_near_pi"] = is_pi

    if "holonomy_dump" not in fwd.columns:
        fwd["holonomy_dump"] = ""

    group_cols = ["loop_e0", "loop_e1", lam0_col, lam1_col, "block", "dim"]
    gb = fwd.groupby(group_cols, dropna=False)

    cell_rows: List[Dict[str, Any]] = []
    registry: List[Dict[str, Any]] = []

    for key, g in gb:
        (e0, e1, l0, l1, block, dim) = key
        e0i = int(e0)
        e1i = int(e1)
        l0f = float(l0)
        l1f = float(l1)
        bi = int(block)
        di = int(dim)

        n = int(len(g))
        if n < int(args.min_n):
            continue

        g_pi = g[g["det_phase_near_pi"] == True]
        n_pi = int(len(g_pi))
        pi_rate = float(n_pi / n) if n else float("nan")

        step_smin_min_min = float(np.nanmin(g.get("step_smin_min", np.nan).values.astype(np.float64)))
        mindist_min = float(np.nanmin(g.get("holonomy_min_dist_to_minus1", np.nan).values.astype(np.float64)))

        event_id = _event_id(e0i, e1i, l0f, l1f, bi, di)

        # Flip-mode identity: use v_minus1 overlaps across pi rows with dumps.
        flip_stats: Dict[str, Any] = {
            "flip_overlap_n": 0,
            "flip_overlap_min": float("nan"),
            "flip_overlap_median": float("nan"),
        }

        dumps = g_pi[g_pi["holonomy_dump"].astype(str).str.len() > 0]
        if di <= int(args.max_dim_for_flip) and len(dumps) >= 2:
            dumps = dumps.copy()
            dumps["_mind"] = dumps.get("holonomy_min_dist_to_minus1", np.nan)
            dumps = dumps.sort_values(["_mind"], ascending=True)

            dump_paths: List[str] = []
            for p in dumps["holonomy_dump"].tolist():
                sp = str(p).replace("\\", "/")
                if os.path.isabs(sp):
                    dump_paths.append(sp)
                else:
                    dump_paths.append(os.path.join(dump_root, sp))
            dump_paths = [p for p in dump_paths if os.path.exists(p)]

            if len(dump_paths) >= 2:
                v_ref = load_v_minus1(dump_paths[0])
                ovs: List[float] = []
                for p in dump_paths:
                    v = load_v_minus1(p)
                    ov = float(abs(np.vdot(v_ref, v)))
                    ovs.append(ov)

                if ovs:
                    flip_stats["flip_overlap_n"] = int(len(ovs))
                    flip_stats["flip_overlap_min"] = float(np.min(ovs))
                    flip_stats["flip_overlap_median"] = float(np.median(ovs))

        cell_row: Dict[str, Any] = {
            "event_id": event_id,
            "loop_e0": e0i,
            "loop_e1": e1i,
            "lambda0": l0f,
            "lambda1": l1f,
            "block": bi,
            "dim": di,
            "n": n,
            "n_pi": n_pi,
            "pi_rate": pi_rate,
            "min_step_smin_min": step_smin_min_min,
            "min_holonomy_min_dist_to_minus1": mindist_min,
            **flip_stats,
        }
        cell_rows.append(cell_row)

        reg_entry = dict(cell_row)
        registry.append(reg_entry)

    if not cell_rows:
        raise SystemExit("No event cells met min_n; try lowering --min_n")

    cells_df = pd.DataFrame(cell_rows).sort_values(
        ["pi_rate", "min_step_smin_min"], ascending=[False, True]
    )

    cells_df.to_csv(os.path.join(out_root, "phase3f_event_cells.csv"), index=False)
    cells_df.head(20).to_csv(os.path.join(out_root, "phase3f_event_cells_top20.csv"), index=False)

    with open(os.path.join(out_root, "phase3f_event_registry.json"), "w", encoding="utf-8") as f:
        json.dump({"event_cells": registry}, f, indent=2)


if __name__ == "__main__":
    main()
