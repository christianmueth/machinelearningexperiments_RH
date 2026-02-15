import argparse
import math
import os
from typing import List

import numpy as np
import pandas as pd


def angle_wrap_pi(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return ((x + np.pi) % (2.0 * np.pi)) - np.pi


def parse_thresholds(s: str) -> List[float]:
    s = (s or "").strip()
    if not s:
        return [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    out: List[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True, help="Folder containing phase3e_elambda_suite_rows.csv")
    ap.add_argument("--rows_csv", default="phase3e_elambda_suite_rows.csv")
    ap.add_argument("--tol", type=float, default=1e-6, help="Quantization tolerance for |phase| near 0/pi")
    ap.add_argument(
        "--thresholds",
        default="",
        help="Comma list of step_smin_min thresholds for conditional pi-rate table (default is a log-ish ladder)",
    )
    args = ap.parse_args()

    out_root = str(args.out_root)
    tol = float(args.tol)
    thresholds = parse_thresholds(str(args.thresholds))

    rows_path = os.path.join(out_root, str(args.rows_csv))
    if not os.path.exists(rows_path):
        raise SystemExit(f"Missing rows CSV: {rows_path}")

    df = pd.read_csv(rows_path)

    # Filter to the same canonical subset used by the suite's quantization grid.
    fwd_rect = df[
        (df.get("loop_type", "") == "rectangle")
        & (df.get("loop_direction", "") == "fwd")
        & (df.get("loop_double", False) == False)
        & (df.get("valid", True) == True)
    ].copy()

    if not len(fwd_rect):
        raise SystemExit("No canonical fwd-rectangle valid rows found; nothing to postprocess")

    if "holonomy_det_phase" not in fwd_rect.columns:
        raise SystemExit("rows CSV missing holonomy_det_phase")

    ph = angle_wrap_pi(fwd_rect["holonomy_det_phase"].values.astype(np.float64))
    absph = np.abs(ph)
    is_pi = np.abs(absph - math.pi) <= tol
    is_0 = absph <= tol

    fwd_rect["det_phase_abs"] = absph
    fwd_rect["det_phase_near_pi"] = is_pi
    fwd_rect["det_phase_near_0"] = is_0

    # Direction A: pi-rate concentration vs conditioning thresholds.
    if "step_smin_min" in fwd_rect.columns:
        smin = fwd_rect["step_smin_min"].values.astype(np.float64)

        rows: List[dict] = []
        n = int(len(fwd_rect))

        for t in thresholds:
            mask = np.isfinite(smin) & (smin <= float(t))
            cov = float(np.mean(mask)) if n else float("nan")
            pi_rate = float(np.mean(is_pi[mask])) if np.any(mask) else float("nan")
            pi_rate_complement = float(np.mean(is_pi[~mask])) if np.any(~mask) else float("nan")
            rows.append(
                {
                    "threshold": float(t),
                    "coverage_frac": cov,
                    "n": int(np.sum(mask)),
                    "pi_rate_given_below": pi_rate,
                    "pi_rate_given_above": pi_rate_complement,
                }
            )

        thresh_df = pd.DataFrame(rows).sort_values(["threshold"], ascending=True).reset_index(drop=True)
        thresh_df.to_csv(os.path.join(out_root, "phase3e_elambda_pi_rate_thresholds_step_smin_min.csv"), index=False)

    # Choose the most meaningful lambda coordinate available.
    # - If lambda_eff0/1 exists (newer runs, incl. deform_alpha sweeps), use that.
    # - Otherwise, fall back to lambda0/1.
    lam0_col = "lambda_eff0" if "lambda_eff0" in fwd_rect.columns else "lambda0"
    lam1_col = "lambda_eff1" if "lambda_eff1" in fwd_rect.columns else "lambda1"

    # Direction B: wall map in (E-pair, lambda-interval) cells.
    # First collapse to per-loop stats (loop_id groups are the natural unit; each loop has multiple blocks).
    by_loop = fwd_rect.groupby(["loop_id", "loop_e0", "loop_e1", lam0_col, lam1_col], dropna=False)

    loop_rows: List[dict] = []
    for (lid, e0, e1, l0, l1), g in by_loop:
        absph_g = g["det_phase_abs"].values.astype(np.float64)
        loop_rows.append(
            {
                "loop_id": int(lid),
                "loop_e0": int(e0),
                "loop_e1": int(e1),
                "lambda0": float(l0),
                "lambda1": float(l1),
                "n_blocks": int(len(g)),
                "frac_near_0": float(np.mean(absph_g <= tol)) if absph_g.size else float("nan"),
                "frac_near_pi": float(np.mean(np.abs(absph_g - math.pi) <= tol)) if absph_g.size else float("nan"),
                "step_smin_min_median": float(np.nanmedian(g["step_smin_min"].values.astype(np.float64)))
                if "step_smin_min" in g.columns
                else float("nan"),
                "step_smin_min_min": float(np.nanmin(g["step_smin_min"].values.astype(np.float64)))
                if "step_smin_min" in g.columns
                else float("nan"),
                "step_smin_median_median": float(np.nanmedian(g["step_smin_median"].values.astype(np.float64)))
                if "step_smin_median" in g.columns
                else float("nan"),
                "step_dLambda_max_median": float(np.nanmedian(g["step_dLambda_max"].values.astype(np.float64)))
                if "step_dLambda_max" in g.columns
                else float("nan"),
                "minus1_eigs_median": float(np.nanmedian(g["holonomy_num_eigs_near_minus1"].values.astype(np.float64)))
                if "holonomy_num_eigs_near_minus1" in g.columns
                else float("nan"),
                "minus1_dist_median": float(np.nanmedian(g["holonomy_min_dist_to_minus1"].values.astype(np.float64)))
                if "holonomy_min_dist_to_minus1" in g.columns
                else float("nan"),
            }
        )

    loop_df = pd.DataFrame(loop_rows)
    loop_df.to_csv(os.path.join(out_root, "phase3e_elambda_loop_wall_stats.csv"), index=False)

    by_cell = loop_df.groupby(["loop_e0", "loop_e1", "lambda0", "lambda1"], dropna=False)
    cell_df = (
        by_cell.agg(
            pi_rate=("frac_near_pi", "mean"),
            zero_rate=("frac_near_0", "mean"),
            n_loops=("frac_near_pi", "size"),
            median_step_smin_min=("step_smin_min_median", "median"),
            min_step_smin_min=("step_smin_min_min", "min"),
            median_step_smin_median=("step_smin_median_median", "median"),
            median_step_dLambda_max=("step_dLambda_max_median", "median"),
            median_minus1_eigs=("minus1_eigs_median", "median"),
            median_minus1_dist=("minus1_dist_median", "median"),
        )
        .reset_index()
        .sort_values(["pi_rate", "min_step_smin_min"], ascending=[False, True])
        .reset_index(drop=True)
    )

    cell_df.to_csv(os.path.join(out_root, "phase3e_elambda_wall_cells.csv"), index=False)
    cell_df.head(20).to_csv(os.path.join(out_root, "phase3e_elambda_wall_cells_top20.csv"), index=False)

    print(f"[phase3e_elambda_wall_postprocess] wrote threshold + wall tables under: {out_root}")


if __name__ == "__main__":
    main()
