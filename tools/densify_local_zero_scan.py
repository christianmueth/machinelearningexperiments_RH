from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import corrected_backend_interface as backend  # type: ignore
import diagnose_functional_equation_family as fe_diag  # type: ignore
import probe_completed_global_object as global_probe  # type: ignore
import probe_frontend_realization as frontend  # type: ignore


def _best_index(vals: np.ndarray) -> int:
    vals = np.asarray(vals, dtype=float).ravel()
    return int(np.argmin(vals)) if vals.size else -1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Densify the local (sigma,t) scan around the persistent t≈28 cluster for the frozen backend plus accepted A3 frontend, "
            "and compare raw versus completion-normalized minima."
        )
    )
    ap.add_argument("--coeff_csv", default=str(backend.DEFAULT_COEFF_CSV))
    ap.add_argument("--u", type=float, default=0.24)
    ap.add_argument("--primes_global", default="2,3,5,7,11,13,17,19,23,29")
    ap.add_argument("--radius_max", type=float, default=0.999)
    ap.add_argument("--n_random", type=int, default=30000)
    ap.add_argument("--local_steps", type=int, default=40)
    ap.add_argument("--w_A1", type=float, default=1.35)
    ap.add_argument("--w_A2", type=float, default=1.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma_min", type=float, default=0.45)
    ap.add_argument("--sigma_max", type=float, default=0.55)
    ap.add_argument("--sigma_steps", type=int, default=101)
    ap.add_argument("--t_min", type=float, default=27.7)
    ap.add_argument("--t_max", type=float, default=28.2)
    ap.add_argument("--t_steps", type=int, default=1001)
    ap.add_argument("--completion_basis", choices=["none", "poly2", "poly2_gamma"], default="poly2")
    ap.add_argument("--out_prefix", default="out/dense_local_zero_scan_u024")
    args = ap.parse_args()

    coeff_csv = Path(str(args.coeff_csv))
    primes_global = frontend._parse_int_csv(str(args.primes_global))
    packets, meta, _ = global_probe._build_packets_for_u(
        coeff_csv=coeff_csv,
        u=float(args.u),
        primes_global=primes_global,
        radius_max=float(args.radius_max),
        n_random=int(args.n_random),
        local_steps=int(args.local_steps),
        w_A1=float(args.w_A1),
        w_A2=float(args.w_A2),
        seed=int(args.seed),
    )

    sigma_grid = np.linspace(float(args.sigma_min), float(args.sigma_max), int(args.sigma_steps), dtype=float)
    t_grid = np.linspace(float(args.t_min), float(args.t_max), int(args.t_steps), dtype=float)
    surface_rows: list[dict[str, float | str]] = []
    sigma_summary_rows: list[dict[str, float]] = []

    global_best_raw = {"sigma": float("nan"), "t": float("nan"), "abs_det": float("inf")}
    global_best_completed = {"sigma": float("nan"), "t": float("nan"), "abs_det_completed": float("inf")}

    for sigma in sigma_grid.tolist():
        det_s, log_s = global_probe._compute_det_grid(packets=packets, sigma=float(sigma), t_grid=t_grid)
        det_1ms, log_1ms = global_probe._compute_det_grid(packets=packets, sigma=float(1.0 - float(sigma)), t_grid=-t_grid)
        log_adj, coef = fe_diag._fit_completion_on_logdet_basis(
            (float(sigma) + 1j * t_grid).astype(np.complex128),
            log_s,
            np.conjugate(log_1ms).astype(np.complex128),
            basis=str(args.completion_basis),
        )
        det_completed = np.exp(log_adj).astype(np.complex128)
        abs_raw = np.abs(det_s).astype(float)
        abs_completed = np.abs(det_completed).astype(float)
        raw_idx = _best_index(abs_raw)
        comp_idx = _best_index(abs_completed)
        sigma_summary_rows.append(
            {
                "u": float(args.u),
                "sigma": float(sigma),
                "t_at_min_raw": float(t_grid[raw_idx]),
                "min_abs_det_raw": float(abs_raw[raw_idx]),
                "t_at_min_completed": float(t_grid[comp_idx]),
                "min_abs_det_completed": float(abs_completed[comp_idx]),
                "completion_coef0_re": float(np.real(coef[0])) if coef.size >= 1 else float("nan"),
                "completion_coef0_im": float(np.imag(coef[0])) if coef.size >= 1 else float("nan"),
                "completion_coef1_re": float(np.real(coef[1])) if coef.size >= 2 else float("nan"),
                "completion_coef1_im": float(np.imag(coef[1])) if coef.size >= 2 else float("nan"),
            }
        )
        if float(abs_raw[raw_idx]) < float(global_best_raw["abs_det"]):
            global_best_raw = {"sigma": float(sigma), "t": float(t_grid[raw_idx]), "abs_det": float(abs_raw[raw_idx])}
        if float(abs_completed[comp_idx]) < float(global_best_completed["abs_det_completed"]):
            global_best_completed = {
                "sigma": float(sigma),
                "t": float(t_grid[comp_idx]),
                "abs_det_completed": float(abs_completed[comp_idx]),
            }
        for t, z_raw, z_comp, a_raw, a_comp in zip(t_grid.tolist(), det_s.tolist(), det_completed.tolist(), abs_raw.tolist(), abs_completed.tolist()):
            surface_rows.append(
                {
                    "u": float(args.u),
                    "sigma": float(sigma),
                    "t": float(t),
                    "abs_det_raw": float(a_raw),
                    "abs_det_completed": float(a_comp),
                    "q_raw_re": float(np.real(z_raw)),
                    "q_raw_im": float(np.imag(z_raw)),
                    "q_completed_re": float(np.real(z_comp)),
                    "q_completed_im": float(np.imag(z_comp)),
                }
            )

    sigma_df = pd.DataFrame(sigma_summary_rows)
    crit_idx = int(np.argmin(np.abs(sigma_df["sigma"].astype(float) - 0.5)))
    summary_row = {
        "u": float(args.u),
        "frontend_family": "A3_exact_frontend",
        "coeff_err_pass": bool(meta["coeff_err_pass"]),
        "err1": float(meta["err1"]),
        "err2": float(meta["err2"]),
        "err3": float(meta["err3"]),
        "spectral_radius": float(meta["spectral_radius"]),
        "sigma_min_raw": float(global_best_raw["sigma"]),
        "t_min_raw": float(global_best_raw["t"]),
        "min_abs_det_raw": float(global_best_raw["abs_det"]),
        "sigma_min_completed": float(global_best_completed["sigma"]),
        "t_min_completed": float(global_best_completed["t"]),
        "min_abs_det_completed": float(global_best_completed["abs_det_completed"]),
        "abs_det_raw_at_sigma_half": float(sigma_df.loc[crit_idx, "min_abs_det_raw"]),
        "abs_det_completed_at_sigma_half": float(sigma_df.loc[crit_idx, "min_abs_det_completed"]),
        "critical_gap_raw": float(sigma_df.loc[crit_idx, "min_abs_det_raw"] - float(global_best_raw["abs_det"])),
        "critical_gap_completed": float(sigma_df.loc[crit_idx, "min_abs_det_completed"] - float(global_best_completed["abs_det_completed"])),
    }

    out_prefix = Path(str(args.out_prefix))
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    surface_path = Path(str(out_prefix) + "_surface.csv")
    sigma_path = Path(str(out_prefix) + "_sigma_summary.csv")
    summary_path = Path(str(out_prefix) + "_summary.csv")
    pd.DataFrame(surface_rows).to_csv(surface_path, index=False)
    sigma_df.to_csv(sigma_path, index=False)
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False)

    print(f"wrote {surface_path}")
    print(f"wrote {sigma_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())