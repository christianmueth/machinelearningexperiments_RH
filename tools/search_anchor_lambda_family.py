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


def _parse_float_csv(s: str) -> list[float]:
    out: list[float] = []
    for part in str(s).split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out


def _inverse_anchor_coeffs(lam: float, order: int) -> dict[int, float]:
    lam = float(lam)
    order = int(order)
    if order not in {4, 6, 8}:
        raise ValueError("order must be one of 4, 6, 8")
    coeffs: dict[int, float] = {}
    for power in range(2, order + 1, 2):
        k = power // 2
        coeffs[power] = float(((-1) ** (k + 1)) * ((2.0 * lam) ** power) / float(power))
    return coeffs


def _objective(*, best_u_by_fe: float, best_u_by_zero: float, best_fe_metric: float, zero_at_u024: float) -> float:
    return abs(float(best_u_by_fe) - float(best_u_by_zero)) + 2.0 * float(best_fe_metric) + 0.25 * float(zero_at_u024)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Search the inverse-anchor centered-even completion family over lambda and truncation order, "
            "measuring whether higher-order anchor curvature changes the FE-versus-zero ranking across the full u-window."
        )
    )
    ap.add_argument("--coeff_csv", default=str(backend.DEFAULT_COEFF_CSV))
    ap.add_argument("--us", default="0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24")
    ap.add_argument("--primes_global", default="2,3,5,7,11,13,17,19,23,29")
    ap.add_argument("--radius_max", type=float, default=0.999)
    ap.add_argument("--n_random", type=int, default=30000)
    ap.add_argument("--local_steps", type=int, default=40)
    ap.add_argument("--w_A1", type=float, default=1.35)
    ap.add_argument("--w_A2", type=float, default=1.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma_fe", type=float, default=0.7)
    ap.add_argument("--sigma_zero", type=float, default=0.5)
    ap.add_argument("--t_min", type=float, default=10.0)
    ap.add_argument("--t_max", type=float, default=30.0)
    ap.add_argument("--n_t", type=int, default=401)
    ap.add_argument("--top_zero_candidates", type=int, default=8)
    ap.add_argument("--lambda_min", type=float, default=0.35)
    ap.add_argument("--lambda_max", type=float, default=0.60)
    ap.add_argument("--lambda_steps", type=int, default=21)
    ap.add_argument("--orders", default="4,6,8")
    ap.add_argument("--u_focus", type=float, default=0.24)
    ap.add_argument("--out_prefix", default="out/anchor_lambda_family_search")
    args = ap.parse_args()

    coeff_csv = Path(str(args.coeff_csv))
    us = _parse_float_csv(str(args.us))
    orders = [int(x) for x in str(args.orders).split(",") if str(x).strip()]
    lambdas = np.linspace(float(args.lambda_min), float(args.lambda_max), int(args.lambda_steps), dtype=float)
    primes_global = frontend._parse_int_csv(str(args.primes_global))
    t_grid = np.linspace(float(args.t_min), float(args.t_max), int(args.n_t), dtype=float)
    w = fe_diag._trapz_weights(t_grid)

    rows: list[dict[str, float | int | bool]] = []
    for order in orders:
        for lam in lambdas.tolist():
            summary_rows: list[dict[str, float | bool]] = []
            coeffs = _inverse_anchor_coeffs(lam=float(lam), order=int(order))
            for idx, u in enumerate(us):
                packets, meta, _factors = global_probe._build_packets_for_u(
                    coeff_csv=coeff_csv,
                    u=float(u),
                    primes_global=primes_global,
                    radius_max=float(args.radius_max),
                    n_random=int(args.n_random),
                    local_steps=int(args.local_steps),
                    w_A1=float(args.w_A1),
                    w_A2=float(args.w_A2),
                    seed=int(args.seed) + idx,
                )
                det_fe, log_fe = global_probe._compute_det_grid(
                    packets=packets,
                    sigma=float(args.sigma_fe),
                    t_grid=t_grid,
                    completion_even_a2=float(coeffs.get(2, 0.0)),
                    completion_even_a4=float(coeffs.get(4, 0.0)),
                    completion_even_a6=float(coeffs.get(6, 0.0)),
                    completion_even_a8=float(coeffs.get(8, 0.0)),
                    completion_even_mode="anchored_real",
                )
                det_sym, log_sym = global_probe._compute_det_grid(
                    packets=packets,
                    sigma=float(1.0 - float(args.sigma_fe)),
                    t_grid=-t_grid,
                    completion_even_a2=float(coeffs.get(2, 0.0)),
                    completion_even_a4=float(coeffs.get(4, 0.0)),
                    completion_even_a6=float(coeffs.get(6, 0.0)),
                    completion_even_a8=float(coeffs.get(8, 0.0)),
                    completion_even_mode="anchored_real",
                )
                m_conj = fe_diag._symmetry_metrics(
                    s_vals=(float(args.sigma_fe) + 1j * t_grid).astype(np.complex128),
                    det_s=det_fe,
                    logdet_s=log_fe,
                    det_sym=np.conjugate(det_sym).astype(np.complex128),
                    logdet_sym=np.conjugate(log_sym).astype(np.complex128),
                    w=w,
                    completion_basis="poly2",
                )
                det_zero, _ = global_probe._compute_det_grid(
                    packets=packets,
                    sigma=float(args.sigma_zero),
                    t_grid=t_grid,
                    completion_even_a2=float(coeffs.get(2, 0.0)),
                    completion_even_a4=float(coeffs.get(4, 0.0)),
                    completion_even_a6=float(coeffs.get(6, 0.0)),
                    completion_even_a8=float(coeffs.get(8, 0.0)),
                    completion_even_mode="anchored_real",
                )
                zero_cands = global_probe._zero_candidates(t_grid=t_grid, det_vals=det_zero, top_k=int(args.top_zero_candidates))
                best_zero_abs = float(zero_cands[0]["abs_det"]) if zero_cands else float(np.min(np.abs(det_zero)))
                best_zero_t = float(zero_cands[0]["t"]) if zero_cands else float(t_grid[int(np.argmin(np.abs(det_zero)))])
                summary_rows.append(
                    {
                        "u": float(u),
                        "completion_rich_rel_l2_logdet_conj1ms": float(m_conj.completion_rich_rel_l2_logdet),
                        "best_zero_candidate_abs_det": float(best_zero_abs),
                        "best_zero_candidate_t": float(best_zero_t),
                        "coeff_err_pass": bool(meta["coeff_err_pass"]),
                    }
                )
            df = pd.DataFrame(summary_rows)
            idx_fe = int(df["completion_rich_rel_l2_logdet_conj1ms"].astype(float).idxmin())
            idx_zero = int(df["best_zero_candidate_abs_det"].astype(float).idxmin())
            u_focus_row = df[np.isclose(df["u"].astype(float), float(args.u_focus), atol=1e-12)].iloc[0]
            row = {
                "order": int(order),
                "lambda": float(lam),
                "a2": float(coeffs.get(2, 0.0)),
                "a4": float(coeffs.get(4, 0.0)),
                "a6": float(coeffs.get(6, 0.0)),
                "a8": float(coeffs.get(8, 0.0)),
                "best_u_by_fe_completion": float(df.loc[idx_fe, "u"]),
                "best_fe_completion_metric": float(df.loc[idx_fe, "completion_rich_rel_l2_logdet_conj1ms"]),
                "best_u_by_zero_candidate": float(df.loc[idx_zero, "u"]),
                "best_zero_candidate_abs_det": float(df.loc[idx_zero, "best_zero_candidate_abs_det"]),
                "u_focus": float(args.u_focus),
                "u_focus_fe_metric": float(u_focus_row["completion_rich_rel_l2_logdet_conj1ms"]),
                "u_focus_zero_abs_det": float(u_focus_row["best_zero_candidate_abs_det"]),
                "n_coeff_err_pass": int(df["coeff_err_pass"].astype(bool).sum()),
            }
            row["objective"] = _objective(
                best_u_by_fe=float(row["best_u_by_fe_completion"]),
                best_u_by_zero=float(row["best_u_by_zero_candidate"]),
                best_fe_metric=float(row["best_fe_completion_metric"]),
                zero_at_u024=float(row["u_focus_zero_abs_det"]),
            )
            rows.append(row)

    out_prefix = Path(str(args.out_prefix))
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    all_path = Path(str(out_prefix) + "_all_results.csv")
    summary_path = Path(str(out_prefix) + "_summary.csv")
    df_all = pd.DataFrame(rows)
    df_all.to_csv(all_path, index=False)
    summary_rows = []
    for order in sorted(set(df_all["order"].astype(int).tolist())):
        sub = df_all[df_all["order"].astype(int) == int(order)].copy()
        summary_rows.append(sub.sort_values(["objective", "u_focus_fe_metric", "u_focus_zero_abs_det"], ascending=True).iloc[0].to_dict())
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"wrote {all_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())