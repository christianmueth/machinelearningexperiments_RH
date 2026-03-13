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
import densify_local_zero_scan as dense_scan  # type: ignore
import diagnose_functional_equation_family as fe_diag  # type: ignore
import probe_completed_global_object as global_probe  # type: ignore
import probe_frontend_realization as frontend  # type: ignore


def _det_logdet_from_eigs(*, s_vals: np.ndarray, eigs: np.ndarray, log_primes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s_vals = np.asarray(s_vals, dtype=np.complex128).ravel()
    eigs = np.asarray(eigs, dtype=np.complex128).ravel()
    log_primes = np.asarray(log_primes, dtype=float).ravel()
    out_det = np.ones((s_vals.size,), dtype=np.complex128)
    out_log = np.zeros((s_vals.size,), dtype=np.complex128)
    for lp in log_primes.tolist():
        z = np.exp(-s_vals * float(lp)).astype(np.complex128)
        for lam in eigs.tolist():
            term = (1.0 - complex(lam) * z).astype(np.complex128)
            out_det *= term
            out_log += np.log(term)
    return out_det, out_log


def _centered_basis(s_vals: np.ndarray, family: str) -> np.ndarray:
    s_vals = np.asarray(s_vals, dtype=np.complex128).ravel()
    h = (s_vals - 0.5).astype(np.complex128)
    family = str(family)
    if family == "centered_poly":
        cols = [np.ones_like(h), h, h * h]
        return np.stack(cols, axis=1).astype(np.complex128)
    if family == "centered_even":
        cols = [np.ones_like(h), h * h]
        return np.stack(cols, axis=1).astype(np.complex128)
    if family == "centered_gamma":
        gamma_odd = (fe_diag._loggamma_vec(s_vals / 2.0) - fe_diag._loggamma_vec((1.0 - s_vals) / 2.0)).astype(np.complex128)
        cols = [np.ones_like(h), h, h * h, gamma_odd]
        return np.stack(cols, axis=1).astype(np.complex128)
    raise ValueError(f"unknown family: {family}")


def _fit_one_sided(log_s: np.ndarray, log_sym_conj: np.ndarray, s_vals: np.ndarray, family: str) -> tuple[np.ndarray, np.ndarray]:
    X = _centered_basis(s_vals, family=str(family))
    y = (np.asarray(log_sym_conj, dtype=np.complex128) - np.asarray(log_s, dtype=np.complex128)).astype(np.complex128)
    coef = fe_diag._fit_completion_complex(X, y)
    adj = (X @ coef.reshape(-1, 1)).reshape(-1).astype(np.complex128)
    return (np.asarray(log_s, dtype=np.complex128) + adj).astype(np.complex128), np.asarray(coef, dtype=np.complex128)


def _fit_selfdual(log_s: np.ndarray, log_1ms: np.ndarray, s_vals: np.ndarray, family: str) -> tuple[np.ndarray, np.ndarray]:
    s_vals = np.asarray(s_vals, dtype=np.complex128).ravel()
    log_s = np.asarray(log_s, dtype=np.complex128).ravel()
    log_1ms = np.asarray(log_1ms, dtype=np.complex128).ravel()
    U = _centered_basis(s_vals, family=str(family))
    s_1ms = (1.0 - s_vals).astype(np.complex128)
    V = _centered_basis(s_1ms, family=str(family))
    y = (log_s - np.conjugate(log_1ms)).astype(np.complex128)

    cols: list[np.ndarray] = []
    for j in range(int(U.shape[1])):
        u = U[:, j].astype(np.complex128)
        v = np.conjugate(V[:, j]).astype(np.complex128)
        cols.append((u - v).astype(np.complex128))
        cols.append((1j * (u + v)).astype(np.complex128))
    X = np.stack(cols, axis=1).astype(np.complex128)
    rhs = (-y).astype(np.complex128)
    coef_realimag = fe_diag._fit_completion_complex(X, rhs)
    coeffs: list[complex] = []
    for j in range(int(U.shape[1])):
        coeffs.append(complex(float(np.real(coef_realimag[2 * j])), float(np.real(coef_realimag[2 * j + 1]))))
    c = np.asarray(coeffs, dtype=np.complex128)
    adj = (U @ c.reshape(-1, 1)).reshape(-1).astype(np.complex128)
    return (log_s + adj).astype(np.complex128), c


def _summarize_surface(*, sigma_grid: np.ndarray, t_grid: np.ndarray, det_surface: np.ndarray) -> dict[str, float]:
    sigma_grid = np.asarray(sigma_grid, dtype=float).ravel()
    t_grid = np.asarray(t_grid, dtype=float).ravel()
    abs_surface = np.abs(np.asarray(det_surface, dtype=np.complex128)).astype(float)
    idx = np.unravel_index(int(np.argmin(abs_surface)), abs_surface.shape)
    sigma_star = float(sigma_grid[idx[0]])
    t_star = float(t_grid[idx[1]])
    min_abs = float(abs_surface[idx])
    crit_idx = int(np.argmin(np.abs(sigma_grid - 0.5)))
    crit_row = abs_surface[crit_idx, :]
    crit_t_idx = int(np.argmin(crit_row))
    crit_abs = float(crit_row[crit_t_idx])
    crit_t = float(t_grid[crit_t_idx])
    return {
        "sigma_star": sigma_star,
        "t_star": t_star,
        "min_abs_det": min_abs,
        "sigma_half_t_star": crit_t,
        "abs_det_at_sigma_half": crit_abs,
        "critical_gap": float(crit_abs - min_abs),
    }


def _fe_cluster_defect(*, sigma_grid: np.ndarray, t_grid: np.ndarray, det_surface: np.ndarray, det_sym_surface: np.ndarray, cluster_t_min: float, cluster_t_max: float) -> tuple[float, float]:
    sigma_grid = np.asarray(sigma_grid, dtype=float).ravel()
    t_grid = np.asarray(t_grid, dtype=float).ravel()
    crit_idx = int(np.argmin(np.abs(sigma_grid - 0.5)))
    tmask = (t_grid >= float(cluster_t_min)) & (t_grid <= float(cluster_t_max))
    lhs = np.asarray(det_surface, dtype=np.complex128)[crit_idx, tmask]
    rhs = np.conjugate(np.asarray(det_sym_surface, dtype=np.complex128)[crit_idx, tmask])
    diff = np.abs(lhs - rhs).astype(float)
    rel = diff / (0.5 * (np.abs(lhs) + np.abs(rhs)) + 1e-300)
    return float(np.median(rel)), float(np.max(rel))


def _surface_for_local_shift(*, sigma_grid: np.ndarray, t_grid: np.ndarray, eigs: np.ndarray, log_primes: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    sigmas = np.asarray(sigma_grid, dtype=float).ravel()
    ts = np.asarray(t_grid, dtype=float).ravel()
    det = np.zeros((sigmas.size, ts.size), dtype=np.complex128)
    det_sym = np.zeros((sigmas.size, ts.size), dtype=np.complex128)
    for i, sigma in enumerate(sigmas.tolist()):
        s_line = (float(sigma) + 1j * ts).astype(np.complex128)
        h_line = (s_line - 0.5).astype(np.complex128)
        s_eff = (s_line + float(alpha) * h_line).astype(np.complex128)
        det[i, :], _ = _det_logdet_from_eigs(s_vals=s_eff, eigs=eigs, log_primes=log_primes)
        s_sym = (1.0 - float(sigma) - 1j * ts).astype(np.complex128)
        h_sym = (s_sym - 0.5).astype(np.complex128)
        s_sym_eff = (s_sym + float(alpha) * h_sym).astype(np.complex128)
        det_sym[i, :], _ = _det_logdet_from_eigs(s_vals=s_sym_eff, eigs=eigs, log_primes=log_primes)
    return det, det_sym


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Focused completion-factor investigation for the frozen canonical object at u=0.24: structured global completion families and a local-placement shift model."
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
    ap.add_argument("--cluster_t_min", type=float, default=27.9)
    ap.add_argument("--cluster_t_max", type=float, default=28.05)
    ap.add_argument("--alpha_min", type=float, default=-2.0)
    ap.add_argument("--alpha_max", type=float, default=2.0)
    ap.add_argument("--alpha_steps", type=int, default=81)
    ap.add_argument("--out_prefix", default="out/completion_layer_investigation_u024")
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
    eigs = np.asarray([np.diag(np.asarray(packets[0].S, dtype=np.complex128))[i] for i in range(3)], dtype=np.complex128)
    log_primes = np.log(np.asarray(primes_global, dtype=float))
    sigma_grid = np.linspace(float(args.sigma_min), float(args.sigma_max), int(args.sigma_steps), dtype=float)
    t_grid = np.linspace(float(args.t_min), float(args.t_max), int(args.t_steps), dtype=float)

    raw_det = np.zeros((sigma_grid.size, t_grid.size), dtype=np.complex128)
    raw_det_sym = np.zeros((sigma_grid.size, t_grid.size), dtype=np.complex128)
    raw_log = np.zeros((sigma_grid.size, t_grid.size), dtype=np.complex128)
    raw_log_sym = np.zeros((sigma_grid.size, t_grid.size), dtype=np.complex128)
    for i, sigma in enumerate(sigma_grid.tolist()):
        s_line = (float(sigma) + 1j * t_grid).astype(np.complex128)
        raw_det[i, :], raw_log[i, :] = _det_logdet_from_eigs(s_vals=s_line, eigs=eigs, log_primes=log_primes)
        s_sym = (1.0 - float(sigma) - 1j * t_grid).astype(np.complex128)
        raw_det_sym[i, :], raw_log_sym[i, :] = _det_logdet_from_eigs(s_vals=s_sym, eigs=eigs, log_primes=log_primes)

    rows: list[dict[str, float | str | bool]] = []
    raw_summary = _summarize_surface(sigma_grid=sigma_grid, t_grid=t_grid, det_surface=raw_det)
    raw_fe_med, raw_fe_max = _fe_cluster_defect(
        sigma_grid=sigma_grid,
        t_grid=t_grid,
        det_surface=raw_det,
        det_sym_surface=raw_det_sym,
        cluster_t_min=float(args.cluster_t_min),
        cluster_t_max=float(args.cluster_t_max),
    )
    rows.append({"experiment": "raw", **raw_summary, "fe_cluster_median": float(raw_fe_med), "fe_cluster_max": float(raw_fe_max)})

    for family in ["centered_poly", "centered_even", "centered_gamma"]:
        det_corr = np.zeros_like(raw_det)
        det_corr_sym = np.zeros_like(raw_det)
        for i, sigma in enumerate(sigma_grid.tolist()):
            s_line = (float(sigma) + 1j * t_grid).astype(np.complex128)
            L = raw_log[i, :].astype(np.complex128)
            Lsym = raw_log_sym[i, :].astype(np.complex128)
            L1, coef1 = _fit_one_sided(L, np.conjugate(Lsym).astype(np.complex128), s_line, family=str(family))
            det1 = np.exp(L1).astype(np.complex128)
            det_corr[i, :] = det1
            det_corr_sym[i, :] = raw_det_sym[i, :]
        summary1 = _summarize_surface(sigma_grid=sigma_grid, t_grid=t_grid, det_surface=det_corr)
        fe_med1, fe_max1 = _fe_cluster_defect(
            sigma_grid=sigma_grid,
            t_grid=t_grid,
            det_surface=det_corr,
            det_sym_surface=det_corr_sym,
            cluster_t_min=float(args.cluster_t_min),
            cluster_t_max=float(args.cluster_t_max),
        )
        rows.append({"experiment": f"global_one_sided_{family}", **summary1, "fe_cluster_median": float(fe_med1), "fe_cluster_max": float(fe_max1)})

        det_sd = np.zeros_like(raw_det)
        det_sd_sym = np.zeros_like(raw_det)
        for i, sigma in enumerate(sigma_grid.tolist()):
            s_line = (float(sigma) + 1j * t_grid).astype(np.complex128)
            L = raw_log[i, :].astype(np.complex128)
            Lsym = raw_log_sym[i, :].astype(np.complex128)
            Lsd, coefsd = _fit_selfdual(L, Lsym, s_line, family=str(family))
            det_sd[i, :] = np.exp(Lsd).astype(np.complex128)
            s_sym = (1.0 - float(sigma) - 1j * t_grid).astype(np.complex128)
            Usym = _centered_basis(s_sym, family=str(family))
            det_sd_sym[i, :] = np.exp((Lsym + (Usym @ coefsd.reshape(-1, 1)).reshape(-1)).astype(np.complex128)).astype(np.complex128)
        summary_sd = _summarize_surface(sigma_grid=sigma_grid, t_grid=t_grid, det_surface=det_sd)
        fe_med_sd, fe_max_sd = _fe_cluster_defect(
            sigma_grid=sigma_grid,
            t_grid=t_grid,
            det_surface=det_sd,
            det_sym_surface=det_sd_sym,
            cluster_t_min=float(args.cluster_t_min),
            cluster_t_max=float(args.cluster_t_max),
        )
        rows.append({"experiment": f"global_selfdual_{family}", **summary_sd, "fe_cluster_median": float(fe_med_sd), "fe_cluster_max": float(fe_max_sd)})

    best_local_row: dict[str, float | str] | None = None
    for alpha in np.linspace(float(args.alpha_min), float(args.alpha_max), int(args.alpha_steps), dtype=float).tolist():
        det_loc, det_loc_sym = _surface_for_local_shift(sigma_grid=sigma_grid, t_grid=t_grid, eigs=eigs, log_primes=log_primes, alpha=float(alpha))
        summary_loc = _summarize_surface(sigma_grid=sigma_grid, t_grid=t_grid, det_surface=det_loc)
        fe_med_loc, fe_max_loc = _fe_cluster_defect(
            sigma_grid=sigma_grid,
            t_grid=t_grid,
            det_surface=det_loc,
            det_sym_surface=det_loc_sym,
            cluster_t_min=float(args.cluster_t_min),
            cluster_t_max=float(args.cluster_t_max),
        )
        row = {"experiment": "local_shift_linear", "alpha": float(alpha), **summary_loc, "fe_cluster_median": float(fe_med_loc), "fe_cluster_max": float(fe_max_loc)}
        rows.append(row)
        objective = abs(float(summary_loc["sigma_star"]) - 0.5) + max(0.0, float(summary_loc["critical_gap"])) + 0.25 * float(fe_med_loc)
        if best_local_row is None:
            best_local_row = dict(row)
            best_local_row["objective"] = float(objective)
        elif float(objective) < float(best_local_row["objective"]):
            best_local_row = dict(row)
            best_local_row["objective"] = float(objective)

    results_df = pd.DataFrame(rows)
    summary_rows: list[dict[str, float | str | bool]] = []
    for experiment in [
        "raw",
        "global_one_sided_centered_poly",
        "global_selfdual_centered_poly",
        "global_selfdual_centered_even",
        "global_selfdual_centered_gamma",
    ]:
        sub = results_df[results_df["experiment"].astype(str) == str(experiment)].copy()
        if sub.empty:
            continue
        summary_rows.append(sub.iloc[0].to_dict())
    if best_local_row is not None:
        summary_rows.append(best_local_row)

    out_prefix = Path(str(args.out_prefix))
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    results_path = Path(str(out_prefix) + "_all_results.csv")
    summary_path = Path(str(out_prefix) + "_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    results_df.to_csv(results_path, index=False)
    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())