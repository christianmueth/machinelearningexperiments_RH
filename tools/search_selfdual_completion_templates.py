from __future__ import annotations

import argparse
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
import investigate_completion_layer as comp_layer  # type: ignore
import probe_completed_global_object as global_probe  # type: ignore
import probe_frontend_realization as frontend  # type: ignore


def _linspace_csv(start: float, stop: float, steps: int) -> list[float]:
    return np.linspace(float(start), float(stop), int(steps), dtype=float).tolist()


def _safe_gamma_pair(s_grid: np.ndarray, mu: float) -> np.ndarray:
    s_grid = np.asarray(s_grid, dtype=np.complex128)
    return (
        fe_diag._loggamma_vec((s_grid.ravel() + float(mu)) / 2.0).reshape(s_grid.shape)
        + fe_diag._loggamma_vec((1.0 - s_grid.ravel() + float(mu)) / 2.0).reshape(s_grid.shape)
    ).astype(np.complex128)


def _build_log_surface(base_log: np.ndarray, basis_terms: list[np.ndarray], coeffs: list[float]) -> np.ndarray:
    out = np.asarray(base_log, dtype=np.complex128).copy()
    for c, term in zip(coeffs, basis_terms):
        out += complex(float(c), 0.0) * np.asarray(term, dtype=np.complex128)
    return out.astype(np.complex128)


def _transverse_summary_from_log(*, sigma_grid: np.ndarray, t_grid: np.ndarray, log_surface: np.ndarray) -> dict[str, float]:
    sigma_grid = np.asarray(sigma_grid, dtype=float).ravel()
    t_grid = np.asarray(t_grid, dtype=float).ravel()
    logabs = np.real(np.asarray(log_surface, dtype=np.complex128)).astype(float)
    crit_idx = int(np.argmin(np.abs(sigma_grid - 0.5)))
    crit_logabs = logabs[crit_idx, :].reshape(1, -1)
    rel_log = (logabs - crit_logabs).astype(float)
    min_idx = np.unravel_index(int(np.argmin(rel_log)), rel_log.shape)
    sigma_star = float(sigma_grid[min_idx[0]])
    t_star = float(t_grid[min_idx[1]])
    min_ratio = float(np.exp(rel_log[min_idx]))
    return {
        "sigma_star": sigma_star,
        "t_star": t_star,
        "min_ratio_to_sigma_half": min_ratio,
        "critical_advantage": float(max(0.0, 1.0 - min_ratio)),
    }


def _safe_det_pair(log_surface: np.ndarray, log_surface_sym: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    log_surface = np.asarray(log_surface, dtype=np.complex128)
    log_surface_sym = np.asarray(log_surface_sym, dtype=np.complex128)
    shift = np.maximum(np.real(log_surface), np.real(log_surface_sym)).astype(float)
    return np.exp(log_surface - shift).astype(np.complex128), np.exp(log_surface_sym - shift).astype(np.complex128)


def _objective(summary: dict[str, float], fe_cluster_median: float) -> float:
    return abs(float(summary["sigma_star"]) - 0.5) + float(summary["critical_advantage"]) + 0.5 * float(fe_cluster_median)


def _candidate_row(
    *,
    family: str,
    sigma_grid: np.ndarray,
    t_grid: np.ndarray,
    log_surface: np.ndarray,
    log_surface_sym: np.ndarray,
    cluster_t_min: float,
    cluster_t_max: float,
    params: dict[str, float],
) -> dict[str, float | str]:
    summary = _transverse_summary_from_log(sigma_grid=sigma_grid, t_grid=t_grid, log_surface=log_surface)
    det_surface, det_surface_sym = _safe_det_pair(log_surface, log_surface_sym)
    fe_med, fe_max = comp_layer._fe_cluster_defect(
        sigma_grid=sigma_grid,
        t_grid=t_grid,
        det_surface=det_surface,
        det_sym_surface=det_surface_sym,
        cluster_t_min=float(cluster_t_min),
        cluster_t_max=float(cluster_t_max),
    )
    row: dict[str, float | str] = {
        "family": str(family),
        "objective": _objective(summary, fe_cluster_median=float(fe_med)),
        **summary,
        "fe_cluster_median": float(fe_med),
        "fe_cluster_max": float(fe_max),
    }
    row.update({k: float(v) for k, v in params.items()})
    return row


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Search explicit self-dual completion templates for the frozen canonical object at u=0.24. "
            "Templates are centered on s-1/2 and scored by normalized transverse shape relative to sigma=1/2, plus FE defect near the t≈28 cluster."
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
    ap.add_argument("--a2_min", type=float, default=-2.0)
    ap.add_argument("--a2_max", type=float, default=2.0)
    ap.add_argument("--a2_steps", type=int, default=41)
    ap.add_argument("--a4_min", type=float, default=-1.0)
    ap.add_argument("--a4_max", type=float, default=1.0)
    ap.add_argument("--a4_steps", type=int, default=21)
    ap.add_argument("--gamma_coeff_min", type=float, default=-1.0)
    ap.add_argument("--gamma_coeff_max", type=float, default=1.0)
    ap.add_argument("--gamma_coeff_steps", type=int, default=41)
    ap.add_argument("--gamma_mus", default="0.0,1.0,2.0")
    ap.add_argument("--out_prefix", default="out/selfdual_completion_templates_u024")
    args = ap.parse_args()

    coeff_csv = Path(str(args.coeff_csv))
    primes_global = frontend._parse_int_csv(str(args.primes_global))
    packets, _meta, _ = global_probe._build_packets_for_u(
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
    eigs = np.asarray(np.diag(np.asarray(packets[0].S, dtype=np.complex128)), dtype=np.complex128)
    log_primes = np.log(np.asarray(primes_global, dtype=float))
    sigma_grid = np.linspace(float(args.sigma_min), float(args.sigma_max), int(args.sigma_steps), dtype=float)
    t_grid = np.linspace(float(args.t_min), float(args.t_max), int(args.t_steps), dtype=float)
    s_mesh = np.asarray([[(float(sigma) + 1j * float(t)) for t in t_grid.tolist()] for sigma in sigma_grid.tolist()], dtype=np.complex128)
    s_sym_mesh = np.asarray([[(1.0 - float(sigma) - 1j * float(t)) for t in t_grid.tolist()] for sigma in sigma_grid.tolist()], dtype=np.complex128)

    raw_log = np.zeros_like(s_mesh, dtype=np.complex128)
    raw_log_sym = np.zeros_like(s_mesh, dtype=np.complex128)
    for i in range(int(sigma_grid.size)):
        _det_line, log_line = comp_layer._det_logdet_from_eigs(s_vals=s_mesh[i, :], eigs=eigs, log_primes=log_primes)
        raw_log[i, :] = log_line
        _det_sym_line, log_sym_line = comp_layer._det_logdet_from_eigs(s_vals=s_sym_mesh[i, :], eigs=eigs, log_primes=log_primes)
        raw_log_sym[i, :] = log_sym_line

    h_mesh = (s_mesh - 0.5).astype(np.complex128)
    h2 = (h_mesh * h_mesh).astype(np.complex128)
    h4 = (h2 * h2).astype(np.complex128)
    h_sym_mesh = (s_sym_mesh - 0.5).astype(np.complex128)
    h2_sym = (h_sym_mesh * h_sym_mesh).astype(np.complex128)
    h4_sym = (h2_sym * h2_sym).astype(np.complex128)

    results: list[dict[str, float | str]] = []
    results.append(
        _candidate_row(
            family="raw",
            sigma_grid=sigma_grid,
            t_grid=t_grid,
            log_surface=raw_log,
            log_surface_sym=raw_log_sym,
            cluster_t_min=float(args.cluster_t_min),
            cluster_t_max=float(args.cluster_t_max),
            params={},
        )
    )

    best_poly: dict[str, float | str] | None = None
    for a2 in _linspace_csv(float(args.a2_min), float(args.a2_max), int(args.a2_steps)):
        for a4 in _linspace_csv(float(args.a4_min), float(args.a4_max), int(args.a4_steps)):
            row = _candidate_row(
                family="even_poly",
                sigma_grid=sigma_grid,
                t_grid=t_grid,
                log_surface=_build_log_surface(raw_log, [h2, h4], [float(a2), float(a4)]),
                log_surface_sym=_build_log_surface(raw_log_sym, [h2_sym, h4_sym], [float(a2), float(a4)]),
                cluster_t_min=float(args.cluster_t_min),
                cluster_t_max=float(args.cluster_t_max),
                params={"a2": float(a2), "a4": float(a4)},
            )
            results.append(row)
            if best_poly is None or float(row["objective"]) < float(best_poly["objective"]):
                best_poly = dict(row)

    mu_vals = [float(x.strip()) for x in str(args.gamma_mus).split(",") if str(x).strip()]
    for mu in mu_vals:
        gamma_pair = _safe_gamma_pair(s_mesh, mu=float(mu))
        gamma_pair_sym = _safe_gamma_pair(s_sym_mesh, mu=float(mu))
        for a in _linspace_csv(float(args.gamma_coeff_min), float(args.gamma_coeff_max), int(args.gamma_coeff_steps)):
            row = _candidate_row(
                family="gamma_pair",
                sigma_grid=sigma_grid,
                t_grid=t_grid,
                log_surface=_build_log_surface(raw_log, [gamma_pair], [float(a)]),
                log_surface_sym=_build_log_surface(raw_log_sym, [gamma_pair_sym], [float(a)]),
                cluster_t_min=float(args.cluster_t_min),
                cluster_t_max=float(args.cluster_t_max),
                params={"mu": float(mu), "a": float(a)},
            )
            results.append(row)
        if best_poly is not None:
            for a in _linspace_csv(float(args.gamma_coeff_min), float(args.gamma_coeff_max), int(args.gamma_coeff_steps)):
                row = _candidate_row(
                    family="even_poly_plus_gamma_pair",
                    sigma_grid=sigma_grid,
                    t_grid=t_grid,
                    log_surface=_build_log_surface(
                        raw_log,
                        [h2, h4, gamma_pair],
                        [float(best_poly.get("a2", 0.0)), float(best_poly.get("a4", 0.0)), float(a)],
                    ),
                    log_surface_sym=_build_log_surface(
                        raw_log_sym,
                        [h2_sym, h4_sym, gamma_pair_sym],
                        [float(best_poly.get("a2", 0.0)), float(best_poly.get("a4", 0.0)), float(a)],
                    ),
                    cluster_t_min=float(args.cluster_t_min),
                    cluster_t_max=float(args.cluster_t_max),
                    params={
                        "a2": float(best_poly.get("a2", 0.0)),
                        "a4": float(best_poly.get("a4", 0.0)),
                        "mu": float(mu),
                        "a": float(a),
                    },
                )
                results.append(row)

    results_df = pd.DataFrame(results).replace([np.inf, -np.inf], np.nan)
    summary_rows: list[dict[str, float | str]] = []
    for family in ["raw", "even_poly", "gamma_pair", "even_poly_plus_gamma_pair"]:
        sub = results_df[results_df["family"].astype(str) == str(family)].copy()
        sub = sub[np.isfinite(sub["objective"].astype(float))]
        if sub.empty:
            continue
        summary_rows.append(sub.sort_values(["objective", "critical_advantage", "fe_cluster_median"], ascending=True).iloc[0].to_dict())

    out_prefix = Path(str(args.out_prefix))
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    results_path = Path(str(out_prefix) + "_all_results.csv")
    summary_path = Path(str(out_prefix) + "_summary.csv")
    results_df.to_csv(results_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())