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


def _packet_params_from_meta(meta: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    eigs = np.asarray(
        [
            complex(float(meta["eig1_re"]), float(meta["eig1_im"])),
            complex(float(meta["eig2_re"]), float(meta["eig2_im"])),
            complex(float(meta["eig3_re"]), float(meta["eig3_im"])),
        ],
        dtype=np.complex128,
    )
    return np.abs(eigs).astype(float), np.angle(eigs).astype(float)


def _eigs_from_params(*, radii: np.ndarray, phases: np.ndarray) -> np.ndarray:
    radii = np.asarray(radii, dtype=float).ravel()
    phases = np.asarray(phases, dtype=float).ravel()
    return (radii * np.exp(1j * phases)).astype(np.complex128)


def _make_packets(*, eigs: np.ndarray, primes_global: list[int]) -> list[frontend.GenericPacket]:
    S = np.diag(np.asarray(eigs, dtype=np.complex128))
    return [
        frontend.GenericPacket(p=int(p), ell=float(math.log(float(p))), S=np.asarray(S, dtype=np.complex128))
        for p in primes_global
    ]


def _fe_surface_rows(
    *,
    packets: list[frontend.GenericPacket],
    sigma_values: list[float],
    t_grid: np.ndarray,
    u: float,
    completion_even_a2: float,
    completion_even_a4: float,
    completion_even_a6: float,
    completion_even_a8: float,
    completion_even_mode: str,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for sigma in sigma_values:
        det_s, log_s = global_probe._compute_det_grid(
            packets=packets,
            sigma=float(sigma),
            t_grid=t_grid,
            completion_even_a2=float(completion_even_a2),
            completion_even_a4=float(completion_even_a4),
            completion_even_a6=float(completion_even_a6),
            completion_even_a8=float(completion_even_a8),
            completion_even_mode=str(completion_even_mode),
        )
        det_sym, log_sym = global_probe._compute_det_grid(
            packets=packets,
            sigma=float(1.0 - float(sigma)),
            t_grid=-t_grid,
            completion_even_a2=float(completion_even_a2),
            completion_even_a4=float(completion_even_a4),
            completion_even_a6=float(completion_even_a6),
            completion_even_a8=float(completion_even_a8),
            completion_even_mode=str(completion_even_mode),
        )
        det_sym = np.conjugate(det_sym).astype(np.complex128)
        log_sym = np.conjugate(log_sym).astype(np.complex128)
        abs_def = np.abs(det_s - det_sym).astype(float)
        rel_def = abs_def / (0.5 * (np.abs(det_s) + np.abs(det_sym)) + 1e-300)
        abs_log_def = np.abs(log_s - log_sym).astype(float)
        for t, ds, d1, ad, rd, ald in zip(t_grid.tolist(), det_s.tolist(), det_sym.tolist(), abs_def.tolist(), rel_def.tolist(), abs_log_def.tolist()):
            rows.append(
                {
                    "u": float(u),
                    "sigma": float(sigma),
                    "t": float(t),
                    "abs_det_defect": float(ad),
                    "rel_det_defect": float(rd),
                    "abs_logdet_defect": float(ald),
                    "det_re": float(np.real(ds)),
                    "det_im": float(np.imag(ds)),
                    "det_sym_re": float(np.real(d1)),
                    "det_sym_im": float(np.imag(d1)),
                }
            )
    return rows


def _summarize_fe_surface(*, fe_df: pd.DataFrame, zero_t: float, zero_band: float) -> dict[str, float]:
    crit = fe_df[np.isclose(fe_df["sigma"].astype(float), 0.5, atol=1e-12)].copy()
    near_zero = fe_df[np.abs(fe_df["t"].astype(float) - float(zero_t)) <= float(zero_band)].copy()
    near_zero_crit = crit[np.abs(crit["t"].astype(float) - float(zero_t)) <= float(zero_band)].copy()
    return {
        "median_abs_det_defect_all": float(fe_df["abs_det_defect"].median()),
        "max_abs_det_defect_all": float(fe_df["abs_det_defect"].max()),
        "median_rel_det_defect_all": float(fe_df["rel_det_defect"].median()),
        "max_rel_det_defect_all": float(fe_df["rel_det_defect"].max()),
        "median_abs_det_defect_critical": float(crit["abs_det_defect"].median()) if not crit.empty else float("nan"),
        "max_abs_det_defect_critical": float(crit["abs_det_defect"].max()) if not crit.empty else float("nan"),
        "median_rel_det_defect_critical": float(crit["rel_det_defect"].median()) if not crit.empty else float("nan"),
        "max_rel_det_defect_critical": float(crit["rel_det_defect"].max()) if not crit.empty else float("nan"),
        "median_abs_det_defect_near_zero": float(near_zero["abs_det_defect"].median()) if not near_zero.empty else float("nan"),
        "max_abs_det_defect_near_zero": float(near_zero["abs_det_defect"].max()) if not near_zero.empty else float("nan"),
        "median_rel_det_defect_near_zero": float(near_zero["rel_det_defect"].median()) if not near_zero.empty else float("nan"),
        "max_rel_det_defect_near_zero": float(near_zero["rel_det_defect"].max()) if not near_zero.empty else float("nan"),
        "median_rel_det_defect_zero_on_critical": float(near_zero_crit["rel_det_defect"].median()) if not near_zero_crit.empty else float("nan"),
        "max_rel_det_defect_zero_on_critical": float(near_zero_crit["rel_det_defect"].max()) if not near_zero_crit.empty else float("nan"),
    }


def _best_cluster_minimum(
    *,
    packets: list[frontend.GenericPacket],
    sigma_zero: float,
    t_grid: np.ndarray,
    cluster_t_min: float,
    cluster_t_max: float,
    completion_even_a2: float,
    completion_even_a4: float,
    completion_even_a6: float,
    completion_even_a8: float,
    completion_even_mode: str,
) -> dict[str, float]:
    det_zero, _ = global_probe._compute_det_grid(
        packets=packets,
        sigma=float(sigma_zero),
        t_grid=t_grid,
        completion_even_a2=float(completion_even_a2),
        completion_even_a4=float(completion_even_a4),
        completion_even_a6=float(completion_even_a6),
        completion_even_a8=float(completion_even_a8),
        completion_even_mode=str(completion_even_mode),
    )
    cand = global_probe._zero_candidates(t_grid=t_grid, det_vals=det_zero, top_k=int(max(8, t_grid.size // 20)))
    cand = [row for row in cand if float(cluster_t_min) <= float(row["t"]) <= float(cluster_t_max)]
    if cand:
        best = min(cand, key=lambda row: float(row["abs_det"]))
        return {"t": float(best["t"]), "abs_det": float(best["abs_det"]), "q_re": float(best["q_re"]), "q_im": float(best["q_im"])}
    mask = (np.asarray(t_grid, dtype=float) >= float(cluster_t_min)) & (np.asarray(t_grid, dtype=float) <= float(cluster_t_max))
    if not np.any(mask):
        idx = int(np.argmin(np.abs(det_zero)))
        return {"t": float(t_grid[idx]), "abs_det": float(abs(det_zero[idx])), "q_re": float(np.real(det_zero[idx])), "q_im": float(np.imag(det_zero[idx]))}
    mags = np.abs(det_zero[mask]).astype(float)
    t_sub = np.asarray(t_grid, dtype=float)[mask]
    idx_local = int(np.argmin(mags))
    return {"t": float(t_sub[idx_local]), "abs_det": float(mags[idx_local]), "q_re": float(np.real(det_zero[mask][idx_local])), "q_im": float(np.imag(det_zero[mask][idx_local]))}


def _sigma_scan_rows(
    *,
    packets: list[frontend.GenericPacket],
    sigma_values: list[float],
    t0: float,
    u: float,
    completion_even_a2: float,
    completion_even_a4: float,
    completion_even_a6: float,
    completion_even_a8: float,
    completion_even_mode: str,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for sigma in sigma_values:
        det_vals, _ = global_probe._compute_det_grid(
            packets=packets,
            sigma=float(sigma),
            t_grid=np.asarray([float(t0)], dtype=float),
            completion_even_a2=float(completion_even_a2),
            completion_even_a4=float(completion_even_a4),
            completion_even_a6=float(completion_even_a6),
            completion_even_a8=float(completion_even_a8),
            completion_even_mode=str(completion_even_mode),
        )
        z = complex(det_vals[0])
        rows.append({"u": float(u), "sigma": float(sigma), "t": float(t0), "abs_det": float(abs(z)), "q_re": float(np.real(z)), "q_im": float(np.imag(z))})
    return rows


def _mangoldt_probe_command(csv_path: Path, out_csv: Path) -> str:
    return (
        f"c:/Users/chris_xht57xv/machinelearningexperiments_RH/.venv/Scripts/python.exe "
        f"tools/mangoldt_logderivative_probe.py --csv {csv_path} --sigma 2.0 --n_max 512 --out_csv {out_csv}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Analytic characterization of the frozen backend plus accepted A3 frontend: FE defect map, packet rigidity, zero tracking, transverse sigma scan, and stability checks."
        )
    )
    ap.add_argument("--coeff_csv", default=str(backend.DEFAULT_COEFF_CSV))
    ap.add_argument("--u_canonical", type=float, default=0.24)
    ap.add_argument("--u_track", default="0.21,0.22,0.23,0.24")
    ap.add_argument("--u_stress", type=float, default=0.2)
    ap.add_argument("--primes_global", default="2,3,5,7,11,13,17,19,23,29")
    ap.add_argument("--primes_alt", default="2,3,5,7,11,13,17,19,23")
    ap.add_argument("--radius_max", type=float, default=0.999)
    ap.add_argument("--n_random", type=int, default=30000)
    ap.add_argument("--local_steps", type=int, default=40)
    ap.add_argument("--w_A1", type=float, default=1.35)
    ap.add_argument("--w_A2", type=float, default=1.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma_values_fe", default="0.35,0.4,0.45,0.5,0.55,0.6,0.65")
    ap.add_argument("--sigma_scan_min", type=float, default=0.45)
    ap.add_argument("--sigma_scan_max", type=float, default=0.55)
    ap.add_argument("--sigma_scan_steps", type=int, default=41)
    ap.add_argument("--sigma_zero", type=float, default=0.5)
    ap.add_argument("--t_min", type=float, default=10.0)
    ap.add_argument("--t_max", type=float, default=30.0)
    ap.add_argument("--n_t", type=int, default=401)
    ap.add_argument("--cluster_t_min", type=float, default=27.7)
    ap.add_argument("--cluster_t_max", type=float, default=28.2)
    ap.add_argument("--zero_band", type=float, default=0.25)
    ap.add_argument("--perturb_radius_frac", type=float, default=0.01)
    ap.add_argument("--perturb_phase", type=float, default=0.03)
    ap.add_argument("--completion_even_a2", type=float, default=0.0)
    ap.add_argument("--completion_even_a4", type=float, default=-0.2)
    ap.add_argument("--completion_even_a6", type=float, default=0.0)
    ap.add_argument("--completion_even_a8", type=float, default=0.0)
    ap.add_argument("--completion_even_mode", choices=["direct", "anchored_real"], default="anchored_real")
    ap.add_argument("--stability_t_min", type=float, default=5.0)
    ap.add_argument("--stability_t_max", type=float, default=40.0)
    ap.add_argument("--stability_n_t", type=int, default=1401)
    ap.add_argument("--out_prefix", default="out/analytic_characterization_a3default")
    args = ap.parse_args()

    coeff_csv = Path(str(args.coeff_csv))
    primes_global = frontend._parse_int_csv(str(args.primes_global))
    primes_alt = frontend._parse_int_csv(str(args.primes_alt))
    sigma_values_fe = _parse_float_csv(str(args.sigma_values_fe))
    sigma_scan_values = np.linspace(float(args.sigma_scan_min), float(args.sigma_scan_max), int(args.sigma_scan_steps), dtype=float).tolist()
    t_grid = np.linspace(float(args.t_min), float(args.t_max), int(args.n_t), dtype=float)
    out_prefix = Path(str(args.out_prefix))
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    packets_canon, meta_canon, _ = global_probe._build_packets_for_u(
        coeff_csv=coeff_csv,
        u=float(args.u_canonical),
        primes_global=primes_global,
        radius_max=float(args.radius_max),
        n_random=int(args.n_random),
        local_steps=int(args.local_steps),
        w_A1=float(args.w_A1),
        w_A2=float(args.w_A2),
        seed=int(args.seed),
    )
    base_min = _best_cluster_minimum(
        packets=packets_canon,
        sigma_zero=float(args.sigma_zero),
        t_grid=t_grid,
        cluster_t_min=float(args.cluster_t_min),
        cluster_t_max=float(args.cluster_t_max),
        completion_even_a2=float(args.completion_even_a2),
        completion_even_a4=float(args.completion_even_a4),
        completion_even_a6=float(args.completion_even_a6),
        completion_even_a8=float(args.completion_even_a8),
        completion_even_mode=str(args.completion_even_mode),
    )

    fe_rows = _fe_surface_rows(
        packets=packets_canon,
        sigma_values=sigma_values_fe,
        t_grid=t_grid,
        u=float(args.u_canonical),
        completion_even_a2=float(args.completion_even_a2),
        completion_even_a4=float(args.completion_even_a4),
        completion_even_a6=float(args.completion_even_a6),
        completion_even_a8=float(args.completion_even_a8),
        completion_even_mode=str(args.completion_even_mode),
    )
    fe_df = pd.DataFrame(fe_rows)
    fe_summary = _summarize_fe_surface(fe_df=fe_df, zero_t=float(base_min["t"]), zero_band=float(args.zero_band))
    fe_summary_row = {
        "u": float(args.u_canonical),
        "frontend_family": "A3_exact_frontend",
        "completion_even_a2": float(args.completion_even_a2),
        "completion_even_a4": float(args.completion_even_a4),
        "completion_even_a6": float(args.completion_even_a6),
        "completion_even_a8": float(args.completion_even_a8),
        "completion_even_mode": str(args.completion_even_mode),
        "zero_candidate_t": float(base_min["t"]),
        "zero_candidate_abs_det": float(base_min["abs_det"]),
        **fe_summary,
    }

    sigma_scan_rows = _sigma_scan_rows(
        packets=packets_canon,
        sigma_values=sigma_scan_values,
        t0=float(base_min["t"]),
        u=float(args.u_canonical),
        completion_even_a2=float(args.completion_even_a2),
        completion_even_a4=float(args.completion_even_a4),
        completion_even_a6=float(args.completion_even_a6),
        completion_even_a8=float(args.completion_even_a8),
        completion_even_mode=str(args.completion_even_mode),
    )
    sigma_scan_df = pd.DataFrame(sigma_scan_rows)
    sigma_min_idx = int(sigma_scan_df["abs_det"].astype(float).idxmin())
    sigma_summary_row = {
        "u": float(args.u_canonical),
        "t_cluster": float(base_min["t"]),
        "sigma_at_min_abs_det": float(sigma_scan_df.loc[sigma_min_idx, "sigma"]),
        "min_abs_det": float(sigma_scan_df.loc[sigma_min_idx, "abs_det"]),
        "abs_det_at_critical": float(sigma_scan_df.loc[np.argmin(np.abs(sigma_scan_df["sigma"].astype(float) - 0.5)), "abs_det"]),
        "critical_line_preference_gap": float(sigma_scan_df.loc[np.argmin(np.abs(sigma_scan_df["sigma"].astype(float) - 0.5)), "abs_det"] - sigma_scan_df.loc[sigma_min_idx, "abs_det"]),
    }

    track_rows: list[dict[str, float | bool]] = []
    for idx, u in enumerate(_parse_float_csv(str(args.u_track))):
        packets_u, meta_u, _ = global_probe._build_packets_for_u(
            coeff_csv=coeff_csv,
            u=float(u),
            primes_global=primes_global,
            radius_max=float(args.radius_max),
            n_random=int(args.n_random),
            local_steps=int(args.local_steps),
            w_A1=float(args.w_A1),
            w_A2=float(args.w_A2),
            seed=int(args.seed) + 100 + idx,
        )
        best_u = _best_cluster_minimum(
            packets=packets_u,
            sigma_zero=float(args.sigma_zero),
            t_grid=t_grid,
            cluster_t_min=float(args.cluster_t_min),
            cluster_t_max=float(args.cluster_t_max),
            completion_even_a2=float(args.completion_even_a2),
            completion_even_a4=float(args.completion_even_a4),
            completion_even_a6=float(args.completion_even_a6),
            completion_even_a8=float(args.completion_even_a8),
            completion_even_mode=str(args.completion_even_mode),
        )
        sigma_rows_u = _sigma_scan_rows(
            packets=packets_u,
            sigma_values=sigma_scan_values,
            t0=float(best_u["t"]),
            u=float(u),
            completion_even_a2=float(args.completion_even_a2),
            completion_even_a4=float(args.completion_even_a4),
            completion_even_a6=float(args.completion_even_a6),
            completion_even_a8=float(args.completion_even_a8),
            completion_even_mode=str(args.completion_even_mode),
        )
        sigma_df_u = pd.DataFrame(sigma_rows_u)
        idx_u = int(sigma_df_u["abs_det"].astype(float).idxmin())
        sigma_star = float(sigma_df_u.loc[idx_u, "sigma"])
        track_rows.append(
            {
                "u": float(u),
                "t_candidate": float(best_u["t"]),
                "abs_det_candidate": float(best_u["abs_det"]),
                "sigma_at_transverse_min": float(sigma_star),
                "abs_det_transverse_min": float(sigma_df_u.loc[idx_u, "abs_det"]),
                "critical_line_preferred": bool(abs(sigma_star - 0.5) <= 0.02),
                "coeff_err_pass": bool(meta_u["coeff_err_pass"]),
                "err1": float(meta_u["err1"]),
                "err2": float(meta_u["err2"]),
                "err3": float(meta_u["err3"]),
            }
        )
    track_df = pd.DataFrame(track_rows)

    radii0, phases0 = _packet_params_from_meta(meta_canon)
    dr = float(args.perturb_radius_frac)
    dth = float(args.perturb_phase)
    perturb_specs = [
        ("base", radii0.copy(), phases0.copy()),
        ("r_all_up", np.clip(radii0 * (1.0 + dr), 1e-6, float(args.radius_max) - 1e-6), phases0.copy()),
        ("r_all_down", np.clip(radii0 * (1.0 - dr), 1e-6, float(args.radius_max) - 1e-6), phases0.copy()),
        ("phase_twist_pos", radii0.copy(), phases0 + np.asarray([dth, -dth, dth], dtype=float)),
        ("phase_twist_neg", radii0.copy(), phases0 + np.asarray([-dth, dth, -dth], dtype=float)),
    ]

    rigidity_rows: list[dict[str, float | str | bool]] = []
    for label, rr, tt in perturb_specs:
        eigs = _eigs_from_params(radii=np.asarray(rr, dtype=float), phases=np.asarray(tt, dtype=float))
        packets = _make_packets(eigs=eigs, primes_global=primes_global)
        det_sigma_half, _ = global_probe._compute_det_grid(
            packets=packets,
            sigma=float(args.sigma_zero),
            t_grid=t_grid,
            completion_even_a2=float(args.completion_even_a2),
            completion_even_a4=float(args.completion_even_a4),
            completion_even_a6=float(args.completion_even_a6),
            completion_even_a8=float(args.completion_even_a8),
            completion_even_mode=str(args.completion_even_mode),
        )
        base_det_sigma_half, _ = global_probe._compute_det_grid(
            packets=packets_canon,
            sigma=float(args.sigma_zero),
            t_grid=t_grid,
            completion_even_a2=float(args.completion_even_a2),
            completion_even_a4=float(args.completion_even_a4),
            completion_even_a6=float(args.completion_even_a6),
            completion_even_a8=float(args.completion_even_a8),
            completion_even_mode=str(args.completion_even_mode),
        )
        fe_rows_p = _fe_surface_rows(
            packets=packets,
            sigma_values=sigma_values_fe,
            t_grid=t_grid,
            u=float(args.u_canonical),
            completion_even_a2=float(args.completion_even_a2),
            completion_even_a4=float(args.completion_even_a4),
            completion_even_a6=float(args.completion_even_a6),
            completion_even_a8=float(args.completion_even_a8),
            completion_even_mode=str(args.completion_even_mode),
        )
        fe_df_p = pd.DataFrame(fe_rows_p)
        fe_summary_p = _summarize_fe_surface(fe_df=fe_df_p, zero_t=float(base_min["t"]), zero_band=float(args.zero_band))
        best_p = _best_cluster_minimum(
            packets=packets,
            sigma_zero=float(args.sigma_zero),
            t_grid=t_grid,
            cluster_t_min=float(args.cluster_t_min),
            cluster_t_max=float(args.cluster_t_max),
            completion_even_a2=float(args.completion_even_a2),
            completion_even_a4=float(args.completion_even_a4),
            completion_even_a6=float(args.completion_even_a6),
            completion_even_a8=float(args.completion_even_a8),
            completion_even_mode=str(args.completion_even_mode),
        )
        B1, B2, B3 = frontend._coeffs_from_eigs(np.asarray(eigs, dtype=np.complex128))
        err1 = float(abs(B1 - complex(float(meta_canon["A1_star"]), 0.0)))
        err2 = float(abs(B2 - complex(float(meta_canon["A2_star"]), 0.0)))
        err3 = float(abs(B3 - complex(float(meta_canon["A3_star"]), 0.0)))
        det_rms = float(np.sqrt(np.mean((np.log(np.abs(det_sigma_half) + 1e-300) - np.log(np.abs(base_det_sigma_half) + 1e-300)) ** 2)))
        rigidity_rows.append(
            {
                "label": str(label),
                "u": float(args.u_canonical),
                "completion_even_a2": float(args.completion_even_a2),
                "completion_even_a4": float(args.completion_even_a4),
                "completion_even_a6": float(args.completion_even_a6),
                "completion_even_a8": float(args.completion_even_a8),
                "completion_even_mode": str(args.completion_even_mode),
                "err1": float(err1),
                "err2": float(err2),
                "err3": float(err3),
                "spectral_radius": float(np.max(np.abs(eigs))),
                "median_rel_det_defect_all": float(fe_summary_p["median_rel_det_defect_all"]),
                "median_rel_det_defect_zero_on_critical": float(fe_summary_p["median_rel_det_defect_zero_on_critical"]),
                "t_candidate": float(best_p["t"]),
                "abs_det_candidate": float(best_p["abs_det"]),
                "delta_t_candidate": float(best_p["t"] - float(base_min["t"])),
                "det_logabs_rms_diff_sigma_half": float(det_rms),
                "coeff_err_pass": bool(err1 < 0.01 and err2 < 0.05 and err3 < 0.05),
            }
        )
        det_df_p = frontend._global_det_line_generic(packets=packets, sigma=2.0, t_grid=t_grid)
        det_df_p.to_csv(Path(str(out_prefix) + f"_rigidity_{label}_det.csv"), index=False)
    rigidity_df = pd.DataFrame(rigidity_rows)

    stability_rows: list[dict[str, float | str | bool]] = []
    for label, t_lo, t_hi, n_t, primes_use in [
        ("base", float(args.t_min), float(args.t_max), int(args.n_t), primes_global),
        ("fine", float(args.t_min), float(args.t_max), 1201, primes_global),
        ("wide", float(args.stability_t_min), float(args.stability_t_max), int(args.stability_n_t), primes_global),
        ("primecut_alt", float(args.t_min), float(args.t_max), int(args.n_t), primes_alt),
    ]:
        packets_s, _, _ = global_probe._build_packets_for_u(
            coeff_csv=coeff_csv,
            u=float(args.u_canonical),
            primes_global=list(primes_use),
            radius_max=float(args.radius_max),
            n_random=int(args.n_random),
            local_steps=int(args.local_steps),
            w_A1=float(args.w_A1),
            w_A2=float(args.w_A2),
            seed=int(args.seed),
        )
        t_grid_s = np.linspace(float(t_lo), float(t_hi), int(n_t), dtype=float)
        best_s = _best_cluster_minimum(
            packets=packets_s,
            sigma_zero=float(args.sigma_zero),
            t_grid=t_grid_s,
            cluster_t_min=float(args.cluster_t_min),
            cluster_t_max=float(args.cluster_t_max),
            completion_even_a2=float(args.completion_even_a2),
            completion_even_a4=float(args.completion_even_a4),
            completion_even_a6=float(args.completion_even_a6),
            completion_even_a8=float(args.completion_even_a8),
            completion_even_mode=str(args.completion_even_mode),
        )
        fe_rows_s = _fe_surface_rows(
            packets=packets_s,
            sigma_values=sigma_values_fe,
            t_grid=t_grid_s,
            u=float(args.u_canonical),
            completion_even_a2=float(args.completion_even_a2),
            completion_even_a4=float(args.completion_even_a4),
            completion_even_a6=float(args.completion_even_a6),
            completion_even_a8=float(args.completion_even_a8),
            completion_even_mode=str(args.completion_even_mode),
        )
        fe_df_s = pd.DataFrame(fe_rows_s)
        fe_summary_s = _summarize_fe_surface(fe_df=fe_df_s, zero_t=float(best_s["t"]), zero_band=float(args.zero_band))
        stability_rows.append(
            {
                "label": str(label),
                "u": float(args.u_canonical),
                "completion_even_a2": float(args.completion_even_a2),
                "completion_even_a4": float(args.completion_even_a4),
                "completion_even_a6": float(args.completion_even_a6),
                "completion_even_a8": float(args.completion_even_a8),
                "completion_even_mode": str(args.completion_even_mode),
                "t_candidate": float(best_s["t"]),
                "abs_det_candidate": float(best_s["abs_det"]),
                "candidate_in_cluster": bool(float(args.cluster_t_min) <= float(best_s["t"]) <= float(args.cluster_t_max)),
                "median_rel_det_defect_all": float(fe_summary_s["median_rel_det_defect_all"]),
                "median_rel_det_defect_zero_on_critical": float(fe_summary_s["median_rel_det_defect_zero_on_critical"]),
                "n_primes": int(len(primes_use)),
                "t_min": float(t_lo),
                "t_max": float(t_hi),
                "n_t": int(n_t),
            }
        )
    stability_df = pd.DataFrame(stability_rows)

    fe_surface_path = Path(str(out_prefix) + "_fe_surface.csv")
    fe_summary_path = Path(str(out_prefix) + "_fe_summary.csv")
    track_path = Path(str(out_prefix) + "_zero_tracking.csv")
    sigma_scan_path = Path(str(out_prefix) + "_sigma_scan.csv")
    sigma_summary_path = Path(str(out_prefix) + "_sigma_scan_summary.csv")
    rigidity_path = Path(str(out_prefix) + "_rigidity.csv")
    stability_path = Path(str(out_prefix) + "_stability.csv")
    fe_df.to_csv(fe_surface_path, index=False)
    pd.DataFrame([fe_summary_row]).to_csv(fe_summary_path, index=False)
    track_df.to_csv(track_path, index=False)
    sigma_scan_df.to_csv(sigma_scan_path, index=False)
    pd.DataFrame([sigma_summary_row]).to_csv(sigma_summary_path, index=False)
    rigidity_df.to_csv(rigidity_path, index=False)
    stability_df.to_csv(stability_path, index=False)

    print(f"wrote {fe_surface_path}")
    print(f"wrote {fe_summary_path}")
    print(f"wrote {track_path}")
    print(f"wrote {sigma_scan_path}")
    print(f"wrote {sigma_summary_path}")
    print(f"wrote {rigidity_path}")
    print(f"wrote {stability_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())