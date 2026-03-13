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
import probe_frontend_realization as frontend  # type: ignore


def _parse_float_csv(s: str) -> list[float]:
    out: list[float] = []
    for part in str(s).split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out


def _centered_even_log_weight(
    *,
    sigma: float,
    t_grid: np.ndarray,
    a2: float,
    a4: float,
    a6: float,
    a8: float,
    mode: str,
) -> np.ndarray:
    sigma = float(sigma)
    t_grid = np.asarray(t_grid, dtype=float)
    mode = str(mode).strip().lower()
    s_vals = (sigma + 1j * t_grid).astype(np.complex128)
    coeffs = {2: float(a2), 4: float(a4), 6: float(a6), 8: float(a8)}
    if mode == "direct":
        h = (s_vals - 0.5).astype(np.complex128)
        out = np.zeros_like(h, dtype=np.complex128)
        for power, coeff in coeffs.items():
            if coeff == 0.0:
                continue
            out += coeff * (h ** int(power))
        return out.astype(np.complex128)
    if mode == "anchored_real":
        h = (complex(float(sigma - 0.5), 0.0) + 1j * t_grid).astype(np.complex128)
        h_crit = (1j * t_grid).astype(np.complex128)
        out = np.zeros_like(t_grid, dtype=float)
        for power, coeff in coeffs.items():
            if coeff == 0.0:
                continue
            out += coeff * np.real((h ** int(power)) - (h_crit ** int(power))).astype(float)
        return np.asarray(out, dtype=np.complex128)
    raise ValueError("completion_even_mode must be one of: direct, anchored_real")


def _build_packets_for_u(
    *,
    coeff_csv: Path,
    u: float,
    primes_global: list[int],
    radius_max: float,
    n_random: int,
    local_steps: int,
    w_A1: float,
    w_A2: float,
    seed: int,
) -> tuple[list[frontend.GenericPacket], dict[str, float], backend.CorrectedFactors]:
    factors = backend.load_corrected_factors(coeff_csv=coeff_csv, u=float(u))
    eigs, meta = frontend._fit_a3_exact_frontend(
        A1=float(factors.A1_star),
        A2=float(factors.A2_star),
        A3=float(factors.A3_star),
        radius_max=float(radius_max),
        n_random=int(n_random),
        local_steps=int(local_steps),
        seed=int(seed),
        w_A1=float(w_A1),
        w_A2=float(w_A2),
    )
    S = np.diag(np.asarray(eigs, dtype=np.complex128))
    packets = [
        frontend.GenericPacket(p=int(p), ell=float(math.log(float(p))), S=np.asarray(S, dtype=np.complex128))
        for p in primes_global
    ]
    meta = dict(meta)
    meta["frontend_family"] = "A3_exact_frontend"
    meta["u"] = float(u)
    meta["A1_star"] = float(factors.A1_star)
    meta["A2_star"] = float(factors.A2_star)
    meta["A3_star"] = float(factors.A3_star)
    meta["beta2"] = float(factors.beta2)
    meta["beta3"] = float(factors.beta3)
    meta["c"] = float(factors.c)
    meta["coeff_err_pass"] = bool(float(meta["err1"]) < 0.01 and float(meta["err2"]) < 0.05 and float(meta["err3"]) < 0.05)
    return packets, meta, factors


def _compute_det_grid(
    *,
    packets: list[frontend.GenericPacket],
    sigma: float,
    t_grid: np.ndarray,
    completion_even_a2: float = 0.0,
    completion_even_a4: float = 0.0,
    completion_even_a6: float = 0.0,
    completion_even_a8: float = 0.0,
    completion_even_mode: str = "direct",
) -> tuple[np.ndarray, np.ndarray]:
    s_vals = (float(sigma) + 1j * np.asarray(t_grid, dtype=float)).astype(np.complex128)
    det = np.zeros_like(s_vals, dtype=np.complex128)
    logdet = np.zeros_like(s_vals, dtype=np.complex128)
    for i, s in enumerate(s_vals.tolist()):
        dim = int(sum(pkt.S.shape[0] for pkt in packets))
        K = np.zeros((dim, dim), dtype=np.complex128)
        ofs = 0
        for pkt in packets:
            d = int(pkt.S.shape[0])
            a = complex(np.exp(-complex(s) * float(pkt.ell)))
            K[ofs : ofs + d, ofs : ofs + d] = (a * np.asarray(pkt.S, dtype=np.complex128)).astype(np.complex128)
            ofs += d
        M = (np.eye(dim, dtype=np.complex128) - K).astype(np.complex128)
        det[i] = complex(np.linalg.det(M))
        try:
            sign, logabs = np.linalg.slogdet(M)
            logdet[i] = complex(np.log(sign) + logabs)
        except Exception:
            logdet[i] = complex("nan")
    if any(abs(float(v)) > 0.0 for v in [completion_even_a2, completion_even_a4, completion_even_a6, completion_even_a8]):
        log_weight = _centered_even_log_weight(
            sigma=float(sigma),
            t_grid=np.asarray(t_grid, dtype=float),
            a2=float(completion_even_a2),
            a4=float(completion_even_a4),
            a6=float(completion_even_a6),
            a8=float(completion_even_a8),
            mode=str(completion_even_mode),
        )
        logdet = (logdet + log_weight).astype(np.complex128)
        det = (det * np.exp(log_weight)).astype(np.complex128)
    return det, logdet


def _zero_candidates(*, t_grid: np.ndarray, det_vals: np.ndarray, top_k: int) -> list[dict[str, float]]:
    t = np.asarray(t_grid, dtype=float).ravel()
    mag = np.abs(np.asarray(det_vals, dtype=np.complex128).ravel()).astype(float)
    rows: list[dict[str, float]] = []
    if t.size < 3:
        return rows
    for i in range(1, int(t.size) - 1):
        if mag[i] <= mag[i - 1] and mag[i] <= mag[i + 1]:
            rows.append(
                {
                    "t": float(t[i]),
                    "abs_det": float(mag[i]),
                    "logabs_det": float(math.log(max(1e-300, mag[i]))),
                    "q_re": float(np.real(det_vals[i])),
                    "q_im": float(np.imag(det_vals[i])),
                }
            )
    rows.sort(key=lambda row: row["abs_det"])
    return rows[: int(top_k)]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build the completed global object from the locked backend and default A3_exact frontend, "
            "then evaluate FE-type symmetry, critical-line zero candidates, and stability across u."
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
    ap.add_argument("--completion_basis", choices=["none", "poly2", "poly2_gamma"], default="poly2")
    ap.add_argument("--completion_even_a2", type=float, default=0.0)
    ap.add_argument("--completion_even_a4", type=float, default=-0.2)
    ap.add_argument("--completion_even_a6", type=float, default=0.0)
    ap.add_argument("--completion_even_a8", type=float, default=0.0)
    ap.add_argument("--completion_even_mode", choices=["direct", "anchored_real"], default="anchored_real")
    ap.add_argument("--t_min", type=float, default=10.0)
    ap.add_argument("--t_max", type=float, default=30.0)
    ap.add_argument("--n_t", type=int, default=401)
    ap.add_argument("--top_zero_candidates", type=int, default=8)
    ap.add_argument("--out_prefix", default="out/completed_global_object")
    args = ap.parse_args()

    coeff_csv = Path(str(args.coeff_csv))
    us = _parse_float_csv(str(args.us))
    primes_global = frontend._parse_int_csv(str(args.primes_global))
    t_grid = np.linspace(float(args.t_min), float(args.t_max), int(args.n_t), dtype=float)
    w = fe_diag._trapz_weights(t_grid)
    out_prefix = Path(str(args.out_prefix))
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, float | str | bool]] = []
    det_rows: list[dict[str, float | str]] = []
    zero_rows: list[dict[str, float | str | bool]] = []

    for idx, u in enumerate(us):
        packets, meta, factors = _build_packets_for_u(
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

        s_fe = (float(args.sigma_fe) + 1j * t_grid).astype(np.complex128)
        s_1ms = (1.0 - float(args.sigma_fe) - 1j * t_grid).astype(np.complex128)
        det_fe = np.zeros_like(s_fe, dtype=np.complex128)
        log_fe = np.zeros_like(s_fe, dtype=np.complex128)
        det_1ms = np.zeros_like(s_fe, dtype=np.complex128)
        log_1ms = np.zeros_like(s_fe, dtype=np.complex128)
        det_fe, log_fe = _compute_det_grid(
            packets=packets,
            sigma=float(args.sigma_fe),
            t_grid=t_grid,
            completion_even_a2=float(args.completion_even_a2),
            completion_even_a4=float(args.completion_even_a4),
            completion_even_a6=float(args.completion_even_a6),
            completion_even_a8=float(args.completion_even_a8),
            completion_even_mode=str(args.completion_even_mode),
        )
        det_1ms, log_1ms = _compute_det_grid(
            packets=packets,
            sigma=float(1.0 - float(args.sigma_fe)),
            t_grid=-t_grid,
            completion_even_a2=float(args.completion_even_a2),
            completion_even_a4=float(args.completion_even_a4),
            completion_even_a6=float(args.completion_even_a6),
            completion_even_a8=float(args.completion_even_a8),
            completion_even_mode=str(args.completion_even_mode),
        )

        m_conj = fe_diag._symmetry_metrics(
            s_vals=s_fe,
            det_s=det_fe,
            logdet_s=log_fe,
            det_sym=np.conjugate(det_1ms).astype(np.complex128),
            logdet_sym=np.conjugate(log_1ms).astype(np.complex128),
            w=w,
            completion_basis=str(args.completion_basis),
        )
        abs_conj = fe_diag._abs_symmetry_metrics(
            s_vals=s_fe,
            logabs_s=fe_diag._logabs(det_fe),
            logabs_sym=fe_diag._logabs(np.conjugate(det_1ms).astype(np.complex128)),
            w=w,
            completion_basis=str(args.completion_basis),
        )

        det_zero, _ = _compute_det_grid(
            packets=packets,
            sigma=float(args.sigma_zero),
            t_grid=t_grid,
            completion_even_a2=float(args.completion_even_a2),
            completion_even_a4=float(args.completion_even_a4),
            completion_even_a6=float(args.completion_even_a6),
            completion_even_a8=float(args.completion_even_a8),
            completion_even_mode=str(args.completion_even_mode),
        )
        zero_cands = _zero_candidates(t_grid=t_grid, det_vals=det_zero, top_k=int(args.top_zero_candidates))
        min_abs_det = float(np.min(np.abs(det_zero))) if det_zero.size else float("nan")
        min_idx = int(np.argmin(np.abs(det_zero))) if det_zero.size else -1

        for row in zero_cands:
            zero_rows.append({"u": float(u), "frontend_family": "A3_exact_frontend", **row})

        for t, z in zip(t_grid.tolist(), det_zero.tolist()):
            det_rows.append(
                {
                    "u": float(u),
                    "sigma": float(args.sigma_zero),
                    "t": float(t),
                    "q_track_re": float(np.real(z)),
                    "q_track_im": float(np.imag(z)),
                    "frontend_family": "A3_exact_frontend",
                }
            )

        summary_rows.append(
            {
                "u": float(u),
                "frontend_family": "A3_exact_frontend",
                "coeff_csv": str(coeff_csv),
                "A1_star": float(factors.A1_star),
                "A2_star": float(factors.A2_star),
                "A3_star": float(factors.A3_star),
                "beta2": float(factors.beta2),
                "beta3": float(factors.beta3),
                "c": float(factors.c),
                "err1": float(meta["err1"]),
                "err2": float(meta["err2"]),
                "err3": float(meta["err3"]),
                "coeff_err_pass": bool(meta["coeff_err_pass"]),
                "spectral_radius": float(meta["spectral_radius"]),
                "rel_l2_det_conj1ms": float(m_conj.rel_l2_det),
                "rel_l2_logdet_conj1ms": float(m_conj.rel_l2_logdet),
                "completion_rich_rel_l2_logdet_conj1ms": float(m_conj.completion_rich_rel_l2_logdet),
                "rel_l2_logabs_det_conj1ms": float(abs_conj.rel_l2_logabs),
                "completion_rich_rel_l2_logabs_det_conj1ms": float(abs_conj.completion_rich_rel_l2_logabs),
                "sigma_fe": float(args.sigma_fe),
                "sigma_zero": float(args.sigma_zero),
                "completion_even_a2": float(args.completion_even_a2),
                "completion_even_a4": float(args.completion_even_a4),
                "completion_even_a6": float(args.completion_even_a6),
                "completion_even_a8": float(args.completion_even_a8),
                "completion_even_mode": str(args.completion_even_mode),
                "min_abs_det_sigma_zero": float(min_abs_det),
                "t_at_min_abs_det_sigma_zero": float(t_grid[min_idx]) if min_idx >= 0 else float("nan"),
                "n_zero_candidates": int(len(zero_cands)),
                "best_zero_candidate_t": float(zero_cands[0]["t"]) if zero_cands else float("nan"),
                "best_zero_candidate_abs_det": float(zero_cands[0]["abs_det"]) if zero_cands else float("nan"),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    zero_df = pd.DataFrame(zero_rows)
    det_df = pd.DataFrame(det_rows)

    stability_row: dict[str, float | str] = {
        "frontend_family": "A3_exact_frontend",
        "u_min": float(min(us)) if us else float("nan"),
        "u_max": float(max(us)) if us else float("nan"),
        "n_u": int(len(us)),
        "spread_rel_l2_logdet_conj1ms": float(summary_df["rel_l2_logdet_conj1ms"].max() - summary_df["rel_l2_logdet_conj1ms"].min()),
        "spread_completion_rich_rel_l2_logdet_conj1ms": float(summary_df["completion_rich_rel_l2_logdet_conj1ms"].max() - summary_df["completion_rich_rel_l2_logdet_conj1ms"].min()),
        "spread_min_abs_det_sigma_zero": float(summary_df["min_abs_det_sigma_zero"].max() - summary_df["min_abs_det_sigma_zero"].min()),
        "n_coeff_err_pass": int(summary_df["coeff_err_pass"].astype(bool).sum()),
        "best_u_by_fe_completion": float(summary_df.loc[summary_df["completion_rich_rel_l2_logdet_conj1ms"].idxmin(), "u"]),
        "best_u_by_zero_candidate": float(summary_df.loc[summary_df["best_zero_candidate_abs_det"].idxmin(), "u"]),
    }

    summary_path = Path(str(out_prefix) + "_summary.csv")
    zero_path = Path(str(out_prefix) + "_zero_candidates.csv")
    det_path = Path(str(out_prefix) + "_critical_line_det.csv")
    stability_path = Path(str(out_prefix) + "_stability.csv")
    summary_df.to_csv(summary_path, index=False)
    zero_df.to_csv(zero_path, index=False)
    det_df.to_csv(det_path, index=False)
    pd.DataFrame([stability_row]).to_csv(stability_path, index=False)

    print(f"wrote {summary_path}")
    print(f"wrote {zero_path}")
    print(f"wrote {det_path}")
    print(f"wrote {stability_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())