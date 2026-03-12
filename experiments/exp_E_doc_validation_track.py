from __future__ import annotations

import argparse
import json
import os
import re
import sys
import heapq
import gc
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mpmath as mp
import numpy as np
import pandas as pd


def _validate_run_contract(out_dir: str, config: dict) -> None:
    """Fail fast if required run artifacts are missing.

    This is intentionally lightweight: it checks *presence* of key files that define
    the experiment contract for reproducibility and audit.
    """

    od = Path(out_dir)
    required: list[str] = [
        "config.json",
        "doc_meta.json",
        "E_summary.md",
    ]

    do_conv = bool(config.get("do_convergence_sweep_only", False))
    boundary_autopick = bool(config.get("boundary_autopick", False))

    if do_conv:
        required += [
            "E_Qlambda_convergence_sweep.csv",
            "E_Qlambda_convergence_fit.csv",
        ]

        if bool(config.get("convergence_do_magphase_diag", True)):
            required.append("E_Qlambda_convergence_magphase.csv")

        # Keep the contract default aligned with the execution default.
        conv_do_winding = bool(config.get("convergence_do_winding", False))
        if conv_do_winding:
            conv_wind_out = str(config.get("convergence_wind_out_csv", "E_Qlambda_convergence_winding.csv"))
            required.append(conv_wind_out)

        # Optional: critical-line fingerprint sweep (Task 1 infrastructure).
        if bool(config.get("do_critical_line_fingerprint", False)):
            fp_points = str(config.get("critical_line_points_out_csv", "E_Qlambda_critical_line_fingerprint_points.csv"))
            fp_summary = str(config.get("critical_line_summary_out_csv", "E_Qlambda_critical_line_fingerprint_summary.csv"))
            required += [fp_points, fp_summary]
            if bool(config.get("critical_line_eval_candidates", False)):
                fp_cands = str(config.get("critical_line_candidates_out_csv", "E_Qlambda_critical_line_candidates_summary.csv"))
                required.append(fp_cands)

            # Optional: pi-ramp diagnostics + alpha-fit witnesses (referee-proof artifacts)
            if bool(config.get("critical_line_do_pi_ramp_diagnostics", False)):
                out_fit = str(config.get("critical_line_pi_ramp_fit_out_csv", "E_Qlambda_pi_ramp_fit.csv"))
                out_res = str(config.get("critical_line_phase_residual_out_csv", "E_Qlambda_phase_residual_table.csv"))
                required += [out_fit, out_res]
                # Optional contracted secondary (relaxed-gamma) witnesses for empty/fragile intersections.
                if bool(config.get("critical_line_pi_ramp_emit_secondary", False)):
                    out_fit2 = str(config.get("critical_line_pi_ramp_fit_secondary_out_csv", "E_Qlambda_pi_ramp_fit_relaxed.csv"))
                    out_res2 = str(config.get("critical_line_phase_residual_secondary_out_csv", "E_Qlambda_phase_residual_table_relaxed.csv"))
                    required += [out_fit2, out_res2]

    if boundary_autopick:
        required += [
            "E_boundary_autopick_candidates.csv",
            "E_boundary_autopick_choice.csv",
            "E_boundary_autopick_acceptance.csv",
            "E_boundary_autopick_policy.md",
        ]

    # Optional diagnostic modules
    if bool(config.get("do_dip_atlas", False)):
        required += [
            "E_dip_atlas_points.csv",
            "E_dip_atlas_lines.csv",
            "E_dip_atlas_suggested_rect.json",
        ]

    if bool(config.get("do_pole_line_scan", False)):
        required.append("E_pole_line_scan.csv")

    if bool(config.get("do_arg_c_emp", True)) and (not do_conv):
        # Only required in the full (non convergence-only) spine where it's computed.
        required.append("E_arg_principle_c_emp.csv")

    if bool(config.get("do_arg_lambda_branch", False)) and (not do_conv):
        required.append("E_divisor_arg_principle_Q_lambda.csv")

    missing = [name for name in required if not (od / name).exists()]
    if missing:
        msg = "Run contract violated: missing required artifacts:\n" + "\n".join([f"- {m}" for m in missing])
        raise RuntimeError(msg)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.configs as cfg
from src.bulk import BulkParams, build_A, eps_prime
from src.dn import dn_map, dn_map_on_indices
from src.logging_utils import make_run_dir, save_run_snapshot, write_json
from src.metrics import cond_number, fro_norm, hermitian_defect, unitarity_defect
from src.scattering import cayley_from_lambda


def _anti_diag_J(n: int) -> np.ndarray:
    n = int(n)
    J = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        J[i, n - 1 - i] = 1.0
    return J


def _sign_diag(n: int) -> np.ndarray:
    n = int(n)
    d = np.ones(n, dtype=np.complex128)
    d[1::2] = -1.0
    return np.diag(d)


def _block_diag(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.complex128)
    B = np.asarray(B, dtype=np.complex128)
    out = np.zeros((A.shape[0] + B.shape[0], A.shape[1] + B.shape[1]), dtype=np.complex128)
    out[: A.shape[0], : A.shape[1]] = A
    out[A.shape[0] :, A.shape[1] :] = B
    return out


def _channel_swap(base_n: int) -> np.ndarray:
    base_n = int(base_n)
    if base_n <= 0:
        raise ValueError("base_n must be positive")
    I = np.eye(base_n, dtype=np.complex128)
    Z = np.zeros((base_n, base_n), dtype=np.complex128)
    top = np.concatenate([Z, I], axis=1)
    bot = np.concatenate([I, Z], axis=1)
    return np.concatenate([top, bot], axis=0)


def _parse_float_list_or_spec(v: object, *, default: str) -> list[float]:
    """Parse either a list/tuple of numbers or a compact spec string.

    Supported spec forms:
    - comma list: "0,0.5,1"
    - range: "start:stop:step" (inclusive-ish; uses np.arange)
    - union: "spec1;spec2"
    """

    if v is None:
        v = default

    if isinstance(v, (list, tuple)):
        out: list[float] = []
        for x in v:
            out.append(float(x))
        return sorted(set(out))

    s = str(v).strip()
    if not s:
        s = str(default)

    out2: list[float] = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            toks = [t.strip() for t in part.split(":") if t.strip()]
            if len(toks) != 3:
                raise ValueError(f"Bad float spec (expected start:stop:step): {part}")
            a0 = float(toks[0])
            a1 = float(toks[1])
            da = float(toks[2])
            if da == 0.0:
                raise ValueError("step must be nonzero")
            arr = np.arange(a0, a1 + 0.5 * da, da, dtype=float)
            out2 += [float(x) for x in arr.tolist()]
        else:
            toks = [t.strip() for t in part.split(",") if t.strip()]
            out2 += [float(t) for t in toks]
    return sorted(set(out2))


def _parse_windows(spec: object) -> list[tuple[float, float]]:
    """Parse windows spec.

    Accepts:
    - list of strings like ["0:25","25:50"]
    - string "0:25;25:50"
    """
    if spec is None:
        return []
    if isinstance(spec, (list, tuple)):
        parts = [str(x) for x in spec]
    else:
        s = str(spec).strip()
        if not s:
            return []
        parts = [p.strip() for p in s.split(";") if p.strip()]
    out: list[tuple[float, float]] = []
    for p in parts:
        toks = [t.strip() for t in str(p).split(":") if t.strip()]
        if len(toks) != 2:
            raise ValueError(f"Bad window spec (expected tmin:tmax): {p}")
        out.append((float(toks[0]), float(toks[1])))
    return out


def _compute_jump_by_N_from_point_rows(point_rows: list[dict]) -> dict[tuple[int, float], float]:
    """Compute lam_raw jump magnitude per point (N,t) using sorted t per N."""
    byN: dict[int, list[tuple[float, complex]]] = {}
    for r in point_rows:
        if str(r.get("status", "")) != "ok":
            continue
        try:
            N0 = int(r.get("N"))
            t0 = float(r.get("t"))
            lam_raw = complex(float(r.get("lam_raw_re")), float(r.get("lam_raw_im")))
        except Exception:
            continue
        byN.setdefault(N0, []).append((t0, lam_raw))

    jump: dict[tuple[int, float], float] = {}
    for N0, arr in byN.items():
        arr2 = sorted(arr, key=lambda x: x[0])
        for i, (t0, lam0) in enumerate(arr2):
            if i == 0:
                jump[(int(N0), float(t0))] = float("nan")
            else:
                jump[(int(N0), float(t0))] = float(abs(lam0 - arr2[i - 1][1]))
    return jump


def _write_pi_ramp_diagnostics_from_points_csv(
    out_dir: str,
    config: dict,
    points_csv_name: str,
    *,
    gamma_override: float | None = None,
    out_fit_override: str | None = None,
    out_res_override: str | None = None,
    relaxed_tag: str = "primary",
) -> None:
    """Emit contracted pi-ramp + residual diagnostic CSV artifacts for the run."""

    od = Path(out_dir)
    points_path = od / str(points_csv_name)
    if not points_path.exists():
        return

    # Settings (kept intentionally explicit for auditability)
    which = str(config.get("critical_line_pi_ramp_which", "healthy"))
    gamma_primary = float(config.get("critical_line_pi_ramp_gamma", 0.05))
    gamma = float(gamma_override) if gamma_override is not None else float(gamma_primary)
    jump_thr = float(config.get("critical_line_pi_ramp_jump", 5.0))
    mask_mode = str(config.get("critical_line_pi_ramp_mask", "intersection_gamma_ginf"))
    loss_name = str(config.get("critical_line_pi_ramp_loss", "phase_top5mean"))
    phase_mode = str(config.get("critical_line_pi_ramp_mode", "unit_phase_err"))
    lambda_mag = float(config.get("critical_line_pi_ramp_lambda_mag", 0.25))
    pi_scales = [int(x) for x in (config.get("critical_line_pi_scales") or [1, 2])]
    windows = _parse_windows(config.get("critical_line_pi_ramp_windows"))
    alpha_grid = _parse_float_list_or_spec(config.get("critical_line_alpha_grid"), default="0,0.5,1,1.5,2")

    out_fit = str(out_fit_override) if out_fit_override is not None else str(config.get("critical_line_pi_ramp_fit_out_csv", "E_Qlambda_pi_ramp_fit.csv"))
    out_res = str(out_res_override) if out_res_override is not None else str(config.get("critical_line_phase_residual_out_csv", "E_Qlambda_phase_residual_table.csv"))

    df = pd.read_csv(points_path)
    rows = df.to_dict(orient="records")
    jumps = _compute_jump_by_N_from_point_rows(rows)

    # Use the same primitives used to create the points; this avoids drift.
    phi_target_mode = str(config.get("phi_target_mode", "modular"))
    phi_select_mode = str(config.get("phi_select_mode", phi_target_mode))
    completion_mode = str(config.get("completion_mode", "none"))
    primes = tuple(int(x) for x in (config.get("primes") or []))

    def _healthy_keyset(*, tmin: float | None = None, tmax: float | None = None) -> set[tuple[int, float]]:
        ok: set[tuple[int, float]] = set()
        for r in rows:
            if str(r.get("status", "")) != "ok":
                continue
            try:
                N0 = int(r.get("N"))
                t0 = float(r.get("t"))
            except Exception:
                continue
            if (tmin is not None and t0 < float(tmin)) or (tmax is not None and t0 > float(tmax)):
                continue
            if which == "healthy":
                try:
                    gap = float(r.get("gap_phi"))
                except Exception:
                    continue
                if (not np.isfinite(gap)) or gap < float(gamma):
                    continue
                j = float(jumps.get((int(N0), float(t0)), float("nan")))
                if (not np.isfinite(j)) or j > float(jump_thr):
                    continue
            ok.add((int(N0), float(t0)))
        return ok

    # Base keyset for full window
    full_ok = _healthy_keyset()

    def q_for_factor(lam_raw: complex, s: complex, fac: complex) -> complex | None:
        try:
            phi_norm = phi_target(s, primes=primes, completion_mode=completion_mode, mode=phi_target_mode)
            if (not is_finite_complex(phi_norm)) or abs(phi_norm) <= 1e-30:
                return None
            lam = complex(lam_raw * fac)
            q = complex(lam / phi_norm)
            return q if is_finite_complex(q) else None
        except Exception:
            return None

    def q_gamma(lam_raw: complex, s: complex) -> complex | None:
        return q_for_factor(lam_raw, s, lambda_norm_factor(s, "gamma_only"))

    def q_ginf(lam_raw: complex, s: complex) -> complex | None:
        return q_for_factor(lam_raw, s, lambda_norm_factor(s, "ginf_multiply"))

    def _iter_points(keyset: set[tuple[int, float]], *, tmin: float | None = None, tmax: float | None = None):
        for r in rows:
            if str(r.get("status", "")) != "ok":
                continue
            try:
                N0 = int(r.get("N"))
                t0 = float(r.get("t"))
                sigma0 = float(r.get("sigma"))
                lam_raw = complex(float(r.get("lam_raw_re")), float(r.get("lam_raw_im")))
            except Exception:
                continue
            if (tmin is not None and t0 < float(tmin)) or (tmax is not None and t0 > float(tmax)):
                continue
            if (int(N0), float(t0)) not in keyset:
                continue
            yield (int(N0), float(t0), complex(float(sigma0), float(t0)), lam_raw)

    def _unwrap_phase_by_N(points_phase: list[tuple[int, float, float]]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        byN: dict[int, list[tuple[float, float]]] = {}
        for N0, t0, ph in points_phase:
            byN.setdefault(int(N0), []).append((float(t0), float(ph)))
        out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for N0, arr in byN.items():
            arr2 = sorted(arr, key=lambda x: x[0])
            ts = np.asarray([a[0] for a in arr2], dtype=float)
            phs = np.asarray([a[1] for a in arr2], dtype=float)
            out[int(N0)] = (ts, np.unwrap(phs))
        return out

    def _robust_linefit_by_N(points_phase: list[tuple[int, float, float]]) -> tuple[float, float, float, dict[int, dict[str, float]]]:
        unwrapped = _unwrap_phase_by_N(points_phase)
        slopes: list[float] = []
        rms_list: list[float] = []
        perN: dict[int, dict[str, float]] = {}
        for N0, (ts, phs) in unwrapped.items():
            mask = np.isfinite(ts) & np.isfinite(phs)
            ts2 = ts[mask]
            phs2 = phs[mask]
            if ts2.size < 8:
                continue
            b, a = np.polyfit(ts2, phs2, 1)
            pred = a + b * ts2
            rms = float(np.sqrt(np.mean((phs2 - pred) ** 2)))
            slopes.append(float(b))
            rms_list.append(float(rms))
            perN[int(N0)] = {"n": float(ts2.size), "b": float(b), "a": float(a), "rms": float(rms)}
        if not slopes:
            return (float("nan"), float("nan"), float("nan"), perN)
        b_med = float(np.median(slopes))
        b_p95abs = float(np.percentile(np.abs(np.asarray(slopes, dtype=float)), 95.0))
        rms_med = float(np.median(rms_list)) if rms_list else float("nan")
        return (b_med, b_p95abs, rms_med, perN)

    def _forced_line_residuals_by_N(points_phase: list[tuple[int, float, float]], *, forced_b: float) -> tuple[float, float, float]:
        """Compute residual slope and RMS after removing a forced slope b, allowing only an intercept per N.

        Returns (b_resid_median, b_resid_p95abs, rms_median).
        """

        unwrapped = _unwrap_phase_by_N(points_phase)
        b_resid: list[float] = []
        rms_list: list[float] = []
        for _, (ts, phs) in unwrapped.items():
            mask = np.isfinite(ts) & np.isfinite(phs)
            ts2 = ts[mask]
            phs2 = phs[mask]
            if ts2.size < 8:
                continue
            a = float(np.mean(phs2 - float(forced_b) * ts2))
            resid = phs2 - (a + float(forced_b) * ts2)
            rms = float(np.sqrt(np.mean(resid**2)))
            br, _ar = np.polyfit(ts2, resid, 1)
            b_resid.append(float(br))
            rms_list.append(float(rms))

        if not b_resid:
            return (float("nan"), float("nan"), float("nan"))
        b_med = float(np.median(b_resid))
        b_p95abs = float(np.percentile(np.abs(np.asarray(b_resid, dtype=float)), 95.0))
        rms_med = float(np.median(rms_list)) if rms_list else float("nan")
        return (b_med, b_p95abs, rms_med)

    def _phase_loss(q: complex) -> float:
        absQ = float(abs(q))
        if not np.isfinite(absQ) or absQ <= 0 or (not is_finite_complex(q)):
            return float("nan")
        if phase_mode == "abs_argQ":
            return float(abs(np.angle(q)))
        unit = q / absQ
        return float(abs(unit - 1.0))

    def _stats(vals: list[float]) -> dict[str, float]:
        a = np.asarray(vals, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return {"n": 0.0, "median": float("nan"), "p95": float("nan"), "top5mean": float("nan"), "max": float("nan")}
        k = max(1, int(np.ceil(0.05 * a.size)))
        top = np.sort(a)[-k:]
        return {
            "n": float(a.size),
            "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95.0)),
            "top5mean": float(np.mean(top)),
            "max": float(np.max(a)),
        }

    def _loss_key(phase_s: dict[str, float], mag_s: dict[str, float]) -> float:
        if loss_name == "phase_p95":
            return float(phase_s["p95"])
        if loss_name == "phase_top5mean":
            return float(phase_s["top5mean"])
        if loss_name == "score_p95":
            return float(phase_s["p95"]) + float(lambda_mag) * float(mag_s["p95"])
        if loss_name == "score_top5mean":
            return float(phase_s["top5mean"]) + float(lambda_mag) * float(mag_s["top5mean"])
        return float("inf")

    # CSV rows
    fit_rows: list[dict] = []
    res_rows: list[dict] = []

    # Windows always include full
    win_list: list[tuple[str, float | None, float | None]] = [("full", None, None)]
    for (a, b) in windows:
        win_list.append((f"{a}:{b}", float(a), float(b)))

    # Step A per-window ramp fit on R(t)=Q_ginf/Q_gamma
    for wname, tmin, tmax in win_list:
        keyset = _healthy_keyset(tmin=tmin, tmax=tmax)
        if mask_mode == "intersection_gamma_ginf":
            keyset2: set[tuple[int, float]] = set()
            for N0, t0, s, lam_raw in _iter_points(keyset, tmin=tmin, tmax=tmax):
                qg = q_gamma(lam_raw, s)
                qf = q_ginf(lam_raw, s)
                if qg is None or qf is None:
                    continue
                if abs(qg) <= 1e-300 or abs(qf) <= 1e-300:
                    continue
                keyset2.add((int(N0), float(t0)))
            keyset = keyset2

        slope_pts: list[tuple[int, float, float]] = []
        mag_dev: list[float] = []
        for N0, t0, s, lam_raw in _iter_points(keyset, tmin=tmin, tmax=tmax):
            qg = q_gamma(lam_raw, s)
            qf = q_ginf(lam_raw, s)
            if qg is None or qf is None or abs(qg) <= 1e-300:
                continue
            R = qf / qg
            if not is_finite_complex(R) or abs(R) <= 1e-300:
                continue
            slope_pts.append((int(N0), float(t0), float(np.angle(R))))
            mag_dev.append(float(abs(abs(R) - 1.0)))

        b_med, b_p95abs, rms_med, perN = _robust_linefit_by_N(slope_pts)
        forced_b = -float(np.log(np.pi))
        b_resid_med, b_resid_p95abs, forced_rms_med = _forced_line_residuals_by_N(slope_pts, forced_b=forced_b)
        mstats = _stats(mag_dev)
        fit_rows.append(
            {
                "source": "pi_ramp_fit",
                "relaxed_tag": str(relaxed_tag),
                "relaxed_gamma_used": bool(gamma_override is not None),
                "gamma_primary": float(gamma_primary),
                "gamma_used": float(gamma),
                "step": "A",
                "ratio": "Q_ginf_over_Q_gamma",
                "mask": str(mask_mode),
                "which": str(which),
                "gamma": float(gamma),
                "jump": float(jump_thr),
                "window": str(wname),
                "t_min": float(tmin) if tmin is not None else float("nan"),
                "t_max": float(tmax) if tmax is not None else float("nan"),
                "b_med": float(b_med),
                "b_p95abs": float(b_p95abs),
                "rms_med": float(rms_med),
                "forced_b": float(forced_b),
                "forced_b_resid_med": float(b_resid_med),
                "forced_b_resid_p95abs": float(b_resid_p95abs),
                "forced_rms_med": float(forced_rms_med),
                "magdev_med": float(mstats["median"]),
                "magdev_p95": float(mstats["p95"]),
                "magdev_max": float(mstats["max"]),
                "n_points": int(mstats["n"]),
                "nN": int(len(perN)),
            }
        )

    # Step B: alpha fit per pi_scale and per window
    lnpi = float(np.log(np.pi))
    for pi_scale in pi_scales:
        for wname, tmin, tmax in win_list:
            keyset = _healthy_keyset(tmin=tmin, tmax=tmax)
            if mask_mode == "intersection_gamma_ginf":
                keyset2: set[tuple[int, float]] = set()
                for N0, t0, s, lam_raw in _iter_points(keyset, tmin=tmin, tmax=tmax):
                    qg = q_gamma(lam_raw, s)
                    qf = q_ginf(lam_raw, s)
                    if qg is None or qf is None:
                        continue
                    if abs(qg) <= 1e-300 or abs(qf) <= 1e-300:
                        continue
                    keyset2.add((int(N0), float(t0)))
                keyset = keyset2

            best_alpha: float | None = None
            best_loss: float = float("inf")
            best_phase: dict[str, float] | None = None
            best_mag: dict[str, float] | None = None
            best_n: int = 0

            # evaluate grid
            for alpha in alpha_grid:
                phase_errs: list[float] = []
                mag_errs: list[float] = []
                for _, _, s, lam_raw in _iter_points(keyset, tmin=tmin, tmax=tmax):
                    fac_gamma = lambda_norm_factor(s, "gamma_only")
                    pi_fac = complex(np.exp(float(alpha) * (0.5 - (float(pi_scale) * s)) * lnpi))
                    q = q_for_factor(lam_raw, s, fac_gamma * pi_fac)
                    if q is None:
                        continue
                    absQ = float(abs(q))
                    if not np.isfinite(absQ) or absQ <= 0:
                        continue
                    phase_err = _phase_loss(q)
                    mag_err = float(abs(absQ - 1.0))
                    if np.isfinite(phase_err):
                        phase_errs.append(float(phase_err))
                    if np.isfinite(mag_err):
                        mag_errs.append(float(mag_err))
                ps = _stats(phase_errs)
                ms = _stats(mag_errs)
                cur_loss = _loss_key(ps, ms)
                if np.isfinite(cur_loss) and cur_loss < best_loss:
                    best_loss = float(cur_loss)
                    best_alpha = float(alpha)
                    best_phase = ps
                    best_mag = ms
                    best_n = int(ps.get("n", 0.0))

            fit_rows.append(
                {
                    "source": "pi_ramp_fit",
                    "relaxed_tag": str(relaxed_tag),
                    "relaxed_gamma_used": bool(gamma_override is not None),
                    "gamma_primary": float(gamma_primary),
                    "gamma_used": float(gamma),
                    "step": "B",
                    "model": "Q_gamma * exp(alpha*(1/2 - pi_scale*s)*log(pi))",
                    "pi_scale": int(pi_scale),
                    "mask": str(mask_mode),
                    "which": str(which),
                    "gamma": float(gamma),
                    "jump": float(jump_thr),
                    "window": str(wname),
                    "t_min": float(tmin) if tmin is not None else float("nan"),
                    "t_max": float(tmax) if tmax is not None else float("nan"),
                    "loss": str(loss_name),
                    "best_alpha": float(best_alpha) if best_alpha is not None else float("nan"),
                    "best_loss": float(best_loss) if np.isfinite(best_loss) else float("nan"),
                    "phase_p95": float(best_phase["p95"]) if best_phase is not None else float("nan"),
                    "phase_top5mean": float(best_phase["top5mean"]) if best_phase is not None else float("nan"),
                    "mag_p95": float(best_mag["p95"]) if best_mag is not None else float("nan"),
                    "mag_top5mean": float(best_mag["top5mean"]) if best_mag is not None else float("nan"),
                    "n_points": int(best_n),
                }
            )

            # Step C: residual slope/curvature table for alpha in {0,1,2,best}
            alphas_res = [0.0, 1.0, 2.0]
            if best_alpha is not None and all(abs(best_alpha - a) > 1e-12 for a in alphas_res):
                alphas_res.append(float(best_alpha))
            for alpha in alphas_res:
                phase_pts: list[tuple[int, float, float]] = []
                for N0, t0, s, lam_raw in _iter_points(keyset, tmin=tmin, tmax=tmax):
                    fac_gamma = lambda_norm_factor(s, "gamma_only")
                    pi_fac = complex(np.exp(float(alpha) * (0.5 - (float(pi_scale) * s)) * lnpi))
                    q = q_for_factor(lam_raw, s, fac_gamma * pi_fac)
                    if q is None:
                        continue
                    absQ = float(abs(q))
                    if not np.isfinite(absQ) or absQ <= 0:
                        continue
                    qu = q / absQ
                    if not is_finite_complex(qu):
                        continue
                    phase_pts.append((int(N0), float(t0), float(np.angle(qu))))
                b_med, b_p95abs, rms_med, perN = _robust_linefit_by_N(phase_pts)
                n_used = int(sum(int(v.get("n", 0.0)) for v in perN.values()))
                res_rows.append(
                    {
                        "source": "phase_residual",
                        "relaxed_tag": str(relaxed_tag),
                        "relaxed_gamma_used": bool(gamma_override is not None),
                        "gamma_primary": float(gamma_primary),
                        "gamma_used": float(gamma),
                        "pi_scale": int(pi_scale),
                        "mask": str(mask_mode),
                        "which": str(which),
                        "gamma": float(gamma),
                        "jump": float(jump_thr),
                        "window": str(wname),
                        "t_min": float(tmin) if tmin is not None else float("nan"),
                        "t_max": float(tmax) if tmax is not None else float("nan"),
                        "alpha": float(alpha),
                        "b_med": float(b_med),
                        "b_p95abs": float(b_p95abs),
                        "rms_med": float(rms_med),
                        "n_used": int(n_used),
                    }
                )

    if fit_rows:
        pd.DataFrame(fit_rows).to_csv(od / out_fit, index=False)
    if res_rows:
        pd.DataFrame(res_rows).to_csv(od / out_res, index=False)


def xi(s: complex) -> complex:
    ss = mp.mpc(float(np.real(s)), float(np.imag(s)))
    # mpmath 1.3.0 has no riemannxi/xi helper and mp.gamma raises at poles.
    # xi(s) is entire; at s=0 and s=1 the expression has removable singularities with xi(0)=xi(1)=1/2.
    if abs(ss) < mp.mpf("1e-30"):
        return complex(mp.mpf("0.5"))
    if abs(ss - 1) < mp.mpf("1e-30"):
        return complex(mp.mpf("0.5"))

    def _raw(z):
        return mp.mpf("0.5") * z * (z - 1) * (mp.pi ** (-z / 2)) * mp.gamma(z / 2) * mp.zeta(z)

    try:
        return complex(_raw(ss))
    except ValueError:
        # symmetric limit across the pole (good enough for validation diagnostics)
        eps = mp.mpf("1e-10")
        return complex((_raw(ss + eps) + _raw(ss - eps)) / 2)


def phi_modular(s: complex) -> complex:
    # modular scattering coefficient for PSL(2,Z): phi(s)=xi(2s-1)/xi(2s)
    return complex(xi(2 * s - 1) / xi(2 * s))


def phi_euler_only(s: complex) -> complex:
    """Euler-only part (no archimedean completion): zeta(2s-1)/zeta(2s)."""

    ss = mp.mpc(float(np.real(s)), float(np.imag(s)))
    try:
        return complex(mp.zeta(2 * ss - 1) / mp.zeta(2 * ss))
    except ValueError:
        # mpmath raises at zeta(1) pole (hits at s=1/2, t=0). The ratio has a well-defined limit 0.
        if abs(2 * ss - 1) < mp.mpf("1e-30"):
            return 0.0 + 0.0j
        eps = mp.mpf("1e-10")
        return complex((mp.zeta(2 * (ss + eps) - 1) / mp.zeta(2 * (ss + eps)) + mp.zeta(2 * (ss - eps) - 1) / mp.zeta(2 * (ss - eps))) / 2)


def g_infty_classical(s: complex) -> complex:
    """Classical archimedean factor for PSL(2,Z) scattering.

    phi(s) = g_infty(s) * zeta(2s-1)/zeta(2s)
    with g_infty(s)=pi^{1/2 - s} * Gamma(s-1/2)/Gamma(s).
    """

    ss = mp.mpc(float(np.real(s)), float(np.imag(s)))
    return complex((mp.pi ** (mp.mpf("0.5") - ss)) * (mp.gamma(ss - mp.mpf("0.5")) / mp.gamma(ss)))


def _pi_pow_half_minus_s(s: complex) -> complex:
    ss = mp.mpc(float(np.real(s)), float(np.imag(s)))
    return complex(mp.pi ** (mp.mpf("0.5") - ss))


def _gamma_ratio_s_minus_half_over_s(s: complex) -> complex:
    ss = mp.mpc(float(np.real(s)), float(np.imag(s)))
    return complex(mp.gamma(ss - mp.mpf("0.5")) / mp.gamma(ss))


def _gamma_ratio_s_over_s_minus_half(s: complex) -> complex:
    ss = mp.mpc(float(np.real(s)), float(np.imag(s)))
    return complex(mp.gamma(ss) / mp.gamma(ss - mp.mpf("0.5")))


def lambda_norm_factor(s: complex, mode: str, *, s_ref: complex | None = None) -> complex:
        """Scalar normalization factor applied to the extracted channel eigenvalue.

        This is an explicit *candidate* Tier-2 normalization map (operator-level dictionary item)
        implemented at the level of the scalar observable:

            lambda_tilde(s) = factor(s) * lambda(s).

        Modes:
            - 'none': factor = 1
            - 'ginf_multiply': factor = g_infty(s)
            - 'ginf_divide': factor = 1/g_infty(s)
            - 'ginf_abs_multiply': factor = |g_infty(s)| (magnitude-only foil)
            - 'ginf_const_multiply': factor = g_infty(s_ref) (s-independent foil)
            - 'ginf_abs_const_multiply': factor = |g_infty(s_ref)| (s-independent magnitude foil)
            - 'ginf_const_divide': factor = 1/g_infty(s_ref)
        """

        mm = str(mode).strip().lower()
        if mm in {"none", "", "0"}:
                return 1.0 + 0.0j
        if mm == "ginf_multiply":
                return complex(g_infty_classical(s))
        if mm == "ginf_divide":
                g = complex(g_infty_classical(s))
                return (1.0 / g) if is_finite_complex(g) and abs(g) > 1e-300 else complex("nan")
        if mm in {"ginf_gamma_only_multiply", "ginf_gamma_multiply", "gamma_only"}:
            return complex(_gamma_ratio_s_minus_half_over_s(s))
        if mm in {"ginf_pi_only_multiply", "ginf_pi_multiply", "pi_only"}:
            return complex(_pi_pow_half_minus_s(s))
        if mm in {"ginf_abs_multiply", "ginf_abs"}:
            g = complex(g_infty_classical(s))
            return (abs(g) + 0.0j) if is_finite_complex(g) else complex("nan")
        if mm in {"ginf_const_multiply", "ginf_multiply_const"}:
            sr = complex(s_ref if s_ref is not None else (2.0 + 0.0j))
            return complex(g_infty_classical(sr))
        if mm in {"ginf_abs_const_multiply", "ginf_const_abs_multiply"}:
            sr = complex(s_ref if s_ref is not None else (2.0 + 0.0j))
            g = complex(g_infty_classical(sr))
            return (abs(g) + 0.0j) if is_finite_complex(g) else complex("nan")
        if mm in {"ginf_const_divide", "ginf_divide_const"}:
            sr = complex(s_ref if s_ref is not None else (2.0 + 0.0j))
            g = complex(g_infty_classical(sr))
            return (1.0 / g) if is_finite_complex(g) and abs(g) > 1e-300 else complex("nan")
        raise ValueError(f"unknown lambda_norm_mode={mode!r}")


def safe_div(a: complex, b: complex) -> complex:
    return a / b if b != 0 else complex("nan")


def is_finite_complex(z: complex) -> bool:
    return bool(np.isfinite(np.real(z)) and np.isfinite(np.imag(z)))


def best_rel_fit(z: complex, targets: dict[str, complex]) -> tuple[str, float]:
    """Return (label, rel_err) for best target fit using |z/target - 1|."""

    if not is_finite_complex(z):
        return "(nan)", float("nan")
    best_label = "(none)"
    best_err = float("inf")
    for label, tgt in targets.items():
        if not is_finite_complex(tgt) or abs(tgt) <= 1e-30:
            continue
        err = abs((z / tgt) - 1.0)
        if err < best_err:
            best_err = float(err)
            best_label = str(label)
    if best_err == float("inf"):
        return "(none)", float("nan")
    return best_label, float(best_err)


def best_rel_fit_over_eigs(eigs: np.ndarray, targets: dict[str, complex]) -> tuple[str, float]:
    """Return best target fit achieved by ANY eigenvalue in eigs."""

    best_label = "(none)"
    best_err = float("inf")
    if eigs is None:
        return best_label, float("nan")
    for lam in np.asarray(eigs).ravel():
        label, err = best_rel_fit(complex(lam), targets)
        if np.isfinite(err) and err < best_err:
            best_err = float(err)
            best_label = str(label)
    if best_err == float("inf"):
        return "(none)", float("nan")
    return best_label, float(best_err)


def _min_relerr_to_target(lams: np.ndarray, target: complex) -> tuple[int, float]:
    """Return (argmin_index, min_relerr) for |lam/target - 1|; falls back to |lam-target| if target ~ 0."""

    lams = np.asarray(lams, dtype=np.complex128).ravel()
    if lams.size == 0:
        return (0, float("nan"))
    if is_finite_complex(target) and abs(target) > 1e-30:
        rel = np.abs((lams / target) - 1.0)
    else:
        rel = np.abs(lams - target)
    j = int(np.argmin(rel))
    return j, float(rel[j])


def _nearest_neighbor_index(lams: np.ndarray, prev: complex) -> int:
    lams = np.asarray(lams, dtype=np.complex128).ravel()
    if lams.size == 0 or not is_finite_complex(prev):
        return 0
    d = np.abs(lams - prev)
    return int(np.argmin(d))


def _spectral_gap(lams: np.ndarray, j: int) -> float:
    lams = np.asarray(lams, dtype=np.complex128).ravel()
    if lams.size < 2:
        return float("nan")
    lam = lams[int(j)]
    d = np.abs(lams - lam)
    d[int(j)] = np.inf
    return float(np.min(d))


def _boundary_points(rect_: tuple[float, float, float, float], n_edge: int) -> list[complex]:
    """Return an ordered, closed list of points on the rectangle boundary."""

    sigma_min, sigma_max, t_min, t_max = rect_
    n = int(n_edge)
    if n < 40:
        n = 40
    q = max(10, n // 4)
    pts: list[complex] = []
    for sig in np.linspace(sigma_min, sigma_max, q, endpoint=False):
        pts.append(complex(float(sig), float(t_min)))
    for tt in np.linspace(t_min, t_max, q, endpoint=False):
        pts.append(complex(float(sigma_max), float(tt)))
    for sig in np.linspace(sigma_max, sigma_min, q, endpoint=False):
        pts.append(complex(float(sig), float(t_max)))
    for tt in np.linspace(t_max, t_min, q, endpoint=False):
        pts.append(complex(float(sigma_min), float(tt)))
    if pts:
        pts.append(pts[0])
    return pts


def _winding_count_from_samples(zs: list[complex]) -> tuple[float, int]:
    """Return (winding_number_estimate, failures) for a (typically) closed loop.

    Uses the robust incremental phase sum:
      winding ≈ (1/(2π)) * Σ_k Arg(z_{k+1}/z_k)
    where Arg is the principal value in (-π, π].

    This avoids the non-integer drift that can happen if you only compare endpoints
    of an unwrapped phase sequence.
    """

    failures = 0
    good_pairs = 0
    total = 0.0

    if len(zs) < 2:
        return float("nan"), len(zs)

    for z0, z1 in zip(zs[:-1], zs[1:]):
        if (not is_finite_complex(z0)) or (not is_finite_complex(z1)) or abs(z0) <= 1e-14 or abs(z1) <= 1e-14:
            failures += 1
            continue
        r = z1 / z0
        if (not is_finite_complex(r)) or abs(r) <= 1e-300:
            failures += 1
            continue
        dtheta = float(np.angle(r))
        total += dtheta
        good_pairs += 1

    if good_pairs < max(10, (len(zs) - 1) // 4):
        return float("nan"), failures

    w = total / (2.0 * np.pi)
    return float(w), failures


def _minmax_abs(zs: list[complex]) -> tuple[float, float, int]:
    """Return (min_abs, max_abs, failures) over finite, nonzero samples."""

    failures = 0
    mags: list[float] = []
    for z in zs:
        if not is_finite_complex(z) or abs(z) <= 1e-14:
            failures += 1
            continue
        mags.append(float(abs(z)))
    if not mags:
        return float("nan"), float("nan"), failures
    a = np.asarray(mags, dtype=float)
    return float(np.min(a)), float(np.max(a)), failures


def _minmax_abs_with_location(zs: list[complex], pts: list[complex]) -> tuple[float, float, complex, complex, int]:
    """Return (min_abs, max_abs, s_at_min, s_at_max, failures) over finite, nonzero samples."""

    failures = 0
    if len(zs) != len(pts):
        return float("nan"), float("nan"), complex("nan"), complex("nan"), len(zs)

    min_mag = float("inf")
    max_mag = 0.0
    s_min = complex("nan")
    s_max = complex("nan")
    for z, s in zip(zs, pts):
        if not is_finite_complex(z) or abs(z) <= 1e-14:
            failures += 1
            continue
        mag = float(abs(z))
        if mag < min_mag:
            min_mag = mag
            s_min = complex(s)
        if mag > max_mag:
            max_mag = mag
            s_max = complex(s)
    if not np.isfinite(min_mag) or max_mag == 0.0:
        return float("nan"), float("nan"), complex("nan"), complex("nan"), failures
    return float(min_mag), float(max_mag), s_min, s_max, failures


def _log10_abs_det(M: np.ndarray) -> float:
    """Return log10(|det(M)|) using slogdet; returns nan on failure, -inf on zero det."""

    try:
        sign, logabs = np.linalg.slogdet(np.asarray(M, dtype=np.complex128))
        if sign == 0 or (not np.isfinite(logabs)):
            return float("-inf")
        return float(logabs / np.log(10.0))
    except Exception:
        return float("nan")


def _svd_sentinels(M: np.ndarray) -> tuple[float, float, float, float, float]:
    """Return (smin, smax, cond, log10_smin, log10_smax) using singular values.

    Uses log10(max(s, 1e-300)) to avoid -inf underflows.
    """

    M = np.asarray(M, dtype=np.complex128)
    if M.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    try:
        svals = np.linalg.svd(M, compute_uv=False)
        if svals.size == 0:
            return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
        smax = float(np.max(svals))
        smin = float(np.min(svals))
        if smin <= 0 or not np.isfinite(smin) or not np.isfinite(smax):
            cond = float("inf") if np.isfinite(smax) else float("nan")
        else:
            cond = float(smax / smin)
        log10_smin = float(np.log10(max(smin, 1e-300))) if np.isfinite(smin) else float("nan")
        log10_smax = float(np.log10(max(smax, 1e-300))) if np.isfinite(smax) else float("nan")
        return smin, smax, cond, log10_smin, log10_smax
    except Exception:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")


def _split_boundary_interior_indices(N: int, b: int, boundary_idx: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    N = int(N)
    b = int(b)
    if boundary_idx is None:
        bidx = np.arange(0, b, dtype=int)
    else:
        bidx = np.asarray(boundary_idx, dtype=int).ravel()
    bset = set(map(int, bidx.tolist()))
    iidx = np.asarray([i for i in range(N) if i not in bset], dtype=int)
    return bidx, iidx


def local_phi_modular(p: int, s: complex) -> complex:
    # unramified local factor for phi(s): (1 - p^{-2s})/(1 - p^{-(2s-1)})
    p = int(p)
    return complex((1.0 - p ** (-2.0 * s)) / (1.0 - p ** (-(2.0 * s - 1.0))))


def phi_euler_partial(primes: tuple[int, ...], s: complex) -> complex:
    """Truncated Euler product target: Π_{p in primes} (1 - p^{-2s})/(1 - p^{-(2s-1)})."""

    prod = 1.0 + 0.0j
    for p in tuple(primes):
        pp = int(p)
        if pp <= 1:
            continue
        prod *= local_phi_modular(pp, s)
    return complex(prod)


def phi_target(s: complex, *, primes: tuple[int, ...], completion_mode: str, mode: str) -> complex:
    """Selectable target used to form Q_lambda(s) = lambda(s)/phi_target(s).

    Modes:
      - 'modular': xi(2s-1)/xi(2s)
      - 'euler_only': zeta(2s-1)/zeta(2s)
      - 'euler_partial': product over provided primes (Tier-1 default)
      - 'euler_partial_completed': g_infty(s) * product over primes
      - 'auto': uses 'euler_partial' when completion_mode=='none', else 'modular'
    """

    mm = str(mode).strip().lower()
    cm = str(completion_mode).strip().lower()
    if mm == "auto":
        mm = "euler_partial" if cm == "none" else "modular"

    if mm == "modular":
        return phi_modular(s)
    if mm == "euler_only":
        return phi_euler_only(s)
    if mm == "euler_partial":
        return phi_euler_partial(primes, s)
    if mm == "euler_partial_completed":
        return complex(g_infty_classical(s) * phi_euler_partial(primes, s))

    raise ValueError(f"unknown phi_target_mode={mode!r}")


@dataclass(frozen=True)
class ModelCfg:
    N: int
    b: int
    eta: float
    jitter: float
    cayley_eps: float
    primes: tuple[int, ...]
    completion_mode: str
    dual_scale: float
    generator_norm: str | None
    generator_norm_target: float
    weight_k: int
    eps_prime_mode: str
    prime_assembly_mode: str
    generator_packetize_mode: str
    generator_packetize_basis_mode: str


def main() -> int:
    ap = argparse.ArgumentParser(description="Experiment E: follow the doc's end-of-file validation track (H1–H6 + duality spine checks).")
    ap.add_argument("--config", type=str, default="configs/exp_E_doc_validation_track.yaml")
    ap.add_argument("--runs_root", type=str, default="runs")
    args = ap.parse_args()

    config = cfg.load_config(args.config)

    mp.mp.dps = int(config.get("mp_dps", 50))

    docx_path = Path(config.get("docx_path", "playing with RH.docx"))
    doc_meta = {
        "docx_path": str(docx_path),
        "exists": docx_path.exists(),
        "size": docx_path.stat().st_size if docx_path.exists() else None,
        "mtime": docx_path.stat().st_mtime if docx_path.exists() else None,
    }

    primes = tuple(map(int, config.get("primes", [2, 3, 5, 7, 11, 13, 17, 19, 23])))
    Ns = list(map(int, config.get("Ns", [48, 64])))
    boundary_frac = float(config.get("boundary_frac", 0.25))
    boundary_rounding = str(config.get("boundary_rounding", "round")).strip().lower()
    boundary_b0_override = config.get("boundary_b0_override", None)
    boundary_b0_delta = int(config.get("boundary_b0_delta", 0))
    if boundary_rounding not in {"round", "floor", "ceil"}:
        raise ValueError("boundary_rounding must be one of: round, floor, ceil")
    if boundary_b0_override is not None:
        boundary_b0_override = int(boundary_b0_override)
    eta = float(config.get("eta", 1.0))
    jitter = float(config.get("schur_jitter", 1e-10))
    cayley_eps = float(config.get("cayley_eps", 0.0))

    bulk_mode = str(config.get("bulk_mode", "one_channel")).strip()
    completion_mode = str(config.get("completion_mode", "dual_1_minus_s")).strip()
    dual_scale = float(config.get("dual_scale", 1.0))

    phi_target_mode = str(config.get("phi_target_mode", "modular")).strip().lower()
    if phi_target_mode not in {"modular", "euler_only", "euler_partial", "euler_partial_completed", "auto"}:
        raise ValueError("phi_target_mode must be one of: modular, euler_only, euler_partial, euler_partial_completed, auto")

    phi_select_mode = str(config.get("phi_select_mode", phi_target_mode)).strip().lower()
    if phi_select_mode not in {"modular", "euler_only", "euler_partial", "euler_partial_completed", "auto"}:
        raise ValueError("phi_select_mode must be one of: modular, euler_only, euler_partial, euler_partial_completed, auto")

    lambda_norm_mode = str(config.get("lambda_norm_mode", "none")).strip().lower()
    if lambda_norm_mode not in {
        "none",
        "ginf_multiply",
        "ginf_divide",
        "ginf_gamma_only_multiply",
        "ginf_gamma_multiply",
        "gamma_only",
        "ginf_pi_only_multiply",
        "ginf_pi_multiply",
        "pi_only",
        "ginf_abs_multiply",
        "ginf_abs",
        "ginf_const_multiply",
        "ginf_multiply_const",
        "ginf_abs_const_multiply",
        "ginf_const_abs_multiply",
        "ginf_const_divide",
        "ginf_divide_const",
    }:
        raise ValueError(
            "lambda_norm_mode must be one of: none, ginf_multiply, ginf_divide, gamma_only, pi_only, ginf_abs_multiply, ginf_const_multiply, ginf_abs_const_multiply, ginf_const_divide"
        )

    lambda_norm_sref_cfg = config.get("lambda_norm_sref", None)
    lambda_norm_sref: complex | None = None
    if lambda_norm_sref_cfg is not None:
        if isinstance(lambda_norm_sref_cfg, (list, tuple)) and len(lambda_norm_sref_cfg) >= 2:
            lambda_norm_sref = complex(float(lambda_norm_sref_cfg[0]), float(lambda_norm_sref_cfg[1]))
        else:
            raise ValueError("lambda_norm_sref must be [sigma, t] if provided")

    generator_norm = config.get("generator_norm", "fro")
    if generator_norm is not None:
        generator_norm = str(generator_norm)
    generator_norm_target = float(config.get("generator_norm_target", 1.0))

    eps_prime_mode = str(config.get("eps_prime_mode", "2s-1")).strip()
    prime_assembly_mode = str(config.get("prime_assembly_mode", "additive")).strip()
    generator_packetize_mode = str(config.get("generator_packetize_mode", "none")).strip()
    generator_packetize_basis_mode = str(config.get("generator_packetize_basis_mode", "sum_norm")).strip()

    weight_k = int(config.get("weight_k", 0))

    sigmas = list(map(float, config.get("sigmas", [0.3, 0.5, 0.7, 1.2])))
    t_grid = list(map(float, config.get("t_grid", [0.0, 0.5, 1.0, 2.0, 5.0, 10.0])))

    s0 = complex(config.get("s0", 2.0 + 0j))

    rect = config.get("rect", [0.55, 1.60, 0.0, 10.0])
    rect = (float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]))
    ap_h = float(config.get("arg_h", 1e-5))
    ap_n_edge = int(config.get("arg_n_edge", 800))
    do_arg_c_emp = bool(config.get("do_arg_c_emp", True))
    do_arg_lambda_branch = bool(config.get("do_arg_lambda_branch", False))
    ap_n_edge_lambda = int(config.get("arg_n_edge_lambda", 200))
    lambda_branch_basis = str(config.get("lambda_branch_basis", "incoming")).strip().lower()

    do_arg_lambda_subrects = bool(config.get("do_arg_lambda_subrects", False))
    ap_n_edge_lambda_subrect = int(config.get("arg_n_edge_lambda_subrect", max(200, ap_n_edge_lambda // 3)))
    subrect_sigma_edges_cfg = config.get("arg_lambda_subrect_sigma_edges", None)
    subrect_t_edges_cfg = config.get("arg_lambda_subrect_t_edges", None)

    do_arg_lambda_spike_dump = bool(config.get("do_arg_lambda_spike_dump", False))
    arg_lambda_spike_topk = int(config.get("arg_lambda_spike_topk", 20))
    do_arg_lambda_hotspots = bool(config.get("do_arg_lambda_hotspots", False))
    lambda_hotspots_cfg = config.get("arg_lambda_hotspots", None)
    lambda_extra_rects_cfg = config.get("arg_lambda_extra_rects", None)

    # Pole-line scan: classify Cayley singularities along a fixed-σ vertical line.
    do_pole_line_scan = bool(config.get("do_pole_line_scan", False))
    pole_line_sigma = float(config.get("pole_line_sigma", 1.075))
    pole_line_t_min = float(config.get("pole_line_t_min", rect[2]))
    pole_line_t_max = float(config.get("pole_line_t_max", rect[3]))
    pole_line_nt = int(config.get("pole_line_nt", 501))
    pole_line_basis = str(config.get("pole_line_basis", "incoming")).strip().lower()
    pole_line_block = str(config.get("pole_line_block", "plus")).strip().lower()
    pole_line_include_free = bool(config.get("pole_line_include_free", True))
    pole_line_include_phi = bool(config.get("pole_line_include_phi", True))

    # Dip atlas: coarse interior line scans to locate remaining Cayley bottlenecks.
    do_dip_atlas = bool(config.get("do_dip_atlas", False))
    dip_rect_cfg = config.get("dip_rect", None)
    dip_rect = rect
    if dip_rect_cfg is not None:
        dip_rect = (
            float(dip_rect_cfg[0]),
            float(dip_rect_cfg[1]),
            float(dip_rect_cfg[2]),
            float(dip_rect_cfg[3]),
        )
    dip_sigma_lines = list(map(float, config.get("dip_sigma_lines", [0.55, 0.8, 1.1, 1.4])))
    dip_t_lines = list(map(float, config.get("dip_t_lines", [0.5, 2.0, 4.0, 6.5])))
    dip_n_line = int(config.get("dip_n_line", 801))
    dip_basis = str(config.get("dip_basis", "incoming")).strip().lower()
    dip_block = str(config.get("dip_block", "plus")).strip().lower()
    dip_smin_thresh = float(config.get("dip_smin_thresh", 1e-2))
    dip_margin_t = float(config.get("dip_margin_t", 0.15))
    dip_margin_sigma = float(config.get("dip_margin_sigma", 0.03))
    dip_suggest_and_wind = bool(config.get("dip_suggest_and_wind", True))
    dip_suggest_wind_n_edge = int(config.get("dip_suggest_wind_n_edge", 3000))
    dip_suggest_interior_n_sigma = int(config.get("dip_suggest_interior_n_sigma", 31))
    dip_suggest_interior_n_t = int(config.get("dip_suggest_interior_n_t", 31))

    # Convergence sweep: compute interior value agreement on a fixed safe rectangle for many N,
    # optionally normalizing by a best-fit constant c_N (least-squares constant = complex mean).
    do_convergence_sweep_only = bool(config.get("do_convergence_sweep_only", False))
    convergence_rect_cfg = config.get("convergence_rect", None)
    if convergence_rect_cfg is None:
        convergence_rect_cfg = config.get("value_agreement_rect", None)
    convergence_rect: tuple[float, float, float, float] | None = None
    if convergence_rect_cfg is not None:
        convergence_rect = (
            float(convergence_rect_cfg[0]),
            float(convergence_rect_cfg[1]),
            float(convergence_rect_cfg[2]),
            float(convergence_rect_cfg[3]),
        )
    convergence_n_sigma = int(config.get("convergence_n_sigma", config.get("value_agreement_n_sigma", dip_suggest_interior_n_sigma)))
    convergence_n_t = int(config.get("convergence_n_t", config.get("value_agreement_n_t", dip_suggest_interior_n_t)))

    # Critical-line fingerprint sweep: dense sigma=1/2 scan to fingerprint subfactor normalizations.
    do_critical_line_fingerprint = bool(config.get("do_critical_line_fingerprint", False))
    critical_line_sigma = float(config.get("critical_line_sigma", 0.5))
    critical_line_t_min = float(config.get("critical_line_t_min", rect[2]))
    critical_line_t_max = float(config.get("critical_line_t_max", rect[3]))
    critical_line_nt = int(config.get("critical_line_nt", 401))
    critical_line_basis = str(config.get("critical_line_basis", "incoming")).strip().lower()
    critical_line_block = str(config.get("critical_line_block", "plus")).strip().lower()
    critical_line_points_out_csv = str(config.get("critical_line_points_out_csv", "E_Qlambda_critical_line_fingerprint_points.csv"))
    critical_line_summary_out_csv = str(config.get("critical_line_summary_out_csv", "E_Qlambda_critical_line_fingerprint_summary.csv"))
    critical_line_eval_candidates = bool(config.get("critical_line_eval_candidates", False))
    critical_line_candidates_out_csv = str(config.get("critical_line_candidates_out_csv", "E_Qlambda_critical_line_candidates_summary.csv"))
    critical_line_candidates_cfg = config.get(
        "critical_line_candidates",
        ["none", "gamma_only", "pi_only", "ginf_multiply", "ginf_abs_multiply"],
    )
    critical_line_candidates: list[str] = [str(x).strip().lower() for x in (critical_line_candidates_cfg if isinstance(critical_line_candidates_cfg, list) else [])]
    if not critical_line_candidates:
        critical_line_candidates = ["none", "gamma_only", "pi_only", "ginf_multiply", "ginf_abs_multiply"]

    boundary_autopick = bool(config.get("boundary_autopick", False))
    boundary_autopick_window = int(config.get("boundary_autopick_window", 2))
    boundary_autopick_objective = str(config.get("boundary_autopick_objective", "p95")).strip().lower()
    boundary_autopick_tiebreak = str(config.get("boundary_autopick_tiebreak", "median")).strip().lower()
    boundary_autopick_channel_mode = str(config.get("boundary_autopick_channel_mode", "single")).strip().lower()
    boundary_autopick_rank_mode = str(config.get("boundary_autopick_rank_mode", "legacy")).strip().lower()
    boundary_autopick_top5_guard = bool(config.get("boundary_autopick_top5_guard", False))
    boundary_autopick_top5_guard_rel = float(config.get("boundary_autopick_top5_guard_rel", 0.0))
    boundary_autopick_top5_guard_abs = float(config.get("boundary_autopick_top5_guard_abs", 0.0))
    boundary_autopick_gap_gamma = float(config.get("boundary_autopick_gap_gamma", 0.0))
    boundary_autopick_basis = str(config.get("boundary_autopick_basis", "incoming")).strip().lower()
    boundary_autopick_block = str(config.get("boundary_autopick_block", "plus")).strip().lower()
    boundary_autopick_eval_n_sigma = int(config.get("boundary_autopick_eval_n_sigma", convergence_n_sigma))
    boundary_autopick_eval_n_t = int(config.get("boundary_autopick_eval_n_t", convergence_n_t))
    boundary_autopick_wind_n_edge_stage1 = int(config.get("boundary_autopick_wind_n_edge_stage1", 1200))
    boundary_autopick_wind_n_edge_stage2 = int(config.get("boundary_autopick_wind_n_edge_stage2", 6000))
    boundary_autopick_wind_confirm_topk = int(config.get("boundary_autopick_wind_confirm_topk", 2))

    # Regularization sweep (highest-signal next step): sweep jitter and Cayley epsilon, optionally eta.
    do_regularization_sweep_only = bool(config.get("do_regularization_sweep_only", False))
    reg_rect_cfg = config.get("reg_sweep_rect", None)
    reg_rect: tuple[float, float, float, float] | None = None
    if reg_rect_cfg is not None:
        reg_rect = (float(reg_rect_cfg[0]), float(reg_rect_cfg[1]), float(reg_rect_cfg[2]), float(reg_rect_cfg[3]))

    reg_jitters = config.get("reg_sweep_jitters", None)
    if reg_jitters is None:
        reg_jitters = [jitter]
    reg_jitters = [float(x) for x in reg_jitters]

    reg_cayley_eps = config.get("reg_sweep_cayley_eps", None)
    if reg_cayley_eps is None:
        reg_cayley_eps = [cayley_eps]
    reg_cayley_eps = [float(x) for x in reg_cayley_eps]

    reg_etas = config.get("reg_sweep_etas", None)
    if reg_etas is None:
        reg_etas = [eta]
    reg_etas = [float(x) for x in reg_etas]

    reg_b0_deltas = config.get("reg_sweep_b0_deltas", None)
    if reg_b0_deltas is None:
        reg_b0_deltas = [int(boundary_b0_delta)]
    reg_b0_deltas = [int(x) for x in reg_b0_deltas]

    reg_n_sigma = int(config.get("reg_sweep_n_sigma", 31))
    reg_n_t = int(config.get("reg_sweep_n_t", 31))
    reg_do_winding = bool(config.get("reg_sweep_do_winding", True))
    reg_wind_n_edge = int(config.get("reg_sweep_wind_n_edge", 1200))
    reg_wind_N = int(config.get("reg_sweep_wind_N", 96))
    reg_wind_basis = str(config.get("reg_sweep_wind_basis", "incoming")).strip().lower()
    reg_wind_block = str(config.get("reg_sweep_wind_block", "plus")).strip().lower()

    print_progress = bool(config.get("print_progress", True))
    progress_every = int(config.get("progress_every", 0))

    out_dir = make_run_dir(args.runs_root, tag="expE_doc_validation_track")
    save_run_snapshot(out_dir, config=config, workspace_root=str(REPO_ROOT))
    write_json(os.path.join(out_dir, "doc_meta.json"), doc_meta)

    def _swap_block_matrix(S: np.ndarray, block: str, b0: int) -> np.ndarray:
        """Return the Swap± reduced b0xb0 matrix from a 2b0x2b0 scattering matrix.

        If S is already b0xb0 (one-channel mode), there is no Swap± structure; we
        return S for either block so downstream diagnostics still run.
        """

        S = np.asarray(S, dtype=np.complex128)
        if S.shape == (b0, b0):
            return S.astype(np.complex128)
        if S.shape[0] < 2 * b0 or S.shape[1] < 2 * b0:
            raise ValueError(f"expected (2*b0)x(2*b0) matrix, got {S.shape} for b0={b0}")

        S11 = S[:b0, :b0]
        S12 = S[:b0, b0 : 2 * b0]
        S21 = S[b0 : 2 * b0, :b0]
        S22 = S[b0 : 2 * b0, b0 : 2 * b0]
        if block in {"+", "plus", "p"}:
            return (0.5 * (S11 + S12 + S21 + S22)).astype(np.complex128)
        if block in {"-", "minus", "m"}:
            return (0.5 * (S11 - S12 - S21 + S22)).astype(np.complex128)
        raise ValueError("block must be 'plus' or 'minus'")

    def _riesz_projector(
        A: np.ndarray,
        *,
        center: complex,
        radius: float,
        n_theta: int = 32,
    ) -> np.ndarray:
        """Numerical Riesz projector (1/2πi)∮(zI-A)^{-1}dz on a circular contour.

        Parameterization: z(θ)=center+radius*e^{iθ}, θ in [0,2π).
        Then dz = i*radius*e^{iθ} dθ and
          P = (1/2π)∫ (zI-A)^{-1} * radius*e^{iθ} dθ.
        """

        A = np.asarray(A, dtype=np.complex128)
        n = int(A.shape[0])
        if A.shape != (n, n):
            raise ValueError("A must be square")
        radius = float(radius)
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("radius must be finite positive")
        n_theta = int(max(8, n_theta))

        I = np.eye(n, dtype=np.complex128)
        thetas = np.linspace(0.0, 2.0 * float(np.pi), n_theta, endpoint=False).astype(float)
        dtheta = (2.0 * float(np.pi)) / float(n_theta)

        P = np.zeros((n, n), dtype=np.complex128)
        for th in thetas.tolist():
            eith = complex(np.cos(th), np.sin(th))
            z = complex(center + radius * eith)
            # weight = (dθ/2π) * radius*e^{iθ}
            w = (dtheta / (2.0 * float(np.pi))) * (radius * eith)
            try:
                R = np.linalg.solve((z * I - A), I)
            except Exception:
                # If the resolvent solve fails at a quadrature node, bail to NaNs.
                return np.full((n, n), complex("nan"), dtype=np.complex128)
            P += (w * R).astype(np.complex128)
        return P.astype(np.complex128)

    def _proj_diagnostics(A: np.ndarray, P: np.ndarray) -> dict[str, float]:
        A = np.asarray(A, dtype=np.complex128)
        P = np.asarray(P, dtype=np.complex128)
        out: dict[str, float] = {}
        try:
            tr = complex(np.trace(P))
            out["proj_trace_re"] = float(np.real(tr)) if is_finite_complex(tr) else float("nan")
            out["proj_trace_im"] = float(np.imag(tr)) if is_finite_complex(tr) else float("nan")
        except Exception:
            out["proj_trace_re"] = float("nan")
            out["proj_trace_im"] = float("nan")

        # Relative idempotency defect ||P^2-P||_F / ||P||_F.
        try:
            denom = float(np.linalg.norm(P, ord="fro"))
            num = float(np.linalg.norm((P @ P - P), ord="fro"))
            out["proj_idem_rel"] = float(num / (denom + 1e-300))
        except Exception:
            out["proj_idem_rel"] = float("nan")

        # Relative commutator defect ||AP-PA||_F / (||A||_F||P||_F).
        try:
            denomA = float(np.linalg.norm(A, ord="fro"))
            denomP = float(np.linalg.norm(P, ord="fro"))
            num = float(np.linalg.norm((A @ P - P @ A), ord="fro"))
            out["proj_comm_rel"] = float(num / ((denomA + 1e-300) * (denomP + 1e-300)))
        except Exception:
            out["proj_comm_rel"] = float("nan")

        return out

    def _cusp_channel_riesz(
        B: np.ndarray,
        *,
        u0: np.ndarray,
        prefer: complex | None,
        overlap_floor: float = 0.9,
        radius_frac: float = 0.45,
        n_theta: int = 32,
    ) -> dict[str, object]:
        """Extract the cusp eigenchannel via overlap selection + Riesz projector.

        - Select candidate eigenvalue by overlap with the constant-mode line span{u0}.
        - If prefer is finite, break ties by nearest-neighbor to prefer.
        - Build a Riesz projector using a gap-based contour radius.
        """

        B = np.asarray(B, dtype=np.complex128)
        n = int(B.shape[0])
        if B.shape != (n, n):
            raise ValueError("B must be square")
        u0 = np.asarray(u0, dtype=np.complex128).ravel()
        if u0.size != n:
            raise ValueError("u0 shape mismatch")
        nu = float(np.linalg.norm(u0))
        if not np.isfinite(nu) or nu <= 0:
            raise ValueError("u0 must be nonzero")
        u0 = (u0 / nu).astype(np.complex128)

        w, V = np.linalg.eig(B)
        w = np.asarray(w, dtype=np.complex128).ravel()
        V = np.asarray(V, dtype=np.complex128)
        if w.size == 0 or V.size == 0:
            raise ValueError("no eig data")

        # Overlap score: |<u0,v>| / ||v||.
        vn = np.sqrt(np.sum(np.abs(V) ** 2, axis=0)).astype(float)
        vn = np.maximum(vn, 1e-300)
        overlaps = (np.abs(u0.conj().T @ V) / vn).astype(float)
        maxov = float(np.max(overlaps)) if overlaps.size else float("nan")
        if not np.isfinite(maxov):
            raise ValueError("bad overlaps")

        # Candidate set: start with overlap-thresholded eigenvalues, but always include a few top overlaps.
        order = np.argsort(-overlaps.astype(float))
        top_k = int(min(max(3, n // 4), n))
        cand_top = order[:top_k]

        cand_thr = np.where(overlaps >= float(overlap_floor) * maxov)[0]
        if cand_thr.size == 0:
            cand = np.asarray(cand_top, dtype=int)
        else:
            cand = np.unique(np.concatenate([cand_thr.astype(int), cand_top.astype(int)])).astype(int)

        if prefer is not None and is_finite_complex(prefer):
            jpref = int(np.argmin(np.abs(w - complex(prefer)).astype(float)))
            cand = np.unique(np.concatenate([cand, np.asarray([jpref], dtype=int)])).astype(int)

        # Evaluate candidates by Riesz-projector overlap p0_overlap = ||P u0||^2.
        best_j: int | None = None
        best_p0 = -float("inf")
        best_gap1 = float("nan")
        best_gap2 = float("nan")
        best_r = float("nan")
        best_P: np.ndarray | None = None
        best_diag: dict[str, float] | None = None
        best_lam_r = complex("nan")

        cand_records: list[tuple[int, float, float]] = []  # (j, p0_overlap, |w-prefer|)
        for j0 in cand.tolist():
            j0 = int(j0)
            lam0_j = complex(w[j0])
            diffs_all = np.abs(w - lam0_j).astype(float)
            diffs_all[j0] = float("inf")
            diffs_sorted = np.sort(diffs_all[np.isfinite(diffs_all)])
            gap1_j = float(diffs_sorted[0]) if diffs_sorted.size >= 1 else float("nan")
            gap2_j = float(diffs_sorted[1]) if diffs_sorted.size >= 2 else float("nan")
            r_j = float(radius_frac) * float(gap1_j) if np.isfinite(gap1_j) and gap1_j > 0 else float("nan")
            if not np.isfinite(r_j) or r_j <= 0:
                continue
            Pj = _riesz_projector(B, center=lam0_j, radius=r_j, n_theta=int(n_theta))
            if not np.isfinite(np.real(Pj)).all() or not np.isfinite(np.imag(Pj)).all():
                continue
            try:
                # Use an orthonormal basis for range(Pj) to get a bounded overlap in [0,1].
                # (Pj itself is typically not Hermitian for non-normal B.)
                U, S, _ = np.linalg.svd(Pj)
                S = np.asarray(S, dtype=float).ravel()
                rnk = int(np.sum(S > 1e-6))
                if rnk <= 0:
                    continue
                Ur = np.asarray(U[:, :rnk], dtype=np.complex128)
                coeff = (Ur.conj().T @ u0.reshape(-1, 1)).ravel()
                p0_ov_j = float(np.sum(np.abs(coeff) ** 2))
            except Exception:
                continue
            if not np.isfinite(p0_ov_j):
                continue

            d_pref = float(abs(lam0_j - complex(prefer))) if (prefer is not None and is_finite_complex(prefer)) else float("inf")
            cand_records.append((j0, p0_ov_j, d_pref))

            if p0_ov_j > best_p0:
                best_p0 = float(p0_ov_j)
                best_j = int(j0)
                best_gap1 = float(gap1_j)
                best_gap2 = float(gap2_j)
                best_r = float(r_j)
                best_P = Pj
                best_diag = _proj_diagnostics(B, Pj)
                try:
                    trP = complex(np.trace(Pj))
                    trPB = complex(np.trace(Pj @ B))
                    best_lam_r = complex(trPB / trP) if is_finite_complex(trPB) and is_finite_complex(trP) and abs(trP) > 1e-12 else complex("nan")
                except Exception:
                    best_lam_r = complex("nan")

        if best_j is None or best_P is None or best_diag is None:
            # Hard fallback: select by overlap (previous behavior) and return NaN projector diagnostics.
            j = int(order[0])
            lam0 = complex(w[j])
            return {
                "lam_raw": lam0,
                "lam_riesz": complex("nan"),
                "gap": float("nan"),
                "sep2": float("nan"),
                "overlap_max": maxov,
                "overlap_sel": float(overlaps[j]) if np.isfinite(overlaps[j]) else float("nan"),
                "radius": float("nan"),
                "proj": np.full((n, n), complex("nan"), dtype=np.complex128),
                "proj_diag": {"proj_trace_re": float("nan"), "proj_trace_im": float("nan"), "proj_idem_rel": float("nan"), "proj_comm_rel": float("nan")},
                "p0_overlap": float("nan"),
            }

        # If prefer is provided, and multiple candidates have near-max overlap, pick the one closest to prefer.
        if prefer is not None and is_finite_complex(prefer) and cand_records:
            near = [(j0, p0, dp) for (j0, p0, dp) in cand_records if p0 >= 0.98 * best_p0]
            if near:
                j2, _, _ = min(near, key=lambda x: x[2])
                if int(j2) != int(best_j):
                    # Recompute projector bundle for the preferred near-max candidate.
                    lam0_j = complex(w[int(j2)])
                    diffs_all = np.abs(w - lam0_j).astype(float)
                    diffs_all[int(j2)] = float("inf")
                    diffs_sorted = np.sort(diffs_all[np.isfinite(diffs_all)])
                    best_gap1 = float(diffs_sorted[0]) if diffs_sorted.size >= 1 else float("nan")
                    best_gap2 = float(diffs_sorted[1]) if diffs_sorted.size >= 2 else float("nan")
                    best_r = float(radius_frac) * float(best_gap1) if np.isfinite(best_gap1) and best_gap1 > 0 else float("nan")
                    best_P = _riesz_projector(B, center=lam0_j, radius=best_r, n_theta=int(n_theta))
                    best_diag = _proj_diagnostics(B, best_P)
                    try:
                        trP = complex(np.trace(best_P))
                        trPB = complex(np.trace(best_P @ B))
                        best_lam_r = complex(trPB / trP) if is_finite_complex(trPB) and is_finite_complex(trP) and abs(trP) > 1e-12 else complex("nan")
                    except Exception:
                        best_lam_r = complex("nan")
                    try:
                        U, S, _ = np.linalg.svd(best_P)
                        S = np.asarray(S, dtype=float).ravel()
                        rnk = int(np.sum(S > 1e-6))
                        if rnk <= 0:
                            best_p0 = float("nan")
                        else:
                            Ur = np.asarray(U[:, :rnk], dtype=np.complex128)
                            coeff = (Ur.conj().T @ u0.reshape(-1, 1)).ravel()
                            best_p0 = float(np.sum(np.abs(coeff) ** 2))
                    except Exception:
                        best_p0 = float("nan")
                    best_j = int(j2)

        lam0 = complex(w[int(best_j)])
        return {
            "lam_raw": lam0,
            "lam_riesz": complex(best_lam_r),
            "gap": float(best_gap1),
            "sep2": float(best_gap2),
            "overlap_max": maxov,
            "overlap_sel": float(overlaps[int(best_j)]) if np.isfinite(overlaps[int(best_j)]) else float("nan"),
            "radius": float(best_r),
            "proj": best_P,
            "proj_diag": best_diag,
            "p0_overlap": float(best_p0),
        }

    def _b_from_frac(N: int, *, b0_override: int | None = None, b0_delta: int = 0) -> int:
        N0 = int(N)
        if b0_override is None:
            b0_override = boundary_b0_override
        if b0_override is not None:
            b = int(b0_override)
        else:
            x = float(boundary_frac) * float(N0)
            if boundary_rounding == "floor":
                b = int(np.floor(x))
            elif boundary_rounding == "ceil":
                b = int(np.ceil(x))
            else:
                b = int(round(x))
        b = int(b) + int(boundary_b0_delta) + int(b0_delta)
        if b <= 0:
            b = 1
        if b >= N0 - 1:
            b = max(1, N0 // 4)
        return b

    def _mk_cfg(
        N: int,
        *,
        eta_override: float | None = None,
        jitter_override: float | None = None,
        cayley_eps_override: float | None = None,
        b0_override: int | None = None,
        b0_delta_override: int = 0,
    ) -> ModelCfg:
        mode = str(bulk_mode).lower().strip()
        if mode not in {"one_channel", "two_channel_symmetric"}:
            raise ValueError("bulk_mode must be 'one_channel' or 'two_channel_symmetric'")

        b0 = _b_from_frac(int(N), b0_override=b0_override, b0_delta=int(b0_delta_override))
        # For 2-channel bulk, the natural boundary is doubled.
        if mode == "two_channel_symmetric":
            b_eff = 2 * int(b0)
            N_eff = 2 * int(N)
        else:
            b_eff = int(b0)
            N_eff = int(N)

        return ModelCfg(
            N=int(N_eff),
            b=int(b_eff),
            eta=float(eta if eta_override is None else eta_override),
            jitter=float(jitter if jitter_override is None else jitter_override),
            cayley_eps=float(cayley_eps if cayley_eps_override is None else cayley_eps_override),
            primes=primes,
            completion_mode=str(completion_mode),
            dual_scale=float(dual_scale),
            generator_norm=generator_norm,
            generator_norm_target=float(generator_norm_target),
            weight_k=int(weight_k),
            eps_prime_mode=str(eps_prime_mode),
            prime_assembly_mode=str(prime_assembly_mode),
            generator_packetize_mode=str(generator_packetize_mode),
            generator_packetize_basis_mode=str(generator_packetize_basis_mode),
        )

    def _boundary_indices(cfgm: ModelCfg) -> np.ndarray | None:
        mode = str(bulk_mode).lower().strip()
        if mode != "two_channel_symmetric":
            return None

        # build_A orders as [ch1 (0..N-1), ch2 (N..2N-1)].
        # We want the boundary to include both channels' first b0 indices.
        base_N = int(cfgm.N // 2)
        b0 = int(cfgm.b // 2)
        if b0 <= 0 or b0 >= base_N:
            raise ValueError("invalid two-channel boundary sizing")

        bidx = np.concatenate(
            [
                np.arange(0, b0, dtype=int),
                np.arange(base_N, base_N + b0, dtype=int),
            ]
        )
        return bidx

    @lru_cache(maxsize=10000)
    def _model_mats(s: complex, cfgm: ModelCfg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Fast path for convergence-only runs in 2-channel mode:
        # avoid assembling the full 2N×2N bulk matrix and avoid dn_map_on_indices' large interior copy.
        # This computes the exact same boundary DN map Λ_b via the (U,W)-structured Schur complement.
        if bool(do_convergence_sweep_only) and str(bulk_mode).lower().strip() == "two_channel_symmetric":
            from src.bulk import build_two_channel_UW  # local import
            from src.dn import dn_map_two_channel_boundary  # local import

            base_N = int(cfgm.N // 2)
            bulk_params = BulkParams(N=int(base_N), weight_k=cfgm.weight_k)
            U, W = build_two_channel_UW(
                s=s,
                primes=list(cfgm.primes),
                comps=None,
                params=bulk_params,
                generator_norm=cfgm.generator_norm,
                generator_norm_target=cfgm.generator_norm_target,
                completion_mode=cfgm.completion_mode,
                dual_scale=cfgm.dual_scale,
                eps_prime_mode=cfgm.eps_prime_mode,
                prime_assembly_mode=cfgm.prime_assembly_mode,
                generator_packetize_mode=cfgm.generator_packetize_mode,
                generator_packetize_basis_mode=cfgm.generator_packetize_basis_mode,
            )
            b0 = int(cfgm.b // 2)
            Lam = dn_map_two_channel_boundary(U, W, b0=b0, jitter=cfgm.jitter)
            S = cayley_from_lambda(Lam, eta=cfgm.eta, eps=cfgm.cayley_eps)
            return np.empty((0, 0), dtype=np.complex128), Lam, S

        # Note: BulkParams.N refers to the base one-channel size. For 2-channel mode we pass N/2.
        base_N = int(cfgm.N // 2) if str(bulk_mode).lower().strip() == "two_channel_symmetric" else int(cfgm.N)
        bulk_params = BulkParams(N=int(base_N), weight_k=cfgm.weight_k)
        A = build_A(
            s=s,
            primes=list(cfgm.primes),
            comps=None,
            params=bulk_params,
            generator_norm=cfgm.generator_norm,
            generator_norm_target=cfgm.generator_norm_target,
            bulk_mode=str(bulk_mode),
            completion_mode=cfgm.completion_mode,
            dual_scale=cfgm.dual_scale,
            eps_prime_mode=cfgm.eps_prime_mode,
            prime_assembly_mode=cfgm.prime_assembly_mode,
            generator_packetize_mode=cfgm.generator_packetize_mode,
            generator_packetize_basis_mode=cfgm.generator_packetize_basis_mode,
        )
        bidx = _boundary_indices(cfgm)
        if bidx is None:
            Lam = dn_map(A, b=cfgm.b, jitter=cfgm.jitter)
        else:
            Lam = dn_map_on_indices(A, boundary_idx=bidx, jitter=cfgm.jitter)
        S = cayley_from_lambda(Lam, eta=cfgm.eta, eps=cfgm.cayley_eps)
        return A, Lam, S

    @lru_cache(maxsize=10000)
    def _free_mats(s: complex, cfgm: ModelCfg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reference/free mats for primes=[] (used for matrix-level normalization)."""
        if bool(do_convergence_sweep_only) and str(bulk_mode).lower().strip() == "two_channel_symmetric":
            from src.bulk import build_two_channel_UW  # local import
            from src.dn import dn_map_two_channel_boundary  # local import

            base_N = int(cfgm.N // 2)
            bulk_params = BulkParams(N=int(base_N), weight_k=cfgm.weight_k)
            U, W = build_two_channel_UW(
                s=s,
                primes=[],
                comps=None,
                params=bulk_params,
                generator_norm=cfgm.generator_norm,
                generator_norm_target=cfgm.generator_norm_target,
                completion_mode=cfgm.completion_mode,
                dual_scale=cfgm.dual_scale,
                eps_prime_mode=cfgm.eps_prime_mode,
                prime_assembly_mode=cfgm.prime_assembly_mode,
                generator_packetize_mode=cfgm.generator_packetize_mode,
                generator_packetize_basis_mode=cfgm.generator_packetize_basis_mode,
            )
            b0 = int(cfgm.b // 2)
            Lam0 = dn_map_two_channel_boundary(U, W, b0=b0, jitter=cfgm.jitter)
            S0 = cayley_from_lambda(Lam0, eta=cfgm.eta, eps=cfgm.cayley_eps)
            return np.empty((0, 0), dtype=np.complex128), Lam0, S0

        base_N = int(cfgm.N // 2) if str(bulk_mode).lower().strip() == "two_channel_symmetric" else int(cfgm.N)
        bulk_params = BulkParams(N=int(base_N), weight_k=cfgm.weight_k)
        A0 = build_A(
            s=s,
            primes=[],
            comps=None,
            params=bulk_params,
            generator_norm=cfgm.generator_norm,
            generator_norm_target=cfgm.generator_norm_target,
            bulk_mode=str(bulk_mode),
            completion_mode=cfgm.completion_mode,
            dual_scale=cfgm.dual_scale,
            eps_prime_mode=cfgm.eps_prime_mode,
            prime_assembly_mode=cfgm.prime_assembly_mode,
            generator_packetize_mode=cfgm.generator_packetize_mode,
            generator_packetize_basis_mode=cfgm.generator_packetize_basis_mode,
        )
        bidx = _boundary_indices(cfgm)
        if bidx is None:
            Lam0 = dn_map(A0, b=cfgm.b, jitter=cfgm.jitter)
        else:
            Lam0 = dn_map_on_indices(A0, boundary_idx=bidx, jitter=cfgm.jitter)
        S0 = cayley_from_lambda(Lam0, eta=cfgm.eta, eps=cfgm.cayley_eps)
        return A0, Lam0, S0

    @lru_cache(maxsize=10000)
    def _model_detS_norm(s: complex, cfgm: ModelCfg) -> complex:
        _, _, S = _model_mats(s, cfgm)
        detS = complex(np.linalg.det(S))

        # free normalization: primes=[]
        base_N = int(cfgm.N // 2) if str(bulk_mode).lower().strip() == "two_channel_symmetric" else int(cfgm.N)
        bulk_params = BulkParams(N=int(base_N), weight_k=cfgm.weight_k)
        A0 = build_A(
            s=s,
            primes=[],
            comps=None,
            params=bulk_params,
            generator_norm=cfgm.generator_norm,
            generator_norm_target=cfgm.generator_norm_target,
            bulk_mode=str(bulk_mode),
            completion_mode=cfgm.completion_mode,
            dual_scale=cfgm.dual_scale,
            eps_prime_mode=cfgm.eps_prime_mode,
            prime_assembly_mode=cfgm.prime_assembly_mode,
            generator_packetize_mode=cfgm.generator_packetize_mode,
            generator_packetize_basis_mode=cfgm.generator_packetize_basis_mode,
        )
        bidx = _boundary_indices(cfgm)
        if bidx is None:
            Lam0 = dn_map(A0, b=cfgm.b, jitter=cfgm.jitter)
        else:
            Lam0 = dn_map_on_indices(A0, boundary_idx=bidx, jitter=cfgm.jitter)
        S0 = cayley_from_lambda(Lam0, eta=cfgm.eta, eps=cfgm.cayley_eps)
        detS0 = complex(np.linalg.det(S0))
        return detS / detS0 if detS0 != 0 else complex("nan")

    def _compute_q_lambda_value_agreement_for_rect(
        rect_in: tuple[float, float, float, float],
        n_sigma: int,
        n_t: int,
        out_csv_name: str,
        source_label: str,
        Ns_list: list[int] | None = None,
        cfg_factory=None,
        *,
        use_cache: bool = True,
        compute_magphase_diag: bool = False,
        magphase_out_csv_name: str = "E_Qlambda_convergence_magphase.csv",
        lambda_norm_mode: str = "none",
    ) -> None:
        """Compute interior-grid value agreement for Q_lambda on a rectangle.

        Reports median/max of |Q-1| for the phi convention (Q=lambda/phi), the inverse convention,
        and the best-of-two. Also computes a best-fit constant c_N = mean(Q_phi) minimizing
        Σ|Q_phi(s)-c|^2, and reports median/max of |Q_phi/c_N - 1|.
        """

        if n_sigma < 3 or n_t < 3:
            return
        if rect_in[3] <= rect_in[2] or rect_in[1] <= rect_in[0]:
            return

        sigs = np.linspace(rect_in[0], rect_in[1], int(n_sigma) + 2, endpoint=True, dtype=float)[1:-1]
        ts = np.linspace(rect_in[2], rect_in[3], int(n_t) + 2, endpoint=True, dtype=float)[1:-1]

        def _finite_stats(vals: list[float]) -> tuple[float, float]:
            if not vals:
                return (float("nan"), float("nan"))
            a = np.asarray(vals, dtype=float)
            a = a[np.isfinite(a)]
            if a.size == 0:
                return (float("nan"), float("nan"))
            return (float(np.median(a)), float(np.max(a)))

        def _finite_percentile(vals: list[float], q: float) -> float:
            if not vals:
                return float("nan")
            a = np.asarray(vals, dtype=float)
            a = a[np.isfinite(a)]
            if a.size == 0:
                return float("nan")
            return float(np.percentile(a, float(q)))

        def _finite_topk_mean(vals: list[float], k: int) -> float:
            if not vals:
                return float("nan")
            kk = int(k)
            if kk <= 0:
                return float("nan")
            a = np.asarray(vals, dtype=float)
            a = a[np.isfinite(a)]
            if a.size == 0:
                return float("nan")
            kk = min(kk, int(a.size))
            # mean of the kk largest values
            part = np.partition(a, int(a.size - kk))
            top = part[int(a.size - kk) :]
            return float(np.mean(top))

        def _finite_corr_and_slope(x: list[float], y: list[float]) -> tuple[float, float, int]:
            """Return (corr, slope) for y ~ a + slope*x over finite pairs."""

            if (not x) or (not y):
                return float("nan"), float("nan"), 0
            xx = np.asarray(x, dtype=float)
            yy = np.asarray(y, dtype=float)
            mask = np.isfinite(xx) & np.isfinite(yy)
            if int(np.sum(mask)) < 3:
                return float("nan"), float("nan"), int(np.sum(mask))
            xx2 = xx[mask]
            yy2 = yy[mask]
            try:
                corr = float(np.corrcoef(xx2, yy2)[0, 1])
            except Exception:
                corr = float("nan")
            try:
                slope, _intercept = np.polyfit(xx2, yy2, deg=1)
                slope = float(slope)
            except Exception:
                slope = float("nan")
            return corr, slope, int(xx2.size)

        def _complex_mean(vals: list[complex]) -> complex:
            if not vals:
                return complex("nan")
            a = np.asarray(vals, dtype=np.complex128)
            a = a[np.isfinite(np.real(a)) & np.isfinite(np.imag(a))]
            if a.size == 0:
                return complex("nan")
            return complex(np.mean(a))

        Ns_use = [int(x) for x in (Ns if Ns_list is None else Ns_list)]
        if cfg_factory is None:
            cfg_factory = _mk_cfg

        # Avoid caching huge matrices when called in a large parameter sweep.
        model_fn = _model_mats if bool(use_cache) else getattr(_model_mats, "__wrapped__", _model_mats)
        free_fn = _free_mats if bool(use_cache) else getattr(_free_mats, "__wrapped__", _free_mats)

        stats_rows: list[dict] = []
        magphase_rows: list[dict] = []
        for N in Ns_use:
            cfgN = cfg_factory(int(N))
            mode_tag = str(bulk_mode).lower().strip()
            b0 = int(cfgN.b // 2) if mode_tag == "two_channel_symmetric" else int(cfgN.b)
            acc: dict[tuple[str, str], dict[str, object]] = {}
            for basis in ["rel", "incoming"]:
                for block2 in ["plus", "minus"]:
                    acc[(basis, block2)] = {
                        "q_phi_samples": [],
                        "errs_phi": [],
                        "errs_inv": [],
                        "errs_best": [],
                        "gaps_phi": [],
                        "phi_wins": 0,
                        "total": 0,
                        "failures": 0,
                        "logabs_q": [],
                        "absabs_q_minus1": [],
                        "absarg_q": [],
                        "unit_phase_err": [],
                        "logabs_ginf": [],
                        "abs_q_times_ginf_minus1": [],
                        "abs_arg_q_times_ginf": [],
                        "unit_phase_err_q_times_ginf": [],
                    }

            for sig in sigs:
                for tt in ts:
                    s = complex(float(sig), float(tt))

                    try:
                        _, Lam, Sm = model_fn(s, cfgN)
                        _, _, S0 = free_fn(s, cfgN)
                        # Avoid explicit inverse: Sm @ inv(S0) = solve(S0.T, Sm.T).T
                        Srel = np.linalg.solve(np.asarray(S0, dtype=np.complex128).T, np.asarray(Sm, dtype=np.complex128).T).T
                    except Exception:
                        for k in acc:
                            acc[k]["failures"] = int(acc[k]["failures"]) + 1
                        continue

                    phi_norm = phi_target(s, primes=cfgN.primes, completion_mode=cfgN.completion_mode, mode=phi_target_mode)
                    phi_sel = phi_target(s, primes=cfgN.primes, completion_mode=cfgN.completion_mode, mode=phi_select_mode)
                    if (
                        (not is_finite_complex(phi_norm))
                        or abs(phi_norm) <= 1e-30
                        or (not is_finite_complex(phi_sel))
                        or abs(phi_sel) <= 1e-30
                    ):
                        for k in acc:
                            acc[k]["failures"] = int(acc[k]["failures"]) + 1
                        continue

                    # Precompute the incoming basis transform (if possible); rel is always available.
                    M_rel = np.asarray(Srel, dtype=np.complex128)
                    M_in: np.ndarray | None
                    try:
                        nb = int(Lam.shape[0])
                        Tin_eta = (Lam + 1j * float(cfgN.eta) * np.eye(nb, dtype=np.complex128)).astype(np.complex128)
                        if float(cfgN.cayley_eps) != 0.0:
                            Tin_eta = (Tin_eta + float(cfgN.cayley_eps) * np.eye(nb, dtype=np.complex128)).astype(np.complex128)
                        M_in = np.linalg.solve(Tin_eta, (M_rel @ Tin_eta)).astype(np.complex128)
                    except Exception:
                        M_in = None

                    for basis in ["rel", "incoming"]:
                        M = M_rel if basis == "rel" else M_in
                        if M is None:
                            for block2 in ["plus", "minus"]:
                                acc[(basis, block2)]["failures"] = int(acc[(basis, block2)]["failures"]) + 1
                            continue

                        for block2 in ["plus", "minus"]:
                            try:
                                B = _swap_block_matrix(M, block2, b0=b0)
                                eigs = np.asarray(np.linalg.eigvals(B), dtype=np.complex128).ravel()
                            except Exception:
                                acc[(basis, block2)]["failures"] = int(acc[(basis, block2)]["failures"]) + 1
                                continue
                            if eigs.size == 0:
                                acc[(basis, block2)]["failures"] = int(acc[(basis, block2)]["failures"]) + 1
                                continue

                            # Convention 1: lambda ~ phi
                            j_phi, _ = _min_relerr_to_target(eigs, phi_sel)
                            lam_phi_raw = complex(eigs[int(j_phi)])

                            lam_fac = lambda_norm_factor(s, lambda_norm_mode, s_ref=(lambda_norm_sref if lambda_norm_sref is not None else s0))
                            if (not is_finite_complex(lam_fac)) or abs(lam_fac) <= 1e-300:
                                acc[(basis, block2)]["failures"] = int(acc[(basis, block2)]["failures"]) + 1
                                continue
                            lam_phi = complex(lam_phi_raw * lam_fac)

                            q_phi = lam_phi / phi_norm
                            err_phi = float(abs(q_phi - 1.0)) if is_finite_complex(q_phi) else float("nan")

                            # Local spectral gap diagnostic for the chosen eigenvalue.
                            try:
                                if eigs.size >= 2:
                                    diffs = np.abs(eigs - lam_phi_raw)
                                    diffs[int(j_phi)] = np.inf
                                    gap = float(np.min(diffs))
                                else:
                                    gap = float("nan")
                            except Exception:
                                gap = float("nan")
                            if np.isfinite(gap):
                                cast_gaps = acc[(basis, block2)]["gaps_phi"]
                                assert isinstance(cast_gaps, list)
                                cast_gaps.append(gap)

                            # Convention 2: lambda ~ 1/phi
                            invphi = (1.0 / phi_sel) if phi_sel != 0 else complex("nan")
                            if is_finite_complex(invphi) and abs(invphi) > 0:
                                j_inv, _ = _min_relerr_to_target(eigs, invphi)
                                lam_inv = complex(eigs[int(j_inv)])

                                # Apply the same candidate normalization to the selected eigenvalue.
                                lam_inv = complex(lam_inv * lam_fac)

                                q_inv = lam_inv * phi_norm
                                err_inv = float(abs(q_inv - 1.0)) if is_finite_complex(q_inv) else float("nan")
                            else:
                                err_inv = float("nan")

                            if (not np.isfinite(err_phi)) and (not np.isfinite(err_inv)):
                                acc[(basis, block2)]["failures"] = int(acc[(basis, block2)]["failures"]) + 1
                                continue

                            acc[(basis, block2)]["total"] = int(acc[(basis, block2)]["total"]) + 1
                            if np.isfinite(err_phi):
                                cast_phi = acc[(basis, block2)]["errs_phi"]
                                assert isinstance(cast_phi, list)
                                cast_phi.append(float(err_phi))
                            if np.isfinite(err_inv):
                                cast_inv = acc[(basis, block2)]["errs_inv"]
                                assert isinstance(cast_inv, list)
                                cast_inv.append(float(err_inv))

                            best = err_phi
                            best_is_phi = True
                            if np.isfinite(err_inv) and (not np.isfinite(best) or err_inv < best):
                                best = err_inv
                                best_is_phi = False
                            if np.isfinite(best):
                                cast_best = acc[(basis, block2)]["errs_best"]
                                assert isinstance(cast_best, list)
                                cast_best.append(float(best))
                            if best_is_phi:
                                acc[(basis, block2)]["phi_wins"] = int(acc[(basis, block2)]["phi_wins"]) + 1

                            if is_finite_complex(q_phi):
                                cast_qs = acc[(basis, block2)]["q_phi_samples"]
                                assert isinstance(cast_qs, list)
                                cast_qs.append(complex(q_phi))

                                if bool(compute_magphase_diag):
                                    try:
                                        mag = float(abs(q_phi))
                                    except Exception:
                                        mag = float("nan")

                                    try:
                                        if np.isfinite(mag) and mag > 0:
                                            cast_logabs_q = acc[(basis, block2)]["logabs_q"]
                                            assert isinstance(cast_logabs_q, list)
                                            cast_logabs_q.append(float(np.log(mag)))
                                    except Exception:
                                        pass

                                    try:
                                        if np.isfinite(mag):
                                            cast_magerr = acc[(basis, block2)]["absabs_q_minus1"]
                                            assert isinstance(cast_magerr, list)
                                            cast_magerr.append(float(abs(mag - 1.0)))
                                    except Exception:
                                        pass

                                    try:
                                        theta = float(np.angle(q_phi))
                                        cast_arg = acc[(basis, block2)]["absarg_q"]
                                        assert isinstance(cast_arg, list)
                                        cast_arg.append(float(abs(theta)))
                                    except Exception:
                                        pass

                                    try:
                                        if np.isfinite(mag) and mag > 0:
                                            unit = q_phi / mag
                                            cast_pe = acc[(basis, block2)]["unit_phase_err"]
                                            assert isinstance(cast_pe, list)
                                            cast_pe.append(float(abs(unit - 1.0)))
                                    except Exception:
                                        pass

                                    try:
                                        ginf = g_infty_classical(s)
                                        gmag = float(abs(ginf)) if is_finite_complex(ginf) else float("nan")
                                        if np.isfinite(gmag) and gmag > 0:
                                            cast_lg = acc[(basis, block2)]["logabs_ginf"]
                                            assert isinstance(cast_lg, list)
                                            cast_lg.append(float(np.log(gmag)))

                                        # Direct test: does q_phi * g_infty(s) ≈ 1?
                                        if is_finite_complex(ginf) and is_finite_complex(q_phi):
                                            qg = complex(q_phi * ginf)
                                            try:
                                                cast_qg = acc[(basis, block2)]["abs_q_times_ginf_minus1"]
                                                assert isinstance(cast_qg, list)
                                                cast_qg.append(float(abs(qg - 1.0)))
                                            except Exception:
                                                pass
                                            try:
                                                cast_arg_qg = acc[(basis, block2)]["abs_arg_q_times_ginf"]
                                                assert isinstance(cast_arg_qg, list)
                                                cast_arg_qg.append(float(abs(np.angle(qg))))
                                            except Exception:
                                                pass
                                            try:
                                                qg_mag = float(abs(qg))
                                                if np.isfinite(qg_mag) and qg_mag > 0:
                                                    unit_qg = qg / qg_mag
                                                    cast_uqg = acc[(basis, block2)]["unit_phase_err_q_times_ginf"]
                                                    assert isinstance(cast_uqg, list)
                                                    cast_uqg.append(float(abs(unit_qg - 1.0)))
                                            except Exception:
                                                pass
                                    except Exception:
                                        pass

            for basis in ["rel", "incoming"]:
                for block2 in ["plus", "minus"]:
                    a = acc[(basis, block2)]
                    q_phi_samples = a["q_phi_samples"]
                    errs_phi = a["errs_phi"]
                    errs_inv = a["errs_inv"]
                    errs_best = a["errs_best"]
                    gaps_phi = a["gaps_phi"]
                    logabs_q = a["logabs_q"]
                    absabs_q_minus1 = a["absabs_q_minus1"]
                    absarg_q = a["absarg_q"]
                    unit_phase_err = a["unit_phase_err"]
                    logabs_ginf = a["logabs_ginf"]
                    abs_q_times_ginf_minus1 = a["abs_q_times_ginf_minus1"]
                    abs_arg_q_times_ginf = a["abs_arg_q_times_ginf"]
                    unit_phase_err_q_times_ginf = a["unit_phase_err_q_times_ginf"]
                    phi_wins = int(a["phi_wins"])
                    total = int(a["total"])
                    failures = int(a["failures"])

                    assert isinstance(q_phi_samples, list)
                    assert isinstance(errs_phi, list)
                    assert isinstance(errs_inv, list)
                    assert isinstance(errs_best, list)
                    assert isinstance(gaps_phi, list)
                    assert isinstance(logabs_q, list)
                    assert isinstance(absabs_q_minus1, list)
                    assert isinstance(absarg_q, list)
                    assert isinstance(unit_phase_err, list)
                    assert isinstance(logabs_ginf, list)
                    assert isinstance(abs_q_times_ginf_minus1, list)
                    assert isinstance(abs_arg_q_times_ginf, list)
                    assert isinstance(unit_phase_err_q_times_ginf, list)

                    phi_med, phi_max = _finite_stats(errs_phi)
                    inv_med, inv_max = _finite_stats(errs_inv)
                    best_med, best_max = _finite_stats(errs_best)
                    gap_med, _ = _finite_stats(gaps_phi)
                    gap_min = float(np.min(np.asarray(gaps_phi, dtype=float))) if gaps_phi else float("nan")

                    cN = _complex_mean(q_phi_samples)
                    if is_finite_complex(cN) and abs(cN) > 1e-30:
                        errs_phi_norm = [float(abs((q / cN) - 1.0)) for q in q_phi_samples if is_finite_complex(q)]
                        phi_norm_med, phi_norm_max = _finite_stats(errs_phi_norm)
                        phi_norm_p95 = _finite_percentile(errs_phi_norm, 95.0)
                        phi_norm_top5 = _finite_topk_mean(errs_phi_norm, 5)
                    else:
                        phi_norm_med, phi_norm_max = float("nan"), float("nan")
                        phi_norm_p95, phi_norm_top5 = float("nan"), float("nan")

                    stats_rows.append(
                        {
                            "source": str(source_label),
                            "N": int(N),
                            "boundary_b": int(cfgN.b),
                            "boundary_b0": int(b0),
                            "basis": str(basis),
                            "block": str(block2),
                            "sigma_min": float(rect_in[0]),
                            "sigma_max": float(rect_in[1]),
                            "t_min": float(rect_in[2]),
                            "t_max": float(rect_in[3]),
                            "n_sigma": int(n_sigma),
                            "n_t": int(n_t),
                            "points_total": int(total),
                            "failures": int(failures),
                            "err_phi_median": float(phi_med),
                            "err_phi_max": float(phi_max),
                            "err_invphi_median": float(inv_med),
                            "err_invphi_max": float(inv_max),
                            "err_best_median": float(best_med),
                            "err_best_max": float(best_max),
                            "phi_wins_frac": float(phi_wins / max(1, total)),
                            "cN_re": float(np.real(cN)) if is_finite_complex(cN) else float("nan"),
                            "cN_im": float(np.imag(cN)) if is_finite_complex(cN) else float("nan"),
                            "cN_abs": float(abs(cN)) if is_finite_complex(cN) else float("nan"),
                            "cN_arg": float(np.angle(cN)) if is_finite_complex(cN) else float("nan"),
                            "err_phi_norm_median": float(phi_norm_med),
                            "err_phi_norm_max": float(phi_norm_max),
                            "err_phi_norm_p95": float(phi_norm_p95),
                            "err_phi_norm_top5_mean": float(phi_norm_top5),
                            "gap_phi_median": float(gap_med),
                            "gap_phi_min": float(gap_min),
                        }
                    )

                    if bool(compute_magphase_diag):
                        # Magnitude/phase diagnostics: does log|Q| track log|g_infty|?
                        abs_logabs_q = [float(abs(x)) for x in logabs_q if np.isfinite(x)]
                        absarg_q_f = [float(x) for x in absarg_q if np.isfinite(x)]
                        unit_phase_err_f = [float(x) for x in unit_phase_err if np.isfinite(x)]
                        magerr_f = [float(x) for x in absabs_q_minus1 if np.isfinite(x)]
                        qg_err_f = [float(x) for x in abs_q_times_ginf_minus1 if np.isfinite(x)]
                        qg_arg_f = [float(x) for x in abs_arg_q_times_ginf if np.isfinite(x)]
                        qg_unit_f = [float(x) for x in unit_phase_err_q_times_ginf if np.isfinite(x)]

                        abslog_med, abslog_max = _finite_stats(abs_logabs_q)
                        abslog_p95 = _finite_percentile(abs_logabs_q, 95.0)
                        arg_med, arg_max = _finite_stats(absarg_q_f)
                        arg_p95 = _finite_percentile(absarg_q_f, 95.0)
                        unit_med, unit_max = _finite_stats(unit_phase_err_f)
                        unit_p95 = _finite_percentile(unit_phase_err_f, 95.0)
                        magerr_med, magerr_max = _finite_stats(magerr_f)
                        magerr_p95 = _finite_percentile(magerr_f, 95.0)

                        qg_err_med, qg_err_max = _finite_stats(qg_err_f)
                        qg_err_p95 = _finite_percentile(qg_err_f, 95.0)
                        qg_arg_med, qg_arg_max = _finite_stats(qg_arg_f)
                        qg_arg_p95 = _finite_percentile(qg_arg_f, 95.0)
                        qg_unit_med, qg_unit_max = _finite_stats(qg_unit_f)
                        qg_unit_p95 = _finite_percentile(qg_unit_f, 95.0)

                        corr, slope, n_pairs = _finite_corr_and_slope(logabs_ginf, logabs_q)
                        g_med, _ = _finite_stats(logabs_ginf)
                        qlog_med, _ = _finite_stats(logabs_q)

                        magphase_rows.append(
                            {
                                "source": str(source_label),
                                "N": int(N),
                                "basis": str(basis),
                                "block": str(block2),
                                "sigma_min": float(rect_in[0]),
                                "sigma_max": float(rect_in[1]),
                                "t_min": float(rect_in[2]),
                                "t_max": float(rect_in[3]),
                                "n_sigma": int(n_sigma),
                                "n_t": int(n_t),
                                "points_total": int(total),
                                "failures": int(failures),
                                "logabs_Q_median": float(qlog_med),
                                "abs_logabs_Q_median": float(abslog_med),
                                "abs_logabs_Q_p95": float(abslog_p95),
                                "abs_logabs_Q_max": float(abslog_max),
                                "abs_absQ_minus1_median": float(magerr_med),
                                "abs_absQ_minus1_p95": float(magerr_p95),
                                "abs_absQ_minus1_max": float(magerr_max),
                                "abs_argQ_median": float(arg_med),
                                "abs_argQ_p95": float(arg_p95),
                                "abs_argQ_max": float(arg_max),
                                "unit_phase_err_median": float(unit_med),
                                "unit_phase_err_p95": float(unit_p95),
                                "unit_phase_err_max": float(unit_max),
                                "abs_q_times_ginf_minus1_median": float(qg_err_med),
                                "abs_q_times_ginf_minus1_p95": float(qg_err_p95),
                                "abs_q_times_ginf_minus1_max": float(qg_err_max),
                                "abs_arg_q_times_ginf_median": float(qg_arg_med),
                                "abs_arg_q_times_ginf_p95": float(qg_arg_p95),
                                "abs_arg_q_times_ginf_max": float(qg_arg_max),
                                "unit_phase_err_q_times_ginf_median": float(qg_unit_med),
                                "unit_phase_err_q_times_ginf_p95": float(qg_unit_p95),
                                "unit_phase_err_q_times_ginf_max": float(qg_unit_max),
                                "logabs_ginf_median": float(g_med),
                                "logabs_corr": float(corr),
                                "logabs_slope": float(slope),
                                "logabs_pairs": int(n_pairs),
                            }
                        )

        if stats_rows:
            pd.DataFrame(stats_rows).to_csv(os.path.join(out_dir, str(out_csv_name)), index=False)

        if bool(compute_magphase_diag) and magphase_rows:
            pd.DataFrame(magphase_rows).to_csv(os.path.join(out_dir, str(magphase_out_csv_name)), index=False)

    def _compute_q_lambda_value_agreement_single(
        rect_in: tuple[float, float, float, float],
        n_sigma: int,
        n_t: int,
        *,
        cfgN: ModelCfg,
        basis: str,
        block2: str,
        use_cache: bool,
    ) -> dict:
        """A single (basis,block) row of the value-agreement stats (faster than the full 4x report)."""

        if n_sigma < 3 or n_t < 3:
            return {}
        if rect_in[3] <= rect_in[2] or rect_in[1] <= rect_in[0]:
            return {}

        basis = str(basis).strip().lower()
        block2 = str(block2).strip().lower()

        sigs = np.linspace(rect_in[0], rect_in[1], int(n_sigma) + 2, endpoint=True, dtype=float)[1:-1]
        ts = np.linspace(rect_in[2], rect_in[3], int(n_t) + 2, endpoint=True, dtype=float)[1:-1]

        mode_tag = str(bulk_mode).lower().strip()
        b0 = int(cfgN.b // 2) if mode_tag == "two_channel_symmetric" else int(cfgN.b)

        # Avoid caching huge matrices when called in an N-by-candidate sweep.
        model_fn = _model_mats if bool(use_cache) else getattr(_model_mats, "__wrapped__", _model_mats)
        free_fn = _free_mats if bool(use_cache) else getattr(_free_mats, "__wrapped__", _free_mats)

        q_phi_samples: list[complex] = []
        errs_phi: list[float] = []
        gaps_phi: list[float] = []
        total = 0
        failures = 0

        for sig in sigs:
            for tt in ts:
                s = complex(float(sig), float(tt))
                try:
                    _, Lam, Sm = model_fn(s, cfgN)
                    _, _, S0 = free_fn(s, cfgN)
                    Srel = (Sm @ np.linalg.inv(S0)).astype(np.complex128)
                    if basis == "incoming":
                        nb = int(Lam.shape[0])
                        Tin_eta = (Lam + 1j * float(cfgN.eta) * np.eye(nb, dtype=np.complex128)).astype(np.complex128)
                        if float(cfgN.cayley_eps) != 0.0:
                            Tin_eta = (Tin_eta + float(cfgN.cayley_eps) * np.eye(nb, dtype=np.complex128)).astype(np.complex128)
                        M = np.linalg.solve(Tin_eta, (Srel @ Tin_eta)).astype(np.complex128)
                    else:
                        M = Srel
                except Exception:
                    failures += 1
                    continue

                phi_norm = phi_target(s, primes=cfgN.primes, completion_mode=cfgN.completion_mode, mode=phi_target_mode)
                phi_sel = phi_target(s, primes=cfgN.primes, completion_mode=cfgN.completion_mode, mode=phi_select_mode)
                if (
                    (not is_finite_complex(phi_norm))
                    or abs(phi_norm) <= 1e-30
                    or (not is_finite_complex(phi_sel))
                    or abs(phi_sel) <= 1e-30
                ):
                    failures += 1
                    continue

                B = _swap_block_matrix(M, block2, b0=b0)
                try:
                    eigs = np.asarray(np.linalg.eigvals(B), dtype=np.complex128).ravel()
                except Exception:
                    failures += 1
                    continue
                if eigs.size == 0:
                    failures += 1
                    continue

                j_phi, _ = _min_relerr_to_target(eigs, phi_sel)
                lam_phi = complex(eigs[int(j_phi)])
                q_phi = lam_phi / phi_norm
                err_phi = float(abs(q_phi - 1.0)) if is_finite_complex(q_phi) else float("nan")
                if not np.isfinite(err_phi):
                    failures += 1
                    continue
                errs_phi.append(float(err_phi))

                try:
                    if eigs.size >= 2:
                        diffs = np.abs(eigs - lam_phi)
                        diffs[int(j_phi)] = np.inf
                        gap = float(np.min(diffs))
                    else:
                        gap = float("nan")
                except Exception:
                    gap = float("nan")
                if np.isfinite(gap):
                    gaps_phi.append(float(gap))

                if is_finite_complex(q_phi):
                    q_phi_samples.append(complex(q_phi))
                total += 1

        if not errs_phi:
            return {
                "points_total": int(total),
                "failures": int(failures),
                "err_phi_norm_median": float("nan"),
                "err_phi_norm_p95": float("nan"),
                "err_phi_norm_max": float("nan"),
                "err_phi_norm_top5_mean": float("nan"),
                "gap_phi_min": float("nan"),
                "gap_phi_median": float("nan"),
            }

        # Best-fit constant c_N and normalized errors.
        cN = complex(np.mean(np.asarray(q_phi_samples, dtype=np.complex128))) if q_phi_samples else complex("nan")
        if is_finite_complex(cN) and abs(cN) > 1e-30:
            errs_phi_norm = [float(abs((q / cN) - 1.0)) for q in q_phi_samples if is_finite_complex(q)]
        else:
            errs_phi_norm = []
        a = np.asarray(errs_phi_norm, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            phi_norm_med = float("nan")
            phi_norm_p95 = float("nan")
            phi_norm_max = float("nan")
            phi_norm_top5 = float("nan")
        else:
            phi_norm_med = float(np.median(a))
            phi_norm_p95 = float(np.percentile(a, 95.0))
            phi_norm_max = float(np.max(a))
            kk = min(5, int(a.size))
            part = np.partition(a, int(a.size - kk))
            phi_norm_top5 = float(np.mean(part[int(a.size - kk) :]))

        gap_arr = np.asarray(gaps_phi, dtype=float)
        gap_arr = gap_arr[np.isfinite(gap_arr)]
        gap_min = float(np.min(gap_arr)) if gap_arr.size else float("nan")
        gap_med = float(np.median(gap_arr)) if gap_arr.size else float("nan")

        return {
            "points_total": int(total),
            "failures": int(failures),
            "err_phi_norm_median": float(phi_norm_med),
            "err_phi_norm_p95": float(phi_norm_p95),
            "err_phi_norm_max": float(phi_norm_max),
            "err_phi_norm_top5_mean": float(phi_norm_top5),
            "gap_phi_min": float(gap_min),
            "gap_phi_median": float(gap_med),
        }

    def _compute_q_lambda_value_agreement_candidate_worst_channel(
        rect_in: tuple[float, float, float, float],
        n_sigma: int,
        n_t: int,
        *,
        cfgN: ModelCfg,
        use_cache: bool,
    ) -> dict:
        """Evaluate value-agreement across all 4 channels and return worst-channel summary.

        Returns max-over-channels metrics (p95/top5/median) along with which (basis,block)
        attains the maxima. Uses a per-point reuse strategy to avoid 4x recomputation.
        """

        if n_sigma < 3 or n_t < 3:
            return {}
        if rect_in[3] <= rect_in[2] or rect_in[1] <= rect_in[0]:
            return {}

        sigs = np.linspace(rect_in[0], rect_in[1], int(n_sigma) + 2, endpoint=True, dtype=float)[1:-1]
        ts = np.linspace(rect_in[2], rect_in[3], int(n_t) + 2, endpoint=True, dtype=float)[1:-1]

        mode_tag = str(bulk_mode).lower().strip()
        b0 = int(cfgN.b // 2) if mode_tag == "two_channel_symmetric" else int(cfgN.b)

        model_fn = _model_mats if bool(use_cache) else getattr(_model_mats, "__wrapped__", _model_mats)
        free_fn = _free_mats if bool(use_cache) else getattr(_free_mats, "__wrapped__", _free_mats)

        acc: dict[tuple[str, str], dict[str, object]] = {}
        for basis in ["rel", "incoming"]:
            for block2 in ["plus", "minus"]:
                acc[(basis, block2)] = {
                    "q_phi_samples": [],
                    "gaps_phi": [],
                    "total": 0,
                    "failures": 0,
                }

        for sig in sigs:
            for tt in ts:
                s = complex(float(sig), float(tt))
                try:
                    _, Lam, Sm = model_fn(s, cfgN)
                    _, _, S0 = free_fn(s, cfgN)
                    Srel = np.linalg.solve(np.asarray(S0, dtype=np.complex128).T, np.asarray(Sm, dtype=np.complex128).T).T
                except Exception:
                    for k in acc:
                        acc[k]["failures"] = int(acc[k]["failures"]) + 1
                    continue

                phi_norm = phi_target(s, primes=cfgN.primes, completion_mode=cfgN.completion_mode, mode=phi_target_mode)
                phi_sel = phi_target(s, primes=cfgN.primes, completion_mode=cfgN.completion_mode, mode=phi_select_mode)
                if (
                    (not is_finite_complex(phi_norm))
                    or abs(phi_norm) <= 1e-30
                    or (not is_finite_complex(phi_sel))
                    or abs(phi_sel) <= 1e-30
                ):
                    for k in acc:
                        acc[k]["failures"] = int(acc[k]["failures"]) + 1
                    continue

                M_rel = np.asarray(Srel, dtype=np.complex128)
                try:
                    nb = int(Lam.shape[0])
                    Tin_eta = (Lam + 1j * float(cfgN.eta) * np.eye(nb, dtype=np.complex128)).astype(np.complex128)
                    if float(cfgN.cayley_eps) != 0.0:
                        Tin_eta = (Tin_eta + float(cfgN.cayley_eps) * np.eye(nb, dtype=np.complex128)).astype(np.complex128)
                    M_in = np.linalg.solve(Tin_eta, (M_rel @ Tin_eta)).astype(np.complex128)
                except Exception:
                    M_in = None

                for basis in ["rel", "incoming"]:
                    M = M_rel if basis == "rel" else M_in
                    if M is None:
                        for block2 in ["plus", "minus"]:
                            acc[(basis, block2)]["failures"] = int(acc[(basis, block2)]["failures"]) + 1
                        continue

                    for block2 in ["plus", "minus"]:
                        try:
                            B = _swap_block_matrix(M, block2, b0=b0)
                            eigs = np.asarray(np.linalg.eigvals(B), dtype=np.complex128).ravel()
                        except Exception:
                            acc[(basis, block2)]["failures"] = int(acc[(basis, block2)]["failures"]) + 1
                            continue
                        if eigs.size == 0:
                            acc[(basis, block2)]["failures"] = int(acc[(basis, block2)]["failures"]) + 1
                            continue

                        j_phi, _ = _min_relerr_to_target(eigs, phi_sel)
                        lam_phi = complex(eigs[int(j_phi)])
                        q_phi = lam_phi / phi_norm
                        if not is_finite_complex(q_phi):
                            acc[(basis, block2)]["failures"] = int(acc[(basis, block2)]["failures"]) + 1
                            continue

                        acc[(basis, block2)]["total"] = int(acc[(basis, block2)]["total"]) + 1
                        cast_qs = acc[(basis, block2)]["q_phi_samples"]
                        assert isinstance(cast_qs, list)
                        cast_qs.append(complex(q_phi))

                        try:
                            if eigs.size >= 2:
                                diffs = np.abs(eigs - lam_phi)
                                diffs[int(j_phi)] = np.inf
                                gap = float(np.min(diffs))
                            else:
                                gap = float("nan")
                        except Exception:
                            gap = float("nan")
                        if np.isfinite(gap):
                            cast_gaps = acc[(basis, block2)]["gaps_phi"]
                            assert isinstance(cast_gaps, list)
                            cast_gaps.append(float(gap))

        per_rows: list[dict] = []
        for (basis, block2), a in acc.items():
            q_phi_samples = a["q_phi_samples"]
            gaps_phi = a["gaps_phi"]
            total = int(a["total"])
            failures = int(a["failures"])
            assert isinstance(q_phi_samples, list)
            assert isinstance(gaps_phi, list)

            cN = complex(np.mean(np.asarray(q_phi_samples, dtype=np.complex128))) if q_phi_samples else complex("nan")
            if is_finite_complex(cN) and abs(cN) > 1e-30:
                errs_phi_norm = [float(abs((q / cN) - 1.0)) for q in q_phi_samples if is_finite_complex(q)]
                aa = np.asarray(errs_phi_norm, dtype=float)
                aa = aa[np.isfinite(aa)]
                if aa.size == 0:
                    med, p95, top5, mx = float("nan"), float("nan"), float("nan"), float("nan")
                else:
                    med = float(np.median(aa))
                    p95 = float(np.percentile(aa, 95.0))
                    mx = float(np.max(aa))
                    kk = min(5, int(aa.size))
                    part = np.partition(aa, int(aa.size - kk))
                    top5 = float(np.mean(part[int(aa.size - kk) :]))
            else:
                med, p95, top5, mx = float("nan"), float("nan"), float("nan"), float("nan")

            gap_min = float(np.min(np.asarray(gaps_phi, dtype=float))) if gaps_phi else float("nan")
            per_rows.append(
                {
                    "basis": str(basis),
                    "block": str(block2),
                    "points_total": int(total),
                    "failures": int(failures),
                    "err_med": float(med),
                    "err_p95": float(p95),
                    "err_top5": float(top5),
                    "err_max": float(mx),
                    "gap_min": float(gap_min),
                }
            )

        if not per_rows:
            return {}

        dfp = pd.DataFrame(per_rows)
        out: dict[str, object] = {}
        out["per_channel_rows"] = per_rows

        def _pick_worst(col: str) -> dict:
            dd = dfp.copy()
            dd[col] = pd.to_numeric(dd[col], errors="coerce")
            dd = dd[dd[col].notna()].copy()
            if dd.empty:
                return {"val": float("nan"), "basis": "(na)", "block": "(na)"}
            j = int(dd[col].astype(float).values.argmax())
            r = dd.iloc[j]
            return {"val": float(r[col]), "basis": str(r["basis"]), "block": str(r["block"])}

        w_p95 = _pick_worst("err_p95")
        w_top5 = _pick_worst("err_top5")
        w_med = _pick_worst("err_med")
        out["worst_p95"] = float(w_p95["val"])
        out["worst_p95_basis"] = str(w_p95["basis"])
        out["worst_p95_block"] = str(w_p95["block"])
        out["worst_top5"] = float(w_top5["val"])
        out["worst_top5_basis"] = str(w_top5["basis"])
        out["worst_top5_block"] = str(w_top5["block"])
        out["worst_med"] = float(w_med["val"])
        out["worst_med_basis"] = str(w_med["basis"])
        out["worst_med_block"] = str(w_med["block"])
        out["gap_min_over_channels"] = float(pd.to_numeric(dfp["gap_min"], errors="coerce").min())
        return out

    def _winding_check_for_candidate(
        *,
        cfgW: ModelCfg,
        rectW: tuple[float, float, float, float],
        n_edge: int,
        basis: str,
        block: str,
        use_cache: bool,
    ) -> tuple[float, int]:
        """Compute winding for Q(s)=lambda/phi along rectangle boundary with branch-continuation/closure."""

        basis = str(basis).strip().lower()
        if basis not in {"incoming", "rel"}:
            raise ValueError("basis must be 'incoming' or 'rel'")
        block_key = "+" if str(block).strip().lower() in {"plus", "+"} else "-"

        ptsW = _boundary_points(rectW, int(n_edge))
        qs: list[complex] = []
        prev = complex("nan")
        first = complex("nan")
        failures = 0

        model_fnW = _model_mats if bool(use_cache) else getattr(_model_mats, "__wrapped__", _model_mats)
        free_fnW = _free_mats if bool(use_cache) else getattr(_free_mats, "__wrapped__", _free_mats)

        mode_tag = str(bulk_mode).lower().strip()
        b0 = int(cfgW.b // 2) if mode_tag == "two_channel_symmetric" else int(cfgW.b)

        for i, s in enumerate(ptsW):
            try:
                _, Lam, Sm = model_fnW(s, cfgW)
                _, _, S0 = free_fnW(s, cfgW)
                Srel = (Sm @ np.linalg.inv(S0)).astype(np.complex128)
                if basis == "incoming":
                    nb = int(Lam.shape[0])
                    Tin = (Lam + 1j * float(cfgW.eta) * np.eye(nb, dtype=np.complex128)).astype(np.complex128)
                    if float(cfgW.cayley_eps) != 0.0:
                        Tin = (Tin + float(cfgW.cayley_eps) * np.eye(nb, dtype=np.complex128)).astype(np.complex128)
                    M = np.linalg.solve(Tin, (Srel @ Tin)).astype(np.complex128)
                else:
                    M = Srel

                phi_norm = phi_target(s, primes=cfgW.primes, completion_mode=cfgW.completion_mode, mode=phi_target_mode)
                phi_sel = phi_target(s, primes=cfgW.primes, completion_mode=cfgW.completion_mode, mode=phi_select_mode)
                if (
                    (not is_finite_complex(phi_norm))
                    or abs(phi_norm) <= 1e-30
                    or (not is_finite_complex(phi_sel))
                    or abs(phi_sel) <= 1e-30
                ):
                    raise ValueError("bad phi")

                B = _swap_block_matrix(M, "plus" if block_key == "+" else "minus", b0=b0)
                eigs = np.asarray(np.linalg.eigvals(B), dtype=np.complex128).ravel()
                if eigs.size == 0:
                    raise ValueError("no eigs")

                if i == 0 or not is_finite_complex(prev):
                    j_idx, _ = _min_relerr_to_target(eigs, phi_sel)
                elif i == (len(ptsW) - 1) and is_finite_complex(first):
                    j_idx = _nearest_neighbor_index(eigs, first)
                else:
                    j_idx = _nearest_neighbor_index(eigs, prev)
                lam = complex(eigs[int(j_idx)])
                if i == 0:
                    first = lam
                prev = lam
                lam_fac = lambda_norm_factor(s, lambda_norm_mode, s_ref=(lambda_norm_sref if lambda_norm_sref is not None else s0))
                if (not is_finite_complex(lam_fac)) or abs(lam_fac) <= 1e-300:
                    raise ValueError("bad lambda_norm_factor")
                qs.append((lam * lam_fac) / phi_norm)
            except Exception:
                qs.append(complex("nan"))
                failures += 1

        w, wfail = _winding_count_from_samples(qs)
        return float(w), int(wfail + failures)

    if do_regularization_sweep_only:
        if reg_rect is None:
            raise ValueError("do_regularization_sweep_only requires reg_sweep_rect")

        if reg_wind_basis not in {"incoming", "rel"}:
            raise ValueError("reg_sweep_wind_basis must be 'incoming' or 'rel'")
        if reg_wind_block not in {"plus", "minus", "+", "-"}:
            raise ValueError("reg_sweep_wind_block must be 'plus' or 'minus'")

        block_key = "+" if reg_wind_block in {"plus", "+"} else "-"

        reg_resume = bool(config.get("reg_sweep_resume", True))
        master_csv = os.path.join(out_dir, "E_Qlambda_regularization_sweep.csv")

        expected_per_triple = int(len([int(x) for x in Ns]) * 4)  # N x (basis∈{rel,incoming}) x (block∈{+,-})

        done_triples: set[tuple[float, float, float, int]] = set()
        if reg_resume and os.path.exists(master_csv):
            try:
                df_prev = pd.read_csv(master_csv)
                if {"eta", "schur_jitter", "cayley_eps", "N", "basis", "block"}.issubset(set(df_prev.columns)):
                    if "boundary_b0_delta" in df_prev.columns:
                        df_prev["_triple"] = list(zip(df_prev["eta"], df_prev["schur_jitter"], df_prev["cayley_eps"], df_prev["boundary_b0_delta"]))
                    else:
                        df_prev["_triple"] = list(zip(df_prev["eta"], df_prev["schur_jitter"], df_prev["cayley_eps"], [0] * int(len(df_prev))))
                    df_prev["_key"] = list(zip(df_prev["N"], df_prev["basis"], df_prev["block"]))
                    g = df_prev.groupby("_triple")["_key"].nunique()
                    for triple, nuniq in g.items():
                        if int(nuniq) >= int(expected_per_triple):
                            done_triples.add((float(triple[0]), float(triple[1]), float(triple[2]), int(triple[3])))
            except Exception:
                done_triples = set()

        def _safe_tag(x: float) -> str:
            s = f"{float(x):.3e}"
            s = s.replace("+", "")
            s = re.sub(r"[^0-9eE\-\.]+", "_", s)
            s = s.replace(".", "p")
            return s

        def _append_df(df: pd.DataFrame) -> None:
            if df is None or df.empty:
                return
            header = not os.path.exists(master_csv)
            df.to_csv(master_csv, mode="a", header=header, index=False)

        for eta_i in reg_etas:
            for jit in reg_jitters:
                for eps_i in reg_cayley_eps:
                    for b0d in reg_b0_deltas:
                        triple = (float(eta_i), float(jit), float(eps_i), int(b0d))
                        if triple in done_triples:
                            continue

                        # Compute value agreement for all Ns at this parameter triple.
                        def _cfgf(N: int) -> ModelCfg:
                            return _mk_cfg(
                                int(N),
                                eta_override=float(eta_i),
                                jitter_override=float(jit),
                                cayley_eps_override=float(eps_i),
                                b0_delta_override=int(b0d),
                            )

                        tmp_name = f"__tmp_reg_{_safe_tag(eta_i)}__{_safe_tag(jit)}__{_safe_tag(eps_i)}__b0d_{int(b0d)}.csv"
                        tmp_path = os.path.join(out_dir, tmp_name)

                        df_tmp = pd.DataFrame([])
                        err_msg: str | None = None
                        try:
                            _compute_q_lambda_value_agreement_for_rect(
                                reg_rect,
                                int(reg_n_sigma),
                                int(reg_n_t),
                                out_csv_name=tmp_name,
                                source_label="reg_sweep",
                                Ns_list=[int(x) for x in Ns],
                                cfg_factory=_cfgf,
                                use_cache=False,
                            )
                            df_tmp = pd.read_csv(tmp_path)
                        except Exception as e:
                            err_msg = str(e)
                            df_tmp = pd.DataFrame([])

                        # Optional winding check on a representative (N,basis,block).
                        wind_val = float("nan")
                        wind_fail = int(-1)
                        if reg_do_winding:
                            cfgW = _cfgf(int(reg_wind_N))
                            ptsW = _boundary_points(reg_rect, int(reg_wind_n_edge))
                            qs: list[complex] = []
                            prev = complex("nan")
                            first = complex("nan")
                            failures = 0
                            # Avoid caching giant matrices in a sweep.
                            model_fnW = getattr(_model_mats, "__wrapped__", _model_mats)
                            free_fnW = getattr(_free_mats, "__wrapped__", _free_mats)
                            for i, s in enumerate(ptsW):
                                try:
                                    _, Lam, Sm = model_fnW(s, cfgW)
                                    _, _, S0 = free_fnW(s, cfgW)
                                    Srel = (Sm @ np.linalg.inv(S0)).astype(np.complex128)
                                    if reg_wind_basis == "incoming":
                                        nb = int(Lam.shape[0])
                                        Tin = (Lam + 1j * float(cfgW.eta) * np.eye(nb, dtype=np.complex128)).astype(np.complex128)
                                        if float(cfgW.cayley_eps) != 0.0:
                                            Tin = (Tin + float(cfgW.cayley_eps) * np.eye(nb, dtype=np.complex128)).astype(np.complex128)
                                        M = np.linalg.solve(Tin, (Srel @ Tin)).astype(np.complex128)
                                    else:
                                        M = Srel

                                    phi_norm = phi_target(s, primes=cfgW.primes, completion_mode=cfgW.completion_mode, mode=phi_target_mode)
                                    phi_sel = phi_target(s, primes=cfgW.primes, completion_mode=cfgW.completion_mode, mode=phi_select_mode)
                                    if (
                                        (not is_finite_complex(phi_norm))
                                        or abs(phi_norm) <= 1e-30
                                        or (not is_finite_complex(phi_sel))
                                        or abs(phi_sel) <= 1e-30
                                    ):
                                        raise ValueError("bad phi")

                                    b0 = int(cfgW.b // 2)
                                    B = _swap_block_matrix(M, "plus" if block_key == "+" else "minus", b0=b0)
                                    eigs = np.asarray(np.linalg.eigvals(B), dtype=np.complex128).ravel()
                                    if eigs.size == 0:
                                        raise ValueError("no eigs")

                                    if i == 0 or not is_finite_complex(prev):
                                        j_idx, _ = _min_relerr_to_target(eigs, phi_sel)
                                    elif i == (len(ptsW) - 1) and is_finite_complex(first):
                                        j_idx = _nearest_neighbor_index(eigs, first)
                                    else:
                                        j_idx = _nearest_neighbor_index(eigs, prev)
                                    lam = complex(eigs[int(j_idx)])
                                    if i == 0:
                                        first = lam
                                    prev = lam
                                    lam_fac = lambda_norm_factor(s, lambda_norm_mode, s_ref=(lambda_norm_sref if lambda_norm_sref is not None else s0))
                                    if (not is_finite_complex(lam_fac)) or abs(lam_fac) <= 1e-300:
                                        raise ValueError("bad lambda_norm_factor")
                                    qs.append((lam * lam_fac) / phi_norm)
                                except Exception:
                                    qs.append(complex("nan"))
                                    failures += 1

                            w, wfail = _winding_count_from_samples(qs)
                            wind_val = float(w)
                            wind_fail = int(wfail + failures)

                        # Assemble rows for this triple and append immediately.
                        if not df_tmp.empty:
                            df_tmp = df_tmp.copy()
                            df_tmp["eta"] = float(eta_i)
                            df_tmp["schur_jitter"] = float(jit)
                            df_tmp["cayley_eps"] = float(eps_i)
                            df_tmp["boundary_b0_delta"] = int(b0d)
                            # Annotate boundary sizing per row (depends on N).
                            try:
                                mode_tag = str(bulk_mode).lower().strip()
                                b_by_N: dict[int, int] = {}
                                b0_by_N: dict[int, int] = {}
                                for Ntag in sorted({int(x) for x in df_tmp["N"].dropna().tolist()}):
                                    cfg_tag = _cfgf(int(Ntag))
                                    b_by_N[int(Ntag)] = int(cfg_tag.b)
                                    if mode_tag == "two_channel_symmetric":
                                        b0_by_N[int(Ntag)] = int(cfg_tag.b // 2)
                                    else:
                                        b0_by_N[int(Ntag)] = int(cfg_tag.b)
                                df_tmp["boundary_b"] = df_tmp["N"].apply(lambda x: float(b_by_N.get(int(x), float("nan"))) if pd.notna(x) else float("nan"))
                                df_tmp["boundary_b0"] = df_tmp["N"].apply(lambda x: float(b0_by_N.get(int(x), float("nan"))) if pd.notna(x) else float("nan"))
                            except Exception:
                                df_tmp["boundary_b"] = float("nan")
                                df_tmp["boundary_b0"] = float("nan")
                            df_tmp["wind_check_N"] = int(reg_wind_N)
                            df_tmp["wind_check_basis"] = str(reg_wind_basis)
                            df_tmp["wind_check_block"] = "plus" if block_key == "+" else "minus"
                            df_tmp["wind_check_n_edge"] = int(reg_wind_n_edge)
                            df_tmp["wind_check_winding"] = float(wind_val)
                            df_tmp["wind_check_failures"] = int(wind_fail)
                            df_tmp["status"] = "ok"
                            df_tmp["error"] = ""
                            _append_df(df_tmp)
                        else:
                            # Emit a sentinel error row so the sweep can be audited/resumed.
                            df_err = pd.DataFrame(
                                [
                                    {
                                        "source": "reg_sweep",
                                        "N": float("nan"),
                                        "basis": "(na)",
                                        "block": "(na)",
                                        "sigma_min": float(reg_rect[0]),
                                        "sigma_max": float(reg_rect[1]),
                                        "t_min": float(reg_rect[2]),
                                        "t_max": float(reg_rect[3]),
                                        "n_sigma": int(reg_n_sigma),
                                        "n_t": int(reg_n_t),
                                        "points_total": 0,
                                        "failures": 0,
                                        "eta": float(eta_i),
                                        "schur_jitter": float(jit),
                                        "cayley_eps": float(eps_i),
                                        "boundary_b0_delta": int(b0d),
                                        "wind_check_N": int(reg_wind_N),
                                        "wind_check_basis": str(reg_wind_basis),
                                        "wind_check_block": "plus" if block_key == "+" else "minus",
                                        "wind_check_n_edge": int(reg_wind_n_edge),
                                        "wind_check_winding": float(wind_val),
                                        "wind_check_failures": int(wind_fail),
                                        "status": "error",
                                        "error": str(err_msg) if err_msg is not None else "(unknown)",
                                    }
                                ]
                            )
                            _append_df(df_err)

                    # Best-effort cleanup and memory control.
                        try:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                        except Exception:
                            pass
                    try:
                        _model_mats.cache_clear()
                        _free_mats.cache_clear()
                    except Exception:
                        pass
                    try:
                        gc.collect()
                    except Exception:
                        pass

        return 0

    if do_convergence_sweep_only:
        if convergence_rect is None:
            raise ValueError("do_convergence_sweep_only requires convergence_rect (or value_agreement_rect)")

        if boundary_autopick:
            # Auto-pick a discrete boundary b0 for each N by scanning a tiny neighborhood around b*.
            if boundary_autopick_window < 0:
                raise ValueError("boundary_autopick_window must be >= 0")
            if boundary_autopick_objective not in {"p95", "median"}:
                raise ValueError("boundary_autopick_objective must be 'p95' or 'median'")
            if boundary_autopick_tiebreak not in {"median", "p95"}:
                raise ValueError("boundary_autopick_tiebreak must be 'median' or 'p95'")
            if boundary_autopick_channel_mode not in {"single", "worst_channel"}:
                raise ValueError("boundary_autopick_channel_mode must be 'single' or 'worst_channel'")
            if boundary_autopick_rank_mode not in {"legacy", "p95_lex_top5"}:
                raise ValueError("boundary_autopick_rank_mode must be 'legacy' or 'p95_lex_top5'")
            if boundary_autopick_basis not in {"incoming", "rel"}:
                raise ValueError("boundary_autopick_basis must be 'incoming' or 'rel'")
            if boundary_autopick_block not in {"plus", "minus", "+", "-"}:
                raise ValueError("boundary_autopick_block must be 'plus' or 'minus'")

            cand_rows: list[dict] = []
            chosen: dict[int, int] = {}
            accept_rows: list[dict] = []

            mode_tag = str(bulk_mode).lower().strip()
            use_cache_pick = False

            for N0 in [int(x) for x in Ns]:
                # b* computed in one-channel coordinates.
                b_star = int(round(float(boundary_frac) * float(int(N0))))
                deltas = list(range(int(-boundary_autopick_window), int(boundary_autopick_window) + 1))

                stage1: list[dict] = []
                for dlt in deltas:
                    cfgC = _mk_cfg(int(N0), b0_delta_override=int(dlt))
                    worst_summary: dict | None = None
                    if boundary_autopick_channel_mode == "worst_channel":
                        worst_summary = _compute_q_lambda_value_agreement_candidate_worst_channel(
                            convergence_rect,
                            int(boundary_autopick_eval_n_sigma),
                            int(boundary_autopick_eval_n_t),
                            cfgN=cfgC,
                            use_cache=use_cache_pick,
                        )
                        gap_min = float(worst_summary.get("gap_min_over_channels", float("nan")))
                        med = float(worst_summary.get("worst_med", float("nan")))
                        p95 = float(worst_summary.get("worst_p95", float("nan")))
                        top5 = float(worst_summary.get("worst_top5", float("nan")))
                        mx = float("nan")
                        worst_p95_basis = str(worst_summary.get("worst_p95_basis", "(na)"))
                        worst_p95_block = str(worst_summary.get("worst_p95_block", "(na)"))
                        worst_top5_basis = str(worst_summary.get("worst_top5_basis", "(na)"))
                        worst_top5_block = str(worst_summary.get("worst_top5_block", "(na)"))
                    else:
                        one = _compute_q_lambda_value_agreement_single(
                            convergence_rect,
                            int(boundary_autopick_eval_n_sigma),
                            int(boundary_autopick_eval_n_t),
                            cfgN=cfgC,
                            basis=str(boundary_autopick_basis),
                            block2=("plus" if boundary_autopick_block in {"plus", "+"} else "minus"),
                            use_cache=use_cache_pick,
                        )
                        gap_min = float(one.get("gap_phi_min", float("nan")))
                        med = float(one.get("err_phi_norm_median", float("nan")))
                        p95 = float(one.get("err_phi_norm_p95", float("nan")))
                        top5 = float(one.get("err_phi_norm_top5_mean", float("nan")))
                        mx = float(one.get("err_phi_norm_max", float("nan")))
                        worst_p95_basis = "(na)"
                        worst_p95_block = "(na)"
                        worst_top5_basis = "(na)"
                        worst_top5_block = "(na)"

                    wind1, wind1_fail = _winding_check_for_candidate(
                        cfgW=cfgC,
                        rectW=convergence_rect,
                        n_edge=int(boundary_autopick_wind_n_edge_stage1),
                        basis=str(boundary_autopick_basis),
                        block=str(boundary_autopick_block),
                        use_cache=use_cache_pick,
                    )

                    gap_ok = np.isfinite(gap_min) and (gap_min >= float(boundary_autopick_gap_gamma))
                    wind_ok = np.isfinite(wind1) and (abs(float(wind1)) < 0.5) and (int(wind1_fail) == 0)
                    metric_ok = np.isfinite(med) and np.isfinite(p95)

                    row = {
                        "N": int(N0),
                        "b_star": int(b_star),
                        "delta": int(dlt),
                        "b0": int(cfgC.b // 2) if mode_tag == "two_channel_symmetric" else int(cfgC.b),
                        "b": int(cfgC.b),
                        "eval_n_sigma": int(boundary_autopick_eval_n_sigma),
                        "eval_n_t": int(boundary_autopick_eval_n_t),
                        "basis": str(boundary_autopick_basis),
                        "block": "plus" if boundary_autopick_block in {"plus", "+"} else "minus",
                        "err_med": float(med),
                        "err_p95": float(p95),
                        "err_top5": float(top5),
                        "err_max": float(mx),
                        "gap_min": float(gap_min),
                        "top5_star": float("nan"),
                        "top5_guard_ok": True,
                        "worst_p95_basis": str(worst_p95_basis),
                        "worst_p95_block": str(worst_p95_block),
                        "worst_top5_basis": str(worst_top5_basis),
                        "worst_top5_block": str(worst_top5_block),
                        "wind_stage1": float(wind1),
                        "wind_stage1_fail": int(wind1_fail),
                        "gap_ok": bool(gap_ok),
                        "wind_ok_stage1": bool(wind_ok),
                        "metric_ok": bool(metric_ok),
                        "wind_stage2": float("nan"),
                        "wind_stage2_fail": int(-1),
                        "wind_ok_stage2": False,
                        "selected": False,
                        "objective": str(boundary_autopick_objective),
                        "tiebreak": str(boundary_autopick_tiebreak),
                        "channel_mode": str(boundary_autopick_channel_mode),
                        "rank_mode": str(boundary_autopick_rank_mode),
                        "top5_guard_rel": float(boundary_autopick_top5_guard_rel),
                        "top5_guard_abs": float(boundary_autopick_top5_guard_abs),
                        "gap_gamma": float(boundary_autopick_gap_gamma),
                        "wind_n_edge_stage1": int(boundary_autopick_wind_n_edge_stage1),
                        "wind_n_edge_stage2": int(boundary_autopick_wind_n_edge_stage2),
                    }
                    stage1.append(row)

                # Optional guard: avoid top5 regression relative to the unshifted b* (delta=0).
                # This is disabled unless guard_rel>0 or guard_abs>0.
                top5_star = float("nan")
                for r in stage1:
                    if int(r.get("delta", 999999)) == 0:
                        top5_star = float(r.get("err_top5", float("nan")))
                        break
                if bool(boundary_autopick_top5_guard) and np.isfinite(top5_star):
                    # Interpret "no regression" as guard_rel==0 and guard_abs==0 meaning strict <= top5_star.
                    rel = float(boundary_autopick_top5_guard_rel)
                    absb = float(boundary_autopick_top5_guard_abs)
                    cap = float(top5_star) * (1.0 + max(0.0, rel)) + max(0.0, absb)
                    for r in stage1:
                        r["top5_star"] = float(top5_star)
                        et = float(r.get("err_top5", float("nan")))
                        r["top5_guard_ok"] = bool(np.isfinite(et) and (et <= cap))
                else:
                    cap = float("nan")

                # Filter admissible candidates.
                admissible = [r for r in stage1 if r["metric_ok"] and r["gap_ok"] and r["wind_ok_stage1"] and bool(r.get("top5_guard_ok", True))]

                # Acceptance-rate audit summary for this N.
                n_total = int(len(stage1))
                n_metric_ok = int(sum(1 for r in stage1 if bool(r.get("metric_ok", False))))
                n_gap_ok = int(sum(1 for r in stage1 if bool(r.get("gap_ok", False))))
                n_wind_ok = int(sum(1 for r in stage1 if bool(r.get("wind_ok_stage1", False))))
                n_guard_ok = int(sum(1 for r in stage1 if bool(r.get("top5_guard_ok", True))))
                n_admissible = int(len(admissible))

                # "Rejected by constraint" counts are reported as independent failures (not disjoint).
                n_fail_metric = int(sum(1 for r in stage1 if not bool(r.get("metric_ok", False))))
                n_fail_gap = int(sum(1 for r in stage1 if not bool(r.get("gap_ok", False))))
                n_fail_wind = int(sum(1 for r in stage1 if not bool(r.get("wind_ok_stage1", False))))
                n_fail_guard = int(sum(1 for r in stage1 if not bool(r.get("top5_guard_ok", True))))

                # Ordered pipeline counts (disjoint stage rejections) for intuitive reporting.
                # Pipeline order: metric_ok → gap_ok → wind_ok_stage1 → top5_guard_ok.
                stage_metric = [r for r in stage1 if bool(r.get("metric_ok", False))]
                stage_gap = [r for r in stage_metric if bool(r.get("gap_ok", False))]
                stage_wind = [r for r in stage_gap if bool(r.get("wind_ok_stage1", False))]
                stage_guard = [r for r in stage_wind if bool(r.get("top5_guard_ok", True))]

                n_after_metric = int(len(stage_metric))
                n_after_gap = int(len(stage_gap))
                n_after_wind = int(len(stage_wind))
                n_after_guard = int(len(stage_guard))

                n_rej_metric = int(n_total - n_after_metric)
                n_rej_gap = int(n_after_metric - n_after_gap)
                n_rej_wind = int(n_after_gap - n_after_wind)
                n_rej_guard = int(n_after_wind - n_after_guard)

                accept_rows.append(
                    {
                        "N": int(N0),
                        "b_star": int(b_star),
                        "deltas": str(deltas),
                        "channel_mode": str(boundary_autopick_channel_mode),
                        "rank_mode": str(boundary_autopick_rank_mode),
                        "basis_wind": str(boundary_autopick_basis),
                        "block_wind": "plus" if boundary_autopick_block in {"plus", "+"} else "minus",
                        "gap_gamma": float(boundary_autopick_gap_gamma),
                        "top5_guard_enabled": bool(boundary_autopick_top5_guard),
                        "top5_star": float(top5_star),
                        "top5_cap": float(cap),
                        "n_candidates": int(n_total),
                        "n_pass_metric": int(n_metric_ok),
                        "n_pass_gap": int(n_gap_ok),
                        "n_pass_wind_stage1": int(n_wind_ok),
                        "n_pass_top5_guard": int(n_guard_ok),
                        "n_pass_all": int(n_admissible),
                        "n_fail_metric": int(n_fail_metric),
                        "n_fail_gap": int(n_fail_gap),
                        "n_fail_wind_stage1": int(n_fail_wind),
                        "n_fail_top5_guard": int(n_fail_guard),
                        "pipeline_order": "metric->gap->wind->guard",
                        "n_after_metric": int(n_after_metric),
                        "n_after_gap": int(n_after_gap),
                        "n_after_wind_stage1": int(n_after_wind),
                        "n_after_top5_guard": int(n_after_guard),
                        "n_rej_metric": int(n_rej_metric),
                        "n_rej_gap": int(n_rej_gap),
                        "n_rej_wind_stage1": int(n_rej_wind),
                        "n_rej_top5_guard": int(n_rej_guard),
                    }
                )

                # Rank by objective then tiebreak.
                def _rank_key(r: dict) -> tuple[float, float, float]:
                    if boundary_autopick_rank_mode == "p95_lex_top5":
                        # Paper-friendly lexicographic objective: p95 first, then top5, then median.
                        return (float(r["err_p95"]), float(r["err_top5"]), float(r["err_med"]))
                    # Legacy: objective + tiebreak
                    primary = float(r["err_p95"]) if boundary_autopick_objective == "p95" else float(r["err_med"])
                    secondary = float(r["err_med"]) if boundary_autopick_tiebreak == "median" else float(r["err_p95"])
                    return (primary, secondary, float(r["err_top5"]))

                admissible_sorted = sorted(admissible, key=_rank_key)

                # Confirm winding at higher resolution for the top candidates (and pick the first that passes).
                confirm_k = max(1, int(boundary_autopick_wind_confirm_topk))
                confirm_list = admissible_sorted[:confirm_k]
                confirmed_good: set[int] = set()
                for r in confirm_list:
                    cfgC = _mk_cfg(int(N0), b0_delta_override=int(r["delta"]))
                    wind2, wind2_fail = _winding_check_for_candidate(
                        cfgW=cfgC,
                        rectW=convergence_rect,
                        n_edge=int(boundary_autopick_wind_n_edge_stage2),
                        basis=str(boundary_autopick_basis),
                        block=str(boundary_autopick_block),
                        use_cache=use_cache_pick,
                    )
                    r["wind_stage2"] = float(wind2)
                    r["wind_stage2_fail"] = int(wind2_fail)
                    r["wind_ok_stage2"] = bool(np.isfinite(wind2) and (abs(float(wind2)) < 0.5) and (int(wind2_fail) == 0))
                    if r["wind_ok_stage2"]:
                        confirmed_good.add(int(r["delta"]))

                # If none confirmed, fall back to stage-1 admissible list; if that is empty, fall back to delta=0.
                pick_delta: int | None = None
                for r in admissible_sorted:
                    if int(r["delta"]) in confirmed_good:
                        pick_delta = int(r["delta"])  # best confirmed
                        break
                if pick_delta is None:
                    if admissible_sorted:
                        pick_delta = int(admissible_sorted[0]["delta"])  # best stage-1
                    else:
                        pick_delta = 0
                chosen[int(N0)] = int(pick_delta)

                # Mark selections and accumulate rows.
                for r in stage1:
                    if int(r["delta"]) == int(pick_delta):
                        r["selected"] = True
                    cand_rows.append(r)

            # Emit auditable candidate table and choice summary.
            if cand_rows:
                pd.DataFrame(cand_rows).to_csv(os.path.join(out_dir, "E_boundary_autopick_candidates.csv"), index=False)
                choice_rows = [{"N": int(Nk), "chosen_delta": int(dv)} for Nk, dv in sorted(chosen.items())]
                pd.DataFrame(choice_rows).to_csv(os.path.join(out_dir, "E_boundary_autopick_choice.csv"), index=False)
                if accept_rows:
                    pd.DataFrame(accept_rows).to_csv(os.path.join(out_dir, "E_boundary_autopick_acceptance.csv"), index=False)

                # Paper-ready algorithm statement for this run.
                try:
                    pol_lines: list[str] = []
                    pol_lines.append("# Boundary auto-pick policy (Experiment E)\n")
                    pol_lines.append("Candidate set (per N):")
                    pol_lines.append(f"- Δ ∈ {{{', '.join([str(x) for x in range(-int(boundary_autopick_window), int(boundary_autopick_window)+1)])}}}\n")
                    pol_lines.append("Constraints:")
                    pol_lines.append("- Divisor-clean: winding(Q)=0 on the safe rectangle boundary (two-stage check)")
                    pol_lines.append(f"- Spectral gap: gap ≥ γ with γ={float(boundary_autopick_gap_gamma)}\n")
                    pol_lines.append("Objective:")
                    if str(boundary_autopick_channel_mode) == "worst_channel" and str(boundary_autopick_rank_mode) == "p95_lex_top5":
                        pol_lines.append("- Minimize worst-channel p95; tie-break by worst-channel top5 mean; then worst-channel median\n")
                    else:
                        pol_lines.append(f"- Legacy: minimize {boundary_autopick_objective}; tie-break by {boundary_autopick_tiebreak}\n")
                    pol_lines.append("Guard:")
                    if bool(boundary_autopick_top5_guard):
                        pol_lines.append("- Reject candidate if worst-channel top5 exceeds Δ=0 baseline (with optional relative/absolute slack)\n")
                    else:
                        pol_lines.append("- (disabled)\n")
                    pol_lines.append("Winding check channel:")
                    pol_lines.append(f"- basis={boundary_autopick_basis}, block={boundary_autopick_block}, n_edge_stage1={int(boundary_autopick_wind_n_edge_stage1)}, n_edge_stage2={int(boundary_autopick_wind_n_edge_stage2)}\n")
                    Path(os.path.join(out_dir, "E_boundary_autopick_policy.md")).write_text("\n".join(pol_lines) + "\n", encoding="utf-8")
                except Exception:
                    pass

            def _cfg_autopicked(N: int) -> ModelCfg:
                return _mk_cfg(int(N), b0_delta_override=int(chosen.get(int(N), 0)))

            cfg_factory_use = _cfg_autopicked
        else:
            cfg_factory_use = None

        conv_do_winding = bool(config.get("convergence_do_winding", False))
        conv_do_magphase_diag = bool(config.get("convergence_do_magphase_diag", True))
        conv_wind_basis = str(config.get("convergence_wind_basis", "incoming")).strip().lower()
        conv_wind_block = str(config.get("convergence_wind_block", "plus")).strip().lower()
        conv_wind_n_edge = int(config.get("convergence_wind_n_edge", 800))
        conv_wind_out = str(config.get("convergence_wind_out_csv", "E_Qlambda_convergence_winding.csv"))
        conv_emit_boundary_samples = bool(config.get("convergence_emit_boundary_samples", False))
        conv_boundary_samples_out = str(
            config.get("convergence_boundary_samples_out_csv", "E_Qlambda_convergence_boundary_samples.csv")
        )
        if conv_wind_basis not in {"incoming", "rel"}:
            raise ValueError("convergence_wind_basis must be 'incoming' or 'rel'")
        if conv_wind_block not in {"plus", "minus", "+", "-"}:
            raise ValueError("convergence_wind_block must be 'plus' or 'minus'")
        conv_block_key = "+" if conv_wind_block in {"plus", "+"} else "-"

        _compute_q_lambda_value_agreement_for_rect(
            convergence_rect,
            int(convergence_n_sigma),
            int(convergence_n_t),
            out_csv_name="E_Qlambda_convergence_sweep.csv",
            source_label="convergence_sweep",
            cfg_factory=cfg_factory_use,
            use_cache=False,
            compute_magphase_diag=conv_do_magphase_diag,
            magphase_out_csv_name="E_Qlambda_convergence_magphase.csv",
            lambda_norm_mode=lambda_norm_mode,
        )

        if bool(do_critical_line_fingerprint):
            if critical_line_nt < 2:
                raise ValueError("critical_line_nt must be >= 2")
            if critical_line_t_max <= critical_line_t_min:
                raise ValueError("critical_line_t_max must exceed critical_line_t_min")
            if critical_line_basis not in {"incoming", "rel"}:
                raise ValueError("critical_line_basis must be 'incoming' or 'rel'")
            if critical_line_block not in {"plus", "minus", "+", "-"}:
                raise ValueError("critical_line_block must be 'plus' or 'minus'")

            fp_block2 = "plus" if critical_line_block in {"plus", "+"} else "minus"
            t_vals = np.linspace(float(critical_line_t_min), float(critical_line_t_max), int(critical_line_nt), dtype=float)

            # Avoid caching huge matrices when doing dense scans.
            model_fn_fp = getattr(_model_mats, "__wrapped__", _model_mats)
            free_fn_fp = getattr(_free_mats, "__wrapped__", _free_mats)

            point_rows: list[dict] = []
            summary_acc: dict[int, dict[str, list[float]]] = {}
            summary_counts: dict[int, dict[str, int]] = {}

            cand_acc: dict[tuple[int, str], dict[str, list[float]]] = {}
            cand_counts: dict[tuple[int, str], dict[str, int]] = {}

            for N in [int(x) for x in Ns]:
                cfgN = (cfg_factory_use(int(N)) if cfg_factory_use is not None else _mk_cfg(int(N)))
                mode_tag = str(bulk_mode).lower().strip()
                b0 = int(cfgN.b // 2) if mode_tag == "two_channel_symmetric" else int(cfgN.b)

                summary_acc[int(N)] = {
                    "abs_absQ_minus1": [],
                    "abs_argQ": [],
                    "unit_phase_err": [],
                    "abs_q_times_ginf_minus1": [],
                    "abs_arg_q_times_ginf": [],
                    "unit_phase_err_q_times_ginf": [],
                    "logabs_q": [],
                    "logabs_ginf": [],
                    "gap_phi": [],
                }
                summary_counts[int(N)] = {"total": 0, "failures": 0}

                if bool(critical_line_eval_candidates):
                    for cm in critical_line_candidates:
                        key = (int(N), str(cm))
                        cand_acc[key] = {
                            "abs_absQ_minus1": [],
                            "abs_argQ": [],
                            "unit_phase_err": [],
                            "abs_q_times_ginf_minus1": [],
                            "logabs_q": [],
                            "logabs_ginf": [],
                        }
                        cand_counts[key] = {"total": 0, "failures": 0}

                for tt in t_vals:
                    s = complex(float(critical_line_sigma), float(tt))
                    summary_counts[int(N)]["total"] += 1
                    if bool(critical_line_eval_candidates):
                        for cm in critical_line_candidates:
                            cand_counts[(int(N), str(cm))]["total"] += 1

                    try:
                        _, Lam, Sm = model_fn_fp(s, cfgN)
                        _, _, S0 = free_fn_fp(s, cfgN)
                        Srel = np.linalg.solve(np.asarray(S0, dtype=np.complex128).T, np.asarray(Sm, dtype=np.complex128).T).T
                    except Exception:
                        summary_counts[int(N)]["failures"] += 1
                        if bool(critical_line_eval_candidates):
                            for cm in critical_line_candidates:
                                cand_counts[(int(N), str(cm))]["failures"] += 1
                        point_rows.append(
                            {
                                "source": "critical_line_fingerprint",
                                "N": int(N),
                                "basis": str(critical_line_basis),
                                "block": str(fp_block2),
                                "sigma": float(np.real(s)),
                                "t": float(np.imag(s)),
                                "status": "error",
                                "err": "model_mats",
                            }
                        )
                        continue

                    phi_norm = phi_target(s, primes=cfgN.primes, completion_mode=cfgN.completion_mode, mode=phi_target_mode)
                    phi_sel = phi_target(s, primes=cfgN.primes, completion_mode=cfgN.completion_mode, mode=phi_select_mode)
                    if (
                        (not is_finite_complex(phi_norm))
                        or abs(phi_norm) <= 1e-30
                        or (not is_finite_complex(phi_sel))
                        or abs(phi_sel) <= 1e-30
                    ):
                        summary_counts[int(N)]["failures"] += 1
                        if bool(critical_line_eval_candidates):
                            for cm in critical_line_candidates:
                                cand_counts[(int(N), str(cm))]["failures"] += 1
                        point_rows.append(
                            {
                                "source": "critical_line_fingerprint",
                                "N": int(N),
                                "basis": str(critical_line_basis),
                                "block": str(fp_block2),
                                "sigma": float(np.real(s)),
                                "t": float(np.imag(s)),
                                "status": "error",
                                "err": "phi_bad",
                                "phi_norm_abs": float(abs(phi_norm)) if is_finite_complex(phi_norm) else float("nan"),
                                "phi_sel_abs": float(abs(phi_sel)) if is_finite_complex(phi_sel) else float("nan"),
                            }
                        )
                        continue

                    M_rel = np.asarray(Srel, dtype=np.complex128)
                    M_in: np.ndarray | None
                    try:
                        nb = int(Lam.shape[0])
                        Tin_eta = (Lam + 1j * float(cfgN.eta) * np.eye(nb, dtype=np.complex128)).astype(np.complex128)
                        if float(cfgN.cayley_eps) != 0.0:
                            Tin_eta = (Tin_eta + float(cfgN.cayley_eps) * np.eye(nb, dtype=np.complex128)).astype(np.complex128)
                        M_in = np.linalg.solve(Tin_eta, (M_rel @ Tin_eta)).astype(np.complex128)
                    except Exception:
                        M_in = None

                    M = M_rel if critical_line_basis == "rel" else M_in
                    if M is None:
                        summary_counts[int(N)]["failures"] += 1
                        if bool(critical_line_eval_candidates):
                            for cm in critical_line_candidates:
                                cand_counts[(int(N), str(cm))]["failures"] += 1
                        point_rows.append(
                            {
                                "source": "critical_line_fingerprint",
                                "N": int(N),
                                "basis": str(critical_line_basis),
                                "block": str(fp_block2),
                                "sigma": float(np.real(s)),
                                "t": float(np.imag(s)),
                                "status": "error",
                                "err": "incoming_unavailable",
                            }
                        )
                        continue

                    try:
                        B = _swap_block_matrix(M, fp_block2, b0=b0)
                        eigs = np.asarray(np.linalg.eigvals(B), dtype=np.complex128).ravel()
                    except Exception:
                        summary_counts[int(N)]["failures"] += 1
                        if bool(critical_line_eval_candidates):
                            for cm in critical_line_candidates:
                                cand_counts[(int(N), str(cm))]["failures"] += 1
                        point_rows.append(
                            {
                                "source": "critical_line_fingerprint",
                                "N": int(N),
                                "basis": str(critical_line_basis),
                                "block": str(fp_block2),
                                "sigma": float(np.real(s)),
                                "t": float(np.imag(s)),
                                "status": "error",
                                "err": "eig",
                            }
                        )
                        continue

                    if eigs.size == 0:
                        summary_counts[int(N)]["failures"] += 1
                        if bool(critical_line_eval_candidates):
                            for cm in critical_line_candidates:
                                cand_counts[(int(N), str(cm))]["failures"] += 1
                        point_rows.append(
                            {
                                "source": "critical_line_fingerprint",
                                "N": int(N),
                                "basis": str(critical_line_basis),
                                "block": str(fp_block2),
                                "sigma": float(np.real(s)),
                                "t": float(np.imag(s)),
                                "status": "error",
                                "err": "no_eigs",
                            }
                        )
                        continue

                    try:
                        j_phi, relerr_sel = _min_relerr_to_target(eigs, phi_sel)
                        lam_raw = complex(eigs[int(j_phi)])
                    except Exception:
                        summary_counts[int(N)]["failures"] += 1
                        if bool(critical_line_eval_candidates):
                            for cm in critical_line_candidates:
                                cand_counts[(int(N), str(cm))]["failures"] += 1
                        point_rows.append(
                            {
                                "source": "critical_line_fingerprint",
                                "N": int(N),
                                "basis": str(critical_line_basis),
                                "block": str(fp_block2),
                                "sigma": float(np.real(s)),
                                "t": float(np.imag(s)),
                                "status": "error",
                                "err": "pick",
                            }
                        )
                        continue

                    try:
                        if eigs.size >= 2:
                            diffs = np.abs(eigs - lam_raw)
                            diffs[int(j_phi)] = np.inf
                            gap = float(np.min(diffs))
                        else:
                            gap = float("nan")
                    except Exception:
                        gap = float("nan")

                    try:
                        lam_fac = lambda_norm_factor(s, lambda_norm_mode, s_ref=(lambda_norm_sref if lambda_norm_sref is not None else s0))
                        if (not is_finite_complex(lam_fac)) or abs(lam_fac) <= 1e-300:
                            raise ValueError("bad lambda normalization factor")
                        lam = complex(lam_raw * lam_fac)
                        q = complex(lam / phi_norm)
                    except Exception:
                        summary_counts[int(N)]["failures"] += 1
                        if bool(critical_line_eval_candidates):
                            for cm in critical_line_candidates:
                                cand_counts[(int(N), str(cm))]["failures"] += 1
                        point_rows.append(
                            {
                                "source": "critical_line_fingerprint",
                                "N": int(N),
                                "basis": str(critical_line_basis),
                                "block": str(fp_block2),
                                "sigma": float(np.real(s)),
                                "t": float(np.imag(s)),
                                "status": "error",
                                "err": "norm_or_q",
                            }
                        )
                        continue

                    ginf = g_infty_classical(s)
                    qg = complex(q * ginf) if is_finite_complex(q) and is_finite_complex(ginf) else complex("nan")

                    if bool(critical_line_eval_candidates):
                        for cm in critical_line_candidates:
                            try:
                                fac_c = lambda_norm_factor(s, str(cm), s_ref=(lambda_norm_sref if lambda_norm_sref is not None else s0))
                                if (not is_finite_complex(fac_c)) or abs(fac_c) <= 1e-300:
                                    raise ValueError("bad candidate factor")
                                qc = complex((lam_raw * fac_c) / phi_norm)
                            except Exception:
                                cand_counts[(int(N), str(cm))]["failures"] += 1
                                continue

                            absQc = float(abs(qc)) if is_finite_complex(qc) else float("nan")
                            abs_absQc_minus1 = float(abs(absQc - 1.0)) if np.isfinite(absQc) else float("nan")
                            abs_argQc = float(abs(np.angle(qc))) if is_finite_complex(qc) else float("nan")
                            if np.isfinite(absQc) and absQc > 0 and is_finite_complex(qc):
                                unitc = qc / absQc
                                unit_phase_err_c = float(abs(unitc - 1.0))
                            else:
                                unit_phase_err_c = float("nan")
                            qgc = complex(qc * ginf) if is_finite_complex(qc) and is_finite_complex(ginf) else complex("nan")
                            abs_qgc_minus1 = float(abs(qgc - 1.0)) if is_finite_complex(qgc) else float("nan")
                            logabs_qc = float(np.log(absQc)) if np.isfinite(absQc) and absQc > 0 else float("nan")
                            try:
                                gmag = float(abs(ginf)) if is_finite_complex(ginf) else float("nan")
                                logabs_g = float(np.log(gmag)) if np.isfinite(gmag) and gmag > 0 else float("nan")
                            except Exception:
                                logabs_g = float("nan")

                            key = (int(N), str(cm))
                            if np.isfinite(abs_absQc_minus1):
                                cand_acc[key]["abs_absQ_minus1"].append(float(abs_absQc_minus1))
                            if np.isfinite(abs_argQc):
                                cand_acc[key]["abs_argQ"].append(float(abs_argQc))
                            if np.isfinite(unit_phase_err_c):
                                cand_acc[key]["unit_phase_err"].append(float(unit_phase_err_c))
                            if np.isfinite(abs_qgc_minus1):
                                cand_acc[key]["abs_q_times_ginf_minus1"].append(float(abs_qgc_minus1))
                            if np.isfinite(logabs_qc):
                                cand_acc[key]["logabs_q"].append(float(logabs_qc))
                            if np.isfinite(logabs_g):
                                cand_acc[key]["logabs_ginf"].append(float(logabs_g))

                    absQ = float(abs(q)) if is_finite_complex(q) else float("nan")
                    abs_absQ_minus1 = float(abs(absQ - 1.0)) if np.isfinite(absQ) else float("nan")
                    abs_argQ = float(abs(np.angle(q))) if is_finite_complex(q) else float("nan")
                    if np.isfinite(absQ) and absQ > 0 and is_finite_complex(q):
                        unit = q / absQ
                        unit_phase_err = float(abs(unit - 1.0))
                    else:
                        unit_phase_err = float("nan")

                    abs_q_times_ginf_minus1 = float(abs(qg - 1.0)) if is_finite_complex(qg) else float("nan")
                    abs_arg_q_times_ginf = float(abs(np.angle(qg))) if is_finite_complex(qg) else float("nan")
                    try:
                        qg_mag = float(abs(qg)) if is_finite_complex(qg) else float("nan")
                        if np.isfinite(qg_mag) and qg_mag > 0:
                            unit_qg = qg / qg_mag
                            unit_phase_err_q_times_ginf = float(abs(unit_qg - 1.0))
                        else:
                            unit_phase_err_q_times_ginf = float("nan")
                    except Exception:
                        unit_phase_err_q_times_ginf = float("nan")

                    logabs_q = float(np.log(absQ)) if np.isfinite(absQ) and absQ > 0 else float("nan")
                    try:
                        gmag = float(abs(ginf)) if is_finite_complex(ginf) else float("nan")
                        logabs_ginf = float(np.log(gmag)) if np.isfinite(gmag) and gmag > 0 else float("nan")
                    except Exception:
                        logabs_ginf = float("nan")

                    # Accumulate summary stats.
                    if np.isfinite(abs_absQ_minus1):
                        summary_acc[int(N)]["abs_absQ_minus1"].append(float(abs_absQ_minus1))
                    if np.isfinite(abs_argQ):
                        summary_acc[int(N)]["abs_argQ"].append(float(abs_argQ))
                    if np.isfinite(unit_phase_err):
                        summary_acc[int(N)]["unit_phase_err"].append(float(unit_phase_err))
                    if np.isfinite(abs_q_times_ginf_minus1):
                        summary_acc[int(N)]["abs_q_times_ginf_minus1"].append(float(abs_q_times_ginf_minus1))
                    if np.isfinite(abs_arg_q_times_ginf):
                        summary_acc[int(N)]["abs_arg_q_times_ginf"].append(float(abs_arg_q_times_ginf))
                    if np.isfinite(unit_phase_err_q_times_ginf):
                        summary_acc[int(N)]["unit_phase_err_q_times_ginf"].append(float(unit_phase_err_q_times_ginf))
                    if np.isfinite(logabs_q):
                        summary_acc[int(N)]["logabs_q"].append(float(logabs_q))
                    if np.isfinite(logabs_ginf):
                        summary_acc[int(N)]["logabs_ginf"].append(float(logabs_ginf))
                    if np.isfinite(gap):
                        summary_acc[int(N)]["gap_phi"].append(float(gap))

                    point_rows.append(
                        {
                            "source": "critical_line_fingerprint",
                            "N": int(N),
                            "basis": str(critical_line_basis),
                            "block": str(fp_block2),
                            "sigma": float(np.real(s)),
                            "t": float(np.imag(s)),
                            "phi_target_mode": str(phi_target_mode),
                            "phi_select_mode": str(phi_select_mode),
                            "lambda_norm_mode": str(lambda_norm_mode),
                            "relerr_sel": float(relerr_sel) if np.isfinite(relerr_sel) else float("nan"),
                            "gap_phi": float(gap) if np.isfinite(gap) else float("nan"),
                            "lam_raw_re": float(np.real(lam_raw)),
                            "lam_raw_im": float(np.imag(lam_raw)),
                            "lam_fac_re": float(np.real(lam_fac)) if is_finite_complex(lam_fac) else float("nan"),
                            "lam_fac_im": float(np.imag(lam_fac)) if is_finite_complex(lam_fac) else float("nan"),
                            "q_re": float(np.real(q)) if is_finite_complex(q) else float("nan"),
                            "q_im": float(np.imag(q)) if is_finite_complex(q) else float("nan"),
                            "absQ": float(absQ),
                            "abs_absQ_minus1": float(abs_absQ_minus1),
                            "abs_argQ": float(abs_argQ),
                            "unit_phase_err": float(unit_phase_err),
                            "abs_q_times_ginf_minus1": float(abs_q_times_ginf_minus1),
                            "abs_arg_q_times_ginf": float(abs_arg_q_times_ginf),
                            "unit_phase_err_q_times_ginf": float(unit_phase_err_q_times_ginf),
                            "logabs_Q": float(logabs_q),
                            "logabs_ginf": float(logabs_ginf),
                            "status": "ok",
                            "err": "",
                        }
                    )

            if point_rows:
                pd.DataFrame(point_rows).to_csv(os.path.join(out_dir, critical_line_points_out_csv), index=False)

            if bool(config.get("critical_line_do_pi_ramp_diagnostics", False)):
                _write_pi_ramp_diagnostics_from_points_csv(
                    out_dir,
                    config,
                    points_csv_name=str(critical_line_points_out_csv),
                )

                # Optional secondary (relaxed-gamma) witness emission. This is meant to make
                # empty/fragile strict intersections explicit while still providing an auditable
                # numerical witness that the ramp exists at a less stringent health threshold.
                if bool(config.get("critical_line_pi_ramp_emit_secondary", False)):
                    gamma2 = float(config.get("critical_line_pi_ramp_gamma_secondary", 0.03))
                    out_fit2 = str(config.get("critical_line_pi_ramp_fit_secondary_out_csv", "E_Qlambda_pi_ramp_fit_relaxed.csv"))
                    out_res2 = str(config.get("critical_line_phase_residual_secondary_out_csv", "E_Qlambda_phase_residual_table_relaxed.csv"))
                    _write_pi_ramp_diagnostics_from_points_csv(
                        out_dir,
                        config,
                        points_csv_name=str(critical_line_points_out_csv),
                        gamma_override=gamma2,
                        out_fit_override=out_fit2,
                        out_res_override=out_res2,
                        relaxed_tag="secondary",
                    )

            def _finite_stats(vals: list[float]) -> tuple[float, float]:
                if not vals:
                    return (float("nan"), float("nan"))
                a = np.asarray(vals, dtype=float)
                a = a[np.isfinite(a)]
                if a.size == 0:
                    return (float("nan"), float("nan"))
                return (float(np.median(a)), float(np.max(a)))

            def _finite_percentile(vals: list[float], q: float) -> float:
                if not vals:
                    return float("nan")
                a = np.asarray(vals, dtype=float)
                a = a[np.isfinite(a)]
                if a.size == 0:
                    return float("nan")
                return float(np.percentile(a, float(q)))

            def _finite_corr(x: list[float], y: list[float]) -> tuple[float, int]:
                if (not x) or (not y):
                    return float("nan"), 0
                xx = np.asarray(x, dtype=float)
                yy = np.asarray(y, dtype=float)
                mask = np.isfinite(xx) & np.isfinite(yy)
                if int(np.sum(mask)) < 3:
                    return float("nan"), int(np.sum(mask))
                xx2 = xx[mask]
                yy2 = yy[mask]
                try:
                    corr = float(np.corrcoef(xx2, yy2)[0, 1])
                except Exception:
                    corr = float("nan")
                return corr, int(xx2.size)

            summary_rows: list[dict] = []
            # Convention bookkeeping: most formulas in this harness are written in s.
            # Accept legacy/short key 'arch_arg' as an alias to prevent ambiguity.
            critical_line_arch_arg = str(config.get("critical_line_arch_arg", config.get("arch_arg", "s")))
            for N in [int(x) for x in Ns]:
                accN = summary_acc.get(int(N), {})
                cntN = summary_counts.get(int(N), {"total": 0, "failures": 0})
                abs_med, abs_max = _finite_stats(list(accN.get("abs_absQ_minus1", [])))
                abs_p95 = _finite_percentile(list(accN.get("abs_absQ_minus1", [])), 95.0)
                arg_med, arg_max = _finite_stats(list(accN.get("abs_argQ", [])))
                arg_p95 = _finite_percentile(list(accN.get("abs_argQ", [])), 95.0)
                unit_med, unit_max = _finite_stats(list(accN.get("unit_phase_err", [])))
                unit_p95 = _finite_percentile(list(accN.get("unit_phase_err", [])), 95.0)
                qg_med, qg_max = _finite_stats(list(accN.get("abs_q_times_ginf_minus1", [])))
                qg_p95 = _finite_percentile(list(accN.get("abs_q_times_ginf_minus1", [])), 95.0)
                gap_med, _ = _finite_stats(list(accN.get("gap_phi", [])))
                gap_min = float(np.min(np.asarray(list(accN.get("gap_phi", [])), dtype=float))) if accN.get("gap_phi") else float("nan")
                corr, n_pairs = _finite_corr(list(accN.get("logabs_ginf", [])), list(accN.get("logabs_q", [])))

                summary_rows.append(
                    {
                        "source": "critical_line_fingerprint",
                        "N": int(N),
                        "basis": str(critical_line_basis),
                        "block": str(fp_block2),
                        "sigma": float(critical_line_sigma),
                        "t_min": float(critical_line_t_min),
                        "t_max": float(critical_line_t_max),
                        "nt": int(critical_line_nt),
                        "phi_target_mode": str(phi_target_mode),
                        "phi_select_mode": str(phi_select_mode),
                        "lambda_norm_mode": str(lambda_norm_mode),
                        "arch_arg": str(critical_line_arch_arg),
                        "points_total": int(cntN.get("total", 0)),
                        "failures": int(cntN.get("failures", 0)),
                        "abs_absQ_minus1_median": float(abs_med),
                        "abs_absQ_minus1_p95": float(abs_p95),
                        "abs_absQ_minus1_max": float(abs_max),
                        "abs_argQ_median": float(arg_med),
                        "abs_argQ_p95": float(arg_p95),
                        "abs_argQ_max": float(arg_max),
                        "unit_phase_err_median": float(unit_med),
                        "unit_phase_err_p95": float(unit_p95),
                        "unit_phase_err_max": float(unit_max),
                        "abs_q_times_ginf_minus1_median": float(qg_med),
                        "abs_q_times_ginf_minus1_p95": float(qg_p95),
                        "abs_q_times_ginf_minus1_max": float(qg_max),
                        "gap_phi_median": float(gap_med),
                        "gap_phi_min": float(gap_min),
                        "logabs_corr": float(corr),
                        "logabs_pairs": int(n_pairs),
                    }
                )

            if summary_rows:
                pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, critical_line_summary_out_csv), index=False)

            if bool(critical_line_eval_candidates) and cand_acc:
                cand_rows: list[dict] = []
                for (N, cm), accC in sorted(cand_acc.items(), key=lambda kv: (kv[0][0], kv[0][1])):
                    cntC = cand_counts.get((int(N), str(cm)), {"total": 0, "failures": 0})
                    abs_med, abs_max = _finite_stats(list(accC.get("abs_absQ_minus1", [])))
                    abs_p95 = _finite_percentile(list(accC.get("abs_absQ_minus1", [])), 95.0)
                    arg_med, arg_max = _finite_stats(list(accC.get("abs_argQ", [])))
                    arg_p95 = _finite_percentile(list(accC.get("abs_argQ", [])), 95.0)
                    unit_med, unit_max = _finite_stats(list(accC.get("unit_phase_err", [])))
                    unit_p95 = _finite_percentile(list(accC.get("unit_phase_err", [])), 95.0)
                    qg_med, qg_max = _finite_stats(list(accC.get("abs_q_times_ginf_minus1", [])))
                    qg_p95 = _finite_percentile(list(accC.get("abs_q_times_ginf_minus1", [])), 95.0)
                    corr, n_pairs = _finite_corr(list(accC.get("logabs_ginf", [])), list(accC.get("logabs_q", [])))

                    cand_rows.append(
                        {
                            "source": "critical_line_candidates",
                            "N": int(N),
                            "basis": str(critical_line_basis),
                            "block": str(fp_block2),
                            "sigma": float(critical_line_sigma),
                            "t_min": float(critical_line_t_min),
                            "t_max": float(critical_line_t_max),
                            "nt": int(critical_line_nt),
                            "phi_target_mode": str(phi_target_mode),
                            "phi_select_mode": str(phi_select_mode),
                            "candidate_lambda_norm_mode": str(cm),
                            "points_total": int(cntC.get("total", 0)),
                            "failures": int(cntC.get("failures", 0)),
                            "abs_absQ_minus1_median": float(abs_med),
                            "abs_absQ_minus1_p95": float(abs_p95),
                            "abs_absQ_minus1_max": float(abs_max),
                            "abs_argQ_median": float(arg_med),
                            "abs_argQ_p95": float(arg_p95),
                            "abs_argQ_max": float(arg_max),
                            "unit_phase_err_median": float(unit_med),
                            "unit_phase_err_p95": float(unit_p95),
                            "unit_phase_err_max": float(unit_max),
                            "abs_q_times_ginf_minus1_median": float(qg_med),
                            "abs_q_times_ginf_minus1_p95": float(qg_p95),
                            "abs_q_times_ginf_minus1_max": float(qg_max),
                            "logabs_corr": float(corr),
                            "logabs_pairs": int(n_pairs),
                        }
                    )

                if cand_rows:
                    pd.DataFrame(cand_rows).to_csv(os.path.join(out_dir, critical_line_candidates_out_csv), index=False)

        # Lightweight fit summaries (trend vs 1/N and power law vs N) for phi errors.
        try:
            df_conv = pd.read_csv(os.path.join(out_dir, "E_Qlambda_convergence_sweep.csv"))
        except Exception:
            df_conv = None

        fit_rows: list[dict] = []
        if df_conv is not None:
            for (basis, block2), g in df_conv.groupby(["basis", "block"], dropna=False):
                gg = g.copy()
                Ns_fit = np.asarray(gg["N"].values, dtype=float)
                invN = 1.0 / Ns_fit

                def _fit_metrics(y: np.ndarray) -> dict:
                    out: dict = {}
                    mask = np.isfinite(y) & np.isfinite(Ns_fit) & (y > 0)
                    if int(np.sum(mask)) < 3:
                        return {"n_fit": int(np.sum(mask))}
                    y2 = y[mask]
                    N2 = Ns_fit[mask]
                    invN2 = invN[mask]

                    # Fit y ~ a + b*(1/N)
                    try:
                        b1, a1 = np.polyfit(invN2, y2, deg=1)
                    except Exception:
                        a1, b1 = float("nan"), float("nan")
                    out["lin_a"] = float(a1)
                    out["lin_b"] = float(b1)

                    # Fit y ~ C * N^{-alpha} via log-log
                    try:
                        coeff = np.polyfit(np.log(N2), np.log(y2), deg=1)
                        alpha = -float(coeff[0])
                        C = float(np.exp(coeff[1]))
                    except Exception:
                        alpha, C = float("nan"), float("nan")
                    out["pow_alpha"] = float(alpha)
                    out["pow_C"] = float(C)
                    out["n_fit"] = int(np.sum(mask))
                    return out

                for metric in [
                    "err_phi_median",
                    "err_phi_max",
                    "err_phi_norm_median",
                    "err_phi_norm_max",
                    "err_phi_norm_p95",
                    "err_phi_norm_top5_mean",
                ]:
                    y = np.asarray(gg[metric].values, dtype=float)
                    fr = _fit_metrics(y)
                    fr.update({"basis": str(basis), "block": str(block2), "metric": str(metric)})
                    fit_rows.append(fr)

        pd.DataFrame(fit_rows).to_csv(os.path.join(out_dir, "E_Qlambda_convergence_fit.csv"), index=False)

        # Optional: boundary diagnostics on the rectangle boundary, per N.
        # Note: boundary sample emission is useful even when winding diagnostics are disabled.
        if conv_do_winding or conv_emit_boundary_samples:
            wind_rows: list[dict] = []
            boundary_rows: list[dict] = []
            ptsW = _boundary_points(convergence_rect, int(conv_wind_n_edge))
            tminW = float(convergence_rect[2])
            tmaxW = float(convergence_rect[3])
            sminW = float(convergence_rect[0])
            smaxW = float(convergence_rect[1])

            def _boundary_side(s: complex) -> str:
                try:
                    sig = float(np.real(s))
                    tt = float(np.imag(s))
                except Exception:
                    return "unknown"
                eps = 1e-12
                if abs(tt - tminW) <= eps:
                    return "bottom"
                if abs(sig - smaxW) <= eps:
                    return "right"
                if abs(tt - tmaxW) <= eps:
                    return "top"
                if abs(sig - sminW) <= eps:
                    return "left"
                return "unknown"

            # Avoid caching giant matrices in a sweep.
            model_fnW = getattr(_model_mats, "__wrapped__", _model_mats)
            free_fnW = getattr(_free_mats, "__wrapped__", _free_mats)
            for N in [int(x) for x in Ns]:
                cfgW = _mk_cfg(int(N))
                qs_track: list[complex] = []
                qs_pointwise: list[complex] = []
                gaps_track: list[float] = []
                gaps_point: list[float] = []
                seps2_track: list[float] = []
                seps2_point: list[float] = []
                det_rel_samples: list[complex] = []
                logdet_Iminus_rel_samples: list[complex] = []
                phi0_plus_samples: list[complex] = []
                phi0_minus_samples: list[complex] = []
                lam_cusp_plus_raw_samples: list[complex] = []
                lam_cusp_minus_raw_samples: list[complex] = []
                lam_cusp_plus_riesz_samples: list[complex] = []
                lam_cusp_minus_riesz_samples: list[complex] = []
                lam_cusp_plus_norm_samples: list[complex] = []
                lam_cusp_minus_norm_samples: list[complex] = []
                cusp_gap_plus_samples: list[float] = []
                cusp_gap_minus_samples: list[float] = []
                cusp_sep2_plus_samples: list[float] = []
                cusp_sep2_minus_samples: list[float] = []
                cusp_p0ov_plus_samples: list[float] = []
                cusp_p0ov_minus_samples: list[float] = []
                cusp_proj_idem_plus_samples: list[float] = []
                cusp_proj_idem_minus_samples: list[float] = []
                cusp_proj_comm_plus_samples: list[float] = []
                cusp_proj_comm_minus_samples: list[float] = []
                cusp_proj_trace_re_plus_samples: list[float] = []
                cusp_proj_trace_im_plus_samples: list[float] = []
                cusp_proj_trace_re_minus_samples: list[float] = []
                cusp_proj_trace_im_minus_samples: list[float] = []
                lam_raw_track_samples: list[complex] = []
                lam_raw_point_samples: list[complex] = []
                comp_idx_track_samples: list[int] = []
                comp_idx_point_samples: list[int] = []
                lam_comp_track_samples: list[complex] = []
                lam_comp_point_samples: list[complex] = []
                jumps_track: list[float] = []
                jumps_point: list[float] = []
                prev_track = complex("nan")
                first_track = complex("nan")
                prev_point = complex("nan")
                failures = 0
                phi_abs_min = float("inf")
                phi_bad = 0
                gap_min_track = float("inf")
                gap_min_point = float("inf")
                jump_max_track = 0.0
                jump_max_point = 0.0
                prev_cusp_plus = complex("nan")
                first_cusp_plus = complex("nan")
                prev_cusp_minus = complex("nan")
                first_cusp_minus = complex("nan")
                for i, s in enumerate(ptsW):
                    gap_t = float("nan")
                    gap_p = float("nan")
                    sep2_t = float("nan")
                    sep2_p = float("nan")
                    jmp_t = float("nan")
                    jmp_p = float("nan")
                    lam_raw_track_val = complex("nan")
                    lam_raw_point_val = complex("nan")
                    comp_idx_t = -1
                    comp_idx_p = -1
                    lam_comp_t = complex("nan")
                    lam_comp_p = complex("nan")
                    det_rel_val = complex("nan")
                    logdet_Iminus_rel_val = complex("nan")
                    phi0_plus_val = complex("nan")
                    phi0_minus_val = complex("nan")
                    lam_cusp_plus_raw = complex("nan")
                    lam_cusp_minus_raw = complex("nan")
                    lam_cusp_plus_riesz = complex("nan")
                    lam_cusp_minus_riesz = complex("nan")
                    lam_cusp_plus_norm = complex("nan")
                    lam_cusp_minus_norm = complex("nan")
                    cusp_gap_plus = float("nan")
                    cusp_gap_minus = float("nan")
                    cusp_sep2_plus = float("nan")
                    cusp_sep2_minus = float("nan")
                    cusp_p0ov_plus = float("nan")
                    cusp_p0ov_minus = float("nan")
                    cusp_proj_idem_plus = float("nan")
                    cusp_proj_idem_minus = float("nan")
                    cusp_proj_comm_plus = float("nan")
                    cusp_proj_comm_minus = float("nan")
                    cusp_proj_trace_re_plus = float("nan")
                    cusp_proj_trace_im_plus = float("nan")
                    cusp_proj_trace_re_minus = float("nan")
                    cusp_proj_trace_im_minus = float("nan")
                    try:
                        _, Lam, Sm = model_fnW(s, cfgW)
                        _, _, S0 = free_fnW(s, cfgW)
                        Srel = (Sm @ np.linalg.inv(S0)).astype(np.complex128)

                        # Incoming-basis conjugation (used both for basis='incoming' observables
                        # and for defining the canonical cusp/constant-mode scalar channel).
                        nb = int(Lam.shape[0])
                        Tin = (Lam + 1j * float(cfgW.eta) * np.eye(nb, dtype=np.complex128)).astype(np.complex128)
                        if float(cfgW.cayley_eps) != 0.0:
                            Tin = (Tin + float(cfgW.cayley_eps) * np.eye(nb, dtype=np.complex128)).astype(np.complex128)
                        M_in = np.linalg.solve(Tin, (Srel @ Tin)).astype(np.complex128)

                        if conv_wind_basis == "incoming":
                            M = M_in
                        else:
                            M = Srel

                        # Determinant/logdet observables for the *full* relative scattering matrix.
                        # Note: det(M) == det(Srel) (similarity invariant in the incoming basis).
                        try:
                            det_rel_val = complex(np.linalg.det(M))
                        except Exception:
                            det_rel_val = complex("nan")
                        try:
                            I = np.eye(int(M.shape[0]), dtype=np.complex128)
                            sign_im, logabs_im = np.linalg.slogdet(I - M)
                            logdet_Iminus_rel_val = complex(np.log(sign_im) + logabs_im)
                        except Exception:
                            logdet_Iminus_rel_val = complex("nan")

                        # Canonical cusp/constant-mode scalar channel:
                        # approximate P0^(2) Π_± by taking the constant vector in the Swap± reduced block.
                        try:
                            b0 = int(cfgW.b // 2)
                            Bp = _swap_block_matrix(M_in, "plus", b0=b0)
                            Bm = _swap_block_matrix(M_in, "minus", b0=b0)
                            v = np.ones((int(b0),), dtype=np.complex128)
                            vH = v.conj().T
                            den = complex(vH @ v)
                            if den != 0:
                                phi0_plus_val = complex((vH @ (Bp @ v)) / den)
                                phi0_minus_val = complex((vH @ (Bm @ v)) / den)

                            # Doc-faithful cusp eigenchannel: overlap-selected eigenvalue + Riesz projector.
                            # We track a continuous branch along the boundary by nearest-neighbor preference,
                            # with closure at the final boundary sample.
                            # Constant-mode line selector in the *current* basis.
                            # In this repo's boundary basis, the physically constant boundary mode corresponds
                            # to the all-ones vector (DFT(e0) in a Fourier-ordered basis).
                            u0 = np.ones((int(b0),), dtype=np.complex128)
                            prefer_plus: complex | None
                            prefer_minus: complex | None
                            if i == 0 or not is_finite_complex(prev_cusp_plus):
                                prefer_plus = None
                            elif i == (len(ptsW) - 1) and is_finite_complex(first_cusp_plus):
                                prefer_plus = complex(first_cusp_plus)
                            else:
                                prefer_plus = complex(prev_cusp_plus)
                            if i == 0 or not is_finite_complex(prev_cusp_minus):
                                prefer_minus = None
                            elif i == (len(ptsW) - 1) and is_finite_complex(first_cusp_minus):
                                prefer_minus = complex(first_cusp_minus)
                            else:
                                prefer_minus = complex(prev_cusp_minus)

                            cup = _cusp_channel_riesz(Bp, u0=u0, prefer=prefer_plus, overlap_floor=0.9, radius_frac=0.45, n_theta=32)
                            cum = _cusp_channel_riesz(Bm, u0=u0, prefer=prefer_minus, overlap_floor=0.9, radius_frac=0.45, n_theta=32)

                            lam_cusp_plus_raw = complex(cup.get("lam_raw", complex("nan")))
                            lam_cusp_minus_raw = complex(cum.get("lam_raw", complex("nan")))
                            lam_cusp_plus_riesz = complex(cup.get("lam_riesz", complex("nan")))
                            lam_cusp_minus_riesz = complex(cum.get("lam_riesz", complex("nan")))

                            cusp_gap_plus = float(cup.get("gap", float("nan")))
                            cusp_gap_minus = float(cum.get("gap", float("nan")))
                            cusp_sep2_plus = float(cup.get("sep2", float("nan")))
                            cusp_sep2_minus = float(cum.get("sep2", float("nan")))
                            cusp_p0ov_plus = float(cup.get("p0_overlap", float("nan")))
                            cusp_p0ov_minus = float(cum.get("p0_overlap", float("nan")))

                            diagp = cup.get("proj_diag", {}) if isinstance(cup.get("proj_diag", {}), dict) else {}
                            diagm = cum.get("proj_diag", {}) if isinstance(cum.get("proj_diag", {}), dict) else {}
                            cusp_proj_idem_plus = float(diagp.get("proj_idem_rel", float("nan")))
                            cusp_proj_comm_plus = float(diagp.get("proj_comm_rel", float("nan")))
                            cusp_proj_trace_re_plus = float(diagp.get("proj_trace_re", float("nan")))
                            cusp_proj_trace_im_plus = float(diagp.get("proj_trace_im", float("nan")))
                            cusp_proj_idem_minus = float(diagm.get("proj_idem_rel", float("nan")))
                            cusp_proj_comm_minus = float(diagm.get("proj_comm_rel", float("nan")))
                            cusp_proj_trace_re_minus = float(diagm.get("proj_trace_re", float("nan")))
                            cusp_proj_trace_im_minus = float(diagm.get("proj_trace_im", float("nan")))

                            if i == 0:
                                first_cusp_plus = complex(lam_cusp_plus_raw)
                                first_cusp_minus = complex(lam_cusp_minus_raw)
                            prev_cusp_plus = complex(lam_cusp_plus_raw)
                            prev_cusp_minus = complex(lam_cusp_minus_raw)
                        except Exception:
                            phi0_plus_val = complex("nan")
                            phi0_minus_val = complex("nan")
                            lam_cusp_plus_raw = complex("nan")
                            lam_cusp_minus_raw = complex("nan")
                            lam_cusp_plus_riesz = complex("nan")
                            lam_cusp_minus_riesz = complex("nan")

                        phi_norm = phi_target(s, primes=cfgW.primes, completion_mode=cfgW.completion_mode, mode=phi_target_mode)
                        phi_sel = phi_target(s, primes=cfgW.primes, completion_mode=cfgW.completion_mode, mode=phi_select_mode)
                        if (
                            (not is_finite_complex(phi_norm))
                            or abs(phi_norm) <= 1e-30
                            or (not is_finite_complex(phi_sel))
                            or abs(phi_sel) <= 1e-30
                        ):
                            raise ValueError("bad phi")

                        try:
                            phi_abs_min = min(phi_abs_min, float(abs(phi_norm)))
                        except Exception:
                            pass

                        b0 = int(cfgW.b // 2)
                        B = _swap_block_matrix(M, "plus" if conv_block_key == "+" else "minus", b0=b0)
                        eigs = np.asarray(np.linalg.eigvals(B), dtype=np.complex128).ravel()
                        if eigs.size == 0:
                            raise ValueError("no eigs")

                        # Tracked branch: initialization by selector, then nearest-neighbor continuation,
                        # with a closure constraint at the final boundary point.
                        if i == 0 or not is_finite_complex(prev_track):
                            j_track, _ = _min_relerr_to_target(eigs, phi_sel)
                        elif i == (len(ptsW) - 1) and is_finite_complex(first_track):
                            j_track = _nearest_neighbor_index(eigs, first_track)
                        else:
                            j_track = _nearest_neighbor_index(eigs, prev_track)
                        lam_raw_track = complex(eigs[int(j_track)])
                        lam_raw_track_val = lam_raw_track
                        if i == 0:
                            first_track = lam_raw_track

                        # Pointwise baseline: always pick best match to selector (no continuation).
                        j_point, _ = _min_relerr_to_target(eigs, phi_sel)
                        lam_raw_point = complex(eigs[int(j_point)])
                        lam_raw_point_val = lam_raw_point

                        # Local spectral gap diagnostics.
                        try:
                            if eigs.size >= 2:
                                diffs_t = np.abs(eigs - lam_raw_track)
                                diffs_t[int(j_track)] = np.inf
                                try:
                                    comp_idx_t = int(np.argmin(diffs_t))
                                    lam_comp_t = complex(eigs[int(comp_idx_t)])
                                except Exception:
                                    comp_idx_t = -1
                                    lam_comp_t = complex("nan")
                                diffs_sorted_t = np.sort(diffs_t[np.isfinite(diffs_t)])
                                if diffs_sorted_t.size >= 1:
                                    gap_t = float(diffs_sorted_t[0])
                                if diffs_sorted_t.size >= 2:
                                    sep2_t = float(diffs_sorted_t[1])
                                if np.isfinite(gap_t):
                                    gap_min_track = min(gap_min_track, gap_t)
                        except Exception:
                            pass
                        try:
                            if eigs.size >= 2:
                                diffs_p = np.abs(eigs - lam_raw_point)
                                diffs_p[int(j_point)] = np.inf
                                try:
                                    comp_idx_p = int(np.argmin(diffs_p))
                                    lam_comp_p = complex(eigs[int(comp_idx_p)])
                                except Exception:
                                    comp_idx_p = -1
                                    lam_comp_p = complex("nan")
                                diffs_sorted_p = np.sort(diffs_p[np.isfinite(diffs_p)])
                                if diffs_sorted_p.size >= 1:
                                    gap_p = float(diffs_sorted_p[0])
                                if diffs_sorted_p.size >= 2:
                                    sep2_p = float(diffs_sorted_p[1])
                                if np.isfinite(gap_p):
                                    gap_min_point = min(gap_min_point, gap_p)
                        except Exception:
                            pass

                        # Jump diagnostics.
                        try:
                            if is_finite_complex(prev_track):
                                jmp_t = float(abs(lam_raw_track - prev_track))
                                jump_max_track = max(jump_max_track, float(jmp_t))
                            if is_finite_complex(prev_point):
                                jmp_p = float(abs(lam_raw_point - prev_point))
                                jump_max_point = max(jump_max_point, float(jmp_p))
                        except Exception:
                            pass

                        # Candidate Tier-2 normalization map applied to the scalar observable.
                        lam_fac = lambda_norm_factor(s, lambda_norm_mode, s_ref=(lambda_norm_sref if lambda_norm_sref is not None else s0))
                        if (not is_finite_complex(lam_fac)) or abs(lam_fac) <= 1e-300:
                            raise ValueError("bad lambda normalization factor")
                        lam_track = complex(lam_raw_track * lam_fac)
                        lam_point = complex(lam_raw_point * lam_fac)

                        # Apply the same normalization-map factor to the cusp eigenchannels.
                        if is_finite_complex(lam_cusp_plus_riesz):
                            lam_cusp_plus_norm = complex(lam_cusp_plus_riesz * lam_fac)
                        if is_finite_complex(lam_cusp_minus_riesz):
                            lam_cusp_minus_norm = complex(lam_cusp_minus_riesz * lam_fac)

                        prev_track = lam_raw_track
                        prev_point = lam_raw_point
                        qs_track.append(lam_track / phi_norm)
                        qs_pointwise.append(lam_point / phi_norm)
                    except Exception:
                        qs_track.append(complex("nan"))
                        qs_pointwise.append(complex("nan"))
                        failures += 1
                        phi_bad += 1

                    det_rel_samples.append(det_rel_val)
                    logdet_Iminus_rel_samples.append(logdet_Iminus_rel_val)
                    phi0_plus_samples.append(phi0_plus_val)
                    phi0_minus_samples.append(phi0_minus_val)

                    lam_cusp_plus_raw_samples.append(lam_cusp_plus_raw)
                    lam_cusp_minus_raw_samples.append(lam_cusp_minus_raw)
                    lam_cusp_plus_riesz_samples.append(lam_cusp_plus_riesz)
                    lam_cusp_minus_riesz_samples.append(lam_cusp_minus_riesz)
                    lam_cusp_plus_norm_samples.append(lam_cusp_plus_norm)
                    lam_cusp_minus_norm_samples.append(lam_cusp_minus_norm)
                    cusp_gap_plus_samples.append(float(cusp_gap_plus) if np.isfinite(cusp_gap_plus) else float("nan"))
                    cusp_gap_minus_samples.append(float(cusp_gap_minus) if np.isfinite(cusp_gap_minus) else float("nan"))
                    cusp_sep2_plus_samples.append(float(cusp_sep2_plus) if np.isfinite(cusp_sep2_plus) else float("nan"))
                    cusp_sep2_minus_samples.append(float(cusp_sep2_minus) if np.isfinite(cusp_sep2_minus) else float("nan"))
                    cusp_p0ov_plus_samples.append(float(cusp_p0ov_plus) if np.isfinite(cusp_p0ov_plus) else float("nan"))
                    cusp_p0ov_minus_samples.append(float(cusp_p0ov_minus) if np.isfinite(cusp_p0ov_minus) else float("nan"))
                    cusp_proj_idem_plus_samples.append(float(cusp_proj_idem_plus) if np.isfinite(cusp_proj_idem_plus) else float("nan"))
                    cusp_proj_idem_minus_samples.append(float(cusp_proj_idem_minus) if np.isfinite(cusp_proj_idem_minus) else float("nan"))
                    cusp_proj_comm_plus_samples.append(float(cusp_proj_comm_plus) if np.isfinite(cusp_proj_comm_plus) else float("nan"))
                    cusp_proj_comm_minus_samples.append(float(cusp_proj_comm_minus) if np.isfinite(cusp_proj_comm_minus) else float("nan"))
                    cusp_proj_trace_re_plus_samples.append(float(cusp_proj_trace_re_plus) if np.isfinite(cusp_proj_trace_re_plus) else float("nan"))
                    cusp_proj_trace_im_plus_samples.append(float(cusp_proj_trace_im_plus) if np.isfinite(cusp_proj_trace_im_plus) else float("nan"))
                    cusp_proj_trace_re_minus_samples.append(float(cusp_proj_trace_re_minus) if np.isfinite(cusp_proj_trace_re_minus) else float("nan"))
                    cusp_proj_trace_im_minus_samples.append(float(cusp_proj_trace_im_minus) if np.isfinite(cusp_proj_trace_im_minus) else float("nan"))

                    gaps_track.append(float(gap_t) if np.isfinite(gap_t) else float("nan"))
                    gaps_point.append(float(gap_p) if np.isfinite(gap_p) else float("nan"))
                    seps2_track.append(float(sep2_t) if np.isfinite(sep2_t) else float("nan"))
                    seps2_point.append(float(sep2_p) if np.isfinite(sep2_p) else float("nan"))
                    lam_raw_track_samples.append(lam_raw_track_val)
                    lam_raw_point_samples.append(lam_raw_point_val)
                    comp_idx_track_samples.append(int(comp_idx_t))
                    comp_idx_point_samples.append(int(comp_idx_p))
                    lam_comp_track_samples.append(lam_comp_t)
                    lam_comp_point_samples.append(lam_comp_p)
                    jumps_track.append(float(jmp_t) if np.isfinite(jmp_t) else float("nan"))
                    jumps_point.append(float(jmp_p) if np.isfinite(jmp_p) else float("nan"))

                if conv_emit_boundary_samples:
                    for i, s in enumerate(ptsW):
                        qtr = qs_track[i] if i < len(qs_track) else complex("nan")
                        qpw = qs_pointwise[i] if i < len(qs_pointwise) else complex("nan")
                        det_rel = det_rel_samples[i] if i < len(det_rel_samples) else complex("nan")
                        logdet_im_rel = logdet_Iminus_rel_samples[i] if i < len(logdet_Iminus_rel_samples) else complex("nan")
                        phi0p = phi0_plus_samples[i] if i < len(phi0_plus_samples) else complex("nan")
                        phi0m = phi0_minus_samples[i] if i < len(phi0_minus_samples) else complex("nan")
                        lcpr = lam_cusp_plus_raw_samples[i] if i < len(lam_cusp_plus_raw_samples) else complex("nan")
                        lcmr = lam_cusp_minus_raw_samples[i] if i < len(lam_cusp_minus_raw_samples) else complex("nan")
                        lcp = lam_cusp_plus_riesz_samples[i] if i < len(lam_cusp_plus_riesz_samples) else complex("nan")
                        lcm = lam_cusp_minus_riesz_samples[i] if i < len(lam_cusp_minus_riesz_samples) else complex("nan")
                        lcpn = lam_cusp_plus_norm_samples[i] if i < len(lam_cusp_plus_norm_samples) else complex("nan")
                        lcmn = lam_cusp_minus_norm_samples[i] if i < len(lam_cusp_minus_norm_samples) else complex("nan")
                        try:
                            qabs = float(abs(qtr)) if is_finite_complex(qtr) else float("nan")
                            logabs = float(np.log(qabs)) if np.isfinite(qabs) and qabs > 0 else float("nan")
                        except Exception:
                            qabs, logabs = float("nan"), float("nan")
                        boundary_rows.append(
                            {
                                "source": ("convergence_winding" if conv_do_winding else "convergence_boundary_samples"),
                                "N": int(N),
                                "basis": str(conv_wind_basis),
                                "block": "plus" if conv_block_key == "+" else "minus",
                                "sigma": float(np.real(s)) if is_finite_complex(s) else float("nan"),
                                "t": float(np.imag(s)) if is_finite_complex(s) else float("nan"),
                                "side": _boundary_side(s),
                                "idx": int(i),
                                "n_edge": int(conv_wind_n_edge),
                                "gap_track": float(gaps_track[i]) if i < len(gaps_track) else float("nan"),
                                "sep_track": float(gaps_track[i]) if i < len(gaps_track) else float("nan"),
                                "sep2_track": float(seps2_track[i]) if i < len(seps2_track) else float("nan"),
                                "gap_pointwise": float(gaps_point[i]) if i < len(gaps_point) else float("nan"),
                                "sep_pointwise": float(gaps_point[i]) if i < len(gaps_point) else float("nan"),
                                "sep2_pointwise": float(seps2_point[i]) if i < len(seps2_point) else float("nan"),
                                "lam_raw_track_re": float(np.real(lam_raw_track_samples[i])) if i < len(lam_raw_track_samples) and is_finite_complex(lam_raw_track_samples[i]) else float("nan"),
                                "lam_raw_track_im": float(np.imag(lam_raw_track_samples[i])) if i < len(lam_raw_track_samples) and is_finite_complex(lam_raw_track_samples[i]) else float("nan"),
                                "lam_comp_track_idx": int(comp_idx_track_samples[i]) if i < len(comp_idx_track_samples) else int(-1),
                                "lam_comp_track_re": float(np.real(lam_comp_track_samples[i])) if i < len(lam_comp_track_samples) and is_finite_complex(lam_comp_track_samples[i]) else float("nan"),
                                "lam_comp_track_im": float(np.imag(lam_comp_track_samples[i])) if i < len(lam_comp_track_samples) and is_finite_complex(lam_comp_track_samples[i]) else float("nan"),
                                "lam_raw_point_re": float(np.real(lam_raw_point_samples[i])) if i < len(lam_raw_point_samples) and is_finite_complex(lam_raw_point_samples[i]) else float("nan"),
                                "lam_raw_point_im": float(np.imag(lam_raw_point_samples[i])) if i < len(lam_raw_point_samples) and is_finite_complex(lam_raw_point_samples[i]) else float("nan"),
                                "lam_comp_point_idx": int(comp_idx_point_samples[i]) if i < len(comp_idx_point_samples) else int(-1),
                                "lam_comp_point_re": float(np.real(lam_comp_point_samples[i])) if i < len(lam_comp_point_samples) and is_finite_complex(lam_comp_point_samples[i]) else float("nan"),
                                "lam_comp_point_im": float(np.imag(lam_comp_point_samples[i])) if i < len(lam_comp_point_samples) and is_finite_complex(lam_comp_point_samples[i]) else float("nan"),
                                "jump_track": float(jumps_track[i]) if i < len(jumps_track) else float("nan"),
                                "jump_pointwise": float(jumps_point[i]) if i < len(jumps_point) else float("nan"),
                                "det_rel_re": float(np.real(det_rel)) if is_finite_complex(det_rel) else float("nan"),
                                "det_rel_im": float(np.imag(det_rel)) if is_finite_complex(det_rel) else float("nan"),
                                "logdet_Iminus_rel_re": float(np.real(logdet_im_rel)) if is_finite_complex(logdet_im_rel) else float("nan"),
                                "logdet_Iminus_rel_im": float(np.imag(logdet_im_rel)) if is_finite_complex(logdet_im_rel) else float("nan"),
                                "phi0_plus_re": float(np.real(phi0p)) if is_finite_complex(phi0p) else float("nan"),
                                "phi0_plus_im": float(np.imag(phi0p)) if is_finite_complex(phi0p) else float("nan"),
                                "phi0_minus_re": float(np.real(phi0m)) if is_finite_complex(phi0m) else float("nan"),
                                "phi0_minus_im": float(np.imag(phi0m)) if is_finite_complex(phi0m) else float("nan"),

                                "lambda_cusp_plus_raw_re": float(np.real(lcpr)) if is_finite_complex(lcpr) else float("nan"),
                                "lambda_cusp_plus_raw_im": float(np.imag(lcpr)) if is_finite_complex(lcpr) else float("nan"),
                                "lambda_cusp_minus_raw_re": float(np.real(lcmr)) if is_finite_complex(lcmr) else float("nan"),
                                "lambda_cusp_minus_raw_im": float(np.imag(lcmr)) if is_finite_complex(lcmr) else float("nan"),
                                "lambda_cusp_plus_re": float(np.real(lcp)) if is_finite_complex(lcp) else float("nan"),
                                "lambda_cusp_plus_im": float(np.imag(lcp)) if is_finite_complex(lcp) else float("nan"),
                                "lambda_cusp_minus_re": float(np.real(lcm)) if is_finite_complex(lcm) else float("nan"),
                                "lambda_cusp_minus_im": float(np.imag(lcm)) if is_finite_complex(lcm) else float("nan"),
                                "lambda_cusp_plus_norm_re": float(np.real(lcpn)) if is_finite_complex(lcpn) else float("nan"),
                                "lambda_cusp_plus_norm_im": float(np.imag(lcpn)) if is_finite_complex(lcpn) else float("nan"),
                                "lambda_cusp_minus_norm_re": float(np.real(lcmn)) if is_finite_complex(lcmn) else float("nan"),
                                "lambda_cusp_minus_norm_im": float(np.imag(lcmn)) if is_finite_complex(lcmn) else float("nan"),

                                "cusp_gap_plus": float(cusp_gap_plus_samples[i]) if i < len(cusp_gap_plus_samples) else float("nan"),
                                "cusp_gap_minus": float(cusp_gap_minus_samples[i]) if i < len(cusp_gap_minus_samples) else float("nan"),
                                "cusp_sep2_plus": float(cusp_sep2_plus_samples[i]) if i < len(cusp_sep2_plus_samples) else float("nan"),
                                "cusp_sep2_minus": float(cusp_sep2_minus_samples[i]) if i < len(cusp_sep2_minus_samples) else float("nan"),
                                "cusp_p0_overlap_plus": float(cusp_p0ov_plus_samples[i]) if i < len(cusp_p0ov_plus_samples) else float("nan"),
                                "cusp_p0_overlap_minus": float(cusp_p0ov_minus_samples[i]) if i < len(cusp_p0ov_minus_samples) else float("nan"),
                                "cusp_proj_idem_rel_plus": float(cusp_proj_idem_plus_samples[i]) if i < len(cusp_proj_idem_plus_samples) else float("nan"),
                                "cusp_proj_idem_rel_minus": float(cusp_proj_idem_minus_samples[i]) if i < len(cusp_proj_idem_minus_samples) else float("nan"),
                                "cusp_proj_comm_rel_plus": float(cusp_proj_comm_plus_samples[i]) if i < len(cusp_proj_comm_plus_samples) else float("nan"),
                                "cusp_proj_comm_rel_minus": float(cusp_proj_comm_minus_samples[i]) if i < len(cusp_proj_comm_minus_samples) else float("nan"),
                                "cusp_proj_trace_re_plus": float(cusp_proj_trace_re_plus_samples[i]) if i < len(cusp_proj_trace_re_plus_samples) else float("nan"),
                                "cusp_proj_trace_im_plus": float(cusp_proj_trace_im_plus_samples[i]) if i < len(cusp_proj_trace_im_plus_samples) else float("nan"),
                                "cusp_proj_trace_re_minus": float(cusp_proj_trace_re_minus_samples[i]) if i < len(cusp_proj_trace_re_minus_samples) else float("nan"),
                                "cusp_proj_trace_im_minus": float(cusp_proj_trace_im_minus_samples[i]) if i < len(cusp_proj_trace_im_minus_samples) else float("nan"),

                                "q_track_re": float(np.real(qtr)) if is_finite_complex(qtr) else float("nan"),
                                "q_track_im": float(np.imag(qtr)) if is_finite_complex(qtr) else float("nan"),
                                "q_point_re": float(np.real(qpw)) if is_finite_complex(qpw) else float("nan"),
                                "q_point_im": float(np.imag(qpw)) if is_finite_complex(qpw) else float("nan"),
                                "absQ_track": float(qabs),
                                "logabsQ_track": float(logabs),
                            }
                        )

                if conv_do_winding:
                    w_track, wfail_track = _winding_count_from_samples(qs_track)
                    w_point, wfail_point = _winding_count_from_samples(qs_pointwise)
                    wind_rows.append(
                        {
                            "source": "convergence_sweep",
                            "N": int(N),
                            "basis": str(conv_wind_basis),
                            "block": "plus" if conv_block_key == "+" else "minus",
                            "sigma_min": float(convergence_rect[0]),
                            "sigma_max": float(convergence_rect[1]),
                            "t_min": float(convergence_rect[2]),
                            "t_max": float(convergence_rect[3]),
                            "n_edge": int(conv_wind_n_edge),
                            "wind_track": float(w_track),
                            "wind_track_abs": float(abs(w_track)) if np.isfinite(w_track) else float("nan"),
                            "wind_pointwise": float(w_point),
                            "wind_pointwise_abs": float(abs(w_point)) if np.isfinite(w_point) else float("nan"),
                            "gap_min_track": float(gap_min_track) if np.isfinite(gap_min_track) and gap_min_track < float("inf") else float("nan"),
                            "gap_min_pointwise": float(gap_min_point) if np.isfinite(gap_min_point) and gap_min_point < float("inf") else float("nan"),
                            "jump_max_track": float(jump_max_track) if np.isfinite(jump_max_track) else float("nan"),
                            "jump_max_pointwise": float(jump_max_point) if np.isfinite(jump_max_point) else float("nan"),
                            "failures": int(failures + wfail_track + wfail_point),
                            "phi_abs_min": float(phi_abs_min) if np.isfinite(phi_abs_min) else float("nan"),
                            "phi_bad": int(phi_bad),
                            "eta": float(cfgW.eta),
                            "schur_jitter": float(cfgW.jitter),
                            "cayley_eps": float(cfgW.cayley_eps),
                        }
                    )

            if conv_do_winding and wind_rows:
                pd.DataFrame(wind_rows).to_csv(os.path.join(out_dir, conv_wind_out), index=False)
            if conv_emit_boundary_samples and boundary_rows:
                pd.DataFrame(boundary_rows).to_csv(os.path.join(out_dir, conv_boundary_samples_out), index=False)

        # Ensure convergence-only runs still satisfy the run contract and remain auditable.
        try:
            lines: list[str] = []
            lines.append("# Experiment E: Q_lambda convergence sweep\n")
            lines.append("This run was executed with `do_convergence_sweep_only: true`.\n")
            lines.append("## Parameters\n")
            lines.append(f"- Ns: {list(map(int, Ns))}")
            lines.append(f"- rectangle: {tuple(map(float, convergence_rect))}")
            lines.append(f"- n_sigma, n_t: {int(convergence_n_sigma)}, {int(convergence_n_t)}")
            lines.append(f"- eta: {float(eta)}  |  schur_jitter: {float(jitter)}  |  cayley_eps: {float(cayley_eps)}")
            lines.append(f"- boundary_frac: {float(boundary_frac)}  |  boundary_b0_delta: {int(boundary_b0_delta)}")
            lines.append(f"- phi_target_mode: {str(phi_target_mode)}")
            lines.append(f"- phi_select_mode: {str(phi_select_mode)}\n")
            lines.append(f"- lambda_norm_mode: {str(lambda_norm_mode)}\n")
            if bool(do_critical_line_fingerprint):
                lines.append("- do_critical_line_fingerprint: true")
                lines.append(f"- critical_line_sigma: {float(critical_line_sigma)}")
                lines.append(f"- critical_line_t_min, critical_line_t_max: {float(critical_line_t_min)}, {float(critical_line_t_max)}")
                lines.append(f"- critical_line_nt: {int(critical_line_nt)}")
                lines.append(f"- critical_line_basis: {str(critical_line_basis)}  |  critical_line_block: {str(critical_line_block)}\n")
                if bool(critical_line_eval_candidates):
                    lines.append(f"- critical_line_eval_candidates: true  |  candidates: {critical_line_candidates}\n")
            lines.append("## Artifacts\n")
            lines.append("- `E_Qlambda_convergence_sweep.csv` — interior-grid agreement and gap diagnostics (per basis/block)")
            lines.append("- `E_Qlambda_convergence_fit.csv` — simple trend fits of key error metrics vs N")
            if conv_do_magphase_diag:
                lines.append("- `E_Qlambda_convergence_magphase.csv` — magnitude/phase diagnostics (Q vs |g_infty| tracking)")
            if conv_do_winding:
                lines.append(f"- `{str(conv_wind_out)}` — boundary winding sanity check per N")
            if bool(do_critical_line_fingerprint):
                lines.append(f"- `{str(critical_line_points_out_csv)}` — dense critical-line fingerprint points (per t)")
                lines.append(f"- `{str(critical_line_summary_out_csv)}` — critical-line fingerprint summary (per N)")
                if bool(critical_line_eval_candidates):
                    lines.append(f"- `{str(critical_line_candidates_out_csv)}` — critical-line candidate table (per N, per candidate)")
            Path(os.path.join(out_dir, "E_summary.md")).write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception:
            pass

        if bool(config.get("run_contract_enforce", True)):
            _validate_run_contract(out_dir, config)

        return 0

    def _pole_line_scan() -> None:
        """Scan t ↦ smin(Λ+iI) at fixed sigma for multiple N.

        Records both model and free/reference (primes=[]) Cayley denominators so we can tell
        whether near-kernel directions are cancelable by free normalization.
        """

        if not do_pole_line_scan:
            return

        if pole_line_nt < 2:
            raise ValueError("pole_line_nt must be >= 2")
        if pole_line_t_max <= pole_line_t_min:
            raise ValueError("pole_line_t_max must exceed pole_line_t_min")
        if pole_line_basis not in {"incoming", "rel"}:
            raise ValueError("pole_line_basis must be 'incoming' or 'rel'")
        if pole_line_block not in {"plus", "minus", "+", "-", "p", "m"}:
            raise ValueError("pole_line_block must be 'plus' or 'minus'")

        t_vals = np.linspace(pole_line_t_min, pole_line_t_max, int(pole_line_nt), dtype=float)
        sigma = float(pole_line_sigma)
        block = "plus" if pole_line_block in {"plus", "+", "p"} else "minus"

        rows: list[dict] = []
        for N in Ns:
            cfgN = _mk_cfg(int(N))
            b0 = int(cfgN.b // 2)

            prev_lam: complex = complex("nan")
            prev_vec: np.ndarray | None = None

            if print_progress:
                print(
                    f"[pole_line_scan] N={int(N)} sigma={sigma} t=[{pole_line_t_min},{pole_line_t_max}] nt={int(pole_line_nt)} basis={pole_line_basis} block={block}"
                )

            for i, t in enumerate(t_vals):
                if print_progress and progress_every and (i % int(progress_every) == 0 or i == len(t_vals) - 1):
                    print(f"[pole_line_scan] N={int(N)} i={i}/{len(t_vals)-1}")

                s = complex(sigma, float(t))
                try:
                    _, Lam, Sm = _model_mats(s, cfgN)
                    _, Lam0, S0 = _free_mats(s, cfgN)
                    Srel = (Sm @ np.linalg.inv(S0)).astype(np.complex128)

                    nb = int(Lam.shape[0])
                    Tin = (Lam + 1j * np.eye(nb, dtype=np.complex128)).astype(np.complex128)
                    Sin = np.linalg.solve(Tin, (Srel @ Tin)).astype(np.complex128)

                    nb0 = int(Lam0.shape[0])
                    Tin0 = (Lam0 + 1j * np.eye(nb0, dtype=np.complex128)).astype(np.complex128)
                except Exception:
                    continue

                phi = (
                    phi_target(s, primes=cfgN.primes, completion_mode=cfgN.completion_mode, mode=phi_target_mode)
                    if pole_line_include_phi
                    else complex("nan")
                )
                abs_phi = float(abs(phi)) if is_finite_complex(phi) else float("nan")

                # Cayley denominator sentinels (model)
                smin_Tin, smax_Tin, cond_Tin, log10_smin_Tin, log10_smax_Tin = _svd_sentinels(Tin)
                log10_det_Tin = float(_log10_abs_det(Tin))

                # Cayley denominator sentinels (free/reference)
                if pole_line_include_free:
                    smin_Tin0, smax_Tin0, cond_Tin0, log10_smin_Tin0, log10_smax_Tin0 = _svd_sentinels(Tin0)
                    log10_det_Tin0 = float(_log10_abs_det(Tin0))
                else:
                    smin_Tin0 = smax_Tin0 = cond_Tin0 = log10_smin_Tin0 = log10_smax_Tin0 = log10_det_Tin0 = float("nan")

                M = Sin if pole_line_basis == "incoming" else Srel
                try:
                    B = _swap_block_matrix(M, block, b0=b0)
                    evals, evecs = np.linalg.eig(B)
                except Exception:
                    continue
                if evals.size == 0:
                    continue

                evals1 = np.asarray(evals, dtype=np.complex128).ravel()
                abs_eigs = np.abs(evals1)
                j_max = int(np.argmax(abs_eigs))
                lam_max = complex(evals1[j_max])
                q_max = lam_max / phi if (pole_line_include_phi and is_finite_complex(phi) and abs(phi) > 1e-30) else complex("nan")

                if pole_line_include_phi and is_finite_complex(phi):
                    j_best, relerr_best = _min_relerr_to_target(evals1, phi)
                    lam_best = complex(evals1[int(j_best)])
                    q_best = lam_best / phi if abs(phi) > 1e-30 else complex("nan")
                else:
                    j_best, relerr_best = 0, float("nan")
                    lam_best = complex("nan")
                    q_best = complex("nan")

                if not is_finite_complex(prev_lam):
                    j = int(j_best) if np.isfinite(relerr_best) else 0
                else:
                    j = _nearest_neighbor_index(evals1, prev_lam)

                lam = complex(evals1[int(j)])
                prev_lam = lam
                gap = _spectral_gap(evals1, int(j))

                overlap = float("nan")
                try:
                    v = np.asarray(evecs)[:, int(j)].astype(np.complex128)
                    nv = float(np.linalg.norm(v))
                    v = v / nv if nv > 0 else v
                    if prev_vec is not None:
                        overlap = float(abs(np.vdot(prev_vec, v)))
                        overlap = float(min(1.0, max(0.0, overlap)))
                    prev_vec = v
                except Exception:
                    prev_vec = None

                q = lam / phi if (pole_line_include_phi and is_finite_complex(phi) and abs(phi) > 1e-30) else complex("nan")

                rows.append(
                    {
                        "N": int(N),
                        "basis": str(pole_line_basis),
                        "block": str(block),
                        "sigma": float(sigma),
                        "t": float(t),
                        "s_re": float(np.real(s)),
                        "s_im": float(np.imag(s)),
                        "log10_smin_Lam_plus_iI": float(log10_smin_Tin),
                        "log10_smax_Lam_plus_iI": float(log10_smax_Tin),
                        "smin_Lam_plus_iI": float(smin_Tin) if np.isfinite(smin_Tin) else float("nan"),
                        "smax_Lam_plus_iI": float(smax_Tin) if np.isfinite(smax_Tin) else float("nan"),
                        "cond_Lam_plus_iI": float(cond_Tin),
                        "log10_abs_det_Lam_plus_iI": float(log10_det_Tin),
                        "log10_smin_Lam0_plus_iI": float(log10_smin_Tin0),
                        "log10_smax_Lam0_plus_iI": float(log10_smax_Tin0),
                        "smin_Lam0_plus_iI": float(smin_Tin0) if np.isfinite(smin_Tin0) else float("nan"),
                        "smax_Lam0_plus_iI": float(smax_Tin0) if np.isfinite(smax_Tin0) else float("nan"),
                        "cond_Lam0_plus_iI": float(cond_Tin0),
                        "log10_abs_det_Lam0_plus_iI": float(log10_det_Tin0),
                        "lam_re": float(np.real(lam)),
                        "lam_im": float(np.imag(lam)),
                        "abs_lam": float(abs(lam)) if is_finite_complex(lam) else float("nan"),
                        "relerr_best_to_phi": float(relerr_best),
                        "lam_best_re": float(np.real(lam_best)),
                        "lam_best_im": float(np.imag(lam_best)),
                        "abs_lam_best": float(abs(lam_best)) if is_finite_complex(lam_best) else float("nan"),
                        "abs_Qlambda_best": float(abs(q_best)) if is_finite_complex(q_best) else float("nan"),
                        "lam_max_re": float(np.real(lam_max)),
                        "lam_max_im": float(np.imag(lam_max)),
                        "abs_lam_max": float(abs(lam_max)) if is_finite_complex(lam_max) else float("nan"),
                        "abs_Qlambda_max": float(abs(q_max)) if is_finite_complex(q_max) else float("nan"),
                        "spectral_gap": float(gap),
                        "evec_overlap": float(overlap),
                        "abs_phi": float(abs_phi),
                        "abs_Qlambda": float(abs(q)) if is_finite_complex(q) else float("nan"),
                    }
                )

        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(out_dir, "E_pole_line_scan.csv"), index=False)

    _pole_line_scan()

    def _dip_atlas_and_optional_winding() -> None:
        if not do_dip_atlas:
            return

        if dip_n_line < 50:
            raise ValueError("dip_n_line should be >= 50")
        if dip_basis not in {"incoming", "rel"}:
            raise ValueError("dip_basis must be 'incoming' or 'rel'")
        if dip_block not in {"plus", "minus", "+", "-", "p", "m"}:
            raise ValueError("dip_block must be 'plus' or 'minus'")

        block = "plus" if dip_block in {"plus", "+", "p"} else "minus"
        sigma_min, sigma_max, t_min, t_max = dip_rect

        # clamp line locations into the rectangle
        sig_lines = [float(x) for x in dip_sigma_lines if sigma_min <= float(x) <= sigma_max]
        t_lines = [float(x) for x in dip_t_lines if t_min <= float(x) <= t_max]
        if not sig_lines:
            sig_lines = [float((sigma_min + sigma_max) / 2.0)]
        if not t_lines:
            t_lines = [float((t_min + t_max) / 2.0)]

        # Precompute sample grids
        t_grid_line = np.linspace(t_min, t_max, int(dip_n_line), dtype=float)
        sigma_grid_line = np.linspace(sigma_min, sigma_max, int(dip_n_line), dtype=float)

        point_rows: list[dict] = []
        line_rows: list[dict] = []

        def _eval_point(N: int, s: complex) -> tuple[float, float, float, float, float, float]:
            """Return (smin_Tin, cond_Tin, absQ_max, absLam_max, absPhi, absQ_track-ish).

            absQ_max is computed over eigenvalues of the Swap± block, not relying on branch continuation.
            """

            cfgN = _mk_cfg(int(N))
            try:
                _, Lam, Sm = _model_mats(s, cfgN)
                _, _, S0 = _free_mats(s, cfgN)
                Srel = (Sm @ np.linalg.inv(S0)).astype(np.complex128)
                M = Srel
                if dip_basis == "incoming":
                    nb = int(Lam.shape[0])
                    Tin_eta = (Lam + 1j * float(cfgN.eta) * np.eye(nb, dtype=np.complex128)).astype(np.complex128)
                    M = np.linalg.solve(Tin_eta, (Srel @ Tin_eta)).astype(np.complex128)
                else:
                    nb = int(Lam.shape[0])
                    Tin_eta = (Lam + 1j * float(cfgN.eta) * np.eye(nb, dtype=np.complex128)).astype(np.complex128)

                smin_Tin, _, cond_Tin, _, _ = _svd_sentinels(Tin_eta)

                phi = phi_target(s, primes=cfgN.primes, completion_mode=cfgN.completion_mode, mode=phi_target_mode)
                abs_phi = float(abs(phi)) if is_finite_complex(phi) else float("nan")
                if (not is_finite_complex(phi)) or abs(phi) <= 1e-30:
                    return float(smin_Tin), float(cond_Tin), float("nan"), float("nan"), float(abs_phi), float("nan")

                b0 = int(cfgN.b // 2)
                B = _swap_block_matrix(M, block, b0=b0)
                eigs = np.linalg.eigvals(B)
                eigs1 = np.asarray(eigs, dtype=np.complex128).ravel()
                if eigs1.size == 0:
                    return float(smin_Tin), float(cond_Tin), float("nan"), float("nan"), float(abs_phi), float("nan")

                abs_lams = np.abs(eigs1)
                j_max = int(np.argmax(abs_lams))
                lam_max = complex(eigs1[j_max])
                abs_lam_max = float(abs(lam_max))
                abs_q_max = float(np.max(np.abs(eigs1 / phi)))

                # also report a "closest-to-phi" quotient magnitude (not continued)
                j_best, _ = _min_relerr_to_target(eigs1, phi)
                lam_best = complex(eigs1[int(j_best)])
                abs_q_best = float(abs(lam_best / phi))

                return float(smin_Tin), float(cond_Tin), float(abs_q_max), float(abs_lam_max), float(abs_phi), float(abs_q_best)
            except Exception:
                return float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

        # Scan vertical lines: sigma fixed, t varies
        for N in Ns:
            for sigma0 in sig_lines:
                if print_progress:
                    print(f"[dip_atlas] N={int(N)} vertical sigma={sigma0:.6g}")
                smins: list[float] = []
                qs: list[float] = []
                for t in t_grid_line:
                    s = complex(float(sigma0), float(t))
                    smin_Tin, cond_Tin, abs_q_max, abs_lam_max, abs_phi, abs_q_best = _eval_point(int(N), s)
                    smins.append(smin_Tin)
                    qs.append(abs_q_max)
                    point_rows.append(
                        {
                            "N": int(N),
                            "line_type": "vertical",
                            "line_value": float(sigma0),
                            "basis": str(dip_basis),
                            "block": str(block),
                            "sigma": float(np.real(s)),
                            "t": float(np.imag(s)),
                            "smin_Lam_plus_iEtaI": float(smin_Tin),
                            "cond_Lam_plus_iEtaI": float(cond_Tin),
                            "abs_Qlambda_max": float(abs_q_max),
                            "abs_lam_max": float(abs_lam_max),
                            "abs_phi": float(abs_phi),
                            "abs_Qlambda_best": float(abs_q_best),
                        }
                    )

                a = np.asarray(smins, dtype=float)
                b = np.asarray(qs, dtype=float)
                if np.isfinite(a).any():
                    jmin = int(np.nanargmin(a))
                    min_smin = float(a[jmin])
                    t_at_min = float(t_grid_line[jmin])
                else:
                    min_smin = float("nan")
                    t_at_min = float("nan")
                if np.isfinite(b).any():
                    jmax = int(np.nanargmax(b))
                    max_absq = float(b[jmax])
                    t_at_maxq = float(t_grid_line[jmax])
                else:
                    max_absq = float("nan")
                    t_at_maxq = float("nan")
                line_rows.append(
                    {
                        "N": int(N),
                        "line_type": "vertical",
                        "line_value": float(sigma0),
                        "basis": str(dip_basis),
                        "block": str(block),
                        "rect_sigma_min": float(sigma_min),
                        "rect_sigma_max": float(sigma_max),
                        "rect_t_min": float(t_min),
                        "rect_t_max": float(t_max),
                        "min_smin": float(min_smin),
                        "t_at_min_smin": float(t_at_min),
                        "max_absQ": float(max_absq),
                        "t_at_max_absQ": float(t_at_maxq),
                    }
                )

        # Scan horizontal lines: t fixed, sigma varies
        for N in Ns:
            for t0 in t_lines:
                if print_progress:
                    print(f"[dip_atlas] N={int(N)} horizontal t={t0:.6g}")
                smins: list[float] = []
                qs: list[float] = []
                for sigma0 in sigma_grid_line:
                    s = complex(float(sigma0), float(t0))
                    smin_Tin, cond_Tin, abs_q_max, abs_lam_max, abs_phi, abs_q_best = _eval_point(int(N), s)
                    smins.append(smin_Tin)
                    qs.append(abs_q_max)
                    point_rows.append(
                        {
                            "N": int(N),
                            "line_type": "horizontal",
                            "line_value": float(t0),
                            "basis": str(dip_basis),
                            "block": str(block),
                            "sigma": float(np.real(s)),
                            "t": float(np.imag(s)),
                            "smin_Lam_plus_iEtaI": float(smin_Tin),
                            "cond_Lam_plus_iEtaI": float(cond_Tin),
                            "abs_Qlambda_max": float(abs_q_max),
                            "abs_lam_max": float(abs_lam_max),
                            "abs_phi": float(abs_phi),
                            "abs_Qlambda_best": float(abs_q_best),
                        }
                    )

                a = np.asarray(smins, dtype=float)
                b = np.asarray(qs, dtype=float)
                if np.isfinite(a).any():
                    jmin = int(np.nanargmin(a))
                    min_smin = float(a[jmin])
                    sigma_at_min = float(sigma_grid_line[jmin])
                else:
                    min_smin = float("nan")
                    sigma_at_min = float("nan")
                if np.isfinite(b).any():
                    jmax = int(np.nanargmax(b))
                    max_absq = float(b[jmax])
                    sigma_at_maxq = float(sigma_grid_line[jmax])
                else:
                    max_absq = float("nan")
                    sigma_at_maxq = float("nan")
                line_rows.append(
                    {
                        "N": int(N),
                        "line_type": "horizontal",
                        "line_value": float(t0),
                        "basis": str(dip_basis),
                        "block": str(block),
                        "rect_sigma_min": float(sigma_min),
                        "rect_sigma_max": float(sigma_max),
                        "rect_t_min": float(t_min),
                        "rect_t_max": float(t_max),
                        "min_smin": float(min_smin),
                        "sigma_at_min_smin": float(sigma_at_min),
                        "max_absQ": float(max_absq),
                        "sigma_at_max_absQ": float(sigma_at_maxq),
                    }
                )

        if point_rows:
            pd.DataFrame(point_rows).to_csv(os.path.join(out_dir, "E_dip_atlas_points.csv"), index=False)
        if line_rows:
            pd.DataFrame(line_rows).to_csv(os.path.join(out_dir, "E_dip_atlas_lines.csv"), index=False)

        # Suggest a conservative safe rectangle by removing neighborhoods where smin dips below threshold.
        if not dip_suggest_and_wind:
            return
        if not point_rows:
            return

        dfp = pd.DataFrame(point_rows)
        dfp = dfp[np.isfinite(dfp["smin_Lam_plus_iEtaI"])].copy()
        if dfp.empty:
            return

        bad = dfp[dfp["smin_Lam_plus_iEtaI"] < float(dip_smin_thresh)].copy()
        suggested = {
            "dip_rect": {"sigma_min": float(sigma_min), "sigma_max": float(sigma_max), "t_min": float(t_min), "t_max": float(t_max)},
            "smin_thresh": float(dip_smin_thresh),
            "margin_t": float(dip_margin_t),
            "margin_sigma": float(dip_margin_sigma),
            "bad_count": int(len(bad)),
        }

        # If no bad points, we can just reuse the original rectangle.
        if bad.empty:
            rect_suggest = (float(sigma_min), float(sigma_max), float(t_min), float(t_max))
            suggested["suggested_rect"] = {"sigma_min": rect_suggest[0], "sigma_max": rect_suggest[1], "t_min": rect_suggest[2], "t_max": rect_suggest[3]}
            write_json(os.path.join(out_dir, "E_dip_atlas_suggested_rect.json"), suggested)
        else:
            # Build safe masks on the sampled grids using the vertical/horizontal line samples.
            # We choose product intervals: a safe t-interval where all vertical lines across all N are >= thresh,
            # and a safe sigma-interval where all horizontal lines across all N are >= thresh.
            safe_t = np.ones_like(t_grid_line, dtype=bool)
            for t in t_grid_line:
                m = bad[np.abs(bad["t"] - float(t)) <= float(dip_margin_t)]
                if len(m) > 0:
                    safe_t[np.where(t_grid_line == t)[0][0]] = False

            # Find longest contiguous safe interval in t
            best_len = 0
            best_i0 = 0
            best_i1 = 0
            i0 = None
            for i, ok in enumerate(safe_t.tolist() + [False]):
                if ok and i0 is None:
                    i0 = i
                if (not ok) and i0 is not None:
                    i1 = i - 1
                    if (i1 - i0 + 1) > best_len:
                        best_len = (i1 - i0 + 1)
                        best_i0, best_i1 = i0, i1
                    i0 = None
            if best_len < 2:
                return
            t_s_min = float(t_grid_line[best_i0])
            t_s_max = float(t_grid_line[best_i1])

            safe_sigma = np.ones_like(sigma_grid_line, dtype=bool)
            for sig in sigma_grid_line:
                m = bad[np.abs(bad["sigma"] - float(sig)) <= float(dip_margin_sigma)]
                if len(m) > 0:
                    safe_sigma[np.where(sigma_grid_line == sig)[0][0]] = False

            best_len = 0
            best_i0 = 0
            best_i1 = 0
            i0 = None
            for i, ok in enumerate(safe_sigma.tolist() + [False]):
                if ok and i0 is None:
                    i0 = i
                if (not ok) and i0 is not None:
                    i1 = i - 1
                    if (i1 - i0 + 1) > best_len:
                        best_len = (i1 - i0 + 1)
                        best_i0, best_i1 = i0, i1
                    i0 = None
            if best_len < 2:
                return
            sig_s_min = float(sigma_grid_line[best_i0])
            sig_s_max = float(sigma_grid_line[best_i1])

            rect_suggest = (sig_s_min, sig_s_max, t_s_min, t_s_max)
            suggested["suggested_rect"] = {"sigma_min": rect_suggest[0], "sigma_max": rect_suggest[1], "t_min": rect_suggest[2], "t_max": rect_suggest[3]}
            write_json(os.path.join(out_dir, "E_dip_atlas_suggested_rect.json"), suggested)

        # If we got a suggested rectangle, run winding on it.
        try:
            sug = json.loads((Path(out_dir) / "E_dip_atlas_suggested_rect.json").read_text(encoding="utf-8"))
            rr = sug.get("suggested_rect", None)
            if rr is None:
                return
            rect2 = (float(rr["sigma_min"]), float(rr["sigma_max"]), float(rr["t_min"]), float(rr["t_max"]))
        except Exception:
            return

        if do_arg_lambda_branch and str(bulk_mode).lower().strip() == "two_channel_symmetric":
            rows2 = _compute_q_lambda_winding_for_rect(rect2, int(dip_suggest_wind_n_edge), source_label="dip_suggested")
            pd.DataFrame(rows2).to_csv(os.path.join(out_dir, "E_divisor_arg_principle_Q_lambda_dip_suggested.csv"), index=False)

            _compute_q_lambda_value_agreement_for_rect(
                rect2,
                int(dip_suggest_interior_n_sigma),
                int(dip_suggest_interior_n_t),
                out_csv_name="E_Qlambda_value_agreement_dip_suggested.csv",
                source_label="dip_suggested",
            )

    # H1 (holomorphic dependence) proxy: scan code for explicit conjugation of s
    bulk_src = (REPO_ROOT / "src" / "bulk.py").read_text(encoding="utf-8")
    bulk_src_nc = "\n".join([ln for ln in bulk_src.splitlines() if not ln.strip().startswith("#")])
    h1_flags = {
        "bulk_mode": str(bulk_mode),
        "mentions_conj_of_s": bool(re.search(r"conj\(s\)|conjugate\(s\)|np\.conj\(s\)", bulk_src_nc)),
        "mentions_s_star": "s*" in bulk_src,
        "completion_mode": completion_mode,
    }
    write_json(os.path.join(out_dir, "E_H1_holomorphic_scan.json"), h1_flags)

    # Normalization constant C(N) at s0
    C_per_N: dict[int, complex] = {}
    for N in Ns:
        cfgN = _mk_cfg(int(N))
        Sm0 = _model_detS_norm(s0, cfgN)
        target0 = phi_modular(s0)
        C_per_N[int(N)] = target0 / Sm0 if Sm0 != 0 else complex("nan")
    write_json(
        os.path.join(out_dir, "E_normalization.json"),
        {
            "s0": {"re": float(np.real(s0)), "im": float(np.imag(s0))},
            "C_per_N": {str(k): {"re": float(np.real(v)), "im": float(np.imag(v))} for k, v in C_per_N.items()},
        },
    )

    # Duality candidates D.
    # For one-channel: use simple involutions that preserve boundary/interior splitting.
    # For two-channel: include channel swap, which is the intended bulk duality there.
    def _duality_candidates(N: int, b: int) -> list[tuple[str, np.ndarray]]:
        mode = str(bulk_mode).lower().strip()
        if mode == "two_channel_symmetric":
            base_n = int(N // 2)
            if 2 * base_n != int(N):
                raise ValueError("two_channel_symmetric expects even N")

            I2 = np.eye(int(N), dtype=np.complex128)
            Swap = _channel_swap(base_n)
            Jb = _anti_diag_J(base_n)
            Sb = _sign_diag(base_n)
            J2 = _block_diag(Jb, Jb)
            S2 = _block_diag(Sb, Sb)

            cands: list[tuple[str, np.ndarray]] = []
            cands.append(("I", I2))
            cands.append(("Swap", Swap))
            cands.append(("J", J2))
            cands.append(("S", S2))
            cands.append(("SJ", S2 @ J2))
            cands.append(("SwapJ", Swap @ J2))
            cands.append(("SwapS", Swap @ S2))
            cands.append(("SwapSJ", Swap @ (S2 @ J2)))
            return cands

        nb = int(b)
        ni = int(N - b)
        Db_I = np.eye(nb, dtype=np.complex128)
        Di_I = np.eye(ni, dtype=np.complex128)
        Db_J = _anti_diag_J(nb)
        Di_J = _anti_diag_J(ni)
        Db_S = _sign_diag(nb)
        Di_S = _sign_diag(ni)

        cands: list[tuple[str, np.ndarray]] = []
        cands.append(("I", _block_diag(Db_I, Di_I)))
        cands.append(("J", _block_diag(Db_J, Di_J)))
        cands.append(("S", _block_diag(Db_S, Di_S)))
        cands.append(("SJ", _block_diag(Db_S @ Db_J, Di_S @ Di_J)))
        return cands

    def _boundary_duality_matrix(cfgm: ModelCfg, name: str) -> np.ndarray:
        name = str(name)
        mode = str(bulk_mode).lower().strip()

        if mode == "two_channel_symmetric":
            b0 = int(cfgm.b // 2)
            if 2 * b0 != int(cfgm.b):
                raise ValueError("two_channel_symmetric expects even boundary size")
            I2 = np.eye(int(cfgm.b), dtype=np.complex128)
            Swap = _channel_swap(b0)
            Jb = _anti_diag_J(b0)
            Sb = _sign_diag(b0)
            J2 = _block_diag(Jb, Jb)
            S2 = _block_diag(Sb, Sb)

            mapping = {
                "I": I2,
                "Swap": Swap,
                "J": J2,
                "S": S2,
                "SJ": S2 @ J2,
                "SwapJ": Swap @ J2,
                "SwapS": Swap @ S2,
                "SwapSJ": Swap @ (S2 @ J2),
            }
            return mapping.get(name, I2)

        nb = int(cfgm.b)
        Db_I = np.eye(nb, dtype=np.complex128)
        Db_J = _anti_diag_J(nb)
        Db_S = _sign_diag(nb)
        mapping = {
            "I": Db_I,
            "J": Db_J,
            "S": Db_S,
            "SJ": Db_S @ Db_J,
        }
        return mapping.get(name, Db_I)

    rows: list[dict] = []
    row_key_to_index: dict[tuple[int, float, float], int] = {}
    spectra_plus: dict[tuple[int, float, float], np.ndarray] = {}
    spectra_minus: dict[tuple[int, float, float], np.ndarray] = {}
    spectra_plus_in: dict[tuple[int, float, float], np.ndarray] = {}
    spectra_minus_in: dict[tuple[int, float, float], np.ndarray] = {}

    for N in Ns:
        cfgN = _mk_cfg(int(N))
        cands = _duality_candidates(cfgN.N, cfgN.b)
        Cn = C_per_N[int(N)]

        for sigma in sigmas:
            for t in t_grid:
                s = complex(float(sigma), float(t))
                s2 = 1.0 - s

                A1, Lam1, S1 = _model_mats(s, cfgN)
                A2, Lam2, S2 = _model_mats(s2, cfgN)

                # Matrix-level free normalization: S_rel(s) = S_model(s) * S_free(s)^{-1}.
                # This is consistent with det normalization detS_norm = det(S_rel).
                try:
                    _, _, S1_0 = _free_mats(s, cfgN)
                    Srel1 = (S1 @ np.linalg.inv(S1_0)).astype(np.complex128)
                except Exception:
                    Srel1 = None

                # DN-native canonical incoming basis on the boundary:
                # T_in = Lambda + i I, and S_in = T_in^{-1} S_rel T_in.
                # This is a pure change-of-basis on boundary data; it often resolves phi vs 1/phi flips.
                Sin1 = None
                try:
                    if Srel1 is not None and Lam1 is not None:
                        nb = int(Lam1.shape[0])
                        Tin = (Lam1 + 1j * np.eye(nb, dtype=np.complex128)).astype(np.complex128)
                        Sin1 = np.linalg.solve(Tin, (Srel1 @ Tin)).astype(np.complex128)
                except Exception:
                    Sin1 = None

                # H2-ish diagnostics for DN: condition of interior inversion
                bidx = _boundary_indices(cfgN)
                if bidx is None:
                    Aii = A1[cfgN.b :, cfgN.b :]
                    Abb = A1[: cfgN.b, : cfgN.b]
                else:
                    mask = np.ones(A1.shape[0], dtype=bool)
                    mask[bidx] = False
                    iidx = np.nonzero(mask)[0]
                    Aii = A1[np.ix_(iidx, iidx)]
                    Abb = A1[np.ix_(bidx, bidx)]

                # Bulk duality residuals: A(1-s) ?= D A(s)^* D^{-1}
                best = None
                for name, D in cands:
                    try:
                        Dinv = np.linalg.inv(D)
                    except np.linalg.LinAlgError:
                        continue
                    R = A2 - (D @ (A1.conj().T) @ Dinv)
                    rel = fro_norm(R) / (fro_norm(A2) + 1e-30)
                    if best is None or rel < best[0]:
                        best = (rel, name)

                best_rel, best_name = best if best is not None else (float("nan"), "(none)")

                # FE diagnostics (det + matrix)
                detS1 = complex(np.linalg.det(S1))
                detS2 = complex(np.linalg.det(S2))
                detS1n = _model_detS_norm(s, cfgN)
                detS2n = _model_detS_norm(s2, cfgN)

                phi1 = phi_modular(s)
                phi2 = phi_modular(s2)

                phi1_euler = phi_euler_only(s)
                phi2_euler = phi_euler_only(s2)

                # Empirical archimedean factor implied by the modular completion
                # (stable even where Gamma has poles): g_inf_emp = phi_mod / phi_euler.
                ginf1_emp = safe_div(phi1, phi1_euler)
                ginf2_emp = safe_div(phi2, phi2_euler)

                fit_targets = {
                    "phi_mod": phi1,
                    "inv_phi_mod": safe_div(1.0 + 0.0j, phi1),
                    "phi_euler": phi1_euler,
                    "inv_phi_euler": safe_div(1.0 + 0.0j, phi1_euler),
                }

                c_emp1 = (detS1n / phi1) if phi1 != 0 else complex("nan")
                c_emp2 = (detS2n / phi2) if phi2 != 0 else complex("nan")
                # On the critical line, we at least expect |c_emp(1/2+it)| ≈ 1 if the mismatch is a unitary phase.
                c_emp_unit_defect = abs(abs(c_emp1) - 1.0) if np.isfinite(c_emp1) else float("nan")
                # Optional involution-style check (depending on intended symmetry conventions):
                # c_emp(1-s) * conj(c_emp(s)) ≈ 1.
                c_emp_FE_dual = abs(c_emp2 * np.conj(c_emp1) - 1.0) if np.isfinite(c_emp1) and np.isfinite(c_emp2) else float("nan")

                # Scalar FE ("modular style"): expects detS_norm(s)*detS_norm(1-s) ≈ 1
                fe_det = abs(detS1n * detS2n - 1.0)
                fe_det_scaled = abs((Cn * detS1n) * (Cn * detS2n) - 1.0)

                # Scalar FE ("duality/Cayley style"): if S(1-s)=D_b (S(s)^*)^{-1} D_b^{-1}
                # then detS_norm(1-s) * conj(detS_norm(s)) ≈ 1.
                fe_det_dual = abs(detS2n * np.conj(detS1n) - 1.0)

                # Scalar target fits (direct vs inverse) for Euler-only and modular targets.
                # These are convenient to see whether we're closer to phi or 1/phi.
                if is_finite_complex(phi1_euler) and abs(phi1_euler) > 1e-30:
                    err_euler = abs((detS1n / phi1_euler) - 1.0)
                else:
                    err_euler = float("nan")
                err_euler_inv = abs(detS1n * phi1_euler - 1.0) if is_finite_complex(phi1_euler) else float("nan")

                err_mod = abs((detS1n / phi1) - 1.0) if is_finite_complex(phi1) and abs(phi1) > 1e-30 else float("nan")
                err_mod_inv = abs(detS1n * phi1 - 1.0) if is_finite_complex(phi1) else float("nan")

                # If detS_norm differs from the modular target only by an archimedean factor,
                # then detS_norm / phi_euler should be close to g_inf_emp (or its inverse).
                det_over_euler = safe_div(detS1n, phi1_euler)
                err_ginf = abs((det_over_euler / ginf1_emp) - 1.0) if is_finite_complex(ginf1_emp) and abs(ginf1_emp) > 1e-30 else float("nan")
                err_ginf_inv = abs(det_over_euler * ginf1_emp - 1.0) if is_finite_complex(ginf1_emp) else float("nan")

                # Two-channel scalarization: compress boundary dof (size 2*b0) into a 2x2 channel-averaged scattering.
                # This tests the hypothesis that the modular scalar lives in one eigenchannel (not det(S)).
                chan_sym = complex("nan")
                chan_anti = complex("nan")
                chan_offdiag_abs = float("nan")
                chan_sym_best, chan_sym_best_err = "(na)", float("nan")
                chan_anti_best, chan_anti_best_err = "(na)", float("nan")
                chan_lam1 = complex("nan")
                chan_lam2 = complex("nan")
                chan_lam1_best, chan_lam1_best_err = "(na)", float("nan")
                chan_lam2_best, chan_lam2_best_err = "(na)", float("nan")

                plus_best, plus_best_err = "(na)", float("nan")
                minus_best, minus_best_err = "(na)", float("nan")
                plus_lam_mod = complex("nan")
                minus_lam_mod = complex("nan")
                plus_lam_mod_relerr = float("nan")
                minus_lam_mod_relerr = float("nan")

                plus_best_in, plus_best_err_in = "(na)", float("nan")
                minus_best_in, minus_best_err_in = "(na)", float("nan")
                plus_lam_mod_in = complex("nan")
                minus_lam_mod_in = complex("nan")
                plus_lam_mod_relerr_in = float("nan")
                minus_lam_mod_relerr_in = float("nan")
                swap_pm_diff_fro = float("nan")
                swap_pm_diff_fro_in = float("nan")

                if str(bulk_mode).lower().strip() == "two_channel_symmetric" and Srel1 is not None:
                    b0 = int(cfgN.b // 2)
                    if b0 > 0 and 2 * b0 == int(cfgN.b):
                        # Exact swap eigenspace reduction (+ and -) at the full b0-dimensional level.
                        # If the modular scalar is one eigenchannel, it should appear as one eigenvalue of S_plus.
                        S11 = Srel1[:b0, :b0]
                        S12 = Srel1[:b0, b0 : 2 * b0]
                        S21 = Srel1[b0 : 2 * b0, :b0]
                        S22 = Srel1[b0 : 2 * b0, b0 : 2 * b0]
                        S_plus = 0.5 * (S11 + S12 + S21 + S22)
                        S_minus = 0.5 * (S11 - S12 - S21 + S22)
                        try:
                            swap_pm_diff_fro = float(np.linalg.norm(S_plus - S_minus, ord="fro"))
                        except Exception:
                            swap_pm_diff_fro = float("nan")
                        eig_plus = None
                        try:
                            eig_plus = np.linalg.eigvals(S_plus)
                            spectra_plus[(int(N), float(sigma), float(t))] = np.asarray(eig_plus, dtype=np.complex128)
                            plus_best, plus_best_err = best_rel_fit_over_eigs(eig_plus, fit_targets)
                            if is_finite_complex(phi1) and abs(phi1) > 1e-30 and eig_plus.size:
                                relerrs = np.abs((eig_plus / phi1) - 1.0)
                                j = int(np.argmin(relerrs))
                                plus_lam_mod = complex(eig_plus[j])
                                plus_lam_mod_relerr = float(relerrs[j])
                        except Exception:
                            pass
                        eig_minus = None
                        try:
                            eig_minus = np.linalg.eigvals(S_minus)
                            spectra_minus[(int(N), float(sigma), float(t))] = np.asarray(eig_minus, dtype=np.complex128)
                            minus_best, minus_best_err = best_rel_fit_over_eigs(eig_minus, fit_targets)
                            if is_finite_complex(phi1) and abs(phi1) > 1e-30 and eig_minus.size:
                                relerrs = np.abs((eig_minus / phi1) - 1.0)
                                j = int(np.argmin(relerrs))
                                minus_lam_mod = complex(eig_minus[j])
                                minus_lam_mod_relerr = float(relerrs[j])
                        except Exception:
                            pass

                        # Incoming-basis version (DN-canonical).
                        if Sin1 is not None:
                            try:
                                S11i = Sin1[:b0, :b0]
                                S12i = Sin1[:b0, b0 : 2 * b0]
                                S21i = Sin1[b0 : 2 * b0, :b0]
                                S22i = Sin1[b0 : 2 * b0, b0 : 2 * b0]
                                S_plus_in = 0.5 * (S11i + S12i + S21i + S22i)
                                S_minus_in = 0.5 * (S11i - S12i - S21i + S22i)
                                swap_pm_diff_fro_in = float(np.linalg.norm(S_plus_in - S_minus_in, ord="fro"))

                                eig_plus_in = np.linalg.eigvals(S_plus_in)
                                spectra_plus_in[(int(N), float(sigma), float(t))] = np.asarray(eig_plus_in, dtype=np.complex128)
                                plus_best_in, plus_best_err_in = best_rel_fit_over_eigs(eig_plus_in, fit_targets)
                                if is_finite_complex(phi1) and abs(phi1) > 1e-30 and eig_plus_in.size:
                                    relerrs = np.abs((eig_plus_in / phi1) - 1.0)
                                    j = int(np.argmin(relerrs))
                                    plus_lam_mod_in = complex(eig_plus_in[j])
                                    plus_lam_mod_relerr_in = float(relerrs[j])

                                eig_minus_in = np.linalg.eigvals(S_minus_in)
                                spectra_minus_in[(int(N), float(sigma), float(t))] = np.asarray(eig_minus_in, dtype=np.complex128)
                                minus_best_in, minus_best_err_in = best_rel_fit_over_eigs(eig_minus_in, fit_targets)
                                if is_finite_complex(phi1) and abs(phi1) > 1e-30 and eig_minus_in.size:
                                    relerrs = np.abs((eig_minus_in / phi1) - 1.0)
                                    j = int(np.argmin(relerrs))
                                    minus_lam_mod_in = complex(eig_minus_in[j])
                                    minus_lam_mod_relerr_in = float(relerrs[j])
                            except Exception:
                                pass

                        u1 = np.zeros((int(cfgN.b),), dtype=np.complex128)
                        u2 = np.zeros((int(cfgN.b),), dtype=np.complex128)
                        u1[:b0] = 1.0 / np.sqrt(float(b0))
                        u2[b0 : 2 * b0] = 1.0 / np.sqrt(float(b0))
                        U = np.stack([u1, u2], axis=1)  # (2b0) x 2
                        S_ch = (U.conj().T @ Srel1 @ U).astype(np.complex128)
                        # symmetric/antisymmetric change of basis
                        P = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
                        S_ch_sa = (P.conj().T @ S_ch @ P).astype(np.complex128)
                        chan_sym = complex(S_ch_sa[0, 0])
                        chan_anti = complex(S_ch_sa[1, 1])
                        chan_offdiag_abs = float(max(abs(S_ch_sa[0, 1]), abs(S_ch_sa[1, 0])))

                        chan_sym_best, chan_sym_best_err = best_rel_fit(chan_sym, fit_targets)
                        chan_anti_best, chan_anti_best_err = best_rel_fit(chan_anti, fit_targets)

                        eigs = np.linalg.eigvals(S_ch)
                        if eigs.size >= 2:
                            chan_lam1 = complex(eigs[0])
                            chan_lam2 = complex(eigs[1])
                            chan_lam1_best, chan_lam1_best_err = best_rel_fit(chan_lam1, fit_targets)
                            chan_lam2_best, chan_lam2_best_err = best_rel_fit(chan_lam2, fit_targets)

                # Cayley + duality gives: S(1-s) = D_b * (S(s)^*)^{-1} * D_b^{-1}
                Db = _boundary_duality_matrix(cfgN, best_name)
                try:
                    Dbinv = np.linalg.inv(Db)
                    fe_mat = float(np.linalg.norm(S2 - (Db @ np.linalg.inv(S1.conj().T) @ Dbinv), ord="fro"))
                except np.linalg.LinAlgError:
                    fe_mat = float("nan")

                rows.append(
                    {
                        "N": int(N),
                        "b": int(cfgN.b),
                        "sigma": float(sigma),
                        "t": float(t),
                        "cond_Aii": float(cond_number(Aii)),
                        "cond_Abb": float(cond_number(Abb)),
                        "hermitian_Lam": float(hermitian_defect(Lam1)),
                        "unitarity_S": float(unitarity_defect(S1)),
                        "bulk_duality_best_rel": float(best_rel),
                        "bulk_duality_best": str(best_name),
                        "FE_det": float(fe_det),
                        "FE_det_scaled": float(fe_det_scaled),
                        "FE_det_dual": float(fe_det_dual),
                        "c_emp_re": float(np.real(c_emp1)),
                        "c_emp_im": float(np.imag(c_emp1)),
                        "c_emp_abs": float(abs(c_emp1)),
                        "c_emp_unit_defect": float(c_emp_unit_defect),
                        "c_emp_FE_dual": float(c_emp_FE_dual),
                        "err_vs_euler": float(err_euler),
                        "err_vs_euler_inv": float(err_euler_inv),
                        "err_vs_mod": float(err_mod),
                        "err_vs_mod_inv": float(err_mod_inv),
                        "ginf_emp_abs": float(abs(ginf1_emp)) if is_finite_complex(ginf1_emp) else float("nan"),
                        "err_vs_ginf": float(err_ginf),
                        "err_vs_ginf_inv": float(err_ginf_inv),
                        "chan_sym_re": float(np.real(chan_sym)) if is_finite_complex(chan_sym) else float("nan"),
                        "chan_sym_im": float(np.imag(chan_sym)) if is_finite_complex(chan_sym) else float("nan"),
                        "chan_sym_abs": float(abs(chan_sym)) if is_finite_complex(chan_sym) else float("nan"),
                        "chan_anti_re": float(np.real(chan_anti)) if is_finite_complex(chan_anti) else float("nan"),
                        "chan_anti_im": float(np.imag(chan_anti)) if is_finite_complex(chan_anti) else float("nan"),
                        "chan_anti_abs": float(abs(chan_anti)) if is_finite_complex(chan_anti) else float("nan"),
                        "chan_offdiag_abs": float(chan_offdiag_abs),
                        "chan_sym_best": str(chan_sym_best),
                        "chan_sym_best_err": float(chan_sym_best_err),
                        "chan_anti_best": str(chan_anti_best),
                        "chan_anti_best_err": float(chan_anti_best_err),
                        "chan_lam1_abs": float(abs(chan_lam1)) if is_finite_complex(chan_lam1) else float("nan"),
                        "chan_lam2_abs": float(abs(chan_lam2)) if is_finite_complex(chan_lam2) else float("nan"),
                        "chan_lam1_best": str(chan_lam1_best),
                        "chan_lam1_best_err": float(chan_lam1_best_err),
                        "chan_lam2_best": str(chan_lam2_best),
                        "chan_lam2_best_err": float(chan_lam2_best_err),
                        "swap_plus_best": str(plus_best),
                        "swap_plus_best_err": float(plus_best_err),
                        "swap_minus_best": str(minus_best),
                        "swap_minus_best_err": float(minus_best_err),
                        "swap_plus_lam_mod_re": float(np.real(plus_lam_mod)) if is_finite_complex(plus_lam_mod) else float("nan"),
                        "swap_plus_lam_mod_im": float(np.imag(plus_lam_mod)) if is_finite_complex(plus_lam_mod) else float("nan"),
                        "swap_plus_lam_mod_abs": float(abs(plus_lam_mod)) if is_finite_complex(plus_lam_mod) else float("nan"),
                        "swap_plus_lam_mod_relerr": float(plus_lam_mod_relerr),
                        "swap_minus_lam_mod_re": float(np.real(minus_lam_mod)) if is_finite_complex(minus_lam_mod) else float("nan"),
                        "swap_minus_lam_mod_im": float(np.imag(minus_lam_mod)) if is_finite_complex(minus_lam_mod) else float("nan"),
                        "swap_minus_lam_mod_abs": float(abs(minus_lam_mod)) if is_finite_complex(minus_lam_mod) else float("nan"),
                        "swap_minus_lam_mod_relerr": float(minus_lam_mod_relerr),
                        "swap_pm_diff_fro": float(swap_pm_diff_fro),
                        "swap_plus_best_in": str(plus_best_in),
                        "swap_plus_best_err_in": float(plus_best_err_in),
                        "swap_minus_best_in": str(minus_best_in),
                        "swap_minus_best_err_in": float(minus_best_err_in),
                        "swap_plus_lam_mod_in_re": float(np.real(plus_lam_mod_in)) if is_finite_complex(plus_lam_mod_in) else float("nan"),
                        "swap_plus_lam_mod_in_im": float(np.imag(plus_lam_mod_in)) if is_finite_complex(plus_lam_mod_in) else float("nan"),
                        "swap_plus_lam_mod_in_abs": float(abs(plus_lam_mod_in)) if is_finite_complex(plus_lam_mod_in) else float("nan"),
                        "swap_plus_lam_mod_in_relerr": float(plus_lam_mod_relerr_in),
                        "swap_minus_lam_mod_in_re": float(np.real(minus_lam_mod_in)) if is_finite_complex(minus_lam_mod_in) else float("nan"),
                        "swap_minus_lam_mod_in_im": float(np.imag(minus_lam_mod_in)) if is_finite_complex(minus_lam_mod_in) else float("nan"),
                        "swap_minus_lam_mod_in_abs": float(abs(minus_lam_mod_in)) if is_finite_complex(minus_lam_mod_in) else float("nan"),
                        "swap_minus_lam_mod_in_relerr": float(minus_lam_mod_relerr_in),
                        "swap_pm_diff_fro_in": float(swap_pm_diff_fro_in),
                        "FE_mat_fro": float(fe_mat),
                        "phi_mod_re": float(np.real(phi1)),
                        "phi_mod_im": float(np.imag(phi1)),
                    }
                )
                row_key_to_index[(int(N), float(sigma), float(t))] = len(rows) - 1

    # Continuous eigenvalue branch tracking in Swap± blocks, per (N, sigma) as t varies.
    # Initializes at the first t by closest-to-phi_mod, then continues by nearest-neighbor in C.
    if str(bulk_mode).lower().strip() == "two_channel_symmetric" and len(spectra_plus) > 0:
        for N in Ns:
            for sigma in sigmas:
                ts_sorted = sorted(float(x) for x in t_grid)
                prev_plus = complex("nan")
                prev_minus = complex("nan")
                for k, t in enumerate(ts_sorted):
                    key = (int(N), float(sigma), float(t))
                    ridx = row_key_to_index.get(key)
                    if ridx is None:
                        continue
                    s = complex(float(sigma), float(t))
                    phi = phi_target(s, primes=cfgN.primes, completion_mode=cfgN.completion_mode, mode=phi_target_mode)
                    phi_e = phi_euler_only(s)
                    targets = {
                        "phi_mod": phi_modular(s),
                        "inv_phi_mod": safe_div(1.0 + 0.0j, phi_modular(s)),
                        "phi_euler": phi_e,
                        "inv_phi_euler": safe_div(1.0 + 0.0j, phi_e),
                    }

                    # PLUS block
                    eigs_p = spectra_plus.get(key)
                    if eigs_p is not None and np.asarray(eigs_p).size:
                        if k == 0 or not is_finite_complex(prev_plus):
                            j, _ = _min_relerr_to_target(eigs_p, phi)
                        else:
                            j = _nearest_neighbor_index(eigs_p, prev_plus)
                        lam = complex(np.asarray(eigs_p).ravel()[int(j)])
                        gap = _spectral_gap(eigs_p, int(j))
                        jump = abs(lam - prev_plus) if is_finite_complex(prev_plus) else float("nan")
                        best_lbl, best_err = best_rel_fit(lam, targets)
                        rows[ridx].update(
                            {
                                "lam_plus_track_re": float(np.real(lam)),
                                "lam_plus_track_im": float(np.imag(lam)),
                                "lam_plus_track_abs": float(abs(lam)),
                                "lam_plus_track_best": str(best_lbl),
                                "lam_plus_track_best_err": float(best_err),
                                "lam_plus_track_relerr_mod": float(abs((lam / phi) - 1.0))
                                if is_finite_complex(phi) and abs(phi) > 1e-30
                                else float("nan"),
                                "lam_plus_track_relerr_invmod": float(abs(lam * phi - 1.0)) if is_finite_complex(phi) else float("nan"),
                                "lam_plus_track_gap": float(gap),
                                "lam_plus_track_jump": float(jump),
                            }
                        )
                        prev_plus = lam
                    else:
                        rows[ridx].update(
                            {
                                "lam_plus_track_re": float("nan"),
                                "lam_plus_track_im": float("nan"),
                                "lam_plus_track_abs": float("nan"),
                                "lam_plus_track_best": "(na)",
                                "lam_plus_track_best_err": float("nan"),
                                "lam_plus_track_relerr_mod": float("nan"),
                                "lam_plus_track_relerr_invmod": float("nan"),
                                "lam_plus_track_gap": float("nan"),
                                "lam_plus_track_jump": float("nan"),
                            }
                        )

                    # MINUS block
                    eigs_m = spectra_minus.get(key)
                    if eigs_m is not None and np.asarray(eigs_m).size:
                        if k == 0 or not is_finite_complex(prev_minus):
                            j, _ = _min_relerr_to_target(eigs_m, phi)
                        else:
                            j = _nearest_neighbor_index(eigs_m, prev_minus)
                        lam = complex(np.asarray(eigs_m).ravel()[int(j)])
                        gap = _spectral_gap(eigs_m, int(j))
                        jump = abs(lam - prev_minus) if is_finite_complex(prev_minus) else float("nan")
                        best_lbl, best_err = best_rel_fit(lam, targets)
                        rows[ridx].update(
                            {
                                "lam_minus_track_re": float(np.real(lam)),
                                "lam_minus_track_im": float(np.imag(lam)),
                                "lam_minus_track_abs": float(abs(lam)),
                                "lam_minus_track_best": str(best_lbl),
                                "lam_minus_track_best_err": float(best_err),
                                "lam_minus_track_relerr_mod": float(abs((lam / phi) - 1.0))
                                if is_finite_complex(phi) and abs(phi) > 1e-30
                                else float("nan"),
                                "lam_minus_track_relerr_invmod": float(abs(lam * phi - 1.0)) if is_finite_complex(phi) else float("nan"),
                                "lam_minus_track_gap": float(gap),
                                "lam_minus_track_jump": float(jump),
                            }
                        )
                        prev_minus = lam
                    else:
                        rows[ridx].update(
                            {
                                "lam_minus_track_re": float("nan"),
                                "lam_minus_track_im": float("nan"),
                                "lam_minus_track_abs": float("nan"),
                                "lam_minus_track_best": "(na)",
                                "lam_minus_track_best_err": float("nan"),
                                "lam_minus_track_relerr_mod": float("nan"),
                                "lam_minus_track_relerr_invmod": float("nan"),
                                "lam_minus_track_gap": float("nan"),
                                "lam_minus_track_jump": float("nan"),
                            }
                        )

    # Same tracking, but for the DN-canonical incoming basis Swap± blocks.
    if str(bulk_mode).lower().strip() == "two_channel_symmetric" and len(spectra_plus_in) > 0:
        for N in Ns:
            for sigma in sigmas:
                ts_sorted = sorted(float(x) for x in t_grid)
                prev_plus = complex("nan")
                prev_minus = complex("nan")
                for k, t in enumerate(ts_sorted):
                    key = (int(N), float(sigma), float(t))
                    ridx = row_key_to_index.get(key)
                    if ridx is None:
                        continue
                    s = complex(float(sigma), float(t))
                    phi = phi_target(s, primes=cfgN.primes, completion_mode=cfgN.completion_mode, mode=phi_target_mode)
                    phi_e = phi_euler_only(s)
                    targets = {
                        "phi_mod": phi_modular(s),
                        "inv_phi_mod": safe_div(1.0 + 0.0j, phi_modular(s)),
                        "phi_euler": phi_e,
                        "inv_phi_euler": safe_div(1.0 + 0.0j, phi_e),
                    }

                    eigs_p = spectra_plus_in.get(key)
                    if eigs_p is not None and np.asarray(eigs_p).size:
                        if k == 0 or not is_finite_complex(prev_plus):
                            j, _ = _min_relerr_to_target(eigs_p, phi)
                        else:
                            j = _nearest_neighbor_index(eigs_p, prev_plus)
                        lam = complex(np.asarray(eigs_p).ravel()[int(j)])
                        gap = _spectral_gap(eigs_p, int(j))
                        jump = abs(lam - prev_plus) if is_finite_complex(prev_plus) else float("nan")
                        best_lbl, best_err = best_rel_fit(lam, targets)
                        rows[ridx].update(
                            {
                                "lam_plus_track_in_re": float(np.real(lam)),
                                "lam_plus_track_in_im": float(np.imag(lam)),
                                "lam_plus_track_in_abs": float(abs(lam)),
                                "lam_plus_track_in_best": str(best_lbl),
                                "lam_plus_track_in_best_err": float(best_err),
                                "lam_plus_track_in_relerr_mod": float(abs((lam / phi) - 1.0))
                                if is_finite_complex(phi) and abs(phi) > 1e-30
                                else float("nan"),
                                "lam_plus_track_in_relerr_invmod": float(abs(lam * phi - 1.0)) if is_finite_complex(phi) else float("nan"),
                                "lam_plus_track_in_gap": float(gap),
                                "lam_plus_track_in_jump": float(jump),
                            }
                        )
                        prev_plus = lam
                    else:
                        rows[ridx].update(
                            {
                                "lam_plus_track_in_re": float("nan"),
                                "lam_plus_track_in_im": float("nan"),
                                "lam_plus_track_in_abs": float("nan"),
                                "lam_plus_track_in_best": "(na)",
                                "lam_plus_track_in_best_err": float("nan"),
                                "lam_plus_track_in_relerr_mod": float("nan"),
                                "lam_plus_track_in_relerr_invmod": float("nan"),
                                "lam_plus_track_in_gap": float("nan"),
                                "lam_plus_track_in_jump": float("nan"),
                            }
                        )

                    eigs_m = spectra_minus_in.get(key)
                    if eigs_m is not None and np.asarray(eigs_m).size:
                        if k == 0 or not is_finite_complex(prev_minus):
                            j, _ = _min_relerr_to_target(eigs_m, phi)
                        else:
                            j = _nearest_neighbor_index(eigs_m, prev_minus)
                        lam = complex(np.asarray(eigs_m).ravel()[int(j)])
                        gap = _spectral_gap(eigs_m, int(j))
                        jump = abs(lam - prev_minus) if is_finite_complex(prev_minus) else float("nan")
                        best_lbl, best_err = best_rel_fit(lam, targets)
                        rows[ridx].update(
                            {
                                "lam_minus_track_in_re": float(np.real(lam)),
                                "lam_minus_track_in_im": float(np.imag(lam)),
                                "lam_minus_track_in_abs": float(abs(lam)),
                                "lam_minus_track_in_best": str(best_lbl),
                                "lam_minus_track_in_best_err": float(best_err),
                                "lam_minus_track_in_relerr_mod": float(abs((lam / phi) - 1.0))
                                if is_finite_complex(phi) and abs(phi) > 1e-30
                                else float("nan"),
                                "lam_minus_track_in_relerr_invmod": float(abs(lam * phi - 1.0)) if is_finite_complex(phi) else float("nan"),
                                "lam_minus_track_in_gap": float(gap),
                                "lam_minus_track_in_jump": float(jump),
                            }
                        )
                        prev_minus = lam
                    else:
                        rows[ridx].update(
                            {
                                "lam_minus_track_in_re": float("nan"),
                                "lam_minus_track_in_im": float("nan"),
                                "lam_minus_track_in_abs": float("nan"),
                                "lam_minus_track_in_best": "(na)",
                                "lam_minus_track_in_best_err": float("nan"),
                                "lam_minus_track_in_relerr_mod": float("nan"),
                                "lam_minus_track_in_relerr_invmod": float("nan"),
                                "lam_minus_track_in_gap": float("nan"),
                                "lam_minus_track_in_jump": float("nan"),
                            }
                        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "E_spine_checks.csv"), index=False)

    # H4-ish local factor shape: prime-by-prime comparison at a couple of s
    sample_s = [complex(0.5, 1.0), complex(0.5, 2.0), complex(1.2, 0.0)]
    rows_local: list[dict] = []
    for N in Ns:
        cfgN = _mk_cfg(int(N))
        base_N = int(cfgN.N // 2) if str(bulk_mode).lower().strip() == "two_channel_symmetric" else int(cfgN.N)
        bulk_params = BulkParams(N=int(base_N), weight_k=cfgN.weight_k)
        for s in sample_s:
            for p in primes[: min(len(primes), 9)]:
                A = build_A(
                    s=s,
                    primes=[int(p)],
                    comps=None,
                    params=bulk_params,
                    generator_norm=cfgN.generator_norm,
                    generator_norm_target=cfgN.generator_norm_target,
                    bulk_mode=str(bulk_mode),
                    completion_mode=cfgN.completion_mode,
                    dual_scale=cfgN.dual_scale,
                    eps_prime_mode=cfgN.eps_prime_mode,
                    prime_assembly_mode=cfgN.prime_assembly_mode,
                    generator_packetize_mode=cfgN.generator_packetize_mode,
                    generator_packetize_basis_mode=cfgN.generator_packetize_basis_mode,
                )
                bidx = _boundary_indices(cfgN)
                if bidx is None:
                    Lam = dn_map(A, b=cfgN.b, jitter=cfgN.jitter)
                else:
                    Lam = dn_map_on_indices(A, boundary_idx=bidx, jitter=cfgN.jitter)
                S = cayley_from_lambda(Lam, eta=cfgN.eta)
                detS = complex(np.linalg.det(S))

                A0 = build_A(
                    s=s,
                    primes=[],
                    comps=None,
                    params=bulk_params,
                    generator_norm=cfgN.generator_norm,
                    generator_norm_target=cfgN.generator_norm_target,
                    bulk_mode=str(bulk_mode),
                    completion_mode=cfgN.completion_mode,
                    dual_scale=cfgN.dual_scale,
                    eps_prime_mode=cfgN.eps_prime_mode,
                    prime_assembly_mode=cfgN.prime_assembly_mode,
                    generator_packetize_mode=cfgN.generator_packetize_mode,
                    generator_packetize_basis_mode=cfgN.generator_packetize_basis_mode,
                )
                if bidx is None:
                    Lam0 = dn_map(A0, b=cfgN.b, jitter=cfgN.jitter)
                else:
                    Lam0 = dn_map_on_indices(A0, boundary_idx=bidx, jitter=cfgN.jitter)
                S0 = cayley_from_lambda(Lam0, eta=cfgN.eta)
                detS0 = complex(np.linalg.det(S0))
                detSn = detS / detS0 if detS0 != 0 else complex("nan")

                target_local = local_phi_modular(int(p), s)
                ratio = detSn / target_local if target_local != 0 else complex("nan")
                rows_local.append(
                    {
                        "N": int(N),
                        "p": int(p),
                        "s_re": float(np.real(s)),
                        "s_im": float(np.imag(s)),
                        "detS_norm_re": float(np.real(detSn)),
                        "detS_norm_im": float(np.imag(detSn)),
                        "phi_local_re": float(np.real(target_local)),
                        "phi_local_im": float(np.imag(target_local)),
                        "ratio_abs": float(abs(ratio)),
                    }
                )

    pd.DataFrame(rows_local).to_csv(os.path.join(out_dir, "E_local_factor_shape.csv"), index=False)

    # H4 divisor agreement (argument principle) for Q(s) := C(N)*S_model(s)/phi_mod(s)

    def _dlog(f, s: complex, h: float) -> complex:
        f0 = f(s)
        if f0 == 0:
            return complex("nan")
        fp = f(s + h)
        fm = f(s - h)
        return (fp - fm) / (2.0 * h * f0)

    rows_ap: list[dict] = []
    pts = _boundary_points(rect, ap_n_edge)
    for N in Ns:
        cfgN = _mk_cfg(int(N))
        Cn = C_per_N[int(N)]

        def Q(s: complex) -> complex:
            Sm = _model_detS_norm(s, cfgN)
            Sm = Cn * Sm
            ph = phi_target(s, primes=cfgN.primes, completion_mode=cfgN.completion_mode, mode=phi_target_mode)
            return Sm / ph if ph != 0 else complex("nan")

        integral = 0.0 + 0.0j
        failures = 0
        for i in range(len(pts) - 1):
            s0p = pts[i]
            s1p = pts[i + 1]
            ds = s1p - s0p
            smid = (s0p + s1p) / 2.0
            try:
                dlog = _dlog(Q, smid, h=ap_h)
                if not np.isfinite(dlog.real) or not np.isfinite(dlog.imag):
                    failures += 1
                    continue
                integral += dlog * ds
            except Exception:
                failures += 1
        count = integral / (2j * np.pi)
        rows_ap.append(
            {
                "N": int(N),
                "sigma_min": float(rect[0]),
                "sigma_max": float(rect[1]),
                "t_min": float(rect[2]),
                "t_max": float(rect[3]),
                "n_edge": int(ap_n_edge),
                "h": float(ap_h),
                "failures": int(failures),
                "count_re": float(np.real(count)),
                "count_im": float(np.imag(count)),
                "count_abs_im": float(abs(np.imag(count))),
            }
        )

    pd.DataFrame(rows_ap).to_csv(os.path.join(out_dir, "E_divisor_arg_principle_Q.csv"), index=False)

    # Optional: winding / argument principle for a tracked swap-eigenspace branch λ(s) via phase unwrapping.
    # We avoid numerical dlog here because branch tracking at s±h is convention-sensitive.
    if do_arg_lambda_branch and str(bulk_mode).lower().strip() == "two_channel_symmetric":
        spike_rows: list[dict] = []

        def _maybe_push_topk(heap: list[tuple[float, dict]], k: int, absq: float, rec: dict) -> None:
            if not np.isfinite(absq):
                return
            if k <= 0:
                return
            item = (float(absq), rec)
            if len(heap) < k:
                heapq.heappush(heap, item)
                return
            if absq > heap[0][0]:
                heapq.heapreplace(heap, item)

        def _compute_q_lambda_winding_for_rect(
            rect_: tuple[float, float, float, float],
            n_edge_: int,
            *,
            source_label: str = "rect",
        ) -> list[dict]:
            rows: list[dict] = []
            pts_lam = _boundary_points(rect_, n_edge_)
            for N in Ns:
                cfgN = _mk_cfg(int(N))
                bidx = _boundary_indices(cfgN)
                _, iidx_use = _split_boundary_interior_indices(int(cfgN.N), int(cfgN.b), bidx)

                def _block_eigs(Srel: np.ndarray, block: str) -> np.ndarray:
                    b0 = int(cfgN.b // 2)
                    S11 = Srel[:b0, :b0]
                    S12 = Srel[:b0, b0 : 2 * b0]
                    S21 = Srel[b0 : 2 * b0, :b0]
                    S22 = Srel[b0 : 2 * b0, b0 : 2 * b0]
                    if block == "+":
                        B = 0.5 * (S11 + S12 + S21 + S22)
                    else:
                        B = 0.5 * (S11 - S12 - S21 + S22)
                    return np.linalg.eigvals(B)

                states = {
                    ("rel", "+"): {"prev": complex("nan"), "qs": []},
                    ("rel", "-"): {"prev": complex("nan"), "qs": []},
                    ("incoming", "+"): {"prev": complex("nan"), "qs": []},
                    ("incoming", "-"): {"prev": complex("nan"), "qs": []},
                }
                first_lam: dict[tuple[str, str], complex] = {k: complex("nan") for k in states.keys()}
                failures_total = {k: 0 for k in states.keys()}

                topk_heaps: dict[tuple[str, str], list[tuple[float, dict]]] = {
                    ("rel", "+"): [],
                    ("rel", "-"): [],
                    ("incoming", "+"): [],
                    ("incoming", "-"): [],
                }

                for i, s in enumerate(pts_lam):
                    if print_progress and progress_every > 0 and (i % progress_every == 0):
                        print(
                            f"[Q_lambda] src={source_label} rect=({rect_[0]:.3g},{rect_[1]:.3g},{rect_[2]:.3g},{rect_[3]:.3g}) "
                            f"N={int(N)} i={i}/{len(pts_lam)-1}"
                        )
                    try:
                        A, Lam, Sm = _model_mats(s, cfgN)
                        _, _, S0 = _free_mats(s, cfgN)
                        Srel = (Sm @ np.linalg.inv(S0)).astype(np.complex128)
                        nb = int(Lam.shape[0])
                        Tin = (Lam + 1j * np.eye(nb, dtype=np.complex128)).astype(np.complex128)
                        Sin = np.linalg.solve(Tin, (Srel @ Tin)).astype(np.complex128)
                    except Exception:
                        for k in states.keys():
                            states[k]["qs"].append(complex("nan"))
                            failures_total[k] += 1
                        continue

                    phi = phi_target(s, primes=cfgN.primes, completion_mode=cfgN.completion_mode, mode=phi_target_mode)
                    if (not is_finite_complex(phi)) or abs(phi) <= 1e-30:
                        for k in states.keys():
                            states[k]["qs"].append(complex("nan"))
                            failures_total[k] += 1
                        continue

                    mats = {"rel": Srel, "incoming": Sin}
                    for basis in ["rel", "incoming"]:
                        M = mats[basis]
                        for block in ["+", "-"]:
                            key = (basis, block)
                            try:
                                eigs = _block_eigs(M, block)
                            except Exception:
                                states[key]["qs"].append(complex("nan"))
                                failures_total[key] += 1
                                continue
                            if eigs.size == 0:
                                states[key]["qs"].append(complex("nan"))
                                failures_total[key] += 1
                                continue

                            prev = states[key]["prev"]
                            if i == 0 or not is_finite_complex(prev):
                                j, _ = _min_relerr_to_target(eigs, phi)
                            elif i == (len(pts_lam) - 1) and is_finite_complex(first_lam.get(key, complex("nan"))):
                                # Force closure: pick the eigenvalue closest to the initial one.
                                j = _nearest_neighbor_index(eigs, first_lam[key])
                            else:
                                j = _nearest_neighbor_index(eigs, prev)
                            lam = complex(np.asarray(eigs).ravel()[int(j)])
                            if i == 0:
                                first_lam[key] = lam
                            states[key]["prev"] = lam
                            q = lam / phi
                            states[key]["qs"].append(q)

                            if do_arg_lambda_spike_dump:
                                absq = float(abs(q)) if is_finite_complex(q) else float("nan")
                                if np.isfinite(absq):
                                    heap = topk_heaps[key]
                                    need = (len(heap) < arg_lambda_spike_topk) or (
                                        arg_lambda_spike_topk > 0 and absq > heap[0][0]
                                    )
                                    if need:
                                        Aii = np.asarray(A, dtype=np.complex128)[np.ix_(iidx_use, iidx_use)]
                                        if Aii.size:
                                            Aii = Aii + (cfgN.jitter * np.eye(Aii.shape[0], dtype=np.complex128))
                                        log10_det_Aii = _log10_abs_det(Aii) if Aii.size else float("nan")
                                        log10_det_Tin = _log10_abs_det(Tin)
                                        _, _, cond_Tin, log10_smin_Tin, log10_smax_Tin = _svd_sentinels(Tin)
                                        rec = {
                                            "source": str(source_label),
                                            "N": int(N),
                                            "basis": str(basis),
                                            "block": "plus" if block == "+" else "minus",
                                            "sigma_min": float(rect_[0]),
                                            "sigma_max": float(rect_[1]),
                                            "t_min": float(rect_[2]),
                                            "t_max": float(rect_[3]),
                                            "n_edge": int(n_edge_),
                                            "s_re": float(np.real(s)),
                                            "s_im": float(np.imag(s)),
                                            "abs_Qlambda": float(absq),
                                            "lam_re": float(np.real(lam)),
                                            "lam_im": float(np.imag(lam)),
                                            "abs_lam": float(abs(lam)) if is_finite_complex(lam) else float("nan"),
                                            "phi_re": float(np.real(phi)),
                                            "phi_im": float(np.imag(phi)),
                                            "abs_phi": float(abs(phi)) if is_finite_complex(phi) else float("nan"),
                                            "log10_abs_det_Aii_jitter": float(log10_det_Aii),
                                            "log10_abs_det_Lam_plus_iI": float(log10_det_Tin),
                                            "cond_Lam_plus_iI": float(cond_Tin),
                                            "log10_smin_Lam_plus_iI": float(log10_smin_Tin),
                                            "log10_smax_Lam_plus_iI": float(log10_smax_Tin),
                                        }
                                        _maybe_push_topk(heap, arg_lambda_spike_topk, absq, rec)

                for basis in ["rel", "incoming"]:
                    for block in ["+", "-"]:
                        key = (basis, block)
                        qs = states[key]["qs"]
                        w, bad_w = _winding_count_from_samples(qs)
                        qmin, qmax, smin, smax, bad_mm = _minmax_abs_with_location(qs, pts_lam)
                        rows.append(
                            {
                                "N": int(N),
                                "block": "plus" if block == "+" else "minus",
                                "basis": str(basis),
                                "sigma_min": float(rect_[0]),
                                "sigma_max": float(rect_[1]),
                                "t_min": float(rect_[2]),
                                "t_max": float(rect_[3]),
                                "n_edge": int(n_edge_),
                                "failures": int(failures_total[key] + bad_w + bad_mm),
                                "winding": float(w),
                                "winding_abs": float(abs(w)) if np.isfinite(w) else float("nan"),
                                "min_abs_Qlambda": float(qmin),
                                "max_abs_Qlambda": float(qmax),
                                "s_at_min_abs_re": float(np.real(smin)),
                                "s_at_min_abs_im": float(np.imag(smin)),
                                "s_at_max_abs_re": float(np.real(smax)),
                                "s_at_max_abs_im": float(np.imag(smax)),
                            }
                        )

                if do_arg_lambda_spike_dump:
                    for (basis, block), heap in topk_heaps.items():
                        heap_sorted = sorted(heap, key=lambda x: x[0], reverse=True)
                        for rank, (absq, rec) in enumerate(heap_sorted, start=1):
                            rec2 = dict(rec)
                            rec2["rank"] = int(rank)
                            rec2["abs_Qlambda"] = float(absq)
                            spike_rows.append(rec2)

            return rows

        rows_ap_lam = _compute_q_lambda_winding_for_rect(rect, ap_n_edge_lambda)
        pd.DataFrame(rows_ap_lam).to_csv(os.path.join(out_dir, "E_divisor_arg_principle_Q_lambda.csv"), index=False)

        # Optional: extra targeted rectangles for winding/spike localization.
        if lambda_extra_rects_cfg is not None:
            extra_rows: list[dict] = []
            extra_rects = lambda_extra_rects_cfg
            if isinstance(extra_rects, dict):
                extra_rects = [extra_rects]
            for j, rr in enumerate(list(extra_rects)):
                try:
                    name = str(rr.get("name", f"extra_{j}"))
                    rect_v = rr.get("rect", None)
                    if rect_v is None:
                        rect_v = [rr.get("sigma_min"), rr.get("sigma_max"), rr.get("t_min"), rr.get("t_max")]
                    r = (float(rect_v[0]), float(rect_v[1]), float(rect_v[2]), float(rect_v[3]))
                    n_edge_r = int(rr.get("n_edge", max(400, ap_n_edge_lambda // 2)))
                except Exception:
                    continue
                rows_r = _compute_q_lambda_winding_for_rect(r, n_edge_r, source_label=f"extra:{name}")
                for row in rows_r:
                    row["rect_name"] = str(name)
                extra_rows.extend(rows_r)

            if extra_rows:
                pd.DataFrame(extra_rows).to_csv(os.path.join(out_dir, "E_divisor_arg_principle_Q_lambda_extra_rects.csv"), index=False)

        if do_arg_lambda_subrects:
            def _parse_edges(x, lo: float, hi: float) -> list[float]:
                if x is None:
                    return [float(lo), float(0.5 * (lo + hi)), float(hi)]
                if isinstance(x, (int, float)):
                    k = int(x)
                    if k < 1:
                        return [float(lo), float(hi)]
                    return list(map(float, np.linspace(lo, hi, k + 1)))
                # assume list-like
                xs = list(map(float, list(x)))
                xs = [v for v in xs if np.isfinite(v)]
                if not xs:
                    return [float(lo), float(hi)]
                xs = sorted(set(xs + [float(lo), float(hi)]))
                # clamp to [lo,hi]
                xs = [min(float(hi), max(float(lo), v)) for v in xs]
                xs = sorted(set(xs))
                return xs

            sigma_edges = _parse_edges(subrect_sigma_edges_cfg, rect[0], rect[1])
            t_edges = _parse_edges(subrect_t_edges_cfg, rect[2], rect[3])
            sub_rows: list[dict] = []
            for si in range(len(sigma_edges) - 1):
                for ti in range(len(t_edges) - 1):
                    r = (float(sigma_edges[si]), float(sigma_edges[si + 1]), float(t_edges[ti]), float(t_edges[ti + 1]))
                    rr = _compute_q_lambda_winding_for_rect(r, ap_n_edge_lambda_subrect, source_label="subrect")
                    for row in rr:
                        row["sigma_bin"] = int(si)
                        row["t_bin"] = int(ti)
                    sub_rows.extend(rr)

            pd.DataFrame(sub_rows).to_csv(os.path.join(out_dir, "E_divisor_arg_principle_Q_lambda_subrects.csv"), index=False)

        if do_arg_lambda_spike_dump and spike_rows:
            pd.DataFrame(spike_rows).to_csv(os.path.join(out_dir, "E_Qlambda_spikes.csv"), index=False)

        if do_arg_lambda_hotspots:
            # Hotspot boundary scan: dump per-boundary-point diagnostics on small rectangles.
            # Config format:
            # arg_lambda_hotspots:
            #   - {name: hot0, sigma: 1.075, t: 9.3222, ds: 0.03, dt: 0.25, n_edge: 2000}
            hs = lambda_hotspots_cfg
            if hs is None:
                hs = [
                    {"name": "hot_sigma1p075_t9p32", "sigma": 1.075, "t": 9.3222, "ds": 0.03, "dt": 0.25, "n_edge": 2000},
                    {"name": "hot_sigma1p6_t7p5", "sigma": 1.6, "t": 7.5, "ds": 0.03, "dt": 0.25, "n_edge": 2000},
                ]

            hot_rows: list[dict] = []
            for h in hs:
                name = str(h.get("name", "hotspot"))
                sigc = float(h.get("sigma", 1.0))
                tc = float(h.get("t", 1.0))
                ds = float(h.get("ds", 0.03))
                dt = float(h.get("dt", 0.25))
                n_edge_h = int(h.get("n_edge", 2000))
                rect_h = (sigc - ds, sigc + ds, tc - dt, tc + dt)
                pts_h = _boundary_points(rect_h, n_edge_h)

                for N in Ns:
                    cfgN = _mk_cfg(int(N))
                    bidx = _boundary_indices(cfgN)
                    _bidx_use, iidx_use = _split_boundary_interior_indices(int(cfgN.N), int(cfgN.b), bidx)

                    prevs: dict[tuple[str, str], complex] = {
                        ("rel", "+"): complex("nan"),
                        ("rel", "-"): complex("nan"),
                        ("incoming", "+"): complex("nan"),
                        ("incoming", "-"): complex("nan"),
                    }

                    for i, s in enumerate(pts_h):
                        if print_progress and progress_every > 0 and (i % progress_every == 0):
                            print(f"[Q_lambda] hotspot={name} N={int(N)} i={i}/{len(pts_h)-1}")
                        try:
                            A, Lam, Sm = _model_mats(s, cfgN)
                            _, _, S0 = _free_mats(s, cfgN)
                            Srel = (Sm @ np.linalg.inv(S0)).astype(np.complex128)
                            nb = int(Lam.shape[0])
                            Tin = (Lam + 1j * np.eye(nb, dtype=np.complex128)).astype(np.complex128)
                            Sin = np.linalg.solve(Tin, (Srel @ Tin)).astype(np.complex128)
                        except Exception:
                            continue

                        phi = phi_target(s, primes=cfgN.primes, completion_mode=cfgN.completion_mode, mode=phi_target_mode)
                        if (not is_finite_complex(phi)) or abs(phi) <= 1e-30:
                            continue

                        # sentinels common to all basis/block at this s
                        Aii = np.asarray(A, dtype=np.complex128)[np.ix_(iidx_use, iidx_use)]
                        if Aii.size:
                            Aii = Aii + (cfgN.jitter * np.eye(Aii.shape[0], dtype=np.complex128))
                        log10_det_Aii = _log10_abs_det(Aii) if Aii.size else float("nan")
                        log10_det_Tin = _log10_abs_det(Tin)
                        _, _, cond_Tin, log10_smin_Tin, log10_smax_Tin = _svd_sentinels(Tin)

                        def _block_eigs_hot(Srel0: np.ndarray, block: str) -> np.ndarray:
                            b0 = int(cfgN.b // 2)
                            S11 = Srel0[:b0, :b0]
                            S12 = Srel0[:b0, b0 : 2 * b0]
                            S21 = Srel0[b0 : 2 * b0, :b0]
                            S22 = Srel0[b0 : 2 * b0, b0 : 2 * b0]
                            if block == "+":
                                B = 0.5 * (S11 + S12 + S21 + S22)
                            else:
                                B = 0.5 * (S11 - S12 - S21 + S22)
                            return np.linalg.eigvals(B)

                        mats = {"rel": Srel, "incoming": Sin}
                        for basis in ["rel", "incoming"]:
                            M = mats[basis]
                            for block in ["+", "-"]:
                                key = (basis, block)
                                try:
                                    eigs = _block_eigs_hot(M, block)
                                except Exception:
                                    continue
                                if eigs.size == 0:
                                    continue
                                prev = prevs[key]
                                if i == 0 or not is_finite_complex(prev):
                                    j, _ = _min_relerr_to_target(eigs, phi)
                                else:
                                    j = _nearest_neighbor_index(eigs, prev)
                                lam = complex(np.asarray(eigs).ravel()[int(j)])
                                prevs[key] = lam
                                q = lam / phi

                                hot_rows.append(
                                    {
                                        "name": str(name),
                                        "N": int(N),
                                        "basis": str(basis),
                                        "block": "plus" if block == "+" else "minus",
                                        "rect_sigma_min": float(rect_h[0]),
                                        "rect_sigma_max": float(rect_h[1]),
                                        "rect_t_min": float(rect_h[2]),
                                        "rect_t_max": float(rect_h[3]),
                                        "n_edge": int(n_edge_h),
                                        "s_re": float(np.real(s)),
                                        "s_im": float(np.imag(s)),
                                        "abs_Qlambda": float(abs(q)) if is_finite_complex(q) else float("nan"),
                                        "abs_lam": float(abs(lam)) if is_finite_complex(lam) else float("nan"),
                                        "abs_phi": float(abs(phi)) if is_finite_complex(phi) else float("nan"),
                                        "log10_abs_det_Aii_jitter": float(log10_det_Aii),
                                        "log10_abs_det_Lam_plus_iI": float(log10_det_Tin),
                                        "cond_Lam_plus_iI": float(cond_Tin),
                                        "log10_smin_Lam_plus_iI": float(log10_smin_Tin),
                                        "log10_smax_Lam_plus_iI": float(log10_smax_Tin),
                                    }
                                )

            if hot_rows:
                pd.DataFrame(hot_rows).to_csv(os.path.join(out_dir, "E_Qlambda_hotspot_boundary.csv"), index=False)

    # Additional: winding / argument principle for the raw scalar mismatch c_emp(s)=detS_norm(s)/phi_mod(s)
    if do_arg_c_emp:
        rows_ap_c: list[dict] = []
        pts = _boundary_points(rect, ap_n_edge)
        for N in Ns:
            cfgN = _mk_cfg(int(N))

            def c_emp(s: complex) -> complex:
                Sm = _model_detS_norm(s, cfgN)
                ph = phi_target(s, primes=cfgN.primes, completion_mode=cfgN.completion_mode, mode=phi_target_mode)
                return Sm / ph if ph != 0 else complex("nan")

            integral = 0.0 + 0.0j
            failures = 0
            for i in range(len(pts) - 1):
                s0p = pts[i]
                s1p = pts[i + 1]
                ds = s1p - s0p
                smid = (s0p + s1p) / 2.0
                try:
                    dlog = _dlog(c_emp, smid, h=ap_h)
                    if not np.isfinite(dlog.real) or not np.isfinite(dlog.imag):
                        failures += 1
                        continue
                    integral = integral + dlog * ds
                except Exception:
                    failures += 1

            count = integral / (2.0j * np.pi)
            rows_ap_c.append(
                {
                    "N": int(N),
                    "sigma_min": float(rect[0]),
                    "sigma_max": float(rect[1]),
                    "t_min": float(rect[2]),
                    "t_max": float(rect[3]),
                    "n_edge": int(ap_n_edge),
                    "h": float(ap_h),
                    "failures": int(failures),
                    "count_re": float(np.real(count)),
                    "count_im": float(np.imag(count)),
                    "count_abs_im": float(abs(np.imag(count))),
                }
            )

        pd.DataFrame(rows_ap_c).to_csv(os.path.join(out_dir, "E_arg_principle_c_emp.csv"), index=False)

    # Summary report mapped to H1–H6
    md = "# Doc validation track (end-of-doc checklist)\n\n"
    md += f"Doc meta: {doc_meta}\n\n"
    md += "## H1 (holomorphic dependence)\n"
    md += f"- completion_mode: `{completion_mode}`\n"
    md += f"- scan: {h1_flags}\n\n"

    md += "## H2 (DN meromorphy proxy)\n"
    md += "- See `E_spine_checks.csv` columns: `cond_Aii`, `cond_Abb` (near-singular blocks indicate poles for DN / boundary reduction).\n\n"

    md += "## H3 (duality / functional equation spine)\n"
    md += "- Bulk duality proxy: minimize `||A(1-s) - D A(s)^* D^{-1}||_F / ||A(1-s)||_F` over candidate block-diagonal D.\n"
    md += "- FE checks: `FE_mat_fro` (matrix), `FE_det_scaled` (scalar det after normalization), `FE_det_dual` (det(1-s)*conj(det(s)) - 1).\n\n"

    md += "## Scalar mismatch c_emp(s)\n"
    md += "- Defined as `c_emp(s) = detS_norm(s) / phi_mod(s)` (raw mismatch before per-N normalization).\n"
    md += "- Unit-modulus proxy: `|c_emp|-1` (see `c_emp_unit_defect`).\n"
    md += "- Involution-style proxy: `c_emp(1-s)*conj(c_emp(s)) - 1` (see `c_emp_FE_dual`).\n"
    md += "- Optional winding test saved in `E_arg_principle_c_emp.csv` when enabled.\n\n"

    md += "## Scalar Fits (Euler vs Completed)\n"
    md += "- `err_vs_euler` compares `detS_norm` to `zeta(2s-1)/zeta(2s)` (direct), `err_vs_euler_inv` compares to its inverse.\n"
    md += "- `err_vs_mod` compares `detS_norm` to `phi_mod=xi(2s-1)/xi(2s)` (direct), `err_vs_mod_inv` compares to its inverse.\n"
    md += "- `ginf_emp_abs` is `|phi_mod/phi_euler|` (archimedean magnitude implied by completion).\n"
    md += "- `err_vs_ginf` compares `detS_norm/phi_euler` to `ginf_emp` (direct), `err_vs_ginf_inv` compares to its inverse.\n\n"

    md += "## Two-Channel Scalarization (2×2 projection)\n"
    md += "- For `two_channel_symmetric`, uses matrix-normalized relative scattering `S_rel = S_model * S_free^{-1}`, then projects to a 2×2 channel-averaged matrix and rotates to symmetric/antisymmetric basis.\n"
    md += "- `chan_sym_best(_err)` and `chan_anti_best(_err)` report which of {phi_mod, 1/phi_mod, phi_euler, 1/phi_euler} best matches that channel scalar.\n"
    md += "- `chan_offdiag_abs` indicates how well the symmetric/antisymmetric basis decouples the two channels.\n\n"

    md += "## Swap-Eigenspace Eigenvalue Fits (b0×b0)\n"
    md += "- Reduces the full boundary scattering into exact Swap ± eigenspaces (dimension b0 each): `S_plus`, `S_minus`.\n"
    md += "- Computed on `S_rel` (so it is comparable to `detS_norm`).\n"
    md += "- `swap_plus_best_err` is the best relative fit achieved by any eigenvalue of `S_plus` vs {phi_mod, 1/phi_mod, phi_euler, 1/phi_euler}.\n"
    md += "- This is the most direct test of the “one eigenchannel is modular” hypothesis.\n\n"

    md += "## H4 (divisor agreement)\n"
    md += "- Argument principle on Q(s)=C(N)*S_model(s)/phi_mod(s): see `E_divisor_arg_principle_Q.csv`.\n"
    md += "- Local factor shape (prime-by-prime): see `E_local_factor_shape.csv` (ratio_abs near 1 is good).\n\n"

    md += "## H5 (growth/order)\n"
    md += "- Finite-N determinants are meromorphic of finite order automatically; the nontrivial task is stability as N increases.\n\n"

    md += "## H6 (normalization at one point)\n"
    md += "- Per-N constants stored in `E_normalization.json` at s0.\n\n"

    # Small pivot tables for quick scan
    try:
        piv = (
            df.groupby(["N", "sigma"], as_index=False)
            .agg(
                bulk_duality_best_rel_med=("bulk_duality_best_rel", "median"),
                FE_det_scaled_med=("FE_det_scaled", "median"),
                FE_det_dual_med=("FE_det_dual", "median"),
                c_emp_abs_med=("c_emp_abs", "median"),
                c_emp_unit_defect_med=("c_emp_unit_defect", "median"),
                c_emp_FE_dual_med=("c_emp_FE_dual", "median"),
                err_vs_euler_med=("err_vs_euler", "median"),
                err_vs_euler_inv_med=("err_vs_euler_inv", "median"),
                err_vs_mod_med=("err_vs_mod", "median"),
                err_vs_mod_inv_med=("err_vs_mod_inv", "median"),
                ginf_emp_abs_med=("ginf_emp_abs", "median"),
                err_vs_ginf_med=("err_vs_ginf", "median"),
                err_vs_ginf_inv_med=("err_vs_ginf_inv", "median"),
                chan_offdiag_abs_med=("chan_offdiag_abs", "median"),
                chan_sym_best_err_med=("chan_sym_best_err", "median"),
                chan_anti_best_err_med=("chan_anti_best_err", "median"),
                swap_plus_best_err_med=("swap_plus_best_err", "median"),
                swap_minus_best_err_med=("swap_minus_best_err", "median"),
                lam_plus_track_relerr_mod_med=("lam_plus_track_relerr_mod", "median"),
                lam_minus_track_relerr_mod_med=("lam_minus_track_relerr_mod", "median"),
                lam_plus_track_gap_med=("lam_plus_track_gap", "median"),
                lam_minus_track_gap_med=("lam_minus_track_gap", "median"),
                lam_plus_track_jump_med=("lam_plus_track_jump", "median"),
                lam_minus_track_jump_med=("lam_minus_track_jump", "median"),
                swap_pm_diff_fro_med=("swap_pm_diff_fro", "median"),
                swap_pm_diff_fro_in_med=("swap_pm_diff_fro_in", "median"),
                swap_plus_best_err_in_med=("swap_plus_best_err_in", "median"),
                swap_minus_best_err_in_med=("swap_minus_best_err_in", "median"),
                lam_plus_track_in_relerr_mod_med=("lam_plus_track_in_relerr_mod", "median"),
                lam_minus_track_in_relerr_mod_med=("lam_minus_track_in_relerr_mod", "median"),
                lam_plus_track_in_gap_med=("lam_plus_track_in_gap", "median"),
                lam_minus_track_in_gap_med=("lam_minus_track_in_gap", "median"),
                lam_plus_track_in_jump_med=("lam_plus_track_in_jump", "median"),
                lam_minus_track_in_jump_med=("lam_minus_track_in_jump", "median"),
                FE_mat_fro_med=("FE_mat_fro", "median"),
                unitarity_med=("unitarity_S", "median"),
            )
            .sort_values(["N", "sigma"], ascending=True)
        )
        md += "## Quick medians over t-grid\n\n"
        md += "| N | sigma | bulk_duality_best_rel_med | FE_det_scaled_med | FE_det_dual_med | c_emp_abs_med | c_emp_unit_defect_med | c_emp_FE_dual_med | err_vs_euler_med | err_vs_euler_inv_med | err_vs_mod_med | err_vs_mod_inv_med | ginf_emp_abs_med | err_vs_ginf_med | err_vs_ginf_inv_med | chan_offdiag_abs_med | chan_sym_best_err_med | chan_anti_best_err_med | swap_plus_best_err_med | swap_minus_best_err_med | lam_plus_relerr_mod_med | lam_minus_relerr_mod_med | lam_plus_gap_med | lam_minus_gap_med | lam_plus_jump_med | lam_minus_jump_med | swap_pm_diff_fro_med | swap_pm_diff_fro_in_med | swap_plus_best_err_in_med | swap_minus_best_err_in_med | lam_plus_in_relerr_mod_med | lam_minus_in_relerr_mod_med | lam_plus_in_gap_med | lam_minus_in_gap_med | lam_plus_in_jump_med | lam_minus_in_jump_med | FE_mat_fro_med | unitarity_med |\n"
        md += "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        for _, r in piv.iterrows():
            md += (
                f"| {int(r['N'])} | {float(r['sigma'])} | {float(r['bulk_duality_best_rel_med']):.3g} |"
                f" {float(r['FE_det_scaled_med']):.3g} | {float(r['FE_det_dual_med']):.3g} | {float(r['c_emp_abs_med']):.3g} |"
                f" {float(r['c_emp_unit_defect_med']):.3g} | {float(r['c_emp_FE_dual_med']):.3g} | {float(r['err_vs_euler_med']):.3g} |"
                f" {float(r['err_vs_euler_inv_med']):.3g} | {float(r['err_vs_mod_med']):.3g} | {float(r['err_vs_mod_inv_med']):.3g} |"
                f" {float(r['ginf_emp_abs_med']):.3g} | {float(r['err_vs_ginf_med']):.3g} | {float(r['err_vs_ginf_inv_med']):.3g} |"
                f" {float(r['chan_offdiag_abs_med']):.3g} | {float(r['chan_sym_best_err_med']):.3g} | {float(r['chan_anti_best_err_med']):.3g} |"
                f" {float(r['swap_plus_best_err_med']):.3g} | {float(r['swap_minus_best_err_med']):.3g} |"
                f" {float(r['lam_plus_track_relerr_mod_med']):.3g} | {float(r['lam_minus_track_relerr_mod_med']):.3g} |"
                f" {float(r['lam_plus_track_gap_med']):.3g} | {float(r['lam_minus_track_gap_med']):.3g} |"
                f" {float(r['lam_plus_track_jump_med']):.3g} | {float(r['lam_minus_track_jump_med']):.3g} |"
                f" {float(r['swap_pm_diff_fro_med']):.3g} | {float(r['swap_pm_diff_fro_in_med']):.3g} |"
                f" {float(r['swap_plus_best_err_in_med']):.3g} | {float(r['swap_minus_best_err_in_med']):.3g} |"
                f" {float(r['lam_plus_track_in_relerr_mod_med']):.3g} | {float(r['lam_minus_track_in_relerr_mod_med']):.3g} |"
                f" {float(r['lam_plus_track_in_gap_med']):.3g} | {float(r['lam_minus_track_in_gap_med']):.3g} |"
                f" {float(r['lam_plus_track_in_jump_med']):.3g} | {float(r['lam_minus_track_in_jump_med']):.3g} |"
                f" {float(r['FE_mat_fro_med']):.3g} | {float(r['unitarity_med']):.3g} |\n"
            )
        md += "\n"
    except Exception:
        pass

    with open(os.path.join(out_dir, "E_summary.md"), "w", encoding="utf-8") as f:
        f.write(md)

    # Run dip-atlas last so the optional suggested-rectangle winding can reuse the
    # Q_lambda winding helper defined in the argument-principle section.
    _dip_atlas_and_optional_winding()

    # Freeze a run contract: fail the run if required artifacts are missing.
    if bool(config.get("run_contract_enforce", True)):
        _validate_run_contract(out_dir, config)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
