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

        conv_do_winding = bool(config.get("convergence_do_winding", True))
        if conv_do_winding:
            conv_wind_out = str(config.get("convergence_wind_out_csv", "E_Qlambda_convergence_winding.csv"))
            required.append(conv_wind_out)

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

    generator_norm = config.get("generator_norm", "fro")
    if generator_norm is not None:
        generator_norm = str(generator_norm)
    generator_norm_target = float(config.get("generator_norm_target", 1.0))

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
        """Return the Swap± reduced b0xb0 matrix from a 2b0x2b0 scattering matrix."""

        S = np.asarray(S, dtype=np.complex128)
        S11 = S[:b0, :b0]
        S12 = S[:b0, b0 : 2 * b0]
        S21 = S[b0 : 2 * b0, :b0]
        S22 = S[b0 : 2 * b0, b0 : 2 * b0]
        if block in {"+", "plus", "p"}:
            return (0.5 * (S11 + S12 + S21 + S22)).astype(np.complex128)
        if block in {"-", "minus", "m"}:
            return (0.5 * (S11 - S12 - S21 + S22)).astype(np.complex128)
        raise ValueError("block must be 'plus' or 'minus'")

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

                    phi = phi_modular(s)
                    if (not is_finite_complex(phi)) or abs(phi) <= 1e-30:
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
                            j_phi, _ = _min_relerr_to_target(eigs, phi)
                            lam_phi = complex(eigs[int(j_phi)])
                            q_phi = lam_phi / phi
                            err_phi = float(abs(q_phi - 1.0)) if is_finite_complex(q_phi) else float("nan")

                            # Local spectral gap diagnostic for the chosen eigenvalue.
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
                                cast_gaps.append(gap)

                            # Convention 2: lambda ~ 1/phi
                            invphi = (1.0 / phi) if phi != 0 else complex("nan")
                            if is_finite_complex(invphi) and abs(invphi) > 0:
                                j_inv, _ = _min_relerr_to_target(eigs, invphi)
                                lam_inv = complex(eigs[int(j_inv)])
                                q_inv = lam_inv * phi
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

            for basis in ["rel", "incoming"]:
                for block2 in ["plus", "minus"]:
                    a = acc[(basis, block2)]
                    q_phi_samples = a["q_phi_samples"]
                    errs_phi = a["errs_phi"]
                    errs_inv = a["errs_inv"]
                    errs_best = a["errs_best"]
                    gaps_phi = a["gaps_phi"]
                    phi_wins = int(a["phi_wins"])
                    total = int(a["total"])
                    failures = int(a["failures"])

                    assert isinstance(q_phi_samples, list)
                    assert isinstance(errs_phi, list)
                    assert isinstance(errs_inv, list)
                    assert isinstance(errs_best, list)
                    assert isinstance(gaps_phi, list)

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

        if stats_rows:
            pd.DataFrame(stats_rows).to_csv(os.path.join(out_dir, str(out_csv_name)), index=False)

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

                phi = phi_modular(s)
                if (not is_finite_complex(phi)) or abs(phi) <= 1e-30:
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

                j_phi, _ = _min_relerr_to_target(eigs, phi)
                lam_phi = complex(eigs[int(j_phi)])
                q_phi = lam_phi / phi
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

                phi = phi_modular(s)
                if (not is_finite_complex(phi)) or abs(phi) <= 1e-30:
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

                        j_phi, _ = _min_relerr_to_target(eigs, phi)
                        lam_phi = complex(eigs[int(j_phi)])
                        q_phi = lam_phi / phi
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

                phi = phi_modular(s)
                if (not is_finite_complex(phi)) or abs(phi) <= 1e-30:
                    raise ValueError("bad phi")

                B = _swap_block_matrix(M, "plus" if block_key == "+" else "minus", b0=b0)
                eigs = np.asarray(np.linalg.eigvals(B), dtype=np.complex128).ravel()
                if eigs.size == 0:
                    raise ValueError("no eigs")

                if i == 0 or not is_finite_complex(prev):
                    j_idx, _ = _min_relerr_to_target(eigs, phi)
                elif i == (len(ptsW) - 1) and is_finite_complex(first):
                    j_idx = _nearest_neighbor_index(eigs, first)
                else:
                    j_idx = _nearest_neighbor_index(eigs, prev)
                lam = complex(eigs[int(j_idx)])
                if i == 0:
                    first = lam
                prev = lam
                qs.append(lam / phi)
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

                                    phi = phi_modular(s)
                                    if (not is_finite_complex(phi)) or abs(phi) <= 1e-30:
                                        raise ValueError("bad phi")

                                    b0 = int(cfgW.b // 2)
                                    B = _swap_block_matrix(M, "plus" if block_key == "+" else "minus", b0=b0)
                                    eigs = np.asarray(np.linalg.eigvals(B), dtype=np.complex128).ravel()
                                    if eigs.size == 0:
                                        raise ValueError("no eigs")

                                    if i == 0 or not is_finite_complex(prev):
                                        j_idx, _ = _min_relerr_to_target(eigs, phi)
                                    elif i == (len(ptsW) - 1) and is_finite_complex(first):
                                        j_idx = _nearest_neighbor_index(eigs, first)
                                    else:
                                        j_idx = _nearest_neighbor_index(eigs, prev)
                                    lam = complex(eigs[int(j_idx)])
                                    if i == 0:
                                        first = lam
                                    prev = lam
                                    qs.append(lam / phi)
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
        conv_wind_basis = str(config.get("convergence_wind_basis", "incoming")).strip().lower()
        conv_wind_block = str(config.get("convergence_wind_block", "plus")).strip().lower()
        conv_wind_n_edge = int(config.get("convergence_wind_n_edge", 800))
        conv_wind_out = str(config.get("convergence_wind_out_csv", "E_Qlambda_convergence_winding.csv"))
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
        )

        # Lightweight fit summaries (trend vs 1/N and power law vs N) for phi errors.
        try:
            df_conv = pd.read_csv(os.path.join(out_dir, "E_Qlambda_convergence_sweep.csv"))
        except Exception:
            return 0

        fit_rows: list[dict] = []
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

        if fit_rows:
            pd.DataFrame(fit_rows).to_csv(os.path.join(out_dir, "E_Qlambda_convergence_fit.csv"), index=False)

        # Optional: winding sanity checks for Q_lambda on the rectangle boundary, per N.
        if conv_do_winding:
            wind_rows: list[dict] = []
            ptsW = _boundary_points(convergence_rect, int(conv_wind_n_edge))
            # Avoid caching giant matrices in a sweep.
            model_fnW = getattr(_model_mats, "__wrapped__", _model_mats)
            free_fnW = getattr(_free_mats, "__wrapped__", _free_mats)
            for N in [int(x) for x in Ns]:
                cfgW = _mk_cfg(int(N))
                qs: list[complex] = []
                prev = complex("nan")
                first = complex("nan")
                failures = 0
                for i, s in enumerate(ptsW):
                    try:
                        _, Lam, Sm = model_fnW(s, cfgW)
                        _, _, S0 = free_fnW(s, cfgW)
                        Srel = (Sm @ np.linalg.inv(S0)).astype(np.complex128)
                        if conv_wind_basis == "incoming":
                            nb = int(Lam.shape[0])
                            Tin = (Lam + 1j * float(cfgW.eta) * np.eye(nb, dtype=np.complex128)).astype(np.complex128)
                            if float(cfgW.cayley_eps) != 0.0:
                                Tin = (Tin + float(cfgW.cayley_eps) * np.eye(nb, dtype=np.complex128)).astype(np.complex128)
                            M = np.linalg.solve(Tin, (Srel @ Tin)).astype(np.complex128)
                        else:
                            M = Srel

                        phi = phi_modular(s)
                        if (not is_finite_complex(phi)) or abs(phi) <= 1e-30:
                            raise ValueError("bad phi")

                        b0 = int(cfgW.b // 2)
                        B = _swap_block_matrix(M, "plus" if conv_block_key == "+" else "minus", b0=b0)
                        eigs = np.asarray(np.linalg.eigvals(B), dtype=np.complex128).ravel()
                        if eigs.size == 0:
                            raise ValueError("no eigs")

                        if i == 0 or not is_finite_complex(prev):
                            j_idx, _ = _min_relerr_to_target(eigs, phi)
                        elif i == (len(ptsW) - 1) and is_finite_complex(first):
                            j_idx = _nearest_neighbor_index(eigs, first)
                        else:
                            j_idx = _nearest_neighbor_index(eigs, prev)
                        lam = complex(eigs[int(j_idx)])
                        if i == 0:
                            first = lam
                        prev = lam
                        qs.append(lam / phi)
                    except Exception:
                        qs.append(complex("nan"))
                        failures += 1

                w, wfail = _winding_count_from_samples(qs)
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
                        "winding": float(w),
                        "winding_abs": float(abs(w)) if np.isfinite(w) else float("nan"),
                        "failures": int(failures + wfail),
                        "eta": float(cfgW.eta),
                        "schur_jitter": float(cfgW.jitter),
                        "cayley_eps": float(cfgW.cayley_eps),
                    }
                )

            if wind_rows:
                pd.DataFrame(wind_rows).to_csv(os.path.join(out_dir, conv_wind_out), index=False)

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

                phi = phi_modular(s) if pole_line_include_phi else complex("nan")
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

                phi = phi_modular(s)
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
                    phi = phi_modular(s)
                    phi_e = phi_euler_only(s)
                    targets = {
                        "phi_mod": phi,
                        "inv_phi_mod": safe_div(1.0 + 0.0j, phi),
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
                    phi = phi_modular(s)
                    phi_e = phi_euler_only(s)
                    targets = {
                        "phi_mod": phi,
                        "inv_phi_mod": safe_div(1.0 + 0.0j, phi),
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
            ph = phi_modular(s)
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

                    phi = phi_modular(s)
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

                        phi = phi_modular(s)
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
                ph = phi_modular(s)
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
