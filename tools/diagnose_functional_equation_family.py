from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import mpmath as mp  # type: ignore

    _HAS_MPMATH = True
except Exception:  # pragma: no cover
    mp = None  # type: ignore
    _HAS_MPMATH = False

# Import simulator module from sibling file (tools/ is not a package).
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
import six_by_six_prime_tower_sim as sim  # type: ignore


def _trapz_weights(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float).ravel()
    n = int(t.size)
    if n <= 0:
        return np.asarray([], dtype=float)
    if n == 1:
        return np.asarray([1.0], dtype=float)
    dt = np.diff(t)
    w = np.zeros(n, dtype=float)
    w[0] = float(dt[0]) * 0.5
    w[-1] = float(dt[-1]) * 0.5
    if n > 2:
        w[1:-1] = 0.5 * (dt[:-1] + dt[1:])
    s = float(np.sum(w))
    if not math.isfinite(s) or s <= 0:
        return np.full(n, 1.0 / float(n), dtype=float)
    return (w / s).astype(float)


def _weighted_norm(w: np.ndarray, z: np.ndarray) -> float:
    w = np.asarray(w, dtype=float).ravel()
    z = np.asarray(z, dtype=np.complex128).ravel()
    if w.size != z.size:
        raise ValueError("w and z must have same length")
    return float(math.sqrt(float(np.sum(w * (np.abs(z) ** 2)))))


def _compute_det_and_logdet(s: complex, packets: list[sim.Packet]) -> tuple[complex, complex]:
    K, _ = sim._global_K(s, packets)
    I = np.eye(K.shape[0], dtype=np.complex128)
    M = (I - K).astype(np.complex128)

    det_val = complex(np.linalg.det(M))
    try:
        sign, logabs = np.linalg.slogdet(M)
        logdet_val = complex(np.log(sign) + logabs)
    except Exception:
        logdet_val = complex("nan")
    return det_val, logdet_val


def _logabs(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.complex128)
    return np.log(np.abs(z) + 1e-300).astype(np.float64)


@dataclass(frozen=True)
class SymmetryMetrics:
    rel_l2_det: float
    rel_l2_logdet: float
    max_abs_det_diff: float
    max_abs_logdet_diff: float
    # Baseline: best-fit completion exp(c0 + c1*s) in log-space.
    completion_lin_rel_l2_logdet: float
    completion_lin_c0_re: float
    completion_lin_c0_im: float
    completion_lin_c1_re: float
    completion_lin_c1_im: float
    # Rich completion: chosen basis (poly2 or poly2+logGamma) in log-space.
    completion_rich_basis: str
    completion_rich_rel_l2_logdet: float
    completion_rich_k: int
    completion_rich_coef0_re: float
    completion_rich_coef0_im: float
    completion_rich_coef1_re: float
    completion_rich_coef1_im: float
    completion_rich_coef2_re: float
    completion_rich_coef2_im: float
    completion_rich_coef3_re: float
    completion_rich_coef3_im: float
    completion_rich_coef4_re: float
    completion_rich_coef4_im: float


@dataclass(frozen=True)
class AbsSymmetryMetrics:
    rel_l2_logabs: float
    max_abs_logabs_diff: float
    # Baseline: best-fit linear completion on logabs: log|D(s)| + a0 + a1*Re(s) ≈ log|D(sym(s))|.
    completion_lin_rel_l2_logabs: float
    completion_lin_a0: float
    completion_lin_a1: float
    # Rich completion on logabs: basis-dependent real least squares.
    completion_rich_basis: str
    completion_rich_rel_l2_logabs: float
    completion_rich_k: int
    completion_rich_coef0: float
    completion_rich_coef1: float
    completion_rich_coef2: float
    completion_rich_coef3: float
    completion_rich_coef4: float


def _loggamma_vec(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.complex128).ravel()
    if not _HAS_MPMATH:
        raise RuntimeError("mpmath is required for completion basis with logGamma terms")
    out = np.zeros_like(z, dtype=np.complex128)
    for i in range(int(z.size)):
        zi = complex(z[i])
        out[i] = complex(mp.loggamma(zi))
    return out.astype(np.complex128)


def _completion_design_logdet(s_vals: np.ndarray, *, basis: str) -> np.ndarray:
    """Return complex design matrix X(s) for completion in log-space.

    The completion adjustment is modeled as: logD(s) + X(s) @ c ≈ logD(sym(s)).
    """

    s_vals = np.asarray(s_vals, dtype=np.complex128).ravel()
    basis = str(basis).strip().lower()
    if basis in {"none", ""}:
        return np.zeros((int(s_vals.size), 0), dtype=np.complex128)
    if basis == "poly2":
        cols = [
            np.ones_like(s_vals),
            s_vals,
            s_vals**2,
        ]
        return np.stack(cols, axis=1).astype(np.complex128)
    if basis == "poly2_gamma":
        lg0 = _loggamma_vec(s_vals / 2.0)
        lg1 = _loggamma_vec((s_vals + 1.0) / 2.0)
        cols = [
            np.ones_like(s_vals),
            s_vals,
            s_vals**2,
            lg0,
            lg1,
        ]
        return np.stack(cols, axis=1).astype(np.complex128)
    raise ValueError("completion_basis must be one of: none, poly2, poly2_gamma")


def _fit_completion_complex(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve X c ≈ y in least squares for complex c (returns c)."""
    X = np.asarray(X, dtype=np.complex128)
    y = np.asarray(y, dtype=np.complex128).ravel()
    if X.shape[0] != y.size:
        raise ValueError("X and y must have compatible shapes")
    if X.size == 0 or X.shape[1] == 0:
        return np.zeros((0,), dtype=np.complex128)

    A_real = np.block(
        [
            [np.real(X), -np.imag(X)],
            [np.imag(X), np.real(X)],
        ]
    )
    y_real = np.concatenate([np.real(y), np.imag(y)], axis=0)
    coef, *_ = np.linalg.lstsq(A_real, y_real.reshape(-1, 1), rcond=None)
    coef = coef.reshape(-1)
    m = int(X.shape[1])
    c_re = coef[:m]
    c_im = coef[m:]
    return (c_re + 1j * c_im).astype(np.complex128)


def _fit_completion_on_logdet_basis(
    s_vals: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    basis: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit completion in log-space on a chosen basis.

    Returns (logdet_adjusted, coef_vector).
    """
    s_vals = np.asarray(s_vals, dtype=np.complex128).ravel()
    a = np.asarray(a, dtype=np.complex128).ravel()
    b = np.asarray(b, dtype=np.complex128).ravel()
    if s_vals.size != a.size or a.size != b.size or a.size == 0:
        return np.full_like(a, np.nan + 1j * np.nan), np.full((0,), np.nan + 1j * np.nan)

    X = _completion_design_logdet(s_vals, basis=str(basis))
    if X.shape[1] == 0:
        return a.astype(np.complex128), np.zeros((0,), dtype=np.complex128)
    y = (b - a).astype(np.complex128)
    c = _fit_completion_complex(X, y)
    adj = (X @ c.reshape(-1, 1)).reshape(-1).astype(np.complex128)
    return (a + adj).astype(np.complex128), c


def _fit_completion_on_logabs(s_vals: np.ndarray, logabs_s: np.ndarray, logabs_sym: np.ndarray) -> tuple[np.ndarray, float, float]:
    s_vals = np.asarray(s_vals, dtype=np.complex128).ravel()
    x = np.real(s_vals).astype(float)
    y = (np.asarray(logabs_sym, dtype=float).ravel() - np.asarray(logabs_s, dtype=float).ravel()).astype(float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 2:
        return np.full_like(np.asarray(logabs_s, dtype=float).ravel(), np.nan), float("nan"), float("nan")

    X = np.stack([np.ones_like(x), x], axis=1).astype(float)
    coef, *_ = np.linalg.lstsq(X, y.reshape(-1, 1), rcond=None)
    coef = coef.reshape(-1)
    a0 = float(coef[0])
    a1 = float(coef[1])

    x_all = np.real(np.asarray(s_vals, dtype=np.complex128).ravel()).astype(float)
    adj = (a0 + a1 * x_all).astype(float)
    return (np.asarray(logabs_s, dtype=float).ravel() + adj).astype(float), a0, a1


def _completion_design_logabs(s_vals: np.ndarray, *, basis: str) -> np.ndarray:
    s_vals = np.asarray(s_vals, dtype=np.complex128).ravel()
    basis = str(basis).strip().lower()
    if basis in {"none", ""}:
        return np.zeros((int(s_vals.size), 0), dtype=float)
    if basis == "poly2":
        x = np.real(s_vals).astype(float)
        cols = [
            np.ones_like(x),
            x,
            x**2,
        ]
        return np.stack(cols, axis=1).astype(float)
    if basis == "poly2_gamma":
        x = np.real(s_vals).astype(float)
        lg0 = np.real(_loggamma_vec(s_vals / 2.0)).astype(float)
        lg1 = np.real(_loggamma_vec((s_vals + 1.0) / 2.0)).astype(float)
        cols = [
            np.ones_like(x),
            x,
            x**2,
            lg0,
            lg1,
        ]
        return np.stack(cols, axis=1).astype(float)
    raise ValueError("completion_basis must be one of: none, poly2, poly2_gamma")


def _fit_completion_on_logabs_basis(
    s_vals: np.ndarray,
    logabs_s: np.ndarray,
    logabs_sym: np.ndarray,
    *,
    basis: str,
) -> tuple[np.ndarray, np.ndarray]:
    s_vals = np.asarray(s_vals, dtype=np.complex128).ravel()
    logabs_s = np.asarray(logabs_s, dtype=float).ravel()
    logabs_sym = np.asarray(logabs_sym, dtype=float).ravel()
    if s_vals.size != logabs_s.size or logabs_s.size != logabs_sym.size or s_vals.size == 0:
        return np.full_like(logabs_s, np.nan), np.full((0,), np.nan)

    X = _completion_design_logabs(s_vals, basis=str(basis))
    if X.shape[1] == 0:
        return logabs_s.astype(float), np.zeros((0,), dtype=float)
    y = (logabs_sym - logabs_s).astype(float)
    coef, *_ = np.linalg.lstsq(X, y.reshape(-1, 1), rcond=None)
    coef = coef.reshape(-1).astype(float)
    adj = (X @ coef.reshape(-1, 1)).reshape(-1).astype(float)
    return (logabs_s + adj).astype(float), coef


def _abs_symmetry_metrics(
    *,
    s_vals: np.ndarray,
    logabs_s: np.ndarray,
    logabs_sym: np.ndarray,
    w: np.ndarray,
    completion_basis: str,
) -> AbsSymmetryMetrics:
    w = np.asarray(w, dtype=float).ravel()
    logabs_s = np.asarray(logabs_s, dtype=float).ravel()
    logabs_sym = np.asarray(logabs_sym, dtype=float).ravel()
    denom = float(math.sqrt(float(np.sum(w * (logabs_s**2))))) + 1e-300
    diff = (logabs_s - logabs_sym).astype(float)
    rel = float(math.sqrt(float(np.sum(w * (diff**2)))) / denom)
    mx = float(np.max(np.abs(logabs_s - logabs_sym)))

    adj, a0, a1 = _fit_completion_on_logabs(s_vals, logabs_s, logabs_sym)
    diff_comp = (np.asarray(adj, dtype=float).ravel() - logabs_sym).astype(float)
    rel_comp = float(math.sqrt(float(np.sum(w * (diff_comp**2)))) / denom)

    rich_adj, rich_coef = _fit_completion_on_logabs_basis(s_vals, logabs_s, logabs_sym, basis=str(completion_basis))
    rich_diff = (np.asarray(rich_adj, dtype=float).ravel() - logabs_sym).astype(float)
    rich_rel = float(math.sqrt(float(np.sum(w * (rich_diff**2)))) / denom)
    rich_coef = np.asarray(rich_coef, dtype=float).reshape(-1)
    rich_pad = np.full((5,), np.nan, dtype=float)
    rich_pad[: min(5, int(rich_coef.size))] = rich_coef[: min(5, int(rich_coef.size))]

    return AbsSymmetryMetrics(
        rel_l2_logabs=float(rel),
        max_abs_logabs_diff=float(mx),
        completion_lin_rel_l2_logabs=float(rel_comp),
        completion_lin_a0=float(a0),
        completion_lin_a1=float(a1),
        completion_rich_basis=str(completion_basis),
        completion_rich_rel_l2_logabs=float(rich_rel),
        completion_rich_k=int(rich_coef.size),
        completion_rich_coef0=float(rich_pad[0]),
        completion_rich_coef1=float(rich_pad[1]),
        completion_rich_coef2=float(rich_pad[2]),
        completion_rich_coef3=float(rich_pad[3]),
        completion_rich_coef4=float(rich_pad[4]),
    )


def _fit_completion_on_logdet(s_vals: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, complex, complex]:
    """Find c0,c1 such that a + c0 + c1*s ≈ b in least squares over complex arrays."""

    s_vals = np.asarray(s_vals, dtype=np.complex128).ravel()
    a = np.asarray(a, dtype=np.complex128).ravel()
    b = np.asarray(b, dtype=np.complex128).ravel()
    if s_vals.size != a.size or a.size != b.size or a.size == 0:
        return np.full_like(a, np.nan + 1j * np.nan), complex("nan"), complex("nan")

    # Solve in real form for stability: [Re,Im] stacked.
    X = np.stack([np.ones_like(s_vals), s_vals], axis=1)  # complex design
    y = (b - a).astype(np.complex128)

    A_real = np.block(
        [
            [np.real(X), -np.imag(X)],
            [np.imag(X), np.real(X)],
        ]
    )
    y_real = np.concatenate([np.real(y), np.imag(y)], axis=0)

    coef, *_ = np.linalg.lstsq(A_real, y_real.reshape(-1, 1), rcond=None)
    coef = coef.reshape(-1)
    c0 = complex(float(coef[0]), float(coef[2]))
    c1 = complex(float(coef[1]), float(coef[3]))

    adj = (c0 + c1 * s_vals).astype(np.complex128)
    return (a + adj).astype(np.complex128), c0, c1


def _symmetry_metrics(
    *,
    s_vals: np.ndarray,
    det_s: np.ndarray,
    logdet_s: np.ndarray,
    det_sym: np.ndarray,
    logdet_sym: np.ndarray,
    w: np.ndarray,
    completion_basis: str,
) -> SymmetryMetrics:
    w = np.asarray(w, dtype=float).ravel()

    denom_det = _weighted_norm(w, det_s) + 1e-300
    denom_log = _weighted_norm(w, logdet_s) + 1e-300

    rel_det = _weighted_norm(w, det_s - det_sym) / denom_det
    rel_log = _weighted_norm(w, logdet_s - logdet_sym) / denom_log
    max_det = float(np.max(np.abs(det_s - det_sym)))
    max_log = float(np.max(np.abs(logdet_s - logdet_sym)))

    # Completion fit on logdet: logD(s) + c0 + c1*s ≈ logD(sym(s))
    logdet_adj, c0, c1 = _fit_completion_on_logdet(s_vals, logdet_s, logdet_sym)
    rel_log_comp = _weighted_norm(w, logdet_adj - logdet_sym) / denom_log

    # Rich completion fit on logdet basis.
    logdet_rich, cvec = _fit_completion_on_logdet_basis(s_vals, logdet_s, logdet_sym, basis=str(completion_basis))
    rel_log_rich = _weighted_norm(w, logdet_rich - logdet_sym) / denom_log
    cvec = np.asarray(cvec, dtype=np.complex128).reshape(-1)
    cpad = np.full((5,), np.nan + 1j * np.nan, dtype=np.complex128)
    cpad[: min(5, int(cvec.size))] = cvec[: min(5, int(cvec.size))]

    return SymmetryMetrics(
        rel_l2_det=float(rel_det),
        rel_l2_logdet=float(rel_log),
        max_abs_det_diff=float(max_det),
        max_abs_logdet_diff=float(max_log),
        completion_lin_rel_l2_logdet=float(rel_log_comp),
        completion_lin_c0_re=float(np.real(c0)),
        completion_lin_c0_im=float(np.imag(c0)),
        completion_lin_c1_re=float(np.real(c1)),
        completion_lin_c1_im=float(np.imag(c1)),
        completion_rich_basis=str(completion_basis),
        completion_rich_rel_l2_logdet=float(rel_log_rich),
        completion_rich_k=int(cvec.size),
        completion_rich_coef0_re=float(np.real(cpad[0])),
        completion_rich_coef0_im=float(np.imag(cpad[0])),
        completion_rich_coef1_re=float(np.real(cpad[1])),
        completion_rich_coef1_im=float(np.imag(cpad[1])),
        completion_rich_coef2_re=float(np.real(cpad[2])),
        completion_rich_coef2_im=float(np.imag(cpad[2])),
        completion_rich_coef3_re=float(np.real(cpad[3])),
        completion_rich_coef3_im=float(np.imag(cpad[3])),
        completion_rich_coef4_re=float(np.real(cpad[4])),
        completion_rich_coef4_im=float(np.imag(cpad[4])),
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Functional-equation diagnostic for the sixby6 determinant family D_u(s)=det(I-K_u(s)). "
            "Computes D_u(s) on a vertical line s=sigma+it and compares to D_u(1-s) and D_u(1-conj(s)). "
            "Also fits a simple 'completion' factor exp(c0 + c1*s) in log-space to see whether a linear completion can "
            "make logD approximately symmetric."
        )
    )

    ap.add_argument("--u_list", default="0,0.05,0.1,0.15,0.2")
    ap.add_argument("--primes_global", default="2,3,5,7,11,13")

    ap.add_argument(
        "--completion_basis",
        choices=["none", "poly2", "poly2_gamma"],
        default="poly2",
        help=(
            "Optional richer completion basis used in addition to the baseline linear completion. "
            "'poly2' fits (1,s,s^2) in log-space; 'poly2_gamma' adds logGamma(s/2) and logGamma((s+1)/2)."
        ),
    )

    ap.add_argument(
        "--sigma",
        type=float,
        default=0.3,
        help=(
            "Real part sigma for s=sigma+it. Using sigma!=0.5 makes the s->1-s comparison nontrivial; "
            "at sigma=0.5, 1-s is just conjugation on the sampled line."
        ),
    )
    ap.add_argument("--t_min", type=float, default=10.0)
    ap.add_argument("--t_max", type=float, default=30.0)
    ap.add_argument("--n_t", type=int, default=401)

    ap.add_argument("--out_csv", required=True)

    # Same local model conventions as the sweep
    ap.add_argument("--sharp", choices=["transpose", "conj_transpose"], default="conj_transpose")
    ap.add_argument("--X_mode", choices=["raw", "det1"], default="det1")
    ap.add_argument("--X_gamma", type=float, default=1.0)
    ap.add_argument("--scattering", choices=["lambda_pm_i", "i_pm_lambda"], default="i_pm_lambda")

    args = ap.parse_args()

    u_list = sim._parse_float_list(str(args.u_list))
    primes_global = sim._parse_int_list(str(args.primes_global))
    if not u_list or not primes_global:
        raise SystemExit("--u_list and --primes_global must be non-empty")

    t_grid = np.linspace(float(args.t_min), float(args.t_max), int(args.n_t), dtype=float)
    w = _trapz_weights(t_grid)

    # Boundary search seed set (reuse per u)
    p_seed = [int(primes_global[0]), int(primes_global[-1])] if len(primes_global) >= 2 else [int(primes_global[0])]
    k_seed = [1]

    rows: list[dict] = []

    for u in u_list:
        u = float(u)
        v = -float(u)

        boundary, schur_sign, seed_hd, seed_ud = sim._boundary_search(
            p_seed,
            k_seed,
            sharp_mode=str(args.sharp),
            x_mode=str(args.X_mode),
            x_gamma=float(args.X_gamma),
            x_shear=float(u),
            x_lower=float(v),
            scattering_mode=str(args.scattering),
        )

        packets = sim._build_packets(
            primes_global,
            1,
            local_model="sixby6",
            boundary=boundary,
            sign=str(schur_sign),
            sharp_mode=str(args.sharp),
            x_mode=str(args.X_mode),
            x_gamma=float(args.X_gamma),
            x_shear=float(u),
            x_lower=float(v),
            scattering_mode=str(args.scattering),
            satake_family="trivial",
            satake_matrix="diag",
            theta_scale=1.0,
            seed=0,
        )

        sigma = float(args.sigma)
        if abs(sigma - 0.5) < 1e-12:
            # Keep this as a runtime message (not an exception) so the user can intentionally run sigma=0.5.
            print("note: sigma=0.5 makes s->1-s coincide with conjugation on this vertical line; use sigma!=0.5 for a nontrivial FE test")
        s_vals = (sigma + 1j * t_grid).astype(np.complex128)
        s_1ms = (1.0 - sigma - 1j * t_grid).astype(np.complex128)
        s_1mcs = (1.0 - sigma + 1j * t_grid).astype(np.complex128)

        det_s = np.zeros_like(s_vals, dtype=np.complex128)
        log_s = np.zeros_like(s_vals, dtype=np.complex128)
        det_1ms = np.zeros_like(s_vals, dtype=np.complex128)
        log_1ms = np.zeros_like(s_vals, dtype=np.complex128)
        det_1mcs = np.zeros_like(s_vals, dtype=np.complex128)
        log_1mcs = np.zeros_like(s_vals, dtype=np.complex128)

        for i in range(int(s_vals.size)):
            det_s[i], log_s[i] = _compute_det_and_logdet(complex(s_vals[i]), packets)
            det_1ms[i], log_1ms[i] = _compute_det_and_logdet(complex(s_1ms[i]), packets)
            det_1mcs[i], log_1mcs[i] = _compute_det_and_logdet(complex(s_1mcs[i]), packets)

        completion_basis = str(args.completion_basis)
        if completion_basis == "poly2_gamma" and not _HAS_MPMATH:
            raise SystemExit("--completion_basis poly2_gamma requires mpmath")

        m_1ms = _symmetry_metrics(
            s_vals=s_vals,
            det_s=det_s,
            logdet_s=log_s,
            det_sym=det_1ms,
            logdet_sym=log_1ms,
            w=w,
            completion_basis=completion_basis,
        )
        m_1mcs = _symmetry_metrics(
            s_vals=s_vals,
            det_s=det_s,
            logdet_s=log_s,
            det_sym=det_1mcs,
            logdet_sym=log_1mcs,
            w=w,
            completion_basis=completion_basis,
        )

        # Often the meaningful critical-line comparison is against conjugation.
        # Report det-based symmetry: D(s) vs conj(D(1-s)).
        det_conj_1ms = np.conjugate(det_1ms).astype(np.complex128)
        log_conj_1ms = np.conjugate(log_1ms).astype(np.complex128)
        m_conj_1ms = _symmetry_metrics(
            s_vals=s_vals,
            det_s=det_s,
            logdet_s=log_s,
            det_sym=det_conj_1ms,
            logdet_sym=log_conj_1ms,
            w=w,
            completion_basis=completion_basis,
        )

        # Magnitude-only symmetry on det.
        abs_1ms = _abs_symmetry_metrics(
            s_vals=s_vals,
            logabs_s=_logabs(det_s),
            logabs_sym=_logabs(det_1ms),
            w=w,
            completion_basis=completion_basis,
        )
        abs_conj_1ms = _abs_symmetry_metrics(
            s_vals=s_vals,
            logabs_s=_logabs(det_s),
            logabs_sym=_logabs(det_conj_1ms),
            w=w,
            completion_basis=completion_basis,
        )

        row = {
            "u": float(u),
            "v": float(v),
            "sigma": float(sigma),
            "completion_basis": str(args.completion_basis),
            "boundary": f"{boundary[0]},{boundary[1]}",
            "schur_sign": str(schur_sign),
            "seed_max_herm_def": float(seed_hd),
            "seed_max_unit_def": float(seed_ud),
            "t_min": float(args.t_min),
            "t_max": float(args.t_max),
            "n_t": int(args.n_t),
            "primes_global": str(args.primes_global),
            # s -> 1 - s
            "rel_l2_det_1ms": float(m_1ms.rel_l2_det),
            "rel_l2_logdet_1ms": float(m_1ms.rel_l2_logdet),
            "max_abs_det_diff_1ms": float(m_1ms.max_abs_det_diff),
            "max_abs_logdet_diff_1ms": float(m_1ms.max_abs_logdet_diff),
            # Baseline linear completion
            "completion_lin_rel_l2_logdet_1ms": float(m_1ms.completion_lin_rel_l2_logdet),
            "completion_lin_c0_re_1ms": float(m_1ms.completion_lin_c0_re),
            "completion_lin_c0_im_1ms": float(m_1ms.completion_lin_c0_im),
            "completion_lin_c1_re_1ms": float(m_1ms.completion_lin_c1_re),
            "completion_lin_c1_im_1ms": float(m_1ms.completion_lin_c1_im),
            # Rich completion
            "completion_rich_rel_l2_logdet_1ms": float(m_1ms.completion_rich_rel_l2_logdet),
            "completion_rich_k_1ms": int(m_1ms.completion_rich_k),
            "completion_rich_coef0_re_1ms": float(m_1ms.completion_rich_coef0_re),
            "completion_rich_coef0_im_1ms": float(m_1ms.completion_rich_coef0_im),
            "completion_rich_coef1_re_1ms": float(m_1ms.completion_rich_coef1_re),
            "completion_rich_coef1_im_1ms": float(m_1ms.completion_rich_coef1_im),
            "completion_rich_coef2_re_1ms": float(m_1ms.completion_rich_coef2_re),
            "completion_rich_coef2_im_1ms": float(m_1ms.completion_rich_coef2_im),
            "completion_rich_coef3_re_1ms": float(m_1ms.completion_rich_coef3_re),
            "completion_rich_coef3_im_1ms": float(m_1ms.completion_rich_coef3_im),
            "completion_rich_coef4_re_1ms": float(m_1ms.completion_rich_coef4_re),
            "completion_rich_coef4_im_1ms": float(m_1ms.completion_rich_coef4_im),
            "rel_l2_logabs_det_1ms": float(abs_1ms.rel_l2_logabs),
            "max_abs_logabs_det_1ms": float(abs_1ms.max_abs_logabs_diff),
            "completion_lin_rel_l2_logabs_det_1ms": float(abs_1ms.completion_lin_rel_l2_logabs),
            "completion_lin_logabs_a0_1ms": float(abs_1ms.completion_lin_a0),
            "completion_lin_logabs_a1_1ms": float(abs_1ms.completion_lin_a1),
            "completion_rich_rel_l2_logabs_det_1ms": float(abs_1ms.completion_rich_rel_l2_logabs),
            "completion_rich_k_logabs_1ms": int(abs_1ms.completion_rich_k),
            "completion_rich_coef0_logabs_1ms": float(abs_1ms.completion_rich_coef0),
            "completion_rich_coef1_logabs_1ms": float(abs_1ms.completion_rich_coef1),
            "completion_rich_coef2_logabs_1ms": float(abs_1ms.completion_rich_coef2),
            "completion_rich_coef3_logabs_1ms": float(abs_1ms.completion_rich_coef3),
            "completion_rich_coef4_logabs_1ms": float(abs_1ms.completion_rich_coef4),
            # s -> 1 - s with conjugation
            "rel_l2_det_conj1ms": float(m_conj_1ms.rel_l2_det),
            "rel_l2_logdet_conj1ms": float(m_conj_1ms.rel_l2_logdet),
            "max_abs_det_diff_conj1ms": float(m_conj_1ms.max_abs_det_diff),
            "max_abs_logdet_diff_conj1ms": float(m_conj_1ms.max_abs_logdet_diff),
            "completion_lin_rel_l2_logdet_conj1ms": float(m_conj_1ms.completion_lin_rel_l2_logdet),
            "completion_rich_rel_l2_logdet_conj1ms": float(m_conj_1ms.completion_rich_rel_l2_logdet),
            "rel_l2_logabs_det_conj1ms": float(abs_conj_1ms.rel_l2_logabs),
            "max_abs_logabs_det_conj1ms": float(abs_conj_1ms.max_abs_logabs_diff),
            "completion_lin_rel_l2_logabs_det_conj1ms": float(abs_conj_1ms.completion_lin_rel_l2_logabs),
            "completion_rich_rel_l2_logabs_det_conj1ms": float(abs_conj_1ms.completion_rich_rel_l2_logabs),
            # s -> 1 - conj(s)
            "rel_l2_det_1mconjs": float(m_1mcs.rel_l2_det),
            "rel_l2_logdet_1mconjs": float(m_1mcs.rel_l2_logdet),
            "max_abs_det_diff_1mconjs": float(m_1mcs.max_abs_det_diff),
            "max_abs_logdet_diff_1mconjs": float(m_1mcs.max_abs_logdet_diff),
            "completion_lin_rel_l2_logdet_1mconjs": float(m_1mcs.completion_lin_rel_l2_logdet),
            "completion_rich_rel_l2_logdet_1mconjs": float(m_1mcs.completion_rich_rel_l2_logdet),
            "completion_lin_c0_re_1mconjs": float(m_1mcs.completion_lin_c0_re),
            "completion_lin_c0_im_1mconjs": float(m_1mcs.completion_lin_c0_im),
            "completion_lin_c1_re_1mconjs": float(m_1mcs.completion_lin_c1_re),
            "completion_lin_c1_im_1mconjs": float(m_1mcs.completion_lin_c1_im),
        }
        rows.append(row)

        print(
            f"u={u:g}  rel_log(1-s)={row['rel_l2_logdet_1ms']:.3e}  "
            f"lin={row['completion_lin_rel_l2_logdet_1ms']:.3e}  "
            f"rich={row['completion_rich_rel_l2_logdet_1ms']:.3e}"
        )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["u"]).to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
