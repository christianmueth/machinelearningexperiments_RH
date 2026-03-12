from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _is_finite_complex(z: complex) -> bool:
    return bool(np.isfinite(np.real(z)) and np.isfinite(np.imag(z)))


def _primes_upto(n: int) -> list[int]:
    n = int(n)
    if n < 2:
        return []
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, int(n**0.5) + 1):
        if sieve[p]:
            start = p * p
            sieve[start : n + 1 : p] = b"\x00" * (((n - start) // p) + 1)
    return [i for i in range(2, n + 1) if sieve[i]]


def _prime_powers_upto(n_max: int) -> list[int]:
    n_max = int(n_max)
    out: set[int] = set()
    for p in _primes_upto(n_max):
        pk = p
        while pk <= n_max:
            out.add(pk)
            if pk > n_max // p:
                break
            pk *= p
    return sorted(out)


def _is_prime_power(n: int, primes: list[int] | None = None) -> tuple[bool, int | None]:
    n = int(n)
    if n < 2:
        return False, None
    if primes is None:
        primes = _primes_upto(int(math.isqrt(n)) + 1)
    for p in primes:
        if p * p > n:
            break
        if n % p == 0:
            m = n
            while m % p == 0:
                m //= p
            return (m == 1), p
    # n is prime
    return True, n


def _von_mangoldt(n: int, primes: list[int] | None = None) -> float:
    ok, p = _is_prime_power(n, primes=primes)
    if not ok or p is None:
        return 0.0
    return float(math.log(float(p)))


def _central_dlog_dt(t: np.ndarray, z: np.ndarray, eps_abs: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_mid, d/dt log z) using a branch-robust central difference.

    Uses (z_{k+1}-z_{k-1})/(t_{k+1}-t_{k-1}) / z_k on interior points.
    """

    t = np.asarray(t, dtype=float).ravel()
    z = np.asarray(z, dtype=np.complex128).ravel()
    if t.size != z.size:
        raise ValueError("t and z must have same length")
    if t.size < 3:
        return np.asarray([], dtype=float), np.asarray([], dtype=np.complex128)

    t_mid = t[1:-1].copy()
    denom_t = (t[2:] - t[:-2]).astype(float)

    # Prefer a log-ratio derivative to avoid catastrophic cancellation when |z| is huge.
    # d/dt log z(t_k) ≈ log(z_{k+1}/z_{k-1}) / (t_{k+1}-t_{k-1}).
    z_prev = z[:-2].astype(np.complex128)
    z_next = z[2:].astype(np.complex128)
    z_mid = z[1:-1].astype(np.complex128)

    out = np.full_like(z_mid, complex("nan"), dtype=np.complex128)

    ok = np.isfinite(t_mid) & np.isfinite(denom_t) & (np.abs(denom_t) > 0)
    ok &= np.isfinite(np.real(z_prev)) & np.isfinite(np.imag(z_prev)) & (np.abs(z_prev) > float(eps_abs))
    ok &= np.isfinite(np.real(z_next)) & np.isfinite(np.imag(z_next)) & (np.abs(z_next) > float(eps_abs))
    ok &= np.isfinite(np.real(z_mid)) & np.isfinite(np.imag(z_mid)) & (np.abs(z_mid) > float(eps_abs))

    ratio = np.full_like(z_mid, complex("nan"), dtype=np.complex128)
    ratio[ok] = z_next[ok] / z_prev[ok]
    out[ok] = np.log(ratio[ok]) / denom_t[ok]
    return t_mid, out


def _central_ddt(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_mid, d/dt y) by a simple central difference on interior points."""

    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(y, dtype=np.complex128).ravel()
    if t.size != y.size:
        raise ValueError("t and y must have same length")
    if t.size < 3:
        return np.asarray([], dtype=float), np.asarray([], dtype=np.complex128)

    t_mid = t[1:-1].copy()
    denom_t = (t[2:] - t[:-2]).astype(float)
    out = (y[2:] - y[:-2]).astype(np.complex128) / denom_t.astype(np.complex128)

    ok = np.isfinite(t_mid) & np.isfinite(denom_t) & (np.abs(denom_t) > 0)
    ok &= np.isfinite(np.real(out)) & np.isfinite(np.imag(out))
    return t_mid[ok], out[ok]


def _ridge_solve(X: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.complex128)
    y = np.asarray(y, dtype=np.complex128)
    ridge = float(max(0.0, ridge))
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must be 1D with same rows as X")

    # Prefer solving via least-squares (QR/SVD) rather than normal equations.
    # Normal equations square the condition number and can destroy coefficient fidelity
    # even when the fitted residual is tiny (exactly the regime this probe cares about).
    if ridge > 0:
        rr = math.sqrt(ridge)
        Xa = np.vstack([X, rr * np.eye(X.shape[1], dtype=np.complex128)])
        ya = np.concatenate([y, np.zeros((X.shape[1],), dtype=np.complex128)])
        a, *_ = np.linalg.lstsq(Xa, ya, rcond=None)
        return np.asarray(a, dtype=np.complex128)

    a, *_ = np.linalg.lstsq(X, y, rcond=None)
    return np.asarray(a, dtype=np.complex128)


@dataclass(frozen=True)
class FitResult:
    basis: str
    sigma: float
    n_max: int
    n_samples: int
    ridge: float
    rel_rmse: float
    rmse: float
    y_rms: float
    coef: dict[int, complex]


def _trapz_weights(t: np.ndarray) -> np.ndarray:
    """Trapezoid-rule weights normalized to sum=1."""
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


def _matched_filter_coeffs(t: np.ndarray, y: np.ndarray, sigma: float, n_list: list[int]) -> dict[int, complex]:
    """Coefficient estimates via matched filtering against n^{-sigma-it}.

    For g(t) = sum a_n n^{-sigma} e^{-i t log n}, estimate
      a_n ≈ n^{sigma} * E_w[ g(t) e^{i t log n} ]
    where E_w is a weighted average over t (trapezoid weights).

    This avoids the severe coefficient instability that can arise in large, highly
    coherent least-squares dictionaries, while still detecting prime-power support
    and ladder structure.
    """
    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(y, dtype=np.complex128).ravel()
    if t.size != y.size:
        raise ValueError("t and y length mismatch")
    if t.size == 0:
        return {}
    w = _trapz_weights(t)

    out: dict[int, complex] = {}
    for n in n_list:
        ln = float(math.log(float(n)))
        ph = np.exp(1j * t * ln)
        est = (float(n) ** float(sigma)) * complex(np.sum(w * y * ph))
        out[int(n)] = complex(est)
    return out


def _matched_filter_coeffs_alpha(
    t: np.ndarray,
    y: np.ndarray,
    *,
    sigma: float,
    n_list: list[int],
    alpha: float,
) -> dict[int, complex]:
    """Matched filtering for alpha-scaled Dirichlet frequencies.

    For g(t) = sum a_n n^{-alpha*sigma} e^{-i t alpha log n}, estimate
      a_n ≈ n^{alpha*sigma} * E_w[ g(t) e^{i t alpha log n} ].
    """

    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(y, dtype=np.complex128).ravel()
    alpha = float(alpha)
    if t.size != y.size:
        raise ValueError("t and y length mismatch")
    if t.size == 0:
        return {}
    w = _trapz_weights(t)

    out: dict[int, complex] = {}
    for n in n_list:
        ln = float(math.log(float(n)))
        ph = np.exp(1j * t * (alpha * ln))
        est = (float(n) ** float(alpha * sigma)) * complex(np.sum(w * y * ph))
        out[int(n)] = complex(est)
    return out


def _matched_filter_omega(t: np.ndarray, y: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Return complex matched-filter values m(ω)=E_w[y(t) * exp(i ω t)] for ω-grid."""
    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(y, dtype=np.complex128).ravel()
    omega = np.asarray(omega, dtype=float).ravel()
    if t.size == 0 or y.size == 0 or omega.size == 0:
        return np.zeros((omega.size,), dtype=np.complex128)
    if t.size != y.size:
        raise ValueError("t and y length mismatch")
    w = _trapz_weights(t)
    # Broadcast: exp(i ω t)
    ph = np.exp(1j * omega.reshape(-1, 1) * t.reshape(1, -1)).astype(np.complex128)
    return (ph @ (w.reshape(-1, 1) * y.reshape(-1, 1))).ravel().astype(np.complex128)


def _alpha_scan_logp_frequencies(
    t: np.ndarray,
    y: np.ndarray,
    *,
    primes: np.ndarray,
    alpha_min: float,
    alpha_max: float,
    alpha_steps: int,
    omega_max: float,
) -> dict[str, float]:
    """Scan α for frequency alignment ω=α log p.

    Returns a few summary stats (best α, score at α=1, etc.).
    The score is mean(|m_p(α)|) normalized by RMS(y), where
      m_p(α) = E_w[y(t) * exp(i * (α log p) * t)].
    """

    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(y, dtype=np.complex128).ravel()
    primes = np.asarray(primes, dtype=int).ravel()
    if t.size == 0 or y.size == 0 or primes.size == 0:
        return {
            "alpha_best": float("nan"),
            "alpha_score_best": float("nan"),
            "alpha_score_at1": float("nan"),
            "alpha_n_primes_used": 0.0,
        }

    y_rms = float(np.sqrt(np.mean(np.abs(y) ** 2)))
    if not math.isfinite(y_rms) or y_rms <= 0:
        y_rms = 1.0

    logp = np.log(primes.astype(float))
    # Only use primes whose frequencies are within the supported omega range.
    # (omega_max is a soft cap so we don't test wildly oscillatory phases on short windows.)
    m_ok = np.isfinite(logp) & (logp > 0) & (logp <= float(max(1e-12, omega_max)))
    logp = logp[m_ok]
    if logp.size < 3:
        return {
            "alpha_best": float("nan"),
            "alpha_score_best": float("nan"),
            "alpha_score_at1": float("nan"),
            "alpha_n_primes_used": float(logp.size),
        }

    alpha_steps = int(max(3, alpha_steps))
    a_grid = np.linspace(float(alpha_min), float(alpha_max), alpha_steps)

    def _score(alpha: float) -> float:
        omega = float(alpha) * logp
        m = _matched_filter_omega(t, y, omega)
        return float(np.mean(np.abs(m)) / (y_rms + 1e-300))

    scores = np.asarray([_score(a) for a in a_grid], dtype=float)
    if scores.size == 0 or not np.any(np.isfinite(scores)):
        return {
            "alpha_best": float("nan"),
            "alpha_score_best": float("nan"),
            "alpha_score_at1": float("nan"),
            "alpha_n_primes_used": float(logp.size),
        }

    j = int(np.nanargmax(scores))
    alpha_best = float(a_grid[j])
    score_best = float(scores[j])
    score_at1 = _score(1.0) if (float(alpha_min) <= 1.0 <= float(alpha_max)) else float("nan")

    return {
        "alpha_best": float(alpha_best),
        "alpha_score_best": float(score_best),
        "alpha_score_at1": float(score_at1),
        "alpha_n_primes_used": float(logp.size),
    }


def _design_matrix(t: np.ndarray, sigma: float, n_list: list[int], *, alpha: float) -> np.ndarray:
    t = np.asarray(t, dtype=float).ravel()
    sigma = float(sigma)
    alpha = float(alpha)
    logs = np.log(np.asarray(n_list, dtype=float))
    # X[k,j] = n^{-alpha*sigma} * exp(-i t_k * alpha * log n)
    phase = np.exp(-1j * t.reshape(-1, 1) * (alpha * logs).reshape(1, -1)).astype(np.complex128)
    amp = np.exp(-(alpha * sigma) * logs).reshape(1, -1).astype(np.complex128)
    return (phase * amp).astype(np.complex128)


def _fit_dirichlet(
    t: np.ndarray,
    y: np.ndarray,
    *,
    sigma: float,
    n_list: list[int],
    ridge: float,
    basis_name: str,
    alpha: float,
) -> FitResult:
    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(y, dtype=np.complex128).ravel()
    if t.size != y.size:
        raise ValueError("t and y length mismatch")

    X = _design_matrix(t, sigma=float(sigma), n_list=n_list, alpha=float(alpha))
    a = _ridge_solve(X, y, ridge=float(ridge))
    y_hat = X @ a

    resid = y_hat - y
    rmse = float(np.sqrt(np.mean(np.abs(resid) ** 2))) if resid.size else float("nan")
    y_rms = float(np.sqrt(np.mean(np.abs(y) ** 2))) if y.size else float("nan")
    rel = float(rmse / (y_rms + 1e-300)) if math.isfinite(rmse) and math.isfinite(y_rms) else float("nan")

    coef = {int(n): complex(a[j]) for j, n in enumerate(n_list)}
    return FitResult(
        basis=str(basis_name),
        sigma=float(sigma),
        n_max=int(max(n_list) if n_list else 0),
        n_samples=int(t.size),
        ridge=float(ridge),
        rel_rmse=float(rel),
        rmse=float(rmse),
        y_rms=float(y_rms),
        coef=coef,
    )


def _best_complex_scalar(y: np.ndarray, x: np.ndarray) -> complex:
    """Return c minimizing ||y - c*x||_2 for complex vectors."""
    y = np.asarray(y, dtype=np.complex128).ravel()
    x = np.asarray(x, dtype=np.complex128).ravel()
    if y.size == 0 or x.size == 0 or y.size != x.size:
        return complex("nan")
    denom = np.vdot(x, x)
    if not _is_finite_complex(denom) or abs(denom) <= 0:
        return complex("nan")
    return complex(np.vdot(x, y) / denom)


def _rel_rmse(y_hat: np.ndarray, y: np.ndarray) -> float:
    y_hat = np.asarray(y_hat, dtype=np.complex128).ravel()
    y = np.asarray(y, dtype=np.complex128).ravel()
    if y.size == 0 or y_hat.size != y.size:
        return float("nan")
    rmse = float(np.sqrt(np.mean(np.abs(y_hat - y) ** 2)))
    y_rms = float(np.sqrt(np.mean(np.abs(y) ** 2)))
    return float(rmse / (y_rms + 1e-300))


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation for finite real vectors; returns NaN if ill-posed."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return float("nan")
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        return float("nan")
    x0 = x - float(np.mean(x))
    y0 = y - float(np.mean(y))
    denom = float(np.sqrt(np.sum(x0 * x0) * np.sum(y0 * y0)))
    if not math.isfinite(denom) or denom <= 0:
        return float("nan")
    return float(np.sum(x0 * y0) / denom)


def _rel_dispersion(a: np.ndarray) -> float:
    """Relative dispersion ||a-mean(a)||/||a|| for complex vectors."""
    a = np.asarray(a, dtype=np.complex128).ravel()
    if a.size == 0:
        return float("nan")
    denom = float(np.sqrt(np.sum(np.abs(a) ** 2)))
    if not math.isfinite(denom) or denom <= 0:
        return float("nan")
    mu = complex(np.mean(a))
    num = float(np.sqrt(np.sum(np.abs(a - mu) ** 2)))
    return float(num / denom)


def _prime_ladder_design_matrix(
    t: np.ndarray,
    sigma: float,
    n_max: int,
    *,
    with_logp: bool,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Design matrix with one column per prime, enforcing ladder constancy across prime powers.

    Column for prime p is sum_{k>=1: p^k<=N} p^{-k*sigma} e^{-i t k log p} * (log p if with_logp else 1).
    Returns (X, primes, logp).
    """
    t = np.asarray(t, dtype=float).ravel()
    sigma = float(sigma)
    alpha = float(alpha)
    n_max = int(n_max)
    primes = np.asarray(_primes_upto(n_max), dtype=int)
    if primes.size == 0 or t.size == 0:
        return np.zeros((t.size, 0), dtype=np.complex128), primes, np.zeros((0,), dtype=float)
    logp = np.log(primes.astype(float))
    X = np.zeros((t.size, primes.size), dtype=np.complex128)
    for j, p in enumerate(primes.tolist()):
        lp = float(math.log(float(p)))
        if not math.isfinite(lp) or lp <= 0:
            continue
        scale = lp if with_logp else 1.0
        col = np.zeros((t.size,), dtype=np.complex128)
        pk = int(p)
        k = 1
        while pk <= n_max:
            log_pk = float(k) * lp
            amp = math.exp(-(alpha * sigma) * log_pk)
            col += (scale * amp) * np.exp(-1j * t * (alpha * log_pk))
            if pk > n_max // p:
                break
            pk *= p
            k += 1
        X[:, j] = col
    return X, primes.astype(int), logp.astype(float)


def _c_fit(a: np.ndarray, w: np.ndarray) -> complex:
    """Best complex scalar c for a ~ c*w in least squares."""
    a = np.asarray(a, dtype=np.complex128).ravel()
    w = np.asarray(w, dtype=float).ravel()
    if a.size == 0 or w.size == 0 or a.size != w.size:
        return complex("nan")
    denom = float(np.sum(w * w))
    if not math.isfinite(denom) or denom <= 0:
        return complex("nan")
    num = np.sum(a * w)
    return complex(num / denom)


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.complex128).ravel()
    b = np.asarray(b, dtype=np.complex128).ravel()
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return float("nan")
    denom = float(np.sqrt(np.sum(np.abs(b) ** 2)))
    if not math.isfinite(denom) or denom <= 0:
        return float("nan")
    num = float(np.sqrt(np.sum(np.abs(a - b) ** 2)))
    return float(num / denom)


def _metrics_from_signal(
    t: np.ndarray,
    y: np.ndarray,
    *,
    sigma: float,
    n_max: int,
    ridge: float,
    rel_rmse_full: float,
    rel_rmse_pp: float,
    alpha: float,
) -> dict[str, float | int | str]:
    """Compute stable structure metrics without relying on ill-conditioned per-n coefficients."""

    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(y, dtype=np.complex128).ravel()

    X_logp, primes, logp = _prime_ladder_design_matrix(
        t, sigma=float(sigma), n_max=int(n_max), with_logp=True, alpha=float(alpha)
    )
    X_nologp, _pr2, _lp2 = _prime_ladder_design_matrix(
        t, sigma=float(sigma), n_max=int(n_max), with_logp=False, alpha=float(alpha)
    )

    # Per-prime ladder fits.
    c_p = _ridge_solve(X_logp, y, ridge=float(ridge)) if X_logp.size else np.asarray([], dtype=np.complex128)
    d_p = _ridge_solve(X_nologp, y, ridge=float(ridge)) if X_nologp.size else np.asarray([], dtype=np.complex128)

    y_hat_ladder_logp = (X_logp @ c_p) if X_logp.size else np.zeros_like(y)
    y_hat_ladder_nologp = (X_nologp @ d_p) if X_nologp.size else np.zeros_like(y)

    rel_rmse_ladder_logp = _rel_rmse(y_hat_ladder_logp, y)
    rel_rmse_ladder_nologp = _rel_rmse(y_hat_ladder_nologp, y)

    # Global Mangoldt template: sum over primes with the ladder+logp basis and unit coefficients.
    if X_logp.size:
        template = X_logp @ np.ones((X_logp.shape[1],), dtype=np.complex128)
        c_template = _best_complex_scalar(y, template)
        y_hat_template = c_template * template
        rel_rmse_template = _rel_rmse(y_hat_template, y)
    else:
        c_template = complex("nan")
        rel_rmse_template = float("nan")

    # Stable penalty ratios (avoid interpreting ill-conditioned coefficient vectors).
    ratio_ladder_over_pp = (
        float(rel_rmse_ladder_logp / (rel_rmse_pp + 1e-300))
        if math.isfinite(rel_rmse_ladder_logp) and math.isfinite(rel_rmse_pp)
        else float("nan")
    )
    ratio_template_over_ladder = (
        float(rel_rmse_template / (rel_rmse_ladder_logp + 1e-300))
        if math.isfinite(rel_rmse_template) and math.isfinite(rel_rmse_ladder_logp)
        else float("nan")
    )

    # Correlation diagnostics for fitted per-prime coefficients.
    # - For perfect Mangoldt behavior with logp ladder: c_p should be ~1 (no correlation with log p).
    # - For Mangoldt behavior recovered via the no-logp ladder: d_p should be ~log p.
    corr_abs_cp_logp = _pearson_corr(np.abs(c_p), logp) if c_p.size and logp.size else float("nan")
    corr_re_cp_logp = _pearson_corr(np.real(c_p), logp) if c_p.size and logp.size else float("nan")
    corr_abs_dp_logp = _pearson_corr(np.abs(d_p), logp) if d_p.size and logp.size else float("nan")
    corr_re_dp_logp = _pearson_corr(np.real(d_p), logp) if d_p.size and logp.size else float("nan")

    # Does d_p look proportional to log p?
    c_dp_logp = _c_fit(d_p, logp) if d_p.size and logp.size else complex("nan")
    relerr_dp_vs_logp = _rel_err(d_p, c_dp_logp * logp) if d_p.size and logp.size else float("nan")

    rel_disp_cp = _rel_dispersion(c_p) if c_p.size else float("nan")
    rel_disp_dp = _rel_dispersion(d_p) if d_p.size else float("nan")

    return {
        "sigma": float(sigma),
        "n_max": int(n_max),
        "n_primes": int(primes.size),
        "rel_rmse_full": float(rel_rmse_full),
        "rel_rmse_pp": float(rel_rmse_pp),
        "rel_rmse_ratio_pp_over_full": float(rel_rmse_pp / (rel_rmse_full + 1e-300))
        if math.isfinite(rel_rmse_pp) and math.isfinite(rel_rmse_full)
        else float("nan"),
        # Weight/ladder diagnostics (stable structured fits)
        "rel_rmse_ladder_per_prime_logp": float(rel_rmse_ladder_logp),
        "rel_rmse_ladder_per_prime_nologp": float(rel_rmse_ladder_nologp),
        "rel_rmse_mangoldt_template_global": float(rel_rmse_template),
        "ratio_ladder_over_pp": float(ratio_ladder_over_pp),
        "ratio_template_over_ladder": float(ratio_template_over_ladder),
        "corr_abs_cp_logp": float(corr_abs_cp_logp),
        "corr_re_cp_logp": float(corr_re_cp_logp),
        "corr_abs_dp_logp": float(corr_abs_dp_logp),
        "corr_re_dp_logp": float(corr_re_dp_logp),
        "rel_disp_cp": float(rel_disp_cp),
        "rel_disp_dp": float(rel_disp_dp),
        "relerr_dp_vs_logp": float(relerr_dp_vs_logp),
    }


def _split_masks(t: np.ndarray, n_splits: int) -> list[np.ndarray]:
    t = np.asarray(t, dtype=float).ravel()
    n_splits = int(max(1, n_splits))
    if n_splits == 1 or t.size == 0:
        return [np.ones_like(t, dtype=bool)]
    qs = [float(q) for q in np.linspace(0.0, 1.0, n_splits + 1)]
    cuts = [np.quantile(t, q) for q in qs]
    masks = []
    for i in range(n_splits):
        lo, hi = cuts[i], cuts[i + 1]
        if i == n_splits - 1:
            m = (t >= lo) & (t <= hi)
        else:
            m = (t >= lo) & (t < hi)
        masks.append(np.asarray(m, dtype=bool))
    return masks


def _infer_complex_cols(df: pd.DataFrame, mode: str, *, quotient: str, rung: str) -> tuple[str, str]:
    mode = str(mode).strip().lower()
    q = str(quotient).strip()
    rung = str(rung).strip()

    if mode == "q_track":
        for re_c, im_c in [("q_track_re", "q_track_im"), ("q_point_re", "q_point_im")]:
            if re_c in df.columns and im_c in df.columns:
                return re_c, im_c
        raise ValueError("could not infer q_track columns; expected q_track_re/q_track_im")

    # projector-style
    # candidates like Q2_tr_N2_re
    base = f"{q}_{rung}_"
    re_col = base + "re"
    im_col = base + "im"
    if re_col in df.columns and im_col in df.columns:
        return re_col, im_col

    # fallback: some files omit underscore before re/im
    re_col2 = f"{q}_{rung}_re"
    im_col2 = f"{q}_{rung}_im"
    if re_col2 in df.columns and im_col2 in df.columns:
        return re_col2, im_col2

    raise ValueError(f"could not infer complex columns for quotient={q} rung={rung}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Mangoldt rigidity probe on a complex observable D(s) sampled on sigma-lines. "
            "Estimates g(s)=-d/ds log D(s) from t-derivatives, then fits Dirichlet bases and scores "
            "prime-power support + log p ladder/weight structure."
        )
    )
    ap.add_argument("--csv", required=True, help="Input CSV with columns sigma,t and complex D(s) re/im")
    ap.add_argument("--side", default="right", help="If column 'side' exists, filter to this side")
    ap.add_argument("--mode", choices=["auto", "q_track", "quotient"], default="auto", help="Which observable columns to use")
    ap.add_argument("--quotient", default="Q2_tr", help="For mode=quotient: Q2_tr, Q2, Q2_phi2")
    ap.add_argument("--rung", choices=["N", "N2"], default="N2", help="For mode=quotient: use N or N2 columns")
    ap.add_argument("--re_col", default="", help="Override real-part column")
    ap.add_argument("--im_col", default="", help="Override imag-part column")

    ap.add_argument("--sigma", type=float, default=float("nan"), help="If finite, restrict to a single sigma value (nearest match)")
    ap.add_argument("--sigma_tol", type=float, default=1e-9, help="Tolerance when matching --sigma")

    ap.add_argument("--require_tier01", type=int, default=0, help="If 1, require Tier01 columns and filter accordingly")
    ap.add_argument("--gate_tin_inv_max", type=float, default=500.0)
    ap.add_argument("--gate_sep2_min", type=float, default=0.01)
    ap.add_argument("--gate_lam_cost_max", type=float, default=0.8)

    ap.add_argument("--n_max", type=int, default=512, help="Dirichlet truncation N (basis uses n<=N)")
    ap.add_argument("--ridge", type=float, default=1e-6, help="Ridge regularization")
    ap.add_argument("--t_splits", type=int, default=4, help="Compute metrics on t-quantile splits and aggregate worst-case")

    ap.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help=(
            "Frequency scaling for the Dirichlet basis: uses n^{-alpha*sigma} exp(-i t alpha log n). "
            "Set alpha=2 for observables naturally expressed in 2s (e.g. xi(2s-1)/xi(2s))."
        ),
    )

    ap.add_argument("--eps_abs", type=float, default=1e-12, help="Drop points where |D|<=eps_abs for log-derivative")
    ap.add_argument(
        "--input_is_log",
        type=int,
        default=0,
        help=(
            "If 1, treat the complex columns as L(s)=log D(s) already (e.g. logdet(I-S)). "
            "Then g(s)=-d/ds log D(s)=i*d/dt L(σ+it) is estimated by a direct central difference."
        ),
    )
    ap.add_argument(
        "--unwrap_imag",
        type=int,
        default=0,
        help=(
            "If 1, unwrap the imaginary part of the chosen complex series as a function of t before differentiation. "
            "Useful for log-valued observables that may jump by ~2π in the principal branch."
        ),
    )
    ap.add_argument("--min_samples", type=int, default=40, help="Minimum samples per sigma to score")

    ap.add_argument(
        "--alpha_scan",
        type=int,
        default=0,
        help="If 1, scan α to test frequency alignment ω=α log p via matched filtering of g(t)",
    )
    ap.add_argument("--alpha_min", type=float, default=0.5)
    ap.add_argument("--alpha_max", type=float, default=2.0)
    ap.add_argument("--alpha_steps", type=int, default=61)
    ap.add_argument(
        "--alpha_omega_max",
        type=float,
        default=6.0,
        help="Only include primes with log p <= alpha_omega_max in the alpha scan",
    )

    ap.add_argument("--out_csv", default="", help="Optional output CSV path")
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.exists():
        raise SystemExit(f"missing --csv: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit("input CSV empty")

    if "t" not in df.columns or "sigma" not in df.columns:
        raise SystemExit("CSV must include columns: sigma, t")

    if args.side and ("side" in df.columns):
        df = df[df["side"].astype(str) == str(args.side)].copy()

    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df["sigma"] = pd.to_numeric(df["sigma"], errors="coerce")
    df = df.dropna(subset=["t", "sigma"]).copy()

    if int(args.require_tier01) == 1:
        need = ["Tin_inv_norm1_est_N2", "sep2_N2", "lam_match_cost"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise SystemExit(f"require_tier01=1 but missing columns: {missing}")
        tier01 = pd.Series(True, index=df.index)
        tier01 &= pd.to_numeric(df["Tin_inv_norm1_est_N2"], errors="coerce") <= float(args.gate_tin_inv_max)
        tier01 &= pd.to_numeric(df["sep2_N2"], errors="coerce") >= float(args.gate_sep2_min)
        tier01 &= pd.to_numeric(df["lam_match_cost"], errors="coerce") <= float(args.gate_lam_cost_max)
        df = df[tier01.astype(bool)].copy()

    if df.empty:
        raise SystemExit("no rows after filtering")

    # Choose complex columns.
    re_col = str(args.re_col).strip()
    im_col = str(args.im_col).strip()
    if not re_col or not im_col:
        mode = str(args.mode).strip().lower()
        if mode == "auto":
            if "q_track_re" in df.columns and "q_track_im" in df.columns:
                mode = "q_track"
            else:
                mode = "quotient"
        re_col, im_col = _infer_complex_cols(df, mode, quotient=str(args.quotient), rung=str(args.rung))

    for c in [re_col, im_col]:
        if c not in df.columns:
            raise SystemExit(f"missing required complex column: {c}")

    df[re_col] = pd.to_numeric(df[re_col], errors="coerce")
    df[im_col] = pd.to_numeric(df[im_col], errors="coerce")

    # If sigma specified, pick nearest.
    if math.isfinite(float(args.sigma)):
        target = float(args.sigma)
        sigs = pd.to_numeric(df["sigma"], errors="coerce")
        sigs = sigs[np.isfinite(sigs)]
        if sigs.empty:
            raise SystemExit("no finite sigma values")
        nearest = float(sigs.iloc[(sigs - target).abs().argmin()])
        df = df[(df["sigma"] - nearest).abs() <= float(args.sigma_tol)].copy()
        if df.empty:
            raise SystemExit("no rows matched --sigma")

    n_max = int(args.n_max)
    if n_max < 4:
        raise SystemExit("--n_max must be >= 4")

    # Score per sigma.
    out_rows: list[dict] = []
    for sigma_val, gdf in df.groupby("sigma"):
        sigma_val = float(sigma_val)
        gdf = gdf.copy().sort_values("t")

        t = gdf["t"].to_numpy(dtype=float)
        z = (gdf[re_col].to_numpy(dtype=float) + 1j * gdf[im_col].to_numpy(dtype=float)).astype(np.complex128)

        # Optional: unwrap imaginary part for continuity along t.
        if int(args.unwrap_imag) == 1:
            zr = np.real(z).astype(float)
            zi = np.unwrap(np.imag(z).astype(float))
            z = (zr + 1j * zi).astype(np.complex128)

        if int(args.input_is_log) == 1:
            # Here z == L(s) = log D(s). Then g(s) = -d/ds log D(s) = -L'(s).
            # Using d/dt L(s) = i L'(s) for analytic L, we get g(s) = i * d/dt L(s).
            t_mid, dL_dt = _central_ddt(t, z)
            g = (1j * dL_dt).astype(np.complex128)
        else:
            t_mid, dlog_dt = _central_dlog_dt(t, z, eps_abs=float(args.eps_abs))
            # g = -d/ds log D(s) = i * d/dt log D(σ+it)
            g = (1j * dlog_dt).astype(np.complex128)

        ok = np.isfinite(t_mid) & np.isfinite(np.real(g)) & np.isfinite(np.imag(g))
        t_mid = t_mid[ok]
        g = g[ok]

        if t_mid.size < int(args.min_samples):
            continue

        # Correlation-style diagnostics are more stable on the full window.
        n_list_full_all = list(range(2, int(n_max) + 1))
        n_list_pp_all = _prime_powers_upto(int(n_max))
        fit_full_all = _fit_dirichlet(
            t_mid,
            g,
            sigma=float(sigma_val),
            n_list=n_list_full_all,
            ridge=float(args.ridge),
            basis_name="full",
            alpha=float(args.alpha),
        )
        fit_pp_all = _fit_dirichlet(
            t_mid,
            g,
            sigma=float(sigma_val),
            n_list=n_list_pp_all,
            ridge=float(args.ridge),
            basis_name="prime_powers",
            alpha=float(args.alpha),
        )
        metrics_all = _metrics_from_signal(
            t_mid,
            g,
            sigma=float(sigma_val),
            n_max=int(n_max),
            ridge=float(args.ridge),
            rel_rmse_full=float(fit_full_all.rel_rmse),
            rel_rmse_pp=float(fit_pp_all.rel_rmse),
            alpha=float(args.alpha),
        )

        # Split-scoring: compute metrics on each split and take worst-case (max) for errors/leakage/var.
        masks = _split_masks(t_mid, n_splits=int(args.t_splits))
        per_split: list[dict] = []
        for m in masks:
            if int(np.sum(m)) < max(20, int(args.min_samples) // 2):
                continue
            tt = t_mid[m]
            yy = g[m]

            n_list_full = list(range(2, int(n_max) + 1))
            n_list_pp = _prime_powers_upto(int(n_max))

            fit_full = _fit_dirichlet(
                tt,
                yy,
                sigma=float(sigma_val),
                n_list=n_list_full,
                ridge=float(args.ridge),
                basis_name="full",
                alpha=float(args.alpha),
            )
            fit_pp = _fit_dirichlet(
                tt,
                yy,
                sigma=float(sigma_val),
                n_list=n_list_pp,
                ridge=float(args.ridge),
                basis_name="prime_powers",
                alpha=float(args.alpha),
            )

            per_split.append(
                _metrics_from_signal(
                    tt,
                    yy,
                    sigma=float(sigma_val),
                    n_max=int(n_max),
                    ridge=float(args.ridge),
                    rel_rmse_full=float(fit_full.rel_rmse),
                    rel_rmse_pp=float(fit_pp.rel_rmse),
                    alpha=float(args.alpha),
                )
            )

        if not per_split:
            continue

        # Aggregate worst-case over splits for "badness" metrics.
        def _max_key(k: str) -> float:
            vals = [float(r.get(k)) for r in per_split if r.get(k) is not None and math.isfinite(float(r.get(k)))]
            return float(max(vals)) if vals else float("nan")

        def _min_key(k: str) -> float:
            vals = [float(r.get(k)) for r in per_split if r.get(k) is not None and math.isfinite(float(r.get(k)))]
            return float(min(vals)) if vals else float("nan")

        row = {
            "sigma": float(sigma_val),
            "n_max": int(n_max),
            "n_samples": int(t_mid.size),
            "t_splits": int(args.t_splits),
            "re_col": str(re_col),
            "im_col": str(im_col),
            "rel_rmse_full_worst": _max_key("rel_rmse_full"),
            "rel_rmse_pp_worst": _max_key("rel_rmse_pp"),
            "rel_rmse_ratio_pp_over_full_worst": _max_key("rel_rmse_ratio_pp_over_full"),
            "rel_rmse_ladder_per_prime_logp_worst": _max_key("rel_rmse_ladder_per_prime_logp"),
            "rel_rmse_ladder_per_prime_nologp_worst": _max_key("rel_rmse_ladder_per_prime_nologp"),
            "rel_rmse_mangoldt_template_global_worst": _max_key("rel_rmse_mangoldt_template_global"),
            "ratio_ladder_over_pp_worst": _max_key("ratio_ladder_over_pp"),
            "ratio_template_over_ladder_worst": _max_key("ratio_template_over_ladder"),
            "rel_rmse_full_best": _min_key("rel_rmse_full"),
            "rel_rmse_pp_best": _min_key("rel_rmse_pp"),
            # Full-window coefficient/weight diagnostics.
            "corr_abs_cp_logp": float(metrics_all.get("corr_abs_cp_logp", float("nan"))),
            "corr_re_cp_logp": float(metrics_all.get("corr_re_cp_logp", float("nan"))),
            "corr_abs_dp_logp": float(metrics_all.get("corr_abs_dp_logp", float("nan"))),
            "corr_re_dp_logp": float(metrics_all.get("corr_re_dp_logp", float("nan"))),
            "rel_disp_cp": float(metrics_all.get("rel_disp_cp", float("nan"))),
            "rel_disp_dp": float(metrics_all.get("rel_disp_dp", float("nan"))),
            "relerr_dp_vs_logp": float(metrics_all.get("relerr_dp_vs_logp", float("nan"))),
        }

        if int(args.alpha_scan) == 1:
            # Use primes up to n_max for the scan; filter by log p <= alpha_omega_max.
            primes = np.asarray(_primes_upto(int(n_max)), dtype=int)
            scan = _alpha_scan_logp_frequencies(
                t_mid,
                g,
                primes=primes,
                alpha_min=float(args.alpha_min),
                alpha_max=float(args.alpha_max),
                alpha_steps=int(args.alpha_steps),
                omega_max=float(args.alpha_omega_max),
            )
            row.update(scan)
        out_rows.append(row)

    out = pd.DataFrame(out_rows).sort_values(["sigma"], ascending=True)
    if out.empty:
        raise SystemExit("no sigma groups produced usable results (try lowering --min_samples or adjusting filters)")

    with pd.option_context("display.width", 200, "display.max_rows", 200):
        show_cols = [
            "sigma",
            "n_samples",
            "n_max",
            "rel_rmse_full_worst",
            "rel_rmse_pp_worst",
            "rel_rmse_ratio_pp_over_full_worst",
            "rel_rmse_ladder_per_prime_logp_worst",
            "rel_rmse_mangoldt_template_global_worst",
            "ratio_ladder_over_pp_worst",
            "ratio_template_over_ladder_worst",
            "corr_abs_cp_logp",
            "corr_abs_dp_logp",
            "relerr_dp_vs_logp",
        ]
        if int(args.alpha_scan) == 1:
            show_cols += ["alpha_best", "alpha_score_best", "alpha_score_at1", "alpha_n_primes_used"]
        print(out[show_cols].to_string(index=False))

    out_csv = str(args.out_csv).strip()
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)
        print(f"wrote {out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
