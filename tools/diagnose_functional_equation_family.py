from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

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
    # best-fit completion exp(c0 + c1*s) applied to D(s) before comparing to D(1-s)
    completion_rel_l2_logdet: float
    completion_c0_re: float
    completion_c0_im: float
    completion_c1_re: float
    completion_c1_im: float


@dataclass(frozen=True)
class AbsSymmetryMetrics:
    rel_l2_logabs: float
    max_abs_logabs_diff: float
    # best-fit linear completion on logabs: log|D(s)| + a0 + a1*Re(s) ≈ log|D(sym(s))|
    completion_rel_l2_logabs: float
    completion_a0: float
    completion_a1: float


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


def _abs_symmetry_metrics(
    *,
    s_vals: np.ndarray,
    logabs_s: np.ndarray,
    logabs_sym: np.ndarray,
    w: np.ndarray,
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

    return AbsSymmetryMetrics(
        rel_l2_logabs=float(rel),
        max_abs_logabs_diff=float(mx),
        completion_rel_l2_logabs=float(rel_comp),
        completion_a0=float(a0),
        completion_a1=float(a1),
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

    return SymmetryMetrics(
        rel_l2_det=float(rel_det),
        rel_l2_logdet=float(rel_log),
        max_abs_det_diff=float(max_det),
        max_abs_logdet_diff=float(max_log),
        completion_rel_l2_logdet=float(rel_log_comp),
        completion_c0_re=float(np.real(c0)),
        completion_c0_im=float(np.imag(c0)),
        completion_c1_re=float(np.real(c1)),
        completion_c1_im=float(np.imag(c1)),
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

        m_1ms = _symmetry_metrics(s_vals=s_vals, det_s=det_s, logdet_s=log_s, det_sym=det_1ms, logdet_sym=log_1ms, w=w)
        m_1mcs = _symmetry_metrics(s_vals=s_vals, det_s=det_s, logdet_s=log_s, det_sym=det_1mcs, logdet_sym=log_1mcs, w=w)

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
        )

        # Magnitude-only symmetry on det.
        abs_1ms = _abs_symmetry_metrics(s_vals=s_vals, logabs_s=_logabs(det_s), logabs_sym=_logabs(det_1ms), w=w)
        abs_conj_1ms = _abs_symmetry_metrics(s_vals=s_vals, logabs_s=_logabs(det_s), logabs_sym=_logabs(det_conj_1ms), w=w)

        row = {
            "u": float(u),
            "v": float(v),
            "sigma": float(sigma),
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
            "completion_rel_l2_logdet_1ms": float(m_1ms.completion_rel_l2_logdet),
            "completion_c0_re_1ms": float(m_1ms.completion_c0_re),
            "completion_c0_im_1ms": float(m_1ms.completion_c0_im),
            "completion_c1_re_1ms": float(m_1ms.completion_c1_re),
            "completion_c1_im_1ms": float(m_1ms.completion_c1_im),
            "rel_l2_logabs_det_1ms": float(abs_1ms.rel_l2_logabs),
            "max_abs_logabs_det_1ms": float(abs_1ms.max_abs_logabs_diff),
            "completion_rel_l2_logabs_det_1ms": float(abs_1ms.completion_rel_l2_logabs),
            "completion_logabs_a0_1ms": float(abs_1ms.completion_a0),
            "completion_logabs_a1_1ms": float(abs_1ms.completion_a1),
            # s -> 1 - s with conjugation
            "rel_l2_det_conj1ms": float(m_conj_1ms.rel_l2_det),
            "rel_l2_logdet_conj1ms": float(m_conj_1ms.rel_l2_logdet),
            "max_abs_det_diff_conj1ms": float(m_conj_1ms.max_abs_det_diff),
            "max_abs_logdet_diff_conj1ms": float(m_conj_1ms.max_abs_logdet_diff),
            "completion_rel_l2_logdet_conj1ms": float(m_conj_1ms.completion_rel_l2_logdet),
            "rel_l2_logabs_det_conj1ms": float(abs_conj_1ms.rel_l2_logabs),
            "max_abs_logabs_det_conj1ms": float(abs_conj_1ms.max_abs_logabs_diff),
            "completion_rel_l2_logabs_det_conj1ms": float(abs_conj_1ms.completion_rel_l2_logabs),
            # s -> 1 - conj(s)
            "rel_l2_det_1mconjs": float(m_1mcs.rel_l2_det),
            "rel_l2_logdet_1mconjs": float(m_1mcs.rel_l2_logdet),
            "max_abs_det_diff_1mconjs": float(m_1mcs.max_abs_det_diff),
            "max_abs_logdet_diff_1mconjs": float(m_1mcs.max_abs_logdet_diff),
            "completion_rel_l2_logdet_1mconjs": float(m_1mcs.completion_rel_l2_logdet),
            "completion_c0_re_1mconjs": float(m_1mcs.completion_c0_re),
            "completion_c0_im_1mconjs": float(m_1mcs.completion_c0_im),
            "completion_c1_re_1mconjs": float(m_1mcs.completion_c1_re),
            "completion_c1_im_1mconjs": float(m_1mcs.completion_c1_im),
        }
        rows.append(row)

        print(
            f"u={u:g}  rel_log(1-s)={row['rel_l2_logdet_1ms']:.3e}  "
            f"rel_log(conj(1-s))={row['rel_l2_logdet_conj1ms']:.3e}  "
            f"rel_logabs(1-s)={row['rel_l2_logabs_det_1ms']:.3e}"
        )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["u"]).to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
