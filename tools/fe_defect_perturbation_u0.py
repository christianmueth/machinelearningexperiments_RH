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

try:
    import mpmath as mp  # type: ignore

    _HAS_MPMATH = True
except Exception:  # pragma: no cover
    mp = None  # type: ignore
    _HAS_MPMATH = False


def _parse_float_list(s: str) -> list[float]:
    out: list[float] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _complex_profile_cols(prefix: str, z: np.ndarray) -> dict[str, np.ndarray]:
    z = np.asarray(z, dtype=np.complex128).ravel()
    return {
        f"{prefix}_re": np.real(z).astype(np.float64),
        f"{prefix}_im": np.imag(z).astype(np.float64),
        f"{prefix}_abs": np.abs(z).astype(np.float64),
    }


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


def _logabs(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.complex128)
    return np.log(np.abs(z) + 1e-300).astype(np.float64)


def _det_I_minus_K(s: complex, packets: list[sim.Packet]) -> complex:
    K, _ = sim._global_K(s, packets)
    I = np.eye(K.shape[0], dtype=np.complex128)
    M = (I - K).astype(np.complex128)
    return complex(np.linalg.det(M))


def _continuous_logdet_from_det(det_vals: np.ndarray) -> np.ndarray:
    """Build a continuous log(det) along an ordered 1D grid by unwrapping arg(det)."""

    det_vals = np.asarray(det_vals, dtype=np.complex128).ravel()
    if det_vals.size == 0:
        return np.asarray([], dtype=np.complex128)
    logabs = _logabs(det_vals).astype(float)
    ang = np.angle(det_vals).astype(float)
    ang_u = np.unwrap(ang)
    return (logabs + 1j * ang_u).astype(np.complex128)


def _continuous_log_of_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Continuous log(num/den) along an ordered 1D grid.

    This avoids inconsistent branch choices that can occur if log(num) and log(den)
    are unwrapped independently and then subtracted.
    """

    num = np.asarray(num, dtype=np.complex128).ravel()
    den = np.asarray(den, dtype=np.complex128).ravel()
    if num.size != den.size:
        raise ValueError("num and den must have same length")
    ratio = num / den
    return _continuous_logdet_from_det(ratio)


def _loggamma_vec(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.complex128).ravel()
    if not _HAS_MPMATH:
        raise RuntimeError("mpmath is required for completion basis with logGamma terms")
    out = np.zeros_like(z, dtype=np.complex128)
    for i in range(int(z.size)):
        out[i] = complex(mp.loggamma(complex(z[i])))
    return out.astype(np.complex128)


def _completion_design_logdet(s_vals: np.ndarray, *, basis: str) -> np.ndarray:
    s_vals = np.asarray(s_vals, dtype=np.complex128).ravel()
    basis = str(basis).strip().lower()
    if basis == "none":
        return np.zeros((int(s_vals.size), 0), dtype=np.complex128)
    if basis == "poly2":
        cols = [np.ones_like(s_vals), s_vals, s_vals**2]
        return np.stack(cols, axis=1).astype(np.complex128)
    if basis == "poly2_gamma":
        lg0 = _loggamma_vec(s_vals / 2.0)
        lg1 = _loggamma_vec((s_vals + 1.0) / 2.0)
        cols = [np.ones_like(s_vals), s_vals, s_vals**2, lg0, lg1]
        return np.stack(cols, axis=1).astype(np.complex128)
    raise ValueError("basis must be one of: none, poly2, poly2_gamma")


def _fixed_logG(s_vals: np.ndarray, *, fixed: str) -> np.ndarray:
    """Return a fixed completion log-factor logG(s) along a grid.

    This is intentionally non-fitted: no regression, only named templates.

    Templates:
      - none: logG(s)=0
      - zeta: logG(s)=logGamma(s/2) - (s/2)log(pi)
      - gl2:  logG(s)=logGamma(s/2)+logGamma((s+1)/2) - s*log(pi)
      - zeta2:logG(s)=2*logGamma(s/2) - s*log(pi)
    """

    fixed = str(fixed).strip().lower()
    s_vals = np.asarray(s_vals, dtype=np.complex128).ravel()
    if fixed == "none":
        return np.zeros_like(s_vals, dtype=np.complex128)
    if not _HAS_MPMATH:
        raise RuntimeError("mpmath is required for fixed completion with logGamma terms")

    logpi = float(math.log(math.pi))
    if fixed == "zeta":
        return (_loggamma_vec(s_vals / 2.0) - 0.5 * s_vals * logpi).astype(np.complex128)
    if fixed == "gl2":
        return (_loggamma_vec(s_vals / 2.0) + _loggamma_vec((s_vals + 1.0) / 2.0) - s_vals * logpi).astype(
            np.complex128
        )
    if fixed == "zeta2":
        return (2.0 * _loggamma_vec(s_vals / 2.0) - s_vals * logpi).astype(np.complex128)
    raise ValueError("fixed completion must be one of: none, zeta, gl2, zeta2")


def _fit_completion_complex(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.complex128)
    y = np.asarray(y, dtype=np.complex128).ravel()
    if X.shape[0] != y.size:
        raise ValueError("X and y must have compatible shapes")
    if X.shape[1] == 0:
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


@dataclass(frozen=True)
class DefectSummary:
    u: float
    sigma: float
    basis: str
    rel_l2_F: float
    max_abs_F: float


def _build_packets_for_u(
    *,
    primes: list[int],
    k_max: int,
    boundary: tuple[int, int],
    schur_sign: str,
    u: float,
    sharp_mode: str,
    x_mode: str,
    x_gamma: float,
    p_mode: str,
    prime_power_mode: str,
    scattering_mode: str,
) -> list[sim.Packet]:
    v = -float(u)
    return sim._build_packets(
        primes,
        int(k_max),
        local_model="sixby6",
        prime_power_mode=str(prime_power_mode),
        boundary=boundary,
        sign=str(schur_sign),
        sharp_mode=str(sharp_mode),
        x_mode=str(x_mode),
        x_gamma=float(x_gamma),
        x_shear=float(u),
        x_lower=float(v),
        p_mode=str(p_mode),
        scattering_mode=str(scattering_mode),
        satake_family="trivial",
        satake_matrix="diag",
        theta_scale=1.0,
        seed=0,
    )


def _hermitian_defect_clamped(A: np.ndarray, *, den_floor: float) -> float:
    A = np.asarray(A, dtype=np.complex128)
    num = float(np.linalg.norm(A - A.conj().T, ord="fro"))
    den = float(np.linalg.norm(A, ord="fro"))
    return float(num / max(float(den_floor), den))


def _choose_boundary_sign_two_point(
    *,
    primes_seed: list[int],
    k_seed: list[int],
    sharp_mode: str,
    x_mode: str,
    x_gamma: float,
    p_mode: str,
    scattering_mode: str,
    u_probe: float,
    hd_tol: float,
    ud_tol: float,
    sens_min: float,
    den_floor: float,
) -> tuple[tuple[int, int], str, float, float, float]:
    """Pick a boundary/sign gauge using both u=0 and u=u_probe.

    Goal: avoid degenerate gauges where S_p(u) is (numerically) constant in u,
    while still keeping Lambda Hermitian and S unitary.

    Returns: (boundary, sign, max_hd, max_ud, sensitivity).
    """

    pairs: list[tuple[int, int]] = [(i, j) for i in range(6) for j in range(i + 1, 6)]
    u_list = [0.0, float(u_probe)]

    candidates: list[tuple[float, float, float, tuple[int, int], str]] = []
    for boundary in pairs:
        for sgn in ["-", "+"]:
            max_hd = 0.0
            max_ud = 0.0
            sens_acc = 0.0
            sens_n = 0

            ok_any = False
            for p in primes_seed:
                for k in k_seed:
                    S0 = None
                    for u in u_list:
                        try:
                            _, A, Ash = sim._local_blocks_for_prime_power(
                                int(p),
                                int(k),
                                sharp_mode=str(sharp_mode),
                                x_mode=str(x_mode),
                                x_gamma=float(x_gamma),
                                x_shear=float(u),
                                x_lower=float(-u),
                                p_mode=str(p_mode),
                            )
                            B = sim._bulk_B_from_A(A, Ash)
                            Lam = sim._schur_complement_Lambda(B, boundary=boundary, sign=str(sgn))
                            S = sim._scattering_from_Lambda(Lam, mode=str(scattering_mode))
                        except Exception:
                            continue

                        ok_any = True
                        max_hd = max(max_hd, _hermitian_defect_clamped(Lam, den_floor=float(den_floor)))
                        max_ud = max(max_ud, float(sim._unitarity_defect(S)))

                        if float(u) == 0.0:
                            S0 = S
                        elif S0 is not None:
                            sens_acc += float(np.linalg.norm(np.asarray(S, dtype=np.complex128) - np.asarray(S0, dtype=np.complex128), ord="fro"))
                            sens_n += 1

            if not ok_any:
                continue
            sensitivity = float(sens_acc / max(1, sens_n))
            candidates.append((sensitivity, max_hd, max_ud, boundary, str(sgn)))

    if not candidates:
        raise RuntimeError("boundary/sign selection failed")

    # Filter by tolerances, then pick the most u-sensitive candidate.
    good = [c for c in candidates if (c[1] <= float(hd_tol) and c[2] <= float(ud_tol) and c[0] >= float(sens_min))]
    if not good:
        good = [c for c in candidates if (c[1] <= float(hd_tol) and c[2] <= float(ud_tol))]
    if not good:
        # Fall back to best sensitivity even if tolerances aren't met.
        good = candidates

    good.sort(key=lambda x: (-x[0], x[1], x[2]))
    sensitivity, max_hd, max_ud, boundary, sgn = good[0]
    return tuple(boundary), str(sgn), float(max_hd), float(max_ud), float(sensitivity)


def _score_boundary_sign_two_point(
    *,
    primes_seed: list[int],
    k_seed: list[int],
    boundary: tuple[int, int],
    schur_sign: str,
    sharp_mode: str,
    x_mode: str,
    x_gamma: float,
    p_mode: str,
    scattering_mode: str,
    u_probe: float,
    den_floor: float,
) -> tuple[float, float, float]:
    """Compute (max_hd,max_ud,sensitivity) for a fixed boundary/sign using u=0 and u=u_probe."""

    boundary = tuple(int(i) for i in boundary)
    schur_sign = str(schur_sign).strip()
    if schur_sign not in {"-", "+"}:
        raise ValueError("schur_sign must be '+' or '-' for fixed gauge")

    u_list = [0.0, float(u_probe)]
    max_hd = 0.0
    max_ud = 0.0
    sens_acc = 0.0
    sens_n = 0

    for p in primes_seed:
        for k in k_seed:
            S0 = None
            for u in u_list:
                _, A, Ash = sim._local_blocks_for_prime_power(
                    int(p),
                    int(k),
                    sharp_mode=str(sharp_mode),
                    x_mode=str(x_mode),
                    x_gamma=float(x_gamma),
                    x_shear=float(u),
                    x_lower=float(-u),
                    p_mode=str(p_mode),
                )
                B = sim._bulk_B_from_A(A, Ash)
                Lam = sim._schur_complement_Lambda(B, boundary=boundary, sign=str(schur_sign))
                S = sim._scattering_from_Lambda(Lam, mode=str(scattering_mode))

                max_hd = max(max_hd, _hermitian_defect_clamped(Lam, den_floor=float(den_floor)))
                max_ud = max(max_ud, float(sim._unitarity_defect(S)))

                if float(u) == 0.0:
                    S0 = S
                elif S0 is not None:
                    sens_acc += float(
                        np.linalg.norm(
                            np.asarray(S, dtype=np.complex128) - np.asarray(S0, dtype=np.complex128),
                            ord="fro",
                        )
                    )
                    sens_n += 1

    sensitivity = float(sens_acc / max(1, sens_n))
    return float(max_hd), float(max_ud), float(sensitivity)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Compute FE defect perturbatively at u=0. "
            "Fits a completion proxy at u=0 (fixed thereafter), then reports F_u(s)=log(D_u(s)G(s)) - log(D_u(1-s)G(1-s)) "
            "and finite-difference derivatives at u=0."
        )
    )

    ap.add_argument("--primes_global", default="2,3,5,7,11,13,17,19,23,29,31,37")
    ap.add_argument(
        "--k_max",
        type=int,
        default=1,
        help="Max prime-power exponent k to include for each prime in the global determinant (uses k=1..k_max)",
    )
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--t_min", type=float, default=10.0)
    ap.add_argument("--t_max", type=float, default=30.0)
    ap.add_argument("--n_t", type=int, default=201)

    ap.add_argument("--h", type=float, default=0.01, help="Finite difference step for u (uses ±h)")
    ap.add_argument(
        "--completion_mode",
        choices=["fit_u0", "fixed"],
        default="fit_u0",
        help=(
            "Completion mode: fit_u0 performs a least-squares completion fit at u=0 (then freezes it); "
            "fixed uses a named non-fitted completion template."
        ),
    )
    ap.add_argument("--completion_basis", choices=["none", "poly2", "poly2_gamma"], default="poly2_gamma")
    ap.add_argument(
        "--fixed_completion",
        choices=["none", "zeta", "gl2", "zeta2"],
        default="zeta",
        help=(
            "Fixed (non-fitted) completion template used when --completion_mode fixed. "
            "zeta is the usual pi^{-s/2}Gamma(s/2) factor in log form."
        ),
    )

    ap.add_argument(
        "--u_list",
        default="",
        help=(
            "Optional comma-list of additional u values to evaluate (completion still frozen from u=0 fit). "
            "Example: --u_list -0.2,-0.1,-0.05,0,0.05,0.1,0.2"
        ),
    )

    ap.add_argument(
        "--u_probe",
        type=float,
        default=0.2,
        help=(
            "Select the boundary/sign gauge using both u=0 and u=u_probe, "
            "to avoid degenerate gauges with S_p(u) constant in u."
        ),
    )
    ap.add_argument("--p_seed_n", type=int, default=8, help="Number of seed primes used for boundary/sign selection")
    ap.add_argument("--k_seed_max", type=int, default=3, help="Use k=1..k_seed_max in boundary/sign selection")
    ap.add_argument("--hd_tol", type=float, default=1e-6, help="Tolerance for clamped Hermitian defect during gauge selection")
    ap.add_argument("--ud_tol", type=float, default=1e-6, help="Tolerance for unitarity defect during gauge selection")
    ap.add_argument("--sens_min", type=float, default=1e-6, help="Minimum u-sensitivity during gauge selection")
    ap.add_argument("--den_floor", type=float, default=1e-9, help="Denominator floor for clamped Hermitian defect")

    ap.add_argument(
        "--boundary",
        default="auto",
        help=(
            "Boundary indices as 'i,j' in 0..5 (2 indices), or 'auto' to select boundary/sign by two-point gauge search. "
            "Using a fixed boundary/sign is recommended when comparing p_mode choices."
        ),
    )
    ap.add_argument("--schur_sign", choices=["-", "+"], default="+", help="Schur sign used when --boundary is not auto")

    # Conventions
    ap.add_argument("--sharp", choices=["transpose", "conj_transpose"], default="conj_transpose")
    ap.add_argument("--X_mode", choices=["raw", "det1"], default="det1")
    ap.add_argument("--X_gamma", type=float, default=1.0)
    ap.add_argument(
        "--p_mode",
        choices=["p", "logp", "p1_over_p", "p_over_p1", "invp", "p_minus1_over_p"],
        default="p",
        help=(
            "How to inject p into the local diagonal factor D. "
            "This is a targeted test for which prime-length clock the geometry prefers."
        ),
    )
    ap.add_argument(
        "--prime_power_mode",
        choices=["direct", "x_power", "bulk_power"],
        default="direct",
        help=(
            "How to generate prime-power packets k>1 for the sixby6 local model. "
            "direct builds each (p,k) independently. "
            "x_power enforces a semigroup at the 2x2 X level: build X_{p,1} then set X_{p,k}=X_{p,1}^k before reduction. "
            "bulk_power enforces a semigroup on the 6x6 bulk generator: build B_{p,1} then set B_{p,k}=B_{p,1}^k before Schur/Cayley reduction."
        ),
    )
    ap.add_argument("--scattering", choices=["lambda_pm_i", "i_pm_lambda"], default="i_pm_lambda")

    ap.add_argument("--out_csv", required=True)
    ap.add_argument(
        "--out_profile_csv",
        default="",
        help=(
            "Optional per-t profile CSV path (writes t, s, |logD0|, and Re/Im/|.| for F0,F±h,dF,d2F)."
        ),
    )

    args = ap.parse_args()

    primes = sim._parse_int_list(str(args.primes_global))
    if not primes:
        raise SystemExit("--primes_global must be non-empty")

    k_max = int(args.k_max)
    if k_max <= 0:
        raise SystemExit("--k_max must be positive")

    sigma = float(args.sigma)
    if abs(sigma - 0.5) < 1e-12:
        print("note: sigma=0.5 makes s->1-s equal conjugation on this line; use sigma!=0.5 for nontrivial FE defect")

    h = float(args.h)
    if not math.isfinite(h) or h <= 0:
        raise SystemExit("--h must be positive")

    completion_mode = str(args.completion_mode)
    basis = str(args.completion_basis)
    fixed_completion = str(args.fixed_completion)
    if completion_mode == "fit_u0":
        if basis == "poly2_gamma" and not _HAS_MPMATH:
            raise SystemExit("--completion_basis poly2_gamma requires mpmath")
    else:
        if fixed_completion != "none" and not _HAS_MPMATH:
            raise SystemExit("--completion_mode fixed with gamma factors requires mpmath")

    t_grid = np.linspace(float(args.t_min), float(args.t_max), int(args.n_t), dtype=float)
    w = _trapz_weights(t_grid)

    u_list_extra = _parse_float_list(str(args.u_list)) if str(args.u_list).strip() else []

    # Pick a boundary/sign gauge using u=0 and a small u-probe.
    # This avoids selecting a degenerate gauge where S_p(u) is constant in u,
    # which would make d/du FE defects identically 0 for purely gauge reasons.
    p_seed_n = int(args.p_seed_n)
    if p_seed_n <= 0:
        raise SystemExit("--p_seed_n must be positive")
    if p_seed_n >= len(primes):
        primes_seed = list(primes)
    else:
        half = max(1, p_seed_n // 2)
        left = list(primes[:half])
        right = list(primes[-(p_seed_n - len(left)) :])
        primes_seed = []
        for p in left + right:
            if int(p) not in primes_seed:
                primes_seed.append(int(p))

    k_seed_max = int(args.k_seed_max)
    if k_seed_max <= 0:
        raise SystemExit("--k_seed_max must be positive")
    k_seed = list(range(1, k_seed_max + 1))

    boundary_arg = str(args.boundary).strip().lower()
    if boundary_arg == "auto":
        boundary, schur_sign, seed_hd, seed_ud, seed_sens = _choose_boundary_sign_two_point(
            primes_seed=primes_seed,
            k_seed=k_seed,
            sharp_mode=str(args.sharp),
            x_mode=str(args.X_mode),
            x_gamma=float(args.X_gamma),
            p_mode=str(args.p_mode),
            scattering_mode=str(args.scattering),
            u_probe=float(args.u_probe),
            hd_tol=float(args.hd_tol),
            ud_tol=float(args.ud_tol),
            sens_min=float(args.sens_min),
            den_floor=float(args.den_floor),
        )
    else:
        parts = sim._parse_int_list(str(args.boundary))
        if len(parts) != 2:
            raise SystemExit("--boundary must be 'auto' or 'i,j' with two indices")
        boundary = (int(parts[0]), int(parts[1]))
        schur_sign = str(args.schur_sign)
        seed_hd, seed_ud, seed_sens = _score_boundary_sign_two_point(
            primes_seed=primes_seed,
            k_seed=k_seed,
            boundary=boundary,
            schur_sign=str(schur_sign),
            sharp_mode=str(args.sharp),
            x_mode=str(args.X_mode),
            x_gamma=float(args.X_gamma),
            p_mode=str(args.p_mode),
            scattering_mode=str(args.scattering),
            u_probe=float(args.u_probe),
            den_floor=float(args.den_floor),
        )

    print(
        "gauge | "
        + " | ".join(
            [
                f"boundary={boundary}",
                f"sign={schur_sign}",
                f"u_probe={float(args.u_probe):g}",
                f"seed_hd={seed_hd:.3e}",
                f"seed_ud={seed_ud:.3e}",
                f"seed_sens={seed_sens:.3e}",
            ]
        )
    )

    packets_0 = _build_packets_for_u(
        primes=primes,
        k_max=int(k_max),
        boundary=boundary,
        schur_sign=str(schur_sign),
        u=0.0,
        sharp_mode=str(args.sharp),
        x_mode=str(args.X_mode),
        x_gamma=float(args.X_gamma),
        p_mode=str(args.p_mode),
        prime_power_mode=str(args.prime_power_mode),
        scattering_mode=str(args.scattering),
    )
    packets_p = _build_packets_for_u(
        primes=primes,
        k_max=int(k_max),
        boundary=boundary,
        schur_sign=str(schur_sign),
        u=+h,
        sharp_mode=str(args.sharp),
        x_mode=str(args.X_mode),
        x_gamma=float(args.X_gamma),
        p_mode=str(args.p_mode),
        prime_power_mode=str(args.prime_power_mode),
        scattering_mode=str(args.scattering),
    )
    packets_m = _build_packets_for_u(
        primes=primes,
        k_max=int(k_max),
        boundary=boundary,
        schur_sign=str(schur_sign),
        u=-h,
        sharp_mode=str(args.sharp),
        x_mode=str(args.X_mode),
        x_gamma=float(args.X_gamma),
        p_mode=str(args.p_mode),
        prime_power_mode=str(args.prime_power_mode),
        scattering_mode=str(args.scattering),
    )

    s_line = (sigma + 1j * t_grid).astype(np.complex128)
    s_reflect = (1.0 - sigma - 1j * t_grid).astype(np.complex128)

    def det_line(packets: list[sim.Packet], svals: np.ndarray) -> np.ndarray:
        out = np.zeros_like(svals, dtype=np.complex128)
        for i in range(int(svals.size)):
            out[i] = complex(_det_I_minus_K(complex(svals[i]), packets))
        return out.astype(np.complex128)

    det0_s = det_line(packets_0, s_line)
    det0_r = det_line(packets_0, s_reflect)

    # Continuous log of the ratio is the stable FE-defect primitive.
    logr0 = _continuous_log_of_ratio(det0_s, det0_r)
    log0_s = _continuous_logdet_from_det(det0_s)
    log0_r = _continuous_logdet_from_det(det0_r)

    if completion_mode == "fit_u0":
        # Fit completion at u=0 so that logD0(s)+logG(s) ≈ logD0(1-s)+logG(1-s).
        Xs = _completion_design_logdet(s_line, basis=basis)
        Xr = _completion_design_logdet(s_reflect, basis=basis)
        A = (Xs - Xr).astype(np.complex128)
        y = (-logr0).astype(np.complex128)
        c = _fit_completion_complex(A, y)

        logG_s = (Xs @ c.reshape(-1, 1)).reshape(-1).astype(np.complex128) if Xs.shape[1] else np.zeros_like(log0_s)
        logG_r = (Xr @ c.reshape(-1, 1)).reshape(-1).astype(np.complex128) if Xr.shape[1] else np.zeros_like(log0_r)
    else:
        # Non-fitted completion: choose a fixed template logG(s).
        # This is the most RH-relevant check because it cannot "explain away" defect by regression.
        logG_s = _fixed_logG(s_line, fixed=fixed_completion)
        logG_r = _fixed_logG(s_reflect, fixed=fixed_completion)

    def defect_for_packets(packets: list[sim.Packet]) -> np.ndarray:
        det_s = det_line(packets, s_line)
        det_r = det_line(packets, s_reflect)
        logr = _continuous_log_of_ratio(det_s, det_r)
        return (logr + (logG_s - logG_r)).astype(np.complex128)

    F0 = defect_for_packets(packets_0)
    Fp = defect_for_packets(packets_p)
    Fm = defect_for_packets(packets_m)

    dF = ((Fp - Fm) / (2.0 * h)).astype(np.complex128)
    d2F = (((Fp - 2.0 * F0 + Fm) / (h * h))).astype(np.complex128)

    denom0 = _weighted_norm(w, log0_s) + 1e-300
    def rel(z: np.ndarray) -> float:
        return float(_weighted_norm(w, z) / denom0)

    rows: list[dict] = []
    for label, u_val, F in [
        ("F0", 0.0, F0),
        ("Fp", +h, Fp),
        ("Fm", -h, Fm),
        ("dF_du_u0", 0.0, dF),
        ("d2F_du2_u0", 0.0, d2F),
    ]:
        rows.append(
            {
                "label": str(label),
                "u": float(u_val),
                "h": float(h),
                "sigma": float(sigma),
                "completion_mode": str(completion_mode),
                "completion_basis": str(basis),
                "fixed_completion": str(fixed_completion),
                "boundary": f"{boundary[0]},{boundary[1]}",
                "schur_sign": str(schur_sign),
                "seed_max_herm_def": float(seed_hd),
                "seed_max_unit_def": float(seed_ud),
                "k_max": int(k_max),
                "p_mode": str(args.p_mode),
                "prime_power_mode": str(args.prime_power_mode),
                "rel_l2": float(rel(F)),
                "max_abs": float(np.max(np.abs(F))),
            }
        )

    # Compare size of F_u(s) against F_0(s) at additional u values.
    for u_val in u_list_extra:
        u_val = float(u_val)
        packets_u = _build_packets_for_u(
            primes=primes,
            k_max=int(k_max),
            boundary=boundary,
            schur_sign=str(schur_sign),
            u=float(u_val),
            sharp_mode=str(args.sharp),
            x_mode=str(args.X_mode),
            x_gamma=float(args.X_gamma),
            p_mode=str(args.p_mode),
            prime_power_mode=str(args.prime_power_mode),
            scattering_mode=str(args.scattering),
        )
        Fu = defect_for_packets(packets_u)
        rows.append(
            {
                "label": "Fu",
                "u": float(u_val),
                "h": float(h),
                "sigma": float(sigma),
                "completion_mode": str(completion_mode),
                "completion_basis": str(basis),
                "fixed_completion": str(fixed_completion),
                "boundary": f"{boundary[0]},{boundary[1]}",
                "schur_sign": str(schur_sign),
                "seed_max_herm_def": float(seed_hd),
                "seed_max_unit_def": float(seed_ud),
                "k_max": int(k_max),
                "p_mode": str(args.p_mode),
                "prime_power_mode": str(args.prime_power_mode),
                "rel_l2": float(rel(Fu)),
                "max_abs": float(np.max(np.abs(Fu))),
            }
        )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)

    profile_path = str(args.out_profile_csv).strip()
    if profile_path:
        prof: dict[str, np.ndarray] = {
            "t": t_grid.astype(np.float64),
            "s_re": np.real(s_line).astype(np.float64),
            "s_im": np.imag(s_line).astype(np.float64),
            "logD0_abs": np.abs(log0_s).astype(np.float64),
        }
        prof.update(_complex_profile_cols("F0", F0))
        prof.update(_complex_profile_cols("Fp", Fp))
        prof.update(_complex_profile_cols("Fm", Fm))
        prof.update(_complex_profile_cols("dF", dF))
        prof.update(_complex_profile_cols("d2F", d2F))

        pth = Path(profile_path)
        pth.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(prof).to_csv(pth, index=False)
        print(f"wrote profile {pth}")

    print(
        " | ".join(
            [
                f"mode={completion_mode}",
                f"basis={basis}",
                f"fixed={fixed_completion}",
                f"sigma={sigma:g}",
                f"h={h:g}",
                f"rel(F0)={rel(F0):.3e}",
                f"rel(dF)={rel(dF):.3e}",
                f"rel(d2F)={rel(d2F):.3e}",
            ]
        )
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
