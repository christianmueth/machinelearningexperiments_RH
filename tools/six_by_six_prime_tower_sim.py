from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
import hashlib
from typing import Mapping

import numpy as np
import pandas as pd


def _parse_int_list(s: str) -> list[int]:
    out: list[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _parse_float_list(s: str) -> list[float]:
    out: list[float] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _cayley(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.complex128)
    n = int(M.shape[0])
    I = np.eye(n, dtype=np.complex128)
    return (M - 1j * I) @ np.linalg.inv(M + 1j * I)


def _unit_circle(z: complex) -> complex:
    z = complex(z)
    a = abs(z)
    if not math.isfinite(a) or a <= 0:
        return complex(1.0)
    return z / a


def _angle_from_prime(p: int, *, seed: int) -> float:
    """Deterministically map a prime -> angle in [0,2π)."""
    h = hashlib.blake2b(f"p={int(p)}|seed={int(seed)}".encode("utf-8"), digest_size=8).digest()
    u = int.from_bytes(h, "little", signed=False)
    return (float(u) / float(2**64)) * (2.0 * math.pi)


def _satake_params_for_prime(
    p: int,
    *,
    family: str,
    theta_scale: float,
    seed: int,
) -> tuple[complex, complex]:
    """Return (alpha_p, beta_p) for an unramified rank-1 style local packet.

    Conventions:
    - For zeta: alpha=beta=1.
    - For a unitary det-1 family: alpha=exp(iθ), beta=exp(-iθ).
    """

    p = int(p)
    family = str(family).strip().lower()
    theta_scale = float(theta_scale)

    if family == "trivial":
        return complex(1.0), complex(1.0)

    if family in {"phase_hash", "unitary_hash"}:
        theta = theta_scale * _angle_from_prime(p, seed=int(seed))
        return complex(math.cos(theta) + 1j * math.sin(theta)), complex(math.cos(theta) - 1j * math.sin(theta))

    if family in {"phase_logp", "unitary_logp"}:
        theta = theta_scale * float(math.log(float(p)))
        theta = float(theta % (2.0 * math.pi))
        return complex(math.cos(theta) + 1j * math.sin(theta)), complex(math.cos(theta) - 1j * math.sin(theta))

    raise ValueError("satake_family must be one of: trivial, phase_hash, phase_logp, table")


def _load_satake_table_csv(path: str | Path) -> dict[int, tuple[complex, complex]]:
    """Load a per-prime Satake table from CSV.

    Supported schemas:
      (A) p, alpha_re, alpha_im, beta_re, beta_im
      (B) p, theta  (interpreted as alpha=exp(i theta), beta=exp(-i theta))
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if df.empty or "p" not in df.columns:
        raise ValueError("satake table CSV must include non-empty column 'p'")

    out: dict[int, tuple[complex, complex]] = {}

    cols = set(df.columns)
    has_ab = {"alpha_re", "alpha_im", "beta_re", "beta_im"}.issubset(cols)
    has_theta = "theta" in cols

    if not has_ab and not has_theta:
        raise ValueError("satake table CSV must include either alpha/beta columns or a theta column")

    for _, r in df.iterrows():
        p = int(r["p"])
        if has_ab:
            a = complex(float(r["alpha_re"]), float(r["alpha_im"]))
            b = complex(float(r["beta_re"]), float(r["beta_im"]))
        else:
            theta = float(r["theta"])
            a = complex(math.cos(theta) + 1j * math.sin(theta))
            b = complex(math.cos(theta) - 1j * math.sin(theta))
        out[p] = (a, b)

    if not out:
        raise ValueError("satake table CSV produced no rows")
    return out


def _satake_matrix_from_params(alpha: complex, beta: complex, *, mode: str) -> np.ndarray:
    """Return a 2x2 local packet with eigenvalues alpha,beta.

    mode:
      - diag: S=diag(alpha,beta) (unitary if |alpha|=|beta|=1).
      - companion: M=[[λ,-μ],[1,0]] with λ=alpha+beta, μ=alpha*beta.
    """
    alpha = complex(alpha)
    beta = complex(beta)
    mode = str(mode).strip().lower()
    if mode == "diag":
        return np.array([[alpha, 0.0], [0.0, beta]], dtype=np.complex128)
    if mode == "companion":
        lam = alpha + beta
        mu = alpha * beta
        return np.array([[lam, -mu], [1.0, 0.0]], dtype=np.complex128)
    raise ValueError("satake_matrix must be 'diag' or 'companion'")


def _J2() -> np.ndarray:
    return np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.complex128)


def _symplectic_partner(A: np.ndarray, *, mode: str) -> np.ndarray:
    A = np.asarray(A, dtype=np.complex128)
    if A.shape != (2, 2):
        raise ValueError("A must be 2x2")
    mode = str(mode).strip().lower()
    if mode not in {"transpose", "conj_transpose"}:
        raise ValueError("mode must be 'transpose' or 'conj_transpose'")
    J = _J2()
    Jinv = -J  # J^{-1} = -J for 2x2 symplectic form
    AT = A.T if mode == "transpose" else A.conj().T
    return (J @ AT @ Jinv).astype(np.complex128)


def _local_blocks_for_prime_power(
    p: int,
    k: int,
    *,
    sharp_mode: str,
    x_mode: str,
    x_gamma: float,
    x_shear: float,
    x_lower: float,
    p_mode: str = "p",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, A, A_sharp) for the prime-power p^k block."""

    p = int(p)
    k = int(k)
    if p < 2 or k < 0:
        raise ValueError("p>=2, k>=0 required")

    x_mode = str(x_mode).strip().lower()
    if x_mode not in {"raw", "det1"}:
        raise ValueError("x_mode must be 'raw' or 'det1'")

    x_gamma = float(x_gamma)
    if not math.isfinite(x_gamma):
        raise ValueError("x_gamma must be finite")

    x_shear = float(x_shear)
    if not math.isfinite(x_shear):
        raise ValueError("x_shear must be finite")

    x_lower = float(x_lower)
    if not math.isfinite(x_lower):
        raise ValueError("x_lower must be finite")

    # Allow alternative prime parameterizations to test which length law the geometry prefers.
    # This changes ONLY how p enters the diagonal factor D; the rest of the construction is unchanged.
    p_mode = str(p_mode).strip().lower()
    p_f = float(p)
    if p_mode in {"p", "prime"}:
        p_eff = p_f
    elif p_mode in {"logp", "log(p)", "lnp"}:
        p_eff = float(math.log(p_f))
    elif p_mode in {"p1_over_p", "(p+1)/p", "one_plus_invp", "1+1/p"}:
        p_eff = (p_f + 1.0) / p_f
    elif p_mode in {"p_over_p1", "p/(p+1)"}:
        p_eff = p_f / (p_f + 1.0)
    elif p_mode in {"invp", "1/p"}:
        p_eff = 1.0 / p_f
    elif p_mode in {"p_minus1_over_p", "(p-1)/p", "1-1/p"}:
        p_eff = (p_f - 1.0) / p_f
    else:
        raise ValueError("p_mode must be one of: p, logp, p1_over_p, p_over_p1, invp, p_minus1_over_p")

    if not (p_eff > 0.0) or not math.isfinite(p_eff):
        raise ValueError(f"invalid effective prime parameter p_eff={p_eff} from p_mode={p_mode}")

    if x_mode == "raw":
        x11 = float(p_eff) ** (float(x_gamma) * float(k))
        x22 = 1.0
    else:
        # Determinant-1 normalization: diag(p^{k/2}, p^{-k/2}).
        # Many SL(2)/symplectic constructions implicitly assume det(X)=1.
        x11 = float(p_eff) ** (0.5 * float(x_gamma) * float(k))
        x22 = float(p_eff) ** (-0.5 * float(x_gamma) * float(k))

    # Full SL(2)-style det-preserving deformation around the diagonal factor:
    #   X = U(u) * D * L(v)
    # where U(u)=[[1,u],[0,1]] and L(v)=[[1,0],[v,1]].
    # det(U)=det(L)=1 so det(X)=det(D)=x11*x22.
    D = np.array([[x11, 0.0], [0.0, x22]], dtype=np.complex128)
    U = np.array([[1.0, x_shear], [0.0, 1.0]], dtype=np.complex128)
    L = np.array([[1.0, 0.0], [x_lower, 1.0]], dtype=np.complex128)
    X = (U @ D @ L).astype(np.complex128)
    A = _cayley(X)
    Ash = _symplectic_partner(A, mode=str(sharp_mode))
    return X, A, Ash


def _bulk_B_from_A(A: np.ndarray, Ash: np.ndarray) -> np.ndarray:
    """Construct the 6x6 bulk matrix B from 2x2 blocks A and A^sharp."""

    A = np.asarray(A, dtype=np.complex128)
    Ash = np.asarray(Ash, dtype=np.complex128)
    if A.shape != (2, 2) or Ash.shape != (2, 2):
        raise ValueError("A and Ash must be 2x2")

    Z = np.zeros((2, 2), dtype=np.complex128)
    row1 = np.concatenate([Z, -Ash, -A], axis=1)
    row2 = np.concatenate([-A, Z, -Ash], axis=1)
    row3 = np.concatenate([-Ash, -A, Z], axis=1)
    return np.concatenate([row1, row2, row3], axis=0).astype(np.complex128)


def _schur_complement_Lambda(B: np.ndarray, *, boundary: tuple[int, int], sign: str) -> np.ndarray:
    """Schur complement Lambda on a chosen 2D boundary subspace.

    boundary: a pair of indices in {0..5}.
    sign:
      - '-' : Lambda = B_bb - B_bi B_ii^{-1} B_ib
      - '+' : Lambda = B_bb + B_bi B_ii^{-1} B_ib

    The theoretical goal is that Lambda is (approximately) self-adjoint in the
    intended boundary pairing.
    """

    B = np.asarray(B, dtype=np.complex128)
    if B.shape != (6, 6):
        raise ValueError("B must be 6x6")

    bidx = tuple(int(i) for i in boundary)
    if len(bidx) != 2 or len(set(bidx)) != 2 or any((i < 0 or i > 5) for i in bidx):
        raise ValueError("boundary must be two distinct indices in 0..5")
    sign = str(sign).strip()
    if sign not in {"-", "+"}:
        raise ValueError("sign must be '+' or '-'")

    all_idx = list(range(6))
    iidx = [i for i in all_idx if i not in bidx]

    bb = B[np.ix_(bidx, bidx)]
    bi = B[np.ix_(bidx, iidx)]
    ib = B[np.ix_(iidx, bidx)]
    ii = B[np.ix_(iidx, iidx)]

    ii_inv = np.linalg.inv(ii)
    corr = (bi @ ii_inv @ ib).astype(np.complex128)
    Lam = (bb - corr) if sign == "-" else (bb + corr)
    return Lam.astype(np.complex128)


def _scattering_from_Lambda(Lam: np.ndarray, *, mode: str) -> np.ndarray:
    Lam = np.asarray(Lam, dtype=np.complex128)
    if Lam.shape != (2, 2):
        raise ValueError("Lambda must be 2x2")
    I = np.eye(2, dtype=np.complex128)
    mode = str(mode).strip().lower()
    if mode not in {"lambda_pm_i", "i_pm_lambda"}:
        raise ValueError("mode must be 'lambda_pm_i' or 'i_pm_lambda'")

    # Two common Cayley conventions.
    # - 'lambda_pm_i' : S=(Λ - iI)(Λ + iI)^{-1} maps Λ=0 -> S=-I.
    # - 'i_pm_lambda' : S=(I - iΛ)(I + iΛ)^{-1} maps Λ=0 -> S= I.
    if mode == "lambda_pm_i":
        return (Lam - 1j * I) @ np.linalg.inv(Lam + 1j * I)
    return (I - 1j * Lam) @ np.linalg.inv(I + 1j * Lam)


def _hermitian_defect(A: np.ndarray) -> float:
    A = np.asarray(A, dtype=np.complex128)
    num = float(np.linalg.norm(A - A.conj().T, ord="fro"))
    den = float(np.linalg.norm(A, ord="fro"))
    return float(num / (den + 1e-300))


def _unitarity_defect(U: np.ndarray) -> float:
    U = np.asarray(U, dtype=np.complex128)
    n = int(U.shape[0])
    I = np.eye(n, dtype=np.complex128)
    num = float(np.linalg.norm(U.conj().T @ U - I, ord="fro"))
    den = float(np.linalg.norm(I, ord="fro"))
    return float(num / (den + 1e-300))


@dataclass(frozen=True)
class Packet:
    p: int
    k: int
    ell: float  # length = k log p
    S: np.ndarray  # 2x2
    Lam: np.ndarray  # 2x2


def _boundary_search(
    packets_seed: list[int],
    k_seed: list[int],
    *,
    sharp_mode: str,
    x_mode: str,
    x_gamma: float,
    x_shear: float,
    x_lower: float,
    p_mode: str = "p",
    scattering_mode: str,
) -> tuple[tuple[int, int], str, float, float]:
    """Search boundary index pair + Schur sign that best enforces Hermiticity/unitarity.

    Returns (boundary_pair, sign, max_herm_def, max_unit_def) for the best candidate.
    """

    # Build a small set of representative B matrices.
    Bs: list[np.ndarray] = []
    for p in packets_seed:
        for k in k_seed:
            _, A, Ash = _local_blocks_for_prime_power(
                int(p),
                int(k),
                sharp_mode=str(sharp_mode),
                x_mode=str(x_mode),
                x_gamma=float(x_gamma),
                x_shear=float(x_shear),
                x_lower=float(x_lower),
                p_mode=str(p_mode),
            )
            Bs.append(_bulk_B_from_A(A, Ash))

    best: tuple[tuple[int, int], str, float, float] | None = None
    best_score = float("inf")

    idx = list(range(6))
    pairs: list[tuple[int, int]] = []
    for i in range(6):
        for j in range(i + 1, 6):
            pairs.append((i, j))

    for pair in pairs:
        for sgn in ["-", "+"]:
            max_hd = 0.0
            max_ud = 0.0
            ok_any = False
            for B in Bs:
                try:
                    Lam = _schur_complement_Lambda(B, boundary=pair, sign=sgn)
                    S = _scattering_from_Lambda(Lam, mode=str(scattering_mode))
                except Exception:
                    continue
                ok_any = True
                max_hd = max(max_hd, _hermitian_defect(Lam))
                max_ud = max(max_ud, _unitarity_defect(S))
            if not ok_any:
                continue

            # Objective: prioritize Hermiticity strongly; unitarity is secondary.
            score = float(math.log10(max_hd + 1e-12) + 0.3 * math.log10(max_ud + 1e-12))
            if score < best_score:
                best_score = score
                best = (pair, sgn, float(max_hd), float(max_ud))

    if best is None:
        raise RuntimeError("boundary search failed")
    return best


def _build_packets(
    primes: list[int],
    k_max: int,
    *,
    local_model: str,
    prime_power_mode: str = "direct",
    boundary: tuple[int, int],
    sign: str,
    sharp_mode: str,
    x_mode: str,
    x_gamma: float,
    x_shear: float,
    x_lower: float,
    p_mode: str = "p",
    scattering_mode: str,
    satake_family: str,
    satake_matrix: str,
    theta_scale: float,
    seed: int,
    satake_table: Mapping[int, tuple[complex, complex]] | None = None,
) -> list[Packet]:
    packets: list[Packet] = []
    local_model = str(local_model).strip().lower()
    if local_model not in {"sixby6", "satake"}:
        raise ValueError("local_model must be 'sixby6' or 'satake'")

    prime_power_mode = str(prime_power_mode).strip().lower()
    if prime_power_mode not in {"direct", "x_power"}:
        raise ValueError("prime_power_mode must be one of: direct, x_power")
    if local_model != "sixby6" and prime_power_mode != "direct":
        raise ValueError("prime_power_mode is only supported for local_model='sixby6'")

    for p in primes:
        if local_model == "sixby6" and prime_power_mode == "x_power":
            # Semigroup-enforced prime-power tower: generate higher k blocks by powering
            # a single upstream generator X_{p,1}.
            X1, _, _ = _local_blocks_for_prime_power(
                int(p),
                1,
                sharp_mode=str(sharp_mode),
                x_mode=str(x_mode),
                x_gamma=float(x_gamma),
                x_shear=float(x_shear),
                x_lower=float(x_lower),
                p_mode=str(p_mode),
            )

        for k in range(1, int(k_max) + 1):
            if local_model == "sixby6":
                if prime_power_mode == "direct":
                    _, A, Ash = _local_blocks_for_prime_power(
                        int(p),
                        int(k),
                        sharp_mode=str(sharp_mode),
                        x_mode=str(x_mode),
                        x_gamma=float(x_gamma),
                        x_shear=float(x_shear),
                        x_lower=float(x_lower),
                        p_mode=str(p_mode),
                    )
                else:
                    Xk = np.linalg.matrix_power(np.asarray(X1, dtype=np.complex128), int(k))
                    A = _cayley(Xk)
                    Ash = _symplectic_partner(A, mode=str(sharp_mode))

                B = _bulk_B_from_A(A, Ash)
                Lam = _schur_complement_Lambda(B, boundary=boundary, sign=str(sign))
                S = _scattering_from_Lambda(Lam, mode=str(scattering_mode))
            else:
                # Satake injection model: build a local 2x2 packet directly from (α_p,β_p).
                if satake_table is not None:
                    ab = satake_table.get(int(p))
                    if ab is None:
                        raise KeyError(f"satake_table missing prime p={int(p)}")
                    alpha, beta = complex(ab[0]), complex(ab[1])
                else:
                    alpha, beta = _satake_params_for_prime(
                        int(p),
                        family=str(satake_family),
                        theta_scale=float(theta_scale),
                        seed=int(seed),
                    )
                S = _satake_matrix_from_params(alpha, beta, mode=str(satake_matrix))
                Lam = np.zeros((2, 2), dtype=np.complex128)
            ell = float(k) * float(math.log(float(p)))
            packets.append(Packet(p=int(p), k=int(k), ell=float(ell), S=S, Lam=Lam))
    return packets


def _global_K(s: complex, packets: list[Packet]) -> tuple[np.ndarray, np.ndarray]:
    """Return (K(s), K'(s)) for K(s)=diag(exp(-s ell) S_{p,k})."""

    s = complex(s)
    n_blocks = int(len(packets))
    dim = 2 * n_blocks

    K = np.zeros((dim, dim), dtype=np.complex128)
    Kp = np.zeros((dim, dim), dtype=np.complex128)

    for i, pkt in enumerate(packets):
        a = complex(np.exp(-s * float(pkt.ell)))
        block = (a * pkt.S).astype(np.complex128)
        # d/ds exp(-s ell) = -ell exp(-s ell)
        blockp = (-(float(pkt.ell)) * a * pkt.S).astype(np.complex128)

        r0 = 2 * i
        K[r0 : r0 + 2, r0 : r0 + 2] = block
        Kp[r0 : r0 + 2, r0 : r0 + 2] = blockp

    return K, Kp


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Implements the 6x6 local construction X_k->A_k->A_k^sharp->B_k, "
            "Schur complement Lambda_k (2x2), Cayley scattering S_k (2x2), then assembles "
            "a global return operator K(s)=diag(exp(-s k log p) S_{p,k}). "
            "Emits D(s)=det(I-K(s)) on a vertical line for Mangoldt-probe fitting."
        )
    )
    ap.add_argument("--primes", default="2,3,5,7,11,13", help="Comma list of primes p")
    ap.add_argument("--k_max", type=int, default=6, help="Max exponent k for each prime (uses k=1..k_max)")

    ap.add_argument("--sigma", type=float, default=2.0, help="Real part of s")
    ap.add_argument("--t_min", type=float, default=10.0)
    ap.add_argument("--t_max", type=float, default=30.0)
    ap.add_argument("--n_t", type=int, default=401)

    ap.add_argument("--out_csv", required=True)

    ap.add_argument("--m_trace", type=int, default=10, help="Also compute trace-power logdet check up to m")

    ap.add_argument(
        "--local_model",
        choices=["sixby6", "satake"],
        default="sixby6",
        help=(
            "Local packet model: 'sixby6' uses the 6x6 construction to produce Lambda,S; "
            "'satake' injects an explicit Hecke/Satake packet S_p with chosen (alpha_p,beta_p)."
        ),
    )

    # Satake injection options (only used if --local_model satake)
    ap.add_argument(
        "--satake_family",
        choices=["trivial", "phase_hash", "phase_logp", "table"],
        default="trivial",
        help=(
            "Satake parameter family: trivial sets alpha=beta=1 (zeta-like); "
            "phase_hash uses a deterministic prime->angle hash; phase_logp uses theta ~ scale*log p."
        ),
    )
    ap.add_argument(
        "--satake_matrix",
        choices=["diag", "companion"],
        default="diag",
        help=(
            "How to realize the Satake packet from (alpha,beta): diag(alpha,beta) or companion matrix [[λ,-μ],[1,0]]."
        ),
    )
    ap.add_argument("--theta_scale", type=float, default=1.0, help="Scale for the Satake phase (used for phase_* families)")
    ap.add_argument("--seed", type=int, default=0, help="Seed for satake_family=phase_hash")
    ap.add_argument(
        "--satake_table_csv",
        default="",
        help=(
            "For --local_model satake with --satake_family table: path to a CSV giving (alpha_p,beta_p) or theta per prime. "
            "Columns: p plus either (alpha_re,alpha_im,beta_re,beta_im) or (theta)."
        ),
    )

    ap.add_argument(
        "--boundary",
        default="auto",
        help=(
            "Boundary indices as 'i,j' in 0..5 (2 indices), or 'auto' to search for a pair that makes Lambda ~ Hermitian. "
            "Default: auto."
        ),
    )
    ap.add_argument("--schur_sign", choices=["-", "+"], default="-", help="Use Lambda = B_bb (+/-) B_bi B_ii^{-1} B_ib")

    ap.add_argument(
        "--sharp",
        choices=["transpose", "conj_transpose"],
        default="conj_transpose",
        help=(
            "Definition of A^sharp to use in the 2x2 symplectic partner: "
            "J A^T J^{-1} (transpose) vs J A^* J^{-1} (conj_transpose)."
        ),
    )

    ap.add_argument(
        "--X_mode",
        choices=["raw", "det1"],
        default="det1",
        help=(
            "Definition of X_k: 'raw' uses diag(p^k,1); 'det1' uses diag(p^{k/2}, p^{-k/2}). "
            "Default: det1 (often required for SL(2)/symplectic normalization)."
        ),
    )

    ap.add_argument(
        "--X_gamma",
        type=float,
        default=1.0,
        help=(
            "Exponent scaling for X_p. For det1: X=diag(p^{(gamma*k)/2}, p^{-(gamma*k)/2}). "
            "For raw: X=diag(p^{gamma*k},1). Default: 1.0."
        ),
    )

    ap.add_argument(
        "--p_mode",
        choices=["p", "logp", "p1_over_p", "p_over_p1", "invp", "p_minus1_over_p"],
        default="p",
        help=(
            "How to inject p into the local diagonal factor D. "
            "This changes ONLY the effective prime parameter used in the local block, and is useful for testing which "
            "prime-length clock (e.g. log p vs log(1+1/p)) the geometry prefers."
        ),
    )

    ap.add_argument(
        "--X_shear",
        type=float,
        default=0.0,
        help=(
            "Upper unipotent shear u used in X = U(u)*diag(x11,x22)*L(v), with U(u)=[[1,u],[0,1]]. "
            "Keeps det(X)=x11*x22. Default 0.0."
        ),
    )

    ap.add_argument(
        "--X_lower",
        type=float,
        default=0.0,
        help=(
            "Lower unipotent shear v used in X = U(u)*diag(x11,x22)*L(v), with L(v)=[[1,0],[v,1]]. "
            "Keeps det(X)=x11*x22. Default 0.0."
        ),
    )

    ap.add_argument(
        "--scattering",
        choices=["lambda_pm_i", "i_pm_lambda"],
        default="i_pm_lambda",
        help=(
            "Cayley convention mapping Lambda->S: "
            "lambda_pm_i gives S=(Λ-iI)(Λ+iI)^{-1} (Λ=0 -> -I); "
            "i_pm_lambda gives S=(I-iΛ)(I+iΛ)^{-1} (Λ=0 -> I)."
        ),
    )

    args = ap.parse_args()

    primes = _parse_int_list(args.primes)
    if not primes:
        raise SystemExit("--primes must be non-empty")
    k_max = int(args.k_max)
    if k_max < 1:
        raise SystemExit("--k_max must be >=1")

    # Choose boundary convention.
    boundary_arg = str(args.boundary).strip().lower()
    if boundary_arg == "auto":
        if str(args.local_model).strip().lower() == "sixby6":
            # Representative seeds to determine a stable convention.
            p_seed = [int(primes[0]), int(primes[len(primes) // 2]), int(primes[-1])] if len(primes) >= 3 else [int(primes[0])]
            k_seed = [1, max(1, k_max // 2), int(k_max)]
            pair, sgn, hd0, ud0 = _boundary_search(
                p_seed,
                k_seed,
                sharp_mode=str(args.sharp),
                x_mode=str(args.X_mode),
                x_gamma=float(args.X_gamma),
                x_shear=float(args.X_shear),
                x_lower=float(args.X_lower),
                p_mode=str(args.p_mode),
                scattering_mode=str(args.scattering),
            )
            boundary = pair
            schur_sign = sgn
            print(
                f"auto boundary selected: boundary={boundary} schur_sign={schur_sign} "
                f"seed_max_herm_def={hd0:.3g} seed_max_unit_def={ud0:.3g}"
            )
        else:
            # Satake injection doesn't need boundary conventions.
            boundary = (0, 1)
            schur_sign = "-"
    else:
        parts = _parse_int_list(str(args.boundary))
        if len(parts) != 2:
            raise SystemExit("--boundary must be 'auto' or 'i,j' with two indices")
        boundary = (int(parts[0]), int(parts[1]))
        schur_sign = str(args.schur_sign)

    satake_table = None
    if str(args.local_model).strip().lower() == "satake" and str(args.satake_family).strip().lower() == "table":
        if not str(args.satake_table_csv).strip():
            raise SystemExit("--satake_table_csv is required when --satake_family table")
        satake_table = _load_satake_table_csv(str(args.satake_table_csv))

    packets = _build_packets(
        primes,
        k_max,
        local_model=str(args.local_model),
        boundary=boundary,
        sign=schur_sign,
        sharp_mode=str(args.sharp),
        x_mode=str(args.X_mode),
        x_gamma=float(args.X_gamma),
        x_shear=float(args.X_shear),
        x_lower=float(args.X_lower),
        p_mode=str(args.p_mode),
        scattering_mode=str(args.scattering),
        satake_family=str(args.satake_family),
        satake_matrix=str(args.satake_matrix),
        theta_scale=float(args.theta_scale),
        seed=int(args.seed),
        satake_table=satake_table,
    )
    if not packets:
        raise SystemExit("no packets built")

    # Local diagnostics summary (max defects across packets)
    lam_hd = max(_hermitian_defect(pkt.Lam) for pkt in packets)
    s_ud = max(_unitarity_defect(pkt.S) for pkt in packets)

    sigma = float(args.sigma)
    t_min = float(args.t_min)
    t_max = float(args.t_max)
    n_t = int(args.n_t)
    if n_t < 5:
        raise SystemExit("--n_t must be >=5")

    t_grid = np.linspace(t_min, t_max, n_t, dtype=float)

    rows: list[dict] = []
    for t in t_grid.tolist():
        s = complex(sigma, float(t))
        K, Kp = _global_K(s, packets)

        I = np.eye(K.shape[0], dtype=np.complex128)
        M = (I - K).astype(np.complex128)

        try:
            det_val = complex(np.linalg.det(M))
        except Exception:
            det_val = complex("nan")

        try:
            sign, logabs = np.linalg.slogdet(M)
            logdet_val = complex(np.log(sign) + logabs)
        except Exception:
            logdet_val = complex("nan")

        # Phi(s) = -d/ds log det(I-K) = Tr((I-K)^{-1} K')
        try:
            Minv = np.linalg.inv(M)
            phi_val = complex(np.trace(Minv @ Kp))
        except Exception:
            phi_val = complex("nan")

        # Trace-power logdet check: logdet(I-K) ?= -sum_{m<=M} Tr(K^m)/m
        m_max = int(args.m_trace)
        series_err = float("nan")
        if m_max >= 1 and np.isfinite(np.real(logdet_val)) and np.isfinite(np.imag(logdet_val)):
            try:
                Km = np.eye(K.shape[0], dtype=np.complex128)
                acc = 0.0 + 0.0j
                for m in range(1, m_max + 1):
                    Km = (Km @ K).astype(np.complex128)
                    acc += complex(np.trace(Km)) / float(m)
                approx = -complex(acc)
                series_err = float(abs(approx - logdet_val) / (abs(logdet_val) + 1e-300))
            except Exception:
                series_err = float("nan")

        rows.append(
            {
                "sigma": float(sigma),
                "t": float(t),
                "D_re": float(np.real(det_val)) if np.isfinite(np.real(det_val)) else float("nan"),
                "D_im": float(np.imag(det_val)) if np.isfinite(np.imag(det_val)) else float("nan"),
                "logD_re": float(np.real(logdet_val)) if np.isfinite(np.real(logdet_val)) else float("nan"),
                "logD_im": float(np.imag(logdet_val)) if np.isfinite(np.imag(logdet_val)) else float("nan"),
                "Phi_re": float(np.real(phi_val)) if np.isfinite(np.real(phi_val)) else float("nan"),
                "Phi_im": float(np.imag(phi_val)) if np.isfinite(np.imag(phi_val)) else float("nan"),
                "trace_logdet_relerr_m": float(series_err),
                "packets": int(len(packets)),
                "lam_hermitian_defect_max": float(lam_hd),
                "S_unitarity_defect_max": float(s_ud),
            }
        )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)

    print(f"wrote {out_path}")
    print(f"packets={len(packets)}  max hermitian_defect(Lambda)={lam_hd:.3g}  max unitarity_defect(S)={s_ud:.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
