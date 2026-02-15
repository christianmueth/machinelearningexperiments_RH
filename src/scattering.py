from __future__ import annotations

import numpy as np

from .dn import dn_map


def cayley_from_lambda(Lam: np.ndarray, eta: float, eps: float = 0.0) -> np.ndarray:
    Lam = np.asarray(Lam, dtype=np.complex128)
    eta = float(eta)
    eps = float(eps)
    n = int(Lam.shape[0])
    I = np.eye(n, dtype=np.complex128)

    A = Lam + 1j * eta * I
    if eps != 0.0:
        A = A + eps * I
    B = Lam - 1j * eta * I
    invA = np.linalg.solve(A, I)
    return B @ invA


def cayley_skew_from_lambda(Lam: np.ndarray, eta: float) -> np.ndarray:
    """Cayley transform suited to skew-Hermitian Lambda.

    If Lam is skew-Hermitian (Lam* = -Lam) and eta is real, then
    U = (I - Lam/eta) (I + Lam/eta)^{-1} is unitary.
    """

    Lam = np.asarray(Lam, dtype=np.complex128)
    eta = float(eta)
    if eta == 0.0:
        raise ValueError("eta must be nonzero for skew Cayley")

    n = int(Lam.shape[0])
    I = np.eye(n, dtype=np.complex128)
    A = I + (Lam / eta)
    B = I - (Lam / eta)
    invA = np.linalg.solve(A, I)
    return B @ invA


def det_phase(M: np.ndarray) -> float:
    """Return arg(det(M)) computed via slogdet.

    For unitary-ish matrices, this is typically the observable of interest.
    """

    M = np.asarray(M, dtype=np.complex128)
    sign, _logabs = np.linalg.slogdet(M)
    return float(np.angle(sign))


def relative_phi_from_K(K: np.ndarray) -> complex:
    """Return det(I+K) as the basic Fredholm proxy."""

    K = np.asarray(K, dtype=np.complex128)
    n = int(K.shape[0])
    I = np.eye(n, dtype=np.complex128)
    return complex(np.linalg.det(I + K))


def logdet_I_plus_K(K: np.ndarray) -> complex:
    K = np.asarray(K, dtype=np.complex128)
    n = int(K.shape[0])
    I = np.eye(n, dtype=np.complex128)
    sign, logabs = np.linalg.slogdet(I + K)
    # sign is complex unit for complex matrices (numpy returns complex)
    return complex(np.log(sign) + logabs)


def build_lambda_and_S_from_A(
    A: np.ndarray, *, b: int, eta: float, schur_jitter: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    Lam = dn_map(A, b=b, jitter=float(schur_jitter))
    S = cayley_from_lambda(Lam, eta=eta)
    return Lam, S
