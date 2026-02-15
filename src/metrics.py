from __future__ import annotations

import numpy as np


def fro_norm(M: np.ndarray) -> float:
    M = np.asarray(M, dtype=np.complex128)
    return float(np.linalg.norm(M, ord="fro"))


def hermitian_defect(M: np.ndarray) -> float:
    """Return ||M - M*||_F (Hermitian error proxy)."""

    M = np.asarray(M, dtype=np.complex128)
    return float(np.linalg.norm(M - M.conj().T, ord="fro"))


def hs_norm_sq(K: np.ndarray) -> float:
    K = np.asarray(K, dtype=np.complex128)
    return float(np.sum(np.abs(K) ** 2))


def top_singular_values(K: np.ndarray, r: int) -> np.ndarray:
    K = np.asarray(K, dtype=np.complex128)
    s = np.linalg.svd(K, compute_uv=False)
    r = int(r)
    if r <= 0:
        raise ValueError("r must be positive")
    return np.asarray(s[:r], dtype=np.float64)


def cond_number(M: np.ndarray) -> float:
    M = np.asarray(M, dtype=np.complex128)
    s = np.linalg.svd(M, compute_uv=False)
    if s[-1] == 0:
        return float("inf")
    return float(s[0] / s[-1])


def unitarity_defect(S: np.ndarray) -> float:
    S = np.asarray(S, dtype=np.complex128)
    n = int(S.shape[0])
    I = np.eye(n, dtype=np.complex128)
    return float(np.linalg.norm(S.conj().T @ S - I, ord="fro"))


def min_dist_to_minus1(S: np.ndarray) -> float:
    S = np.asarray(S, dtype=np.complex128)
    eig = np.linalg.eigvals(S)
    return float(np.min(np.abs(eig + 1)))
