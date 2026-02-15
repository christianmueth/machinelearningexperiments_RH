from __future__ import annotations

import numpy as np


def dn_map_on_indices(A: np.ndarray, *, boundary_idx: list[int] | np.ndarray, jitter: float = 0.0) -> np.ndarray:
    """Dirichlet–Neumann-style boundary reduction for an arbitrary boundary index set.

    boundary_idx defines the boundary degrees of freedom (in the desired order).
    The interior is the complementary index set.

    Returns Λ_b = A_bb - A_bi (A_ii + jitter I)^{-1} A_ib.

    This is useful when the natural boundary is non-contiguous in the matrix ordering
    (e.g. multi-channel constructions).
    """

    A = np.asarray(A, dtype=np.complex128)
    N = int(A.shape[0])
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")

    bidx = np.asarray(boundary_idx, dtype=int).reshape(-1)
    if bidx.size == 0:
        raise ValueError("boundary_idx must be non-empty")

    if np.any(bidx < 0) or np.any(bidx >= N):
        raise ValueError("boundary_idx entries must be in [0, N)")
    if np.unique(bidx).size != bidx.size:
        raise ValueError("boundary_idx must not contain duplicates")

    mask = np.ones(N, dtype=bool)
    mask[bidx] = False
    iidx = np.nonzero(mask)[0]
    if iidx.size == 0:
        raise ValueError("boundary_idx cannot include all indices")

    Abb = A[np.ix_(bidx, bidx)]
    Abi = A[np.ix_(bidx, iidx)]
    Aib = A[np.ix_(iidx, bidx)]
    Aii = A[np.ix_(iidx, iidx)]

    jitter = float(jitter)
    if jitter != 0.0:
        Aii = Aii + (jitter * np.eye(iidx.size, dtype=np.complex128))

    X = np.linalg.solve(Aii, Aib)
    Lam_b = Abb - Abi @ X
    return np.asarray(Lam_b, dtype=np.complex128)


def dn_map(A: np.ndarray, *, b: int, jitter: float = 0.0) -> np.ndarray:
    """Dirichlet–Neumann-style boundary reduction via Schur complement.

    Indices 0..b-1 are treated as boundary, b..N-1 as interior.

    Returns Λ_b = A_bb - A_bi (A_ii + jitter I)^{-1} A_ib.

    jitter is a diagonal regularization applied only to the interior block.
    """

    A = np.asarray(A, dtype=np.complex128)
    N = int(A.shape[0])
    b = int(b)
    jitter = float(jitter)
    if b <= 0 or b >= N:
        raise ValueError("b must satisfy 0 < b < N")

    Abb = A[:b, :b]
    Abi = A[:b, b:]
    Aib = A[b:, :b]
    Aii = A[b:, b:]

    if jitter != 0.0:
        Aii = Aii + (jitter * np.eye(N - b, dtype=np.complex128))

    X = np.linalg.solve(Aii, Aib)  # Aii^{-1} Aib
    Lam_b = Abb - Abi @ X
    return np.asarray(Lam_b, dtype=np.complex128)


def schur_complement(A: np.ndarray, *, b: int, jitter: float = 0.0) -> np.ndarray:
    """Schur complement of a block-partitioned matrix A.

    Indices 0..b-1 are treated as boundary, b..N-1 as interior.

    Returns Λ = A_ii - A_ib (A_bb + jitter I)^{-1} A_bi.

    jitter is a diagonal regularization applied only to the boundary block.
    """

    A = np.asarray(A, dtype=np.complex128)
    N = int(A.shape[0])
    b = int(b)
    jitter = float(jitter)
    if b <= 0 or b >= N:
        raise ValueError("b must satisfy 0 < b < N")

    Abb = A[:b, :b]
    Abi = A[:b, b:]
    Aib = A[b:, :b]
    Aii = A[b:, b:]

    if jitter != 0.0:
        Abb = Abb + (jitter * np.eye(b, dtype=np.complex128))

    X = np.linalg.solve(Abb, Abi)  # Abb^{-1} Abi
    Lam = Aii - Aib @ X
    return np.asarray(Lam, dtype=np.complex128)
