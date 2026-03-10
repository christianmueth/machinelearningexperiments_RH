from __future__ import annotations

import numpy as np


def dn_map_destructive(A: np.ndarray, *, b: int, jitter: float = 0.0) -> np.ndarray:
    """Low-copy Dirichlet–Neumann reduction that may overwrite A in-place.

    This is equivalent to dn_map, but it is allowed to overwrite the interior
    block A[b:, b:] (and the RHS view A[b:, :b]) during the linear solve when
    SciPy is available. This can materially reduce peak memory for large N.

    If SciPy isn't available (or the fast path fails), falls back to dn_map.
    """

    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    N = int(A.shape[0])
    b = int(b)
    jitter = float(jitter)
    if b <= 0 or b >= N:
        raise ValueError("b must satisfy 0 < b < N")

    # Ensure complex dtype. We don't force complex128 here to avoid an extra copy.
    if not np.iscomplexobj(A):
        A = np.asarray(A, dtype=np.complex128)

    # Ensure writeable (we may overwrite views into A).
    if not A.flags.writeable:
        A = np.array(A, copy=True)

    try:
        import scipy.linalg  # type: ignore

        Abb = A[:b, :b]
        Abi = A[:b, b:]
        Aib = A[b:, :b]
        Aii = A[b:, b:]

        if jitter != 0.0:
            # In-place diagonal jitter to avoid allocating an eye().
            m = int(N - b)
            Aii.flat[:: m + 1] += complex(jitter)

        # Factorize in-place, solve with RHS view (overwriting RHS in-place).
        lu, piv = scipy.linalg.lu_factor(Aii, overwrite_a=True, check_finite=False)
        X = scipy.linalg.lu_solve((lu, piv), Aib, overwrite_b=True, check_finite=False)
        Lam_b = Abb - Abi @ X
        return np.asarray(Lam_b)
    except Exception:
        return dn_map(A, b=b, jitter=jitter)


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


def dn_map_two_channel_boundary(
    U: np.ndarray,
    W: np.ndarray,
    *,
    b0: int,
    jitter: float = 0.0,
) -> np.ndarray:
    """Two-channel structure-aware Dirichlet–Neumann map on the boundary.

    This computes the same Schur complement as dn_map_on_indices() would for the
    special two-channel bulk matrix

        A2 = [[I, U],
              [W, I]]

    with boundary indices [0..b0-1] in each channel.

    Crucially, it avoids constructing the full 2n×2n matrix A2 and avoids copying
    a huge 2m×2m interior block (m = n-b0) via fancy indexing.

    The interior jitter convention matches dn_map_on_indices(): it adds jitter*I
    to the full 2m×2m interior block, i.e. each diagonal block becomes
    d*I_m with d = 1 + jitter.

    Returns Lambda_b in boundary ordering [ch1 boundary, ch2 boundary], shape (2*b0, 2*b0).
    """

    U = np.asarray(U, dtype=np.complex128)
    W = np.asarray(W, dtype=np.complex128)
    if U.ndim != 2 or W.ndim != 2 or U.shape[0] != U.shape[1] or W.shape[0] != W.shape[1]:
        raise ValueError("U and W must be square 2D arrays")
    if U.shape != W.shape:
        raise ValueError("U and W must have the same shape")

    n = int(U.shape[0])
    b0 = int(b0)
    if b0 <= 0 or b0 >= n:
        raise ValueError("b0 must satisfy 0 < b0 < n")
    m = int(n - b0)
    if m <= 0:
        raise ValueError("interior dimension must be positive")

    jitter = float(jitter)
    d = complex(1.0 + jitter)
    if d == 0:
        raise ValueError("invalid jitter: 1+jitter must be nonzero")

    # Partition U,W into boundary/interior blocks (per channel).
    U_bb = U[:b0, :b0]
    U_bi = U[:b0, b0:]
    U_ib = U[b0:, :b0]
    U_ii = U[b0:, b0:]

    W_bb = W[:b0, :b0]
    W_bi = W[:b0, b0:]
    W_ib = W[b0:, :b0]
    W_ii = W[b0:, b0:]

    invd = complex(1.0) / d

    # Build S1 = d I - (1/d) U_ii W_ii and S2 = d I - (1/d) W_ii U_ii.
    # Do this with minimal temporaries.
    S1 = (-(invd) * (U_ii @ W_ii)).astype(np.complex128)
    S2 = (-(invd) * (W_ii @ U_ii)).astype(np.complex128)
    # Add d on the diagonal in-place.
    S1.flat[:: m + 1] += d
    S2.flat[:: m + 1] += d

    def _solve(M: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Solve M X = B with a SciPy LU fast path when available."""
        try:
            import scipy.linalg  # type: ignore

            lu, piv = scipy.linalg.lu_factor(M, overwrite_a=False, check_finite=False)
            X = scipy.linalg.lu_solve((lu, piv), B, overwrite_b=False, check_finite=False)
            return np.asarray(X, dtype=np.complex128)
        except Exception:
            return np.asarray(np.linalg.solve(M, B), dtype=np.complex128)

    # X = S1^{-1} U_ib, Y = S2^{-1} W_ib.
    X = _solve(S1, U_ib)
    Y = _solve(S2, W_ib)

    # Assemble Lambda blocks.
    I_b = np.eye(b0, dtype=np.complex128)

    Lam11 = (I_b - (U_bi @ Y)).astype(np.complex128)
    Lam22 = (I_b - (W_bi @ X)).astype(np.complex128)

    # Off-diagonals: reuse intermediates.
    WX = (W_ii @ X).astype(np.complex128)  # m×b
    UY = (U_ii @ Y).astype(np.complex128)  # m×b
    Lam12 = (U_bb + (invd * (U_bi @ WX))).astype(np.complex128)
    Lam21 = (W_bb + (invd * (W_bi @ UY))).astype(np.complex128)

    top = np.concatenate([Lam11, Lam12], axis=1)
    bot = np.concatenate([Lam21, Lam22], axis=1)
    Lam_b = np.concatenate([top, bot], axis=0)
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
