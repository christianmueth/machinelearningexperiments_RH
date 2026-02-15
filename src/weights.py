from __future__ import annotations

import numpy as np


def cusp_weights_diag(n_indices: np.ndarray, beta: float) -> np.ndarray:
    """Return diagonal weights W_beta where (W_beta e_n) = n^beta e_n.

    n_indices should be 1-based indices (e.g. global interior indices).
    """

    n_indices = np.asarray(n_indices, dtype=np.float64)
    beta = float(beta)
    return np.power(n_indices, beta)


def apply_similarity_weight(K: np.ndarray, n_indices_1based: np.ndarray, beta: float) -> np.ndarray:
    """Compute W^{-1} K W for diagonal W_beta."""

    K = np.asarray(K, dtype=np.complex128)
    w = cusp_weights_diag(n_indices_1based, beta=beta)
    Winv = 1.0 / w

    # W^{-1} K W: scale rows by Winv and cols by w
    return (Winv[:, None] * K) * (w[None, :])
