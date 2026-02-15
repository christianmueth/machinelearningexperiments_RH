from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HeckeParams:
    """Parameters for a finite-section Hecke operator model.

    This implements the classical divisor-sum action on Fourier-coefficient basis
    for weight k, on indices 1..N.

    (T_n a)(m) = sum_{d | gcd(m,n)} d^{k-1} a(m n / d^2)

    Truncation: indices > N are dropped.
    """

    N: int
    weight_k: int = 0


def hecke_Tn(n: int, params: HeckeParams) -> np.ndarray:
    N = int(params.N)
    k = int(params.weight_k)

    if n <= 0:
        raise ValueError("n must be positive")

    T = np.zeros((N, N), dtype=np.float64)

    # basis vector e_j corresponds to coefficient a(j+1)
    # output index i corresponds to m=i+1
    for m in range(1, N + 1):
        g = np.gcd(m, n)
        # enumerate divisors of g
        # (g is small-ish for our usage; keep simple)
        for d in range(1, g + 1):
            if g % d != 0:
                continue
            j = (m * n) // (d * d)
            if 1 <= j <= N:
                w = float(d ** (k - 1)) if k != 1 else 1.0
                T[m - 1, j - 1] += w

    return T


def hecke_Tp(p: int, params: HeckeParams) -> np.ndarray:
    return hecke_Tn(int(p), params)


def prime_power_recursion_Tpows(p: int, r_max: int, params: HeckeParams) -> list[np.ndarray]:
    """Return [T_{p^0}, T_{p^1}, ..., T_{p^r_max}] using the standard recursion.

    Recursion (weight k):
      T_{p^{r+1}} = T_p T_{p^r} - p^{k-1} T_{p^{r-1}}

    This is the algebra relation; it provides a good internal consistency check
    for the truncation model.
    """

    N = int(params.N)
    k = int(params.weight_k)
    p = int(p)
    r_max = int(r_max)
    if r_max < 0:
        raise ValueError("r_max must be >= 0")

    I = np.eye(N, dtype=np.float64)
    Tp = hecke_Tp(p, params)

    out: list[np.ndarray] = [I]
    if r_max == 0:
        return out

    out.append(Tp)
    if r_max == 1:
        return out

    factor = float(p ** (k - 1)) if k != 1 else 1.0
    for r in range(1, r_max):
        T_prev = out[r]
        T_prev2 = out[r - 1]
        out.append(Tp @ T_prev - factor * T_prev2)

    return out
