from __future__ import annotations

"""Möbius inversion utilities.

This is *number-theoretic* Möbius inversion (divisor poset), not “Möbius” as in
linear-fractional (Cayley/Möbius) transforms.

If an "aggregate tower" observable satisfies a divisor-sum identity

  F(n) = sum_{d|n} f(d)

then Möbius inversion recovers primitive contributions:

  f(n) = sum_{d|n} mu(d) * F(n/d)

These functions are intentionally small-n friendly (trial division), which matches
our typical experiment sizes.
"""

from dataclasses import dataclass
from typing import TypeVar

import numpy as np


def _is_squarefree_and_prime_factors(n: int) -> tuple[bool, int]:
    """Return (squarefree, omega) where omega=#distinct prime factors."""

    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")

    omega = 0
    p = 2
    while p * p <= n:
        if n % p != 0:
            p = 3 if p == 2 else p + 2
            continue
        omega += 1
        n //= p
        if n % p == 0:
            return False, omega
        while n % p == 0:
            # (should be unreachable due to early return)
            n //= p
        p = 3 if p == 2 else p + 2

    if n > 1:
        omega += 1
    return True, omega


def mobius_mu(n: int) -> int:
    """Return the classical Möbius function μ(n)."""

    n = int(n)
    if n == 1:
        return 1
    if n <= 0:
        raise ValueError("n must be positive")

    squarefree, omega = _is_squarefree_and_prime_factors(n)
    if not squarefree:
        return 0
    return -1 if (omega % 2 == 1) else 1


def divisors(n: int) -> list[int]:
    """List all positive divisors of n."""

    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")

    small: list[int] = []
    large: list[int] = []
    d = 1
    while d * d <= n:
        if n % d == 0:
            small.append(d)
            q = n // d
            if q != d:
                large.append(q)
        d += 1

    return small + large[::-1]


@dataclass(frozen=True)
class MobiusInversionResult:
    f: list[float]
    F_reconstructed: list[float]
    max_abs_reconstruction_error: float


def mobius_invert_divisor_sum(F: list[float]) -> MobiusInversionResult:
    """Given F[1..N] with F(n)=sum_{d|n} f(d), recover f[1..N].

    Input:
      F: list of length N+1, where F[0] is ignored and F[n] is the aggregate.

    Returns:
      MobiusInversionResult with arrays in the same indexing convention.
    """

    if not isinstance(F, list):
        raise TypeError("F must be a Python list (indexing is part of the API)")
    if len(F) < 2:
        raise ValueError("F must have length >= 2 and be indexed from 0..N")

    N = len(F) - 1
    f = [0.0] * (N + 1)

    # f(n) = sum_{d|n} mu(d) F(n/d)
    for n in range(1, N + 1):
        acc = 0.0
        for d in divisors(n):
            acc += float(mobius_mu(d)) * float(F[n // d])
        f[n] = float(acc)

    # reconstruct
    Frec = [0.0] * (N + 1)
    for n in range(1, N + 1):
        acc = 0.0
        for d in divisors(n):
            acc += float(f[d])
        Frec[n] = float(acc)

    errs = [abs(float(Frec[n]) - float(F[n])) for n in range(1, N + 1)]
    max_err = max(errs) if errs else 0.0

    return MobiusInversionResult(f=f, F_reconstructed=Frec, max_abs_reconstruction_error=float(max_err))


T = TypeVar("T")


def _zero_like(x: T) -> T:
    if isinstance(x, np.ndarray):
        return np.zeros_like(x)
    # Works for int/float/complex/numpy scalar types.
    return type(x)(0)  # type: ignore[call-arg,return-value]


def mobius_invert_divisor_sum_linear(F: list[T]) -> list[T]:
    """Möbius inversion for divisor-sum aggregates, for any additive linear type.

    If F(n) = sum_{d|n} f(d), then f(n) = sum_{d|n} μ(d) F(n/d).

    Unlike `mobius_invert_divisor_sum`, this does not coerce to float; it can be
    used with complex scalars or numpy arrays (e.g., 2x2 matrices).

    Input:
      F: list of length N+1 indexed from 0..N (F[0] ignored).

    Returns:
      f: list of length N+1 with the same indexing.
    """

    if not isinstance(F, list):
        raise TypeError("F must be a Python list (indexing is part of the API)")
    if len(F) < 2:
        raise ValueError("F must have length >= 2 and be indexed from 0..N")

    N = len(F) - 1
    z = _zero_like(F[1])
    f: list[T] = [z for _ in range(N + 1)]

    for n in range(1, N + 1):
        acc = _zero_like(F[1])
        for d in divisors(n):
            acc = acc + (int(mobius_mu(d)) * F[n // d])  # type: ignore[operator]
        f[n] = acc
    return f
