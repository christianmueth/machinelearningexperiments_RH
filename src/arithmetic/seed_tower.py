from __future__ import annotations

"""Dyadic seed-tower scaffolding.

This module is intentionally conservative: it does NOT try to guess the full
mathematical mechanism that turns a dyadic ladder into primes.

Instead it defines the *shape* of the objects we want in the refactor:
  - a seed tower indexed by k (the dyadic ladder 2^k)
  - an aggregate observable on the divisor poset (suitable for Möbius inversion)
  - a place to attach “primitive extraction” outputs

The intent is to feed this layer into a future Hecke/BC propagation engine, and
only then into Schur/Cayley reductions.
"""

import math
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class DyadicSeedTower:
    """A seed ladder indexed by k with intrinsic additive length ℓ_k = k log 2."""

    k_max: int

    def lengths(self) -> list[float]:
        k_max = int(self.k_max)
        if k_max < 0:
            raise ValueError("k_max must be >= 0")
        log2 = float(math.log(2.0))
        return [0.0] + [float(k) * log2 for k in range(1, k_max + 1)]

    def values_on_n(self, *, N: int, value_at_power_of_two: Callable[[int], float]) -> list[float]:
        """Return f_seed[0..N] supported only on powers of two.

        f_seed(n) = value_at_power_of_two(k) if n=2^k else 0.

        This is a convenient representation for downstream divisor-sum identities.
        """

        N = int(N)
        if N < 1:
            raise ValueError("N must be >= 1")

        out = [0.0] * (N + 1)
        k = 0
        n = 1
        while n <= N:
            if k > 0:
                out[n] = float(value_at_power_of_two(int(k)))
            k += 1
            n *= 2
        return out


def divisor_sum_aggregate_from_primitives(f: list[float]) -> list[float]:
    """Given primitives f[1..N], return F(n)=sum_{d|n} f(d) as F[1..N]."""

    if not isinstance(f, list):
        raise TypeError("f must be a list indexed from 0..N")
    if len(f) < 2:
        raise ValueError("f must have length >= 2")

    N = len(f) - 1
    F = [0.0] * (N + 1)
    for d in range(1, N + 1):
        fd = float(f[d])
        if fd == 0.0:
            continue
        for m in range(d, N + 1, d):
            F[m] += fd
    return F
