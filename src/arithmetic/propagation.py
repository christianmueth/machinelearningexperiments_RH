from __future__ import annotations

"""Propagation scaffolding (Hecke / semigroup host).

This is *not* a Bost–Connes implementation.

What it does provide:
  - A small interface for “primitive data -> (p^k) family” propagation.
  - A concrete, existing propagation mechanism already in the repo: finite-section
    Hecke operators in `src/hecke.py`.

The missing mathematical step (dyadic seed -> prime-indexed primitives) is kept
explicitly out of scope here so we don’t accidentally smuggle in hand-coded prime
rules.
"""

from dataclasses import dataclass

import numpy as np

from ..hecke import HeckeParams, hecke_Tn, prime_power_recursion_Tpows


@dataclass(frozen=True)
class PrimePowerOperatorFamily:
    """A small container for a prime-power family {T_{p^r}}_{r=0..r_max}."""

    p: int
    r_max: int
    T_pows: list[np.ndarray]

    def invariants(self) -> dict[str, float]:
        """Cheap summary invariants for debugging/diagnostics."""

        tr = [float(np.trace(T).real) for T in self.T_pows]
        fro = [float(np.linalg.norm(T, ord="fro")) for T in self.T_pows]
        return {
            "p": float(self.p),
            "r_max": float(self.r_max),
            "trace_T_p": tr[1] if len(tr) > 1 else float("nan"),
            "fro_T_p": fro[1] if len(fro) > 1 else float("nan"),
        }


def hecke_prime_power_family(*, p: int, r_max: int, params: HeckeParams) -> PrimePowerOperatorFamily:
    """Build {T_{p^r}} using the internal Hecke recursion.

    This is the “multiplicative propagation law” that exists today in the repo.
    It is useful as a downstream host once primitives are identified.
    """

    p = int(p)
    r_max = int(r_max)
    T_pows = prime_power_recursion_Tpows(int(p), int(r_max), params)
    return PrimePowerOperatorFamily(p=int(p), r_max=int(r_max), T_pows=[np.asarray(T) for T in T_pows])


def hecke_T_of_n(*, n: int, params: HeckeParams) -> np.ndarray:
    """Direct truncated Hecke operator T_n."""

    return np.asarray(hecke_Tn(int(n), params), dtype=np.float64)
