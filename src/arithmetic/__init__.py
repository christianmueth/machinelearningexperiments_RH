"""Arithmetic derivation layer (scaffolding).

Goal (intended architecture):
  dyadic 2^k seed tower
    -> Möbius inversion / primitive extraction
    -> arithmetic semigroup propagation (Hecke / Bost–Connes host)
    -> prime / prime-power packet families

The current codebase has two largely separate tracks:
  - `tools/six_by_six_prime_tower_sim.py`: directly injects prime dependence via `p_mode`.
  - `src/hecke.py` + `src/bulk.py`: a finite-section Hecke operator model.

This package is the start of a refactor that makes the “seed → primitives → propagation”
layer explicit and testable, without changing existing experiment outputs.
"""

from .mobius import mobius_mu, mobius_invert_divisor_sum
