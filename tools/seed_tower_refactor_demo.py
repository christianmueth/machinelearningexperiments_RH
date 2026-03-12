from __future__ import annotations

"""Demo: dyadic seed -> divisor-sum aggregate -> Möbius primitive recovery.

This is a *refactor exploration* tool.

It demonstrates the missing architectural seam you described:
  - a true seed ladder (2^k)
  - an aggregate tower observable mixing imprimitive contributions
  - Möbius inversion to recover primitives

It does NOT yet implement the “seed -> primes” conversion (BC/Hecke host).
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `import src.*` works when executed as
# `python tools/<script>.py`.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.arithmetic.mobius import mobius_invert_divisor_sum
from src.arithmetic.seed_tower import DyadicSeedTower, divisor_sum_aggregate_from_primitives


def main() -> int:
    ap = argparse.ArgumentParser(description="Demo Möbius inversion on a dyadic seed tower.")
    ap.add_argument("--N", type=int, default=128, help="Compute arithmetic functions on n=1..N")
    ap.add_argument("--k_max", type=int, default=10, help="Max dyadic exponent included (2^k)")
    ap.add_argument(
        "--seed_value",
        type=float,
        default=1.0,
        help="Value assigned to each dyadic primitive (for n=2^k, k>=1).",
    )
    args = ap.parse_args()

    N = int(args.N)
    k_max = int(args.k_max)
    seed_value = float(args.seed_value)

    tower = DyadicSeedTower(k_max=k_max)

    # Define dyadic primitives f(n): supported only on n=2^k.
    f = tower.values_on_n(N=N, value_at_power_of_two=lambda k: seed_value)

    # Aggregate observable mixing imprimitive divisor structure: F(n)=sum_{d|n} f(d)
    F = divisor_sum_aggregate_from_primitives(f)

    inv = mobius_invert_divisor_sum(F)

    print("Dyadic seed Möbius inversion demo")
    print(f"N={N}  k_max={k_max}  seed_value={seed_value}")
    print(f"max_abs_reconstruction_error={inv.max_abs_reconstruction_error:.3g}")
    print()

    # Show a few values to make the primitive/imprimitive separation visible.
    print(" n | F(n)=aggregate | f(n)=true primitive | f_hat(n)=mobius-inverted")
    print("---|--------------|-------------------|-----------------------")
    for n in range(1, min(N, 64) + 1):
        if n in {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64}:
            print(f"{n:2d} | {F[n]:12.6g} | {f[n]:17.6g} | {inv.f[n]:21.6g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
