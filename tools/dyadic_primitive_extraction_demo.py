"""Dyadic primitive/imprimitive separation demo.

This tool is deliberately arithmetic-only: it demonstrates the divisor/period
structure extracted by Möbius inversion for a dyadic seed tower.

Given an aggregate sequence F(k) (k=1..k_max), define primitive contributions
P(k) by

  P(k) = sum_{d|k} μ(d) F(k/d)

This is the standard primitive-period extractor used in combinatorics and
dynamical zeta contexts.

It does NOT claim to produce primes.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _build_F(k_max: int, mode: str) -> list[float]:
    k_max = int(k_max)
    if k_max < 1:
        raise ValueError("k_max must be >= 1")

    mode = str(mode).strip().lower()
    F = [0.0] * (k_max + 1)
    for k in range(1, k_max + 1):
        if mode == "2^k":
            F[k] = float(2**k)
        elif mode == "2^k-1":
            F[k] = float((2**k) - 1)
        elif mode == "tr_Kk":
            # For K_k = diag(2^k, 1)
            F[k] = float((2**k) + 1)
        else:
            raise ValueError("mode must be one of: 2^k, 2^k-1, tr_Kk")
    return F


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Compute primitive sequence P(k) from a dyadic aggregate F(k) via Möbius inversion. "
            "This is a primitive/imprimitive (period) separation demo, not a prime generator."
        )
    )
    ap.add_argument("--k_max", type=int, default=16)
    ap.add_argument("--mode", choices=["2^k", "2^k-1", "tr_Kk"], default="2^k-1")
    ap.add_argument("--out_csv", default="", help="Optional output CSV (columns: k,F_k,P_k)")

    args = ap.parse_args(argv)
    _ensure_repo_root_on_path()

    from src.arithmetic.mobius import mobius_invert_divisor_sum_linear  # noqa: E402

    k_max = int(args.k_max)
    F = _build_F(k_max, str(args.mode))
    P = mobius_invert_divisor_sum_linear(F)

    # Print a small table.
    print(f"mode={args.mode}  k_max={k_max}")
    print("k  F(k)  P(k)")
    for k in range(1, k_max + 1):
        print(f"{k:2d}  {F[k]:.0f}  {P[k]:.0f}")

    out_csv = str(args.out_csv).strip()
    if out_csv:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["k", "F_k", "P_k"])
            w.writeheader()
            for k in range(1, k_max + 1):
                w.writerow({"k": int(k), "F_k": float(F[k]), "P_k": float(P[k])})
        print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
