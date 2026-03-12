"""Dyadic seed wrapper for six_by_six_prime_tower_sim.

This is intentionally minimal infrastructure: generate the dyadic seed tower n=2^k
and forward execution into tools/six_by_six_prime_tower_sim.py using its new
--ns mode.

It does NOT attempt any dyadic→prime propagation, Möbius primitive extraction,
or Bost–Connes/Hecke arithmetic generation.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _pow2_ns(k_min: int, k_max: int) -> list[int]:
    if k_min < 1:
        raise ValueError("k_min must be >= 1")
    if k_max < k_min:
        raise ValueError("k_max must be >= k_min")
    ns: list[int] = []
    for k in range(int(k_min), int(k_max) + 1):
        ns.append(int(2**k))
    return ns


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Generate a dyadic seed tower n=2^k and run six_by_six_prime_tower_sim.py using --ns. "
            "Any additional args after '--' are forwarded to the underlying simulator."
        )
    )

    ap.add_argument("--k_max", type=int, default=6, help="Max exponent k (generates n=2^1..2^k_max)")
    ap.add_argument("--k_min", type=int, default=1, help="Min exponent k (default 1)")

    ap.add_argument(
        "--out_csv",
        required=True,
        help=(
            "Output CSV path for the underlying simulator. "
            "(This wrapper does not write its own output format.)"
        ),
    )

    ap.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        help="Use '-- <args>' to forward args to six_by_six_prime_tower_sim.py",
    )

    args = ap.parse_args(argv)

    ns = _pow2_ns(int(args.k_min), int(args.k_max))
    ns_arg = ",".join(str(n) for n in ns)

    repo_root = Path(__file__).resolve().parents[1]
    sim_path = repo_root / "tools" / "six_by_six_prime_tower_sim.py"

    if not sim_path.exists():
        raise FileNotFoundError(f"missing simulator at {sim_path}")

    out_csv = Path(str(args.out_csv))
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    passthrough = list(args.passthrough)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    cmd = [
        sys.executable,
        str(sim_path),
        "--ns",
        ns_arg,
        "--out_csv",
        str(out_csv),
        *passthrough,
    ]

    print(f"dyadic ns: {ns_arg}")
    print("exec:", " ".join(cmd))

    proc = subprocess.run(cmd)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
