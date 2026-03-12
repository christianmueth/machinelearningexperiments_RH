"""Run FE discrimination sweeps across k_max and packet construction modes.

Designed to answer: does semigroup enforcement drive rankings toward stability
as k_max increases?

Outputs:
- Writes per-run CSVs under runs_fe/ (ignored by git).
- These CSVs can be parsed by tools/parse_fe_ranking_table.py.

Example:
  python tools/run_fe_kmax_transition_sweep.py --k_list 2,4,5

Defaults match the existing discrimination setup used in the current thread:
- fixed completion, basis poly2_gamma, sigma=0.3
- t in [10,30], n_t=201
- gauge boundary (0,5), schur_sign '+'
- u_list = [-0.2, +0.2]
- primes_global = 2..37
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_fe"
SCRIPT = ROOT / "tools" / "fe_defect_perturbation_u0.py"


DEFAULT_PRIMES = "2,3,5,7,11,13,17,19,23,29,31,37"
DEFAULT_PMODES = ["p", "invp", "p1_over_p", "logp"]
DEFAULT_COMPLETIONS = ["zeta", "gl2"]
DEFAULT_PPMODES = ["direct", "x_power", "bulk_power"]


def _out_name(*, completion: str, pp_mode: str, p_mode: str, k_max: int) -> str:
    if pp_mode == "direct":
        prefix = "disc_direct"
    elif pp_mode == "x_power":
        prefix = "disc_semigroup"
    elif pp_mode == "bulk_power":
        prefix = "disc_bulkpower"
    else:
        raise ValueError(f"unknown prime_power_mode: {pp_mode}")
    return f"{prefix}_{completion}_pmode_{p_mode}_b05_k{k_max}.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--k_list",
        default="2,4,5",
        help="Comma-separated k_max values to run (e.g. 2,3,4,5).",
    )
    ap.add_argument(
        "--p_modes",
        default=",".join(DEFAULT_PMODES),
        help="Comma-separated p_modes to evaluate.",
    )
    ap.add_argument(
        "--completions",
        default=",".join(DEFAULT_COMPLETIONS),
        help="Comma-separated fixed completions (zeta,gl2).",
    )
    ap.add_argument(
        "--prime_power_modes",
        default=",".join(DEFAULT_PPMODES),
        help="Comma-separated prime_power_mode values (direct,x_power).",
    )
    ap.add_argument(
        "--primes_global",
        default=DEFAULT_PRIMES,
        help="Comma-separated primes to include in the global determinant.",
    )
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--t_min", type=float, default=10.0)
    ap.add_argument("--t_max", type=float, default=30.0)
    ap.add_argument("--n_t", type=int, default=201)
    ap.add_argument("--boundary", default="0,5")
    ap.add_argument("--schur_sign", default="+")
    ap.add_argument("--u_list", default="-0.2,0.2")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if output CSV already exists.",
    )
    args = ap.parse_args()

    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
    p_modes = [x.strip() for x in args.p_modes.split(",") if x.strip()]
    completions = [x.strip() for x in args.completions.split(",") if x.strip()]
    pp_modes = [x.strip() for x in args.prime_power_modes.split(",") if x.strip()]

    RUNS.mkdir(exist_ok=True)

    total = len(k_list) * len(p_modes) * len(completions) * len(pp_modes)
    done = 0

    for completion in completions:
        for k_max in k_list:
            for pp_mode in pp_modes:
                for p_mode in p_modes:
                    out_csv = RUNS / _out_name(
                        completion=completion,
                        pp_mode=pp_mode,
                        p_mode=p_mode,
                        k_max=k_max,
                    )
                    if out_csv.exists() and not args.force:
                        done += 1
                        print(f"[{done:>3}/{total}] skip exists: {out_csv.name}")
                        continue

                    cmd = [
                        sys.executable,
                        str(SCRIPT),
                        "--primes_global",
                        args.primes_global,
                        "--k_max",
                        str(k_max),
                        "--sigma",
                        str(args.sigma),
                        "--t_min",
                        str(args.t_min),
                        "--t_max",
                        str(args.t_max),
                        "--n_t",
                        str(args.n_t),
                        "--completion_mode",
                        "fixed",
                        "--fixed_completion",
                        completion,
                        "--completion_basis",
                        "poly2_gamma",
                        "--prime_power_mode",
                        pp_mode,
                        "--boundary",
                        args.boundary,
                        "--schur_sign",
                        args.schur_sign,
                        f"--u_list={args.u_list}",
                        "--p_mode",
                        p_mode,
                        "--out_csv",
                        str(out_csv),
                    ]

                    done += 1
                    print(
                        f"[{done:>3}/{total}] run: completion={completion} k_max={k_max} "
                        f"prime_power_mode={pp_mode} p_mode={p_mode} -> {out_csv.name}"
                    )
                    subprocess.check_call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    main()
