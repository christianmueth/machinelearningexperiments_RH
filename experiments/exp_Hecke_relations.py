from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.hecke import HeckeParams, hecke_Tn, prime_power_recursion_Tpows
from src.logging_utils import make_run_dir, save_run_snapshot
from src.metrics import fro_norm


def _divisors(n: int) -> list[int]:
    n = int(n)
    if n <= 0:
        return []
    out: list[int] = []
    r = int(np.sqrt(n))
    for d in range(1, r + 1):
        if n % d == 0:
            out.append(d)
            if d * d != n:
                out.append(n // d)
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Experiment: Hecke algebra finite-section sanity checks.")
    ap.add_argument("--config", type=str, default="configs/exp_Hecke_relations.yaml")
    ap.add_argument("--runs_root", type=str, default="runs")
    args = ap.parse_args()

    import src.configs as cfg

    config = cfg.load_config(args.config)

    Ns = list(map(int, config.get("Ns", [32, 48, 64, 96, 128, 256])))
    weight_k = int(config.get("weight_k", 0))

    pairs = config.get("pairs", [[2, 3], [2, 4], [3, 6], [4, 6], [6, 10]])
    pairs = [(int(a), int(b)) for a, b in pairs]

    pp_tests = config.get("prime_power_tests", [[2, 6], [3, 5], [5, 4]])
    pp_tests = [(int(p), int(rmax)) for p, rmax in pp_tests]

    out_dir = make_run_dir(args.runs_root, tag="expHecke_relations")
    save_run_snapshot(out_dir, config=config, workspace_root=str(Path(__file__).resolve().parents[1]))

    rows_rel: list[dict] = []
    rows_pp: list[dict] = []

    for N in Ns:
        params = HeckeParams(N=int(N), weight_k=int(weight_k))

        # Hecke relation: T_m T_n = sum_{d | gcd(m,n)} d^{k-1} T_{mn/d^2}
        for m, n in pairs:
            Tm = np.asarray(hecke_Tn(m, params), dtype=np.float64)
            Tn = np.asarray(hecke_Tn(n, params), dtype=np.float64)
            left = Tm @ Tn

            g = int(np.gcd(m, n))
            right = np.zeros_like(left)
            for d in _divisors(g):
                coeff = float(d ** (weight_k - 1)) if weight_k != 1 else 1.0
                idx = (m * n) // (d * d)
                right = right + coeff * np.asarray(hecke_Tn(idx, params), dtype=np.float64)

            resid = left - right
            denom = fro_norm(left)
            rel = float(fro_norm(resid) / denom) if denom > 0 else float("nan")

            comm = (Tm @ Tn) - (Tn @ Tm)
            comm_rel = float(fro_norm(comm) / (fro_norm(Tm @ Tn) + 1e-30))

            rows_rel.append(
                {
                    "N": int(N),
                    "weight_k": int(weight_k),
                    "m": int(m),
                    "n": int(n),
                    "gcd": int(g),
                    "rel_residual_fro": float(rel),
                    "commutator_rel_fro": float(comm_rel),
                }
            )

        # Prime-power recursion consistency
        for p, rmax in pp_tests:
            Tpows_rec = prime_power_recursion_Tpows(p, r_max=rmax, params=params)
            for r in range(0, int(rmax) + 1):
                T_direct = np.asarray(hecke_Tn(int(p) ** int(r), params), dtype=np.float64)
                T_rec = np.asarray(Tpows_rec[r], dtype=np.float64)
                resid = T_direct - T_rec
                denom = fro_norm(T_direct)
                rel = float(fro_norm(resid) / denom) if denom > 0 else float("nan")
                rows_pp.append(
                    {
                        "N": int(N),
                        "weight_k": int(weight_k),
                        "p": int(p),
                        "r": int(r),
                        "rel_residual_fro": float(rel),
                    }
                )

    pd.DataFrame(rows_rel).to_csv(os.path.join(out_dir, "Hecke_relations.csv"), index=False)
    pd.DataFrame(rows_pp).to_csv(os.path.join(out_dir, "Hecke_prime_power_recursion.csv"), index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
