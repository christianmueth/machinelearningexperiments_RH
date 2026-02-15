from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.bulk import BulkParams, build_A
from src.logging_utils import make_run_dir, save_run_snapshot
from src.metrics import top_singular_values
from src.weights import apply_similarity_weight


def main() -> int:
    ap = argparse.ArgumentParser(description="Experiment B2: singular value decay for weighted interior K_beta(s).")
    ap.add_argument("--config", type=str, default="configs/exp_B_singular_values.yaml")
    ap.add_argument("--runs_root", type=str, default="runs")
    args = ap.parse_args()

    import src.configs as cfg

    config = cfg.load_config(args.config)

    primes = list(map(int, config.get("primes", [2, 3, 5, 7, 11])))
    Ns = list(map(int, config.get("Ns", [256, 512, 1024])))
    beta = float(config.get("beta", 0.5))
    sigma = float(config.get("sigma", 0.8))
    E = float(config.get("E", 0.0))
    r = int(config.get("r", 50))

    boundary_frac = float(config.get("boundary_frac", 0.125))

    out_dir = make_run_dir(args.runs_root, tag="expB_singular_values")
    save_run_snapshot(out_dir, config=config, workspace_root=str(Path(__file__).resolve().parents[1]))

    rows: list[dict] = []

    for N in Ns:
        b = int(round(boundary_frac * int(N)))
        if b <= 0:
            b = 1
        if b >= N - 1:
            b = max(1, N // 4)

        params = BulkParams(N=int(N), weight_k=int(config.get("weight_k", 0)))
        s = complex(float(sigma), float(E))
        A = build_A(s=s, primes=primes, comps=None, params=params, prime_scale=float(config.get("prime_scale", 1.0)))

        Aii = np.asarray(A[b:, b:], dtype=np.complex128)
        K0 = Aii - np.eye(Aii.shape[0], dtype=np.complex128)
        idx = np.arange(b + 1, N + 1, dtype=np.float64)
        Kb = apply_similarity_weight(K0, n_indices_1based=idx, beta=float(beta))

        svals = top_singular_values(Kb, r=r)
        for j, sv in enumerate(svals, start=1):
            rows.append({"N": int(N), "b": int(b), "sigma": sigma, "E": E, "beta": beta, "k": int(j), "sv": float(sv)})

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "B_singular_values.csv"), index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
