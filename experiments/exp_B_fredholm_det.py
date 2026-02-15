from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.bulk import BulkParams, build_A
from src.logging_utils import make_run_dir, save_run_snapshot
from src.metrics import cond_number
from src.scattering import logdet_I_plus_K
from src.weights import apply_similarity_weight


def main() -> int:
    ap = argparse.ArgumentParser(description="Experiment B3: Fredholm determinant stability for det(I+K_beta(s)).")
    ap.add_argument("--config", type=str, default="configs/exp_B_fredholm_det.yaml")
    ap.add_argument("--runs_root", type=str, default="runs")
    args = ap.parse_args()

    import src.configs as cfg

    config = cfg.load_config(args.config)

    primes = list(map(int, config.get("primes", [2, 3, 5, 7, 11])))
    Ns = list(map(int, config.get("Ns", [256, 512, 1024])))
    betas = list(map(float, config.get("betas", [0.0, 0.5, 1.0])))
    sigmas = list(map(float, config.get("sigmas", [1.2, 1.0, 0.8, 0.6])))
    E = float(config.get("E", 0.0))

    boundary_frac = float(config.get("boundary_frac", 0.125))

    out_dir = make_run_dir(args.runs_root, tag="expB_fredholm_det")
    save_run_snapshot(out_dir, config=config, workspace_root=str(Path(__file__).resolve().parents[1]))

    rows: list[dict] = []

    for N in Ns:
        b = int(round(boundary_frac * int(N)))
        if b <= 0:
            b = 1
        if b >= N - 1:
            b = max(1, N // 4)

        params = BulkParams(N=int(N), weight_k=int(config.get("weight_k", 0)))

        for sigma in sigmas:
            s = complex(float(sigma), float(E))
            A = build_A(s=s, primes=primes, comps=None, params=params, prime_scale=float(config.get("prime_scale", 1.0)))

            Aii = np.asarray(A[b:, b:], dtype=np.complex128)
            K0 = Aii - np.eye(Aii.shape[0], dtype=np.complex128)
            idx = np.arange(b + 1, N + 1, dtype=np.float64)

            for beta in betas:
                Kb = apply_similarity_weight(K0, n_indices_1based=idx, beta=float(beta))
                logdet = logdet_I_plus_K(Kb)
                cond = cond_number(np.eye(Kb.shape[0], dtype=np.complex128) + Kb)
                rows.append(
                    {
                        "N": int(N),
                        "b": int(b),
                        "sigma": float(sigma),
                        "E": float(E),
                        "beta": float(beta),
                        "logdet": logdet,
                        "logdet_abs": abs(logdet),
                        "cond_IplusK": float(cond),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "B_fredholm_det.csv"), index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
