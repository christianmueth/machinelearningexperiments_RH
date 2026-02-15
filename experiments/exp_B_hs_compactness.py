from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.bulk import BulkParams, build_A
from src.logging_utils import make_run_dir, save_run_snapshot
from src.metrics import hs_norm_sq
from src.weights import apply_similarity_weight


def main() -> int:
    ap = argparse.ArgumentParser(description="Experiment B1: HS proxy scaling for weighted interior perturbation K_beta(s).")
    ap.add_argument("--config", type=str, default="configs/exp_B_hs_compactness.yaml")
    ap.add_argument("--runs_root", type=str, default="runs")
    args = ap.parse_args()

    import src.configs as cfg

    config = cfg.load_config(args.config)

    primes = list(map(int, config.get("primes", [2, 3, 5, 7, 11])))
    comps = list(map(int, config.get("composites", [])))

    Ns = list(map(int, config.get("Ns", [256, 512, 1024])))
    betas = list(map(float, config.get("betas", [0.0, 0.25, 0.5, 0.75, 1.0])))
    sigmas = list(map(float, config.get("sigmas", [1.2, 1.0, 0.8, 0.6])))
    E = float(config.get("E", 0.0))

    boundary_frac = float(config.get("boundary_frac", 0.125))
    prime_scale = float(config.get("prime_scale", 1.0))
    comp_scale = float(config.get("comp_scale", 1.0))

    out_dir = make_run_dir(args.runs_root, tag="expB_hs_compactness")
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
            A = build_A(s=s, primes=primes, comps=(comps if comps else None), params=params, prime_scale=prime_scale, comp_scale=comp_scale)

            # interior-only perturbation relative to identity
            Aii = np.asarray(A[b:, b:], dtype=np.complex128)
            K0 = Aii - np.eye(Aii.shape[0], dtype=np.complex128)

            # weights use GLOBAL index n = b+1..N
            idx = np.arange(b + 1, N + 1, dtype=np.float64)

            for beta in betas:
                Kb = apply_similarity_weight(K0, n_indices_1based=idx, beta=float(beta))
                hs2 = hs_norm_sq(Kb)
                rows.append(
                    {
                        "N": int(N),
                        "b": int(b),
                        "boundary_frac": boundary_frac,
                        "sigma": float(sigma),
                        "E": float(E),
                        "beta": float(beta),
                        "hs_norm_sq": float(hs2),
                        "hs_norm": float(np.sqrt(hs2)),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "B_hs_compactness.csv"), index=False)

    # quick pivot for sanity
    piv = df.pivot_table(index=["sigma", "beta"], columns=["N"], values=["hs_norm"], aggfunc="mean")
    piv.to_csv(os.path.join(out_dir, "B_hs_compactness_pivot.csv"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
