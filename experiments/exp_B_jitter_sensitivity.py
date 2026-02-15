from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.bulk import BulkParams, build_A
from src.logging_utils import make_run_dir, save_run_snapshot
from src.metrics import cond_number, hermitian_defect, hs_norm_sq, top_singular_values, unitarity_defect
from src.scattering import build_lambda_and_S_from_A, det_phase
from src.weights import apply_similarity_weight


def _first_primes(k: int) -> list[int]:
    k = int(k)
    if k <= 0:
        return []
    out: list[int] = []
    n = 2
    while len(out) < k:
        is_prime = True
        r = int(np.sqrt(n))
        for p in out:
            if p > r:
                break
            if n % p == 0:
                is_prime = False
                break
        if is_prime:
            out.append(n)
        n += 1
    return out


def _diag_part(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.complex128)
    return np.diag(np.diag(M))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Experiment: schur_jitter sensitivity sweep on a small set of representative points. "
            "Treat jitter as a diagnostic axis, not a phenomenon axis."
        )
    )
    ap.add_argument("--config", type=str, default="configs/exp_B_jitter_sensitivity.yaml")
    ap.add_argument("--runs_root", type=str, default="runs")
    args = ap.parse_args()

    import src.configs as cfg

    config = cfg.load_config(args.config)

    primes_cfg = list(map(int, config.get("primes", [2, 3, 5, 7, 11])))
    prime_support_k = int(config.get("prime_support_k", len(primes_cfg)))
    if len(primes_cfg) < prime_support_k:
        primes_cfg = _first_primes(prime_support_k)
    primes = primes_cfg[:prime_support_k]

    comps = list(map(int, config.get("composites", [])))
    comps_mode = str(config.get("composites_mode", "all"))

    Ns = list(map(int, config.get("Ns", [24, 64])))
    sigmas = list(map(float, config.get("sigmas", [0.60, 0.70])))
    Es = list(map(float, config.get("Es", [0.0])))
    betas = list(map(float, config.get("betas", [0.0, 1.0])))
    boundary_fracs = list(map(float, config.get("boundary_fracs", [0.20, 1.0 / 3.0])))
    jitters = list(map(float, config.get("jitters", [0.0, 1e-12, 1e-10, 1e-8])))

    eta = float(config.get("eta", 1.0))

    prime_scale = float(config.get("prime_scale", 1.0))
    comp_scale = float(config.get("comp_scale", 1.0))
    match_comp_to_prime = bool(config.get("match_comp_to_prime", False))

    generator_norm = config.get("generator_norm", None)
    if generator_norm is not None:
        generator_norm = str(generator_norm)
    generator_norm_target = float(config.get("generator_norm_target", 1.0))

    reference_subtraction = str(config.get("reference_subtraction", "diag")).lower().strip()

    r = int(config.get("r", 10))

    out_dir = make_run_dir(args.runs_root, tag="expB_jitter_sensitivity")
    save_run_snapshot(out_dir, config=config, workspace_root=str(Path(__file__).resolve().parents[1]))

    rows: list[dict] = []

    for boundary_frac in boundary_fracs:
        for N in Ns:
            b = int(round(float(boundary_frac) * int(N)))
            if b <= 0:
                b = 1
            if b >= N - 1:
                b = max(1, N // 4)

            params = BulkParams(N=int(N), weight_k=int(config.get("weight_k", 0)))

            for sigma in sigmas:
                for E in Es:
                    s = complex(float(sigma), float(E))

                    A = build_A(
                        s=s,
                        primes=primes,
                        comps=(comps if comps else None),
                        comps_mode=comps_mode,
                        params=params,
                        prime_scale=prime_scale,
                        comp_scale=comp_scale,
                        match_comp_to_prime=match_comp_to_prime,
                        generator_norm=generator_norm,
                        generator_norm_target=generator_norm_target,
                    )

                    for jitter in jitters:
                        Lam, S = build_lambda_and_S_from_A(A, b=b, eta=eta, schur_jitter=float(jitter))
                        K_raw = Lam - np.eye(Lam.shape[0], dtype=np.complex128)

                        if reference_subtraction in {"diag", "diagonal"}:
                            K_rel_raw = K_raw - _diag_part(K_raw)
                        elif reference_subtraction == "none":
                            K_rel_raw = K_raw
                        else:
                            raise ValueError(f"unknown reference_subtraction: {reference_subtraction}")

                        idx = np.arange(b + 1, N + 1, dtype=np.float64)

                        herm_err = hermitian_defect(Lam)
                        unit_err = unitarity_defect(S)
                        phase = det_phase(S)

                        for beta in betas:
                            K_abs = apply_similarity_weight(K_raw, n_indices_1based=idx, beta=float(beta))
                            K_rel = apply_similarity_weight(K_rel_raw, n_indices_1based=idx, beta=float(beta))

                            hs_abs = float(np.sqrt(hs_norm_sq(K_abs)))
                            hs_rel = float(np.sqrt(hs_norm_sq(K_rel)))

                            svals_rel = top_singular_values(K_rel, r=r)
                            cond_rel = cond_number(np.eye(K_rel.shape[0], dtype=np.complex128) + K_rel)

                            row: dict = {
                                "prime_support_k": int(prime_support_k),
                                "sigma": float(sigma),
                                "E": float(E),
                                "beta": float(beta),
                                "N": int(N),
                                "b": int(b),
                                "boundary_frac": float(boundary_frac),
                                "eta": float(eta),
                                "schur_jitter": float(jitter),
                                "reference_subtraction": reference_subtraction,
                                "hs_abs": float(hs_abs),
                                "hs_rel": float(hs_rel),
                                "cond_IplusK_rel": float(cond_rel),
                                "hermitian_err": float(herm_err),
                                "unitarity_err": float(unit_err),
                                "phase": float(phase),
                            }
                            for j in range(r):
                                row[f"sv_rel_{j+1}"] = float(svals_rel[j])

                            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "B_jitter_sensitivity.csv"), index=False)

    # quick best-jitter per point (min unitarity_err then min hs_rel)
    keys = ["prime_support_k", "sigma", "E", "beta", "boundary_frac", "N"]
    best = (
        df.sort_values(["unitarity_err", "hs_rel"], ascending=True)
        .groupby(keys, as_index=False)
        .head(1)
    )
    best.to_csv(os.path.join(out_dir, "B_jitter_sensitivity_best.csv"), index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
