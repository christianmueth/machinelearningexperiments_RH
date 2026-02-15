from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.bulk import BulkParams, build_A
from src.logging_utils import make_run_dir, save_run_snapshot
from src.metrics import fro_norm


def _logdet_series_terms_for_single_generator(T: np.ndarray, eps: complex, m_max: int) -> list[dict]:
    """Compute ladder-like series terms for logdet(I + eps T).

    Uses logdet(I+X) = sum_{m>=1} (-1)^{m+1} tr(X^m)/m.
    Here X = eps*T, so terms are proportional to eps^m.
    """

    T = np.asarray(T, dtype=np.complex128)
    out = []
    X = eps * T
    Xm = np.eye(T.shape[0], dtype=np.complex128)
    for m in range(1, int(m_max) + 1):
        Xm = Xm @ X
        tr = complex(np.trace(Xm))
        term = ((-1) ** (m + 1)) * tr / float(m)
        out.append({"m": int(m), "trace_Xm": tr, "series_term": term, "series_term_abs": abs(term)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Experiment A: local Euler-architecture validations (toy Hecke bulk).")
    ap.add_argument("--config", type=str, default="configs/exp_A_local_euler.yaml")
    ap.add_argument("--runs_root", type=str, default="runs")
    args = ap.parse_args()

    import src.configs as cfg
    from src.hecke import HeckeParams, hecke_Tn

    config = cfg.load_config(args.config)

    N = int(config.get("N", 256))
    s = complex(config.get("s", 0.8))
    primes = list(map(int, config.get("primes", [2, 3, 5, 7, 11])))
    comps = list(map(int, config.get("composites", [4, 6, 8, 9, 10, 12])))
    prime_scale = float(config.get("prime_scale", 1.0))
    comp_scale = float(config.get("comp_scale", 1.0))
    comps_mode = str(config.get("composites_mode", "all"))
    match_comp_to_prime = bool(config.get("match_comp_to_prime", False))
    generator_norm = config.get("generator_norm", None)
    if generator_norm is not None:
        generator_norm = str(generator_norm)
    generator_norm_target = float(config.get("generator_norm_target", 1.0))

    def logdet(M: np.ndarray) -> complex:
        M = np.asarray(M, dtype=np.complex128)
        sign, logabs = np.linalg.slogdet(M)
        return complex(np.log(sign) + logabs)

    out_dir = make_run_dir(args.runs_root, tag="expA_local_euler")
    save_run_snapshot(out_dir, config=config, workspace_root=str(Path(__file__).resolve().parents[1]))

    bulk_params = BulkParams(N=N, weight_k=int(config.get("weight_k", 0)))

    # Prime-only vs composite “pollution”
    A_prime, Mprime_prime, Mcomp_prime = build_A(
        s=s,
        primes=primes,
        comps=None,
        params=bulk_params,
        prime_scale=prime_scale,
        generator_norm=generator_norm,
        generator_norm_target=generator_norm_target,
        return_components=True,
    )

    # Composite injection: (i) prime-powers only, (ii) all composites
    A_pp, Mprime_pp, Mcomp_pp = build_A(
        s=s,
        primes=primes,
        comps=comps,
        comps_mode="prime_powers",
        params=bulk_params,
        prime_scale=prime_scale,
        comp_scale=comp_scale,
        match_comp_to_prime=match_comp_to_prime,
        generator_norm=generator_norm,
        generator_norm_target=generator_norm_target,
        return_components=True,
    )
    A_comp, Mprime_comp, Mcomp_comp = build_A(
        s=s,
        primes=primes,
        comps=comps,
        comps_mode=comps_mode,
        params=bulk_params,
        prime_scale=prime_scale,
        comp_scale=comp_scale,
        match_comp_to_prime=match_comp_to_prime,
        generator_norm=generator_norm,
        generator_norm_target=generator_norm_target,
        return_components=True,
    )

    logphi_prime = logdet(A_prime)
    logphi_pp = logdet(A_pp)
    logphi_comp = logdet(A_comp)

    rows = [
        {
            "model": "prime_only",
            "N": N,
            "s": s,
            "logdet": logphi_prime,
            "logdet_abs": abs(logphi_prime),
            "fro_prime_block": fro_norm(Mprime_prime),
            "fro_comp_block": fro_norm(Mcomp_prime),
        },
        {
            "model": "prime_plus_prime_powers",
            "N": N,
            "s": s,
            "logdet": logphi_pp,
            "logdet_abs": abs(logphi_pp),
            "fro_prime_block": fro_norm(Mprime_pp),
            "fro_comp_block": fro_norm(Mcomp_pp),
        },
        {
            "model": f"prime_plus_{comps_mode}",
            "N": N,
            "s": s,
            "logdet": logphi_comp,
            "logdet_abs": abs(logphi_comp),
            "fro_prime_block": fro_norm(Mprime_comp),
            "fro_comp_block": fro_norm(Mcomp_comp),
        },
        {
            "model": "delta(pp - prime)",
            "N": N,
            "s": s,
            "logdet": (logphi_pp - logphi_prime),
            "logdet_abs": abs(logphi_pp - logphi_prime),
            "fro_prime_block": fro_norm(Mprime_pp),
            "fro_comp_block": fro_norm(Mcomp_pp),
        },
        {
            "model": f"delta({comps_mode} - prime)",
            "N": N,
            "s": s,
            "logdet": (logphi_comp - logphi_prime),
            "logdet_abs": abs(logphi_comp - logphi_prime),
            "fro_prime_block": fro_norm(Mprime_comp),
            "fro_comp_block": fro_norm(Mcomp_comp),
        },
    ]

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "A_prime_vs_composite.csv"), index=False)

    # Prime-power ladder (single prime)
    p0 = int(config.get("ladder_prime", primes[0]))
    hecke_params = HeckeParams(N=N, weight_k=int(config.get("weight_k", 0)))
    Tp = hecke_Tn(p0, hecke_params)

    from src.bulk import eps_prime

    eps = complex(prime_scale) * eps_prime(p0, s)
    m_max = int(config.get("ladder_m_max", 12))
    ladder = _logdet_series_terms_for_single_generator(Tp, eps=eps, m_max=m_max)
    df_ladder = pd.DataFrame(ladder)
    df_ladder["p"] = p0
    df_ladder["s"] = s
    df_ladder.to_csv(os.path.join(out_dir, "A_prime_power_ladder.csv"), index=False)

    # Coprime multiplicativity probe (two primes): delta = logdet(A_{p,q}) - logdet(A_p) - logdet(A_q)
    p = int(config.get("two_primes", [2, 3])[0])
    q = int(config.get("two_primes", [2, 3])[1])
    A_p = build_A(
        s=s,
        primes=[p],
        comps=None,
        params=bulk_params,
        prime_scale=prime_scale,
        generator_norm=generator_norm,
        generator_norm_target=generator_norm_target,
    )
    A_q = build_A(
        s=s,
        primes=[q],
        comps=None,
        params=bulk_params,
        prime_scale=prime_scale,
        generator_norm=generator_norm,
        generator_norm_target=generator_norm_target,
    )
    A_pq = build_A(
        s=s,
        primes=[p, q],
        comps=None,
        params=bulk_params,
        prime_scale=prime_scale,
        generator_norm=generator_norm,
        generator_norm_target=generator_norm_target,
    )

    delta = logdet(A_pq) - logdet(A_p) - logdet(A_q)
    pd.DataFrame(
        [
            {"p": p, "q": q, "N": N, "s": s, "delta_logdet": delta, "abs": abs(delta)},
        ]
    ).to_csv(os.path.join(out_dir, "A_coprime_delta.csv"), index=False)

    # Note: boundary renormalization is done in Exp B (K_beta), where b/N sweeps are explicit.

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
