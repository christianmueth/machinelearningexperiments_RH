from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.bulk import BulkParams, build_A
from src.logging_utils import make_run_dir, save_run_snapshot
from src.metrics import cond_number, fro_norm, hermitian_defect, hs_norm_sq, top_singular_values, unitarity_defect
from src.scattering import build_lambda_and_S_from_A, det_phase
from src.weights import apply_similarity_weight


def _wrap_phase(x: float) -> float:
    # wrap to (-pi, pi]
    return float((x + np.pi) % (2.0 * np.pi) - np.pi)


def _diag_part(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.complex128)
    return np.diag(np.diag(M))


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


def _md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "(empty)\n"
    view = df.head(int(max_rows)).copy()
    cols = list(view.columns)
    header = "| " + " | ".join(cols) + " |\n"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |\n"
    rows: list[str] = []
    for _, r in view.iterrows():
        cells: list[str] = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                if np.isnan(v):
                    cells.append("nan")
                else:
                    cells.append(f"{v:.6g}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |\n")
    return header + sep + "".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Experiment B-map: combined feasibility grid over (sigma,beta,boundary_frac,N,E), "
            "logging absolute + relative metrics (with optional reference subtraction)."
        )
    )
    ap.add_argument("--config", type=str, default="configs/exp_B_feasibility_map.yaml")
    ap.add_argument("--runs_root", type=str, default="runs")
    args = ap.parse_args()

    import src.configs as cfg

    config = cfg.load_config(args.config)

    primes_cfg = list(map(int, config.get("primes", [2, 3, 5, 7, 11])))
    comps = list(map(int, config.get("composites", [])))

    prime_support_sizes = config.get("prime_support_sizes", None)
    if prime_support_sizes is None:
        prime_support_sizes = [len(primes_cfg)]
    prime_support_sizes = [int(x) for x in prime_support_sizes]
    if any(k <= 0 for k in prime_support_sizes):
        raise ValueError("prime_support_sizes must be positive")
    max_k = int(max(prime_support_sizes))

    prime_support_mode = str(config.get("prime_support_mode", "prefix")).lower().strip()
    if prime_support_mode not in {"prefix", "first_primes"}:
        raise ValueError("prime_support_mode must be 'prefix' or 'first_primes'")

    if prime_support_mode == "prefix":
        if len(primes_cfg) < max_k:
            base = set(primes_cfg)
            for p in _first_primes(max_k):
                if p not in base:
                    primes_cfg.append(p)
                    base.add(p)
                if len(primes_cfg) >= max_k:
                    break
    else:
        primes_cfg = _first_primes(max_k)

    Ns = list(map(int, config.get("Ns", [16, 24, 32, 48, 64, 96, 128])))
    betas = list(map(float, config.get("betas", [0.0, 0.25, 0.5, 0.75, 1.0])))
    sigmas = list(map(float, config.get("sigmas", [0.55, 0.60, 0.65, 0.70, 0.75])))
    Es = list(map(float, config.get("Es", [0.0])))
    boundary_fracs = list(map(float, config.get("boundary_fracs", [0.10, 0.20, 0.25, 1.0 / 3.0])))

    eta = float(config.get("eta", 1.0))
    schur_jitter = float(config.get("schur_jitter", 0.0))

    prime_scale = float(config.get("prime_scale", 1.0))
    comp_scale = float(config.get("comp_scale", 1.0))
    comps_mode = str(config.get("composites_mode", "all"))
    match_comp_to_prime = bool(config.get("match_comp_to_prime", False))

    generator_norm = config.get("generator_norm", None)
    if generator_norm is not None:
        generator_norm = str(generator_norm)
    generator_norm_target = float(config.get("generator_norm_target", 1.0))

    use_schur = bool(config.get("use_schur", True))
    reference_subtraction = str(config.get("reference_subtraction", "none")).lower().strip()

    r = int(config.get("r", 10))

    # trust gate + top pockets scoring
    alpha_unitarity = float(config.get("score_alpha_unitarity", 1.0))
    beta_hs_delta = float(config.get("score_beta_hs_delta", 1.0))
    top_k = int(config.get("top_pockets_k", 10))

    gate_max_hermitian = float(config.get("gate_max_hermitian_err", float("inf")))
    gate_max_unitarity = float(config.get("gate_max_unitarity_err", float("inf")))
    gate_max_cond = float(config.get("gate_max_cond", float("inf")))

    out_dir = make_run_dir(args.runs_root, tag="expB_feasibility_map")
    save_run_snapshot(out_dir, config=config, workspace_root=str(Path(__file__).resolve().parents[1]))

    prev_phase: dict[tuple[float, float, float, float, float], float] = {}

    rows: list[dict] = []

    for prime_support_k in prime_support_sizes:
        primes = primes_cfg[: int(prime_support_k)]
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

                    if use_schur:
                        Lam, S = build_lambda_and_S_from_A(A, b=b, eta=eta, schur_jitter=schur_jitter)
                        K_raw = Lam - np.eye(Lam.shape[0], dtype=np.complex128)
                        interior_start = b + 1
                        interior_end = N
                    else:
                        # interior-only baseline (older B1/B2/B3 definition)
                        Aii = np.asarray(A[b:, b:], dtype=np.complex128)
                        K_raw = Aii - np.eye(Aii.shape[0], dtype=np.complex128)
                        Lam = Aii
                        S = np.full((Aii.shape[0], Aii.shape[0]), np.nan + 1j * np.nan)
                        interior_start = b + 1
                        interior_end = N

                    # Reference subtraction axis
                    if reference_subtraction == "none":
                        K_rel_raw = K_raw
                    elif reference_subtraction in {"diag", "diagonal"}:
                        K_rel_raw = K_raw - _diag_part(K_raw)
                    else:
                        raise ValueError(f"unknown reference_subtraction: {reference_subtraction}")

                    idx = np.arange(interior_start, interior_end + 1, dtype=np.float64)

                    for beta in betas:
                        hs_raw = float(np.sqrt(hs_norm_sq(K_raw)))
                        hs_rel_raw = float(np.sqrt(hs_norm_sq(K_rel_raw)))

                        K_abs = apply_similarity_weight(K_raw, n_indices_1based=idx, beta=float(beta))
                        K_rel = apply_similarity_weight(K_rel_raw, n_indices_1based=idx, beta=float(beta))

                        hs_abs = float(np.sqrt(hs_norm_sq(K_abs)))
                        hs_rel = float(np.sqrt(hs_norm_sq(K_rel)))

                        svals_abs = top_singular_values(K_abs, r=r)
                        svals_rel = top_singular_values(K_rel, r=r)

                        I_abs = np.eye(K_abs.shape[0], dtype=np.complex128)
                        I_rel = np.eye(K_rel.shape[0], dtype=np.complex128)

                        cond_abs = cond_number(I_abs + K_abs)
                        cond_rel = cond_number(I_rel + K_rel)

                        # Scattering invariants (only meaningful in Schur mode)
                        herm_err = hermitian_defect(Lam)
                        unit_err = unitarity_defect(S) if use_schur else float("nan")
                        phase = det_phase(S) if use_schur else float("nan")

                        key = (float(prime_support_k), float(sigma), float(beta), float(boundary_frac), float(E))
                        phase_prev = prev_phase.get(key)
                        phase_diff = float("nan")
                        if phase_prev is not None and np.isfinite(phase) and np.isfinite(phase_prev):
                            phase_diff = _wrap_phase(float(phase - phase_prev))
                        if np.isfinite(phase):
                            prev_phase[key] = float(phase)

                        row: dict = {
                            "prime_support_k": int(prime_support_k),
                            "sigma": float(sigma),
                            "E": float(E),
                            "beta": float(beta),
                            "N": int(N),
                            "b": int(b),
                            "boundary_frac": float(boundary_frac),
                            "use_schur": bool(use_schur),
                            "schur_jitter": float(schur_jitter),
                            "eta": float(eta),
                            "reference_subtraction": reference_subtraction,
                            "hs_raw": float(hs_raw),
                            "hs_rel_raw": float(hs_rel_raw),
                            "hs_abs": float(hs_abs),
                            "hs_rel": float(hs_rel),
                            "cond_IplusK_abs": float(cond_abs),
                            "cond_IplusK_rel": float(cond_rel),
                            "hermitian_err": float(herm_err),
                            "unitarity_err": float(unit_err),
                            "phase": float(phase),
                            "phase_diff_vs_prevN": float(phase_diff),
                            "fro_K_raw": float(fro_norm(K_raw)),
                            "fro_K_rel_raw": float(fro_norm(K_rel_raw)),
                        }

                        # wide svals columns (k=1..r)
                        for j in range(r):
                            row[f"sv_abs_{j+1}"] = float(svals_abs[j])
                            row[f"sv_rel_{j+1}"] = float(svals_rel[j])

                        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "B_feasibility_map.csv"), index=False)

    # Tiny summary view: best (smallest) hs_rel per (prime_support_k,sigma,beta,boundary_frac,E)
    summ = (
        df.groupby(["prime_support_k", "sigma", "E", "beta", "boundary_frac"], as_index=False)
        .agg(hs_rel_min=("hs_rel", "min"), hs_abs_min=("hs_abs", "min"), cond_rel_min=("cond_IplusK_rel", "min"))
        .sort_values(["prime_support_k", "sigma", "E", "beta", "boundary_frac"], ascending=True)
    )
    summ.to_csv(os.path.join(out_dir, "B_feasibility_map_summary.csv"), index=False)

    # Collaborator-ready “top pockets” summary (trust gate + score).
    df2 = df.copy()
    df2 = df2.sort_values(["prime_support_k", "sigma", "E", "beta", "boundary_frac", "N"], ascending=True)
    df2["hs_rel_deltaN"] = df2.groupby(["prime_support_k", "sigma", "E", "beta", "boundary_frac"])["hs_rel"].diff(1)
    df2["phase_drift_abs"] = df2["phase_diff_vs_prevN"].abs()

    df2["gate_pass"] = (
        (df2["hermitian_err"] <= gate_max_hermitian)
        & (df2["unitarity_err"].fillna(0.0) <= gate_max_unitarity)
        & (df2["cond_IplusK_rel"] <= gate_max_cond)
    )

    df_last = df2.groupby(["prime_support_k", "sigma", "E", "beta", "boundary_frac"], as_index=False).tail(1)
    df_last["score"] = (
        df_last["phase_drift_abs"].fillna(1e9)
        + alpha_unitarity * df_last["unitarity_err"].fillna(1e9)
        + beta_hs_delta * df_last["hs_rel_deltaN"].abs().fillna(1e9)
    )

    df_top = (
        df_last[df_last["gate_pass"]]
        .sort_values(["score"], ascending=True)
        .head(int(top_k))
        .loc[
            :,
            [
                "score",
                "prime_support_k",
                "sigma",
                "E",
                "beta",
                "boundary_frac",
                "N",
                "hs_rel",
                "hs_rel_deltaN",
                "phase_drift_abs",
                "unitarity_err",
                "hermitian_err",
                "cond_IplusK_rel",
            ],
        ]
    )
    df_top.to_csv(os.path.join(out_dir, "B_top_pockets.csv"), index=False)

    gate_counts = df2.groupby(["prime_support_k"], as_index=False).agg(n_total=("gate_pass", "size"), n_pass=("gate_pass", "sum"))
    gate_counts.to_csv(os.path.join(out_dir, "B_trust_gate_counts.csv"), index=False)

    md = "# Top pockets\n\n"
    md += "Trust gate thresholds:\n"
    md += f"- gate_max_hermitian_err: {gate_max_hermitian}\n"
    md += f"- gate_max_unitarity_err: {gate_max_unitarity}\n"
    md += f"- gate_max_cond: {gate_max_cond}\n\n"
    md += "Score definition (lower is better):\n"
    md += f"- score = |phase_drift| + {alpha_unitarity}·unitarity_err + {beta_hs_delta}·|Δ_N hs_rel|\n\n"
    md += "## Gate counts\n\n"
    md += _md_table(gate_counts, max_rows=100) + "\n"
    md += "## Top pockets (at max N per group)\n\n"
    md += _md_table(df_top, max_rows=int(top_k))

    with open(os.path.join(out_dir, "B_top_pockets.md"), "w", encoding="utf-8") as f:
        f.write(md)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
