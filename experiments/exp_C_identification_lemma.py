from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mpmath as mp
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bulk import BulkParams, build_A
from src.dn import dn_map
from src.logging_utils import make_run_dir, save_run_snapshot
from src.metrics import cond_number, hermitian_defect, min_dist_to_minus1, unitarity_defect
from src.scattering import cayley_from_lambda, cayley_skew_from_lambda


def _complex_slogdet(M: np.ndarray) -> complex:
    M = np.asarray(M, dtype=np.complex128)
    sign, logabs = np.linalg.slogdet(M)
    return complex(np.log(sign) + logabs)


def _det_from_logdet(logdet: complex) -> complex:
    return complex(np.exp(logdet))


def _md_table(df: pd.DataFrame, max_rows: int = 25) -> str:
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


@dataclass(frozen=True)
class ModelCfg:
    primes: tuple[int, ...]
    prime_support_k: int
    N: int
    boundary_frac: float
    eta: float
    schur_jitter: float
    weight_k: int

    prime_scale: float
    comp_scale: float
    generator_norm: str | None
    generator_norm_target: float

    completion_mode: str
    dual_scale: float

    lambda_prefactor: str

    normalize_free: bool


def _build_primes(primes_cfg: list[int], prime_support_k: int, mode: str) -> list[int]:
    mode = str(mode).lower().strip()
    prime_support_k = int(prime_support_k)
    if mode == "prefix":
        primes = list(map(int, primes_cfg))
        if len(primes) < prime_support_k:
            primes = _first_primes(prime_support_k)
        return primes[:prime_support_k]
    if mode == "first_primes":
        return _first_primes(prime_support_k)
    raise ValueError("prime_support_mode must be 'prefix' or 'first_primes'")


def xi(s: complex) -> complex:
    ss = mp.mpc(float(np.real(s)), float(np.imag(s)))
    # mpmath 1.3.0 has no riemannxi/xi helper and mp.gamma raises at poles.
    # xi(s) is entire; at s=0 and s=1 the expression has removable singularities with xi(0)=xi(1)=1/2.
    if abs(ss) < mp.mpf("1e-30"):
        return complex(mp.mpf("0.5"))
    if abs(ss - 1) < mp.mpf("1e-30"):
        return complex(mp.mpf("0.5"))

    def _raw(z):
        return mp.mpf("0.5") * z * (z - 1) * (mp.pi ** (-z / 2)) * mp.gamma(z / 2) * mp.zeta(z)

    try:
        return complex(_raw(ss))
    except ValueError:
        eps = mp.mpf("1e-10")
        return complex((_raw(ss + eps) + _raw(ss - eps)) / 2)


def S_modular(s: complex) -> complex:
    # phi(s) = xi(2s-1)/xi(2s)
    return complex(xi(2 * s - 1) / xi(2 * s))


def main() -> int:
    ap = argparse.ArgumentParser(description="Experiment C: analytic identification lemma checklist (FE/unitarity/arg-principle/normalization).")
    ap.add_argument("--config", type=str, default="configs/exp_C_identification_lemma.yaml")
    ap.add_argument("--runs_root", type=str, default="runs")
    args = ap.parse_args()

    import src.configs as cfg

    config = cfg.load_config(args.config)

    primes_cfg = list(map(int, config.get("primes", [2, 3, 5, 7, 11])))
    prime_support_k = int(config.get("prime_support_k", len(primes_cfg)))
    prime_support_mode = str(config.get("prime_support_mode", "prefix"))
    primes = _build_primes(primes_cfg, prime_support_k=prime_support_k, mode=prime_support_mode)

    Ns = list(map(int, config.get("Ns", [64, 96, 128])))
    sigmas = list(map(float, config.get("sigmas", [0.3, 0.5, 0.7, 1.2])))
    t_max = float(config.get("t_max", 50.0))
    t_steps = int(config.get("t_steps", 101))

    boundary_frac = float(config.get("boundary_frac", 0.25))
    eta = float(config.get("eta", 1.0))
    schur_jitter = float(config.get("schur_jitter", 1e-10))

    prime_scale = float(config.get("prime_scale", 1.0))
    comp_scale = float(config.get("comp_scale", 1.0))

    generator_norm = config.get("generator_norm", "fro")
    if generator_norm is not None:
        generator_norm = str(generator_norm)
    generator_norm_target = float(config.get("generator_norm_target", 1.0))

    completion_mode = str(config.get("completion_mode", "dual_1_minus_s"))
    dual_scale = float(config.get("dual_scale", 1.0))

    lambda_prefactor = str(config.get("lambda_prefactor", "none"))

    normalize_free = bool(config.get("normalize_free", True))

    mp.mp.dps = int(config.get("mp_dps", 50))
    compare_modular = bool(config.get("compare_modular", True))

    s0 = complex(config.get("s0", 2.0 + 0j))

    do_fe_unitarity = bool(config.get("do_fe_unitarity", True))
    do_arg_principle = bool(config.get("do_arg_principle", True))
    do_quotient = bool(config.get("do_quotient", True))

    rects = config.get("rects", [[0.55, 1.60, 0.0, 40.0]])
    rects = [(float(a), float(b), float(c), float(d)) for a, b, c, d in rects]

    ap_h = float(config.get("arg_h", 1e-5))
    ap_n_edge = int(config.get("arg_n_edge", 1200))

    out_dir = make_run_dir(args.runs_root, tag="expC_identification_lemma")
    save_run_snapshot(out_dir, config=config, workspace_root=str(Path(__file__).resolve().parents[1]))

    t_grid = np.linspace(0.0, t_max, t_steps)

    def _mk_cfg(N: int) -> ModelCfg:
        return ModelCfg(
            primes=tuple(primes),
            prime_support_k=int(prime_support_k),
            N=int(N),
            boundary_frac=float(boundary_frac),
            eta=float(eta),
            schur_jitter=float(schur_jitter),
            weight_k=int(config.get("weight_k", 0)),
            prime_scale=float(prime_scale),
            comp_scale=float(comp_scale),
            generator_norm=generator_norm,
            generator_norm_target=float(generator_norm_target),
            completion_mode=str(completion_mode),
            dual_scale=float(dual_scale),
            lambda_prefactor=str(lambda_prefactor),
            normalize_free=bool(normalize_free),
        )

    def _b_from_frac(N: int, boundary_frac_: float) -> int:
        b = int(round(float(boundary_frac_) * int(N)))
        if b <= 0:
            b = 1
        if b >= N - 1:
            b = max(1, N // 4)
        return b

    @lru_cache(maxsize=10000)
    def _S_model_scalar(s: complex, cfgm: ModelCfg) -> tuple[complex, dict]:
        # interacting
        bulk_params = BulkParams(N=int(cfgm.N), weight_k=int(cfgm.weight_k))
        b = _b_from_frac(cfgm.N, cfgm.boundary_frac)

        pref_mode = str(cfgm.lambda_prefactor).lower().strip()
        if pref_mode not in {"none", "2s-1", "2s1"}:
            raise ValueError("lambda_prefactor must be 'none' or '2s-1'")
        if pref_mode in {"2s-1", "2s1"}:
            lam_scale = (2.0 * s - 1.0)
            cayley = cayley_skew_from_lambda
        else:
            lam_scale = 1.0 + 0j
            cayley = cayley_from_lambda

        A = build_A(
            s=s,
            primes=list(cfgm.primes),
            comps=None,
            params=bulk_params,
            prime_scale=cfgm.prime_scale,
            comp_scale=cfgm.comp_scale,
            generator_norm=cfgm.generator_norm,
            generator_norm_target=cfgm.generator_norm_target,
            completion_mode=cfgm.completion_mode,
            dual_scale=cfgm.dual_scale,
        )

        Lam_raw = dn_map(A, b=b, jitter=float(cfgm.schur_jitter))
        Lam = lam_scale * Lam_raw
        S = cayley(Lam, eta=cfgm.eta)
        logdetS = _complex_slogdet(S)
        detS = _det_from_logdet(logdetS)

        meta = {
            "b": int(b),
            "Lam_dim": int(Lam.shape[0]),
            "lambda_prefactor": str(cfgm.lambda_prefactor),
            "hermitian_err": float(hermitian_defect(Lam_raw)),
            "unitarity_err": float(unitarity_defect(S)),
            "minus1_dist": float(min_dist_to_minus1(S)),
        }

        if not cfgm.normalize_free:
            return detS, meta

        A0 = build_A(
            s=s,
            primes=[],
            comps=None,
            params=bulk_params,
            prime_scale=1.0,
            comp_scale=0.0,
            generator_norm=cfgm.generator_norm,
            generator_norm_target=cfgm.generator_norm_target,
            completion_mode=cfgm.completion_mode,
            dual_scale=cfgm.dual_scale,
        )

        Lam0_raw = dn_map(A0, b=b, jitter=float(cfgm.schur_jitter))
        Lam0 = lam_scale * Lam0_raw
        S0 = cayley(Lam0, eta=cfgm.eta)
        detS0 = _det_from_logdet(_complex_slogdet(S0))
        meta["hermitian_err_free"] = float(hermitian_defect(Lam0_raw))
        meta["unitarity_err_free"] = float(unitarity_defect(S0))

        if detS0 == 0:
            return complex("nan"), meta
        return detS / detS0, meta

    # Normalization at one point
    cfg_ref = _mk_cfg(int(Ns[-1]))
    Sm0, meta0 = _S_model_scalar(s0, cfg_ref)
    target0 = S_modular(s0) if compare_modular else (1.0 + 0j)
    C_ref = target0 / Sm0 if Sm0 != 0 else complex("nan")

    # Per-N normalization constants (helps separate finite-N scaling drift from FE/continuation diagnostics).
    C_per_N: dict[int, complex] = {}
    for N in Ns:
        cfgN = _mk_cfg(int(N))
        Sm0N, _ = _S_model_scalar(s0, cfgN)
        C_per_N[int(N)] = (target0 / Sm0N) if Sm0N != 0 else complex("nan")

    with open(os.path.join(out_dir, "C_normalization.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "s0": {"re": float(np.real(s0)), "im": float(np.imag(s0))},
                "N_ref": int(cfg_ref.N),
                "S_model(s0)": {"re": float(np.real(Sm0)), "im": float(np.imag(Sm0))},
                "S_mod(s0)": {"re": float(np.real(target0)), "im": float(np.imag(target0))},
                "scale_C_ref": {"re": float(np.real(C_ref)), "im": float(np.imag(C_ref))},
                "scale_C_per_N": {
                    str(int(N)): {"re": float(np.real(Cn)), "im": float(np.imag(Cn))} for N, Cn in C_per_N.items()
                },
                "meta": meta0,
            },
            f,
            indent=2,
        )

    rows_fe: list[dict] = []

    if do_fe_unitarity:
        for N in Ns:
            cfgN = _mk_cfg(int(N))
            Cn = C_per_N.get(int(N), C_ref)
            for sigma in sigmas:
                for t in t_grid:
                    s = complex(float(sigma), float(t))
                    s2 = 1.0 - s
                    try:
                        S1, m1 = _S_model_scalar(s, cfgN)
                        S2, m2 = _S_model_scalar(s2, cfgN)
                        err_fe = abs(S1 * S2 - 1.0)
                        err_fe_scaled = abs((Cn * S1) * (Cn * S2) - 1.0)
                        sym_err = abs(S1 - S2)
                        unit_scalar = float(abs(abs(S1) - 1.0)) if abs(float(sigma) - 0.5) < 1e-12 else float("nan")
                        rows_fe.append(
                            {
                                "N": int(N),
                                "sigma": float(sigma),
                                "t": float(t),
                                "S_re": float(np.real(S1)),
                                "S_im": float(np.imag(S1)),
                                "S_1ms_re": float(np.real(S2)),
                                "S_1ms_im": float(np.imag(S2)),
                                "err_FE": float(err_fe),
                                "err_FE_scaled": float(err_fe_scaled),
                                "sym_err": float(sym_err),
                                "unit_err_scalar": float(unit_scalar),
                                "unitarity_err_matrix": float(m1.get("unitarity_err", float("nan"))),
                                "hermitian_err": float(m1.get("hermitian_err", float("nan"))),
                                "minus1_dist": float(m1.get("minus1_dist", float("nan"))),
                            }
                        )
                    except Exception as e:
                        rows_fe.append(
                            {
                                "N": int(N),
                                "sigma": float(sigma),
                                "t": float(t),
                                "error": repr(e),
                            }
                        )

        pd.DataFrame(rows_fe).to_csv(os.path.join(out_dir, "C_FE_unitarity.csv"), index=False)

    # Quotient vs modular target
    if do_quotient:
        rows_q: list[dict] = []
        for N in Ns:
            cfgN = _mk_cfg(int(N))
            Cn = C_per_N.get(int(N), C_ref)
            for sigma in sigmas:
                for t in t_grid:
                    s = complex(float(sigma), float(t))
                    try:
                        Sm, _ = _S_model_scalar(s, cfgN)
                        Sm = Cn * Sm
                        if compare_modular:
                            Sm_mod = S_modular(s)
                            Q = Sm / Sm_mod
                        else:
                            Sm_mod = complex("nan")
                            Q = Sm

                        s2 = 1.0 - s
                        Sm2, _ = _S_model_scalar(s2, cfgN)
                        Sm2 = Cn * Sm2
                        if compare_modular:
                            Q2 = Sm2 / S_modular(s2)
                        else:
                            Q2 = Sm2

                        rows_q.append(
                            {
                                "N": int(N),
                                "sigma": float(sigma),
                                "t": float(t),
                                "Q_re": float(np.real(Q)),
                                "Q_im": float(np.imag(Q)),
                                "absQ": float(abs(Q)),
                                "Q_FE_err": float(abs(Q * Q2 - 1.0)),
                            }
                        )
                    except Exception as e:
                        rows_q.append({"N": int(N), "sigma": float(sigma), "t": float(t), "error": repr(e)})

        pd.DataFrame(rows_q).to_csv(os.path.join(out_dir, "C_quotient_Q.csv"), index=False)

    # Argument principle counting
    rows_ap: list[dict] = []

    def _dlogS(s: complex, cfgm: ModelCfg, h: float) -> complex:
        # d/ds log S(s) ≈ (S(s+h)-S(s-h)) / (2h * S(s)) with h real.
        S0, _ = _S_model_scalar(s, cfgm)
        if S0 == 0:
            return complex("nan")
        Sp, _ = _S_model_scalar(s + h, cfgm)
        Sm, _ = _S_model_scalar(s - h, cfgm)
        return (Sp - Sm) / (2.0 * h * S0)

    def _boundary_points(rect: tuple[float, float, float, float], n_edge: int) -> list[complex]:
        sigma_min, sigma_max, t_min, t_max = rect
        n = int(n_edge)
        if n < 40:
            n = 40
        q = n // 4

        pts: list[complex] = []
        # bottom
        for sig in np.linspace(sigma_min, sigma_max, q, endpoint=False):
            pts.append(complex(float(sig), float(t_min)))
        # right
        for tt in np.linspace(t_min, t_max, q, endpoint=False):
            pts.append(complex(float(sigma_max), float(tt)))
        # top
        for sig in np.linspace(sigma_max, sigma_min, q, endpoint=False):
            pts.append(complex(float(sig), float(t_max)))
        # left
        for tt in np.linspace(t_max, t_min, q, endpoint=False):
            pts.append(complex(float(sigma_min), float(tt)))

        pts.append(pts[0])
        return pts

    if do_arg_principle:
        for N in Ns:
            cfgN = _mk_cfg(int(N))
            for rect in rects:
                pts = _boundary_points(rect, ap_n_edge)
                integral = 0.0 + 0.0j
                failures = 0
                for i in range(len(pts) - 1):
                    s0p = pts[i]
                    s1p = pts[i + 1]
                    ds = s1p - s0p
                    smid = (s0p + s1p) / 2.0
                    try:
                        dlog = _dlogS(smid, cfgN, h=ap_h)
                        if not np.isfinite(dlog.real) or not np.isfinite(dlog.imag):
                            failures += 1
                            continue
                        integral += dlog * ds
                    except Exception:
                        failures += 1

                count = integral / (2j * np.pi)
                rows_ap.append(
                    {
                        "N": int(N),
                        "sigma_min": float(rect[0]),
                        "sigma_max": float(rect[1]),
                        "t_min": float(rect[2]),
                        "t_max": float(rect[3]),
                        "n_edge": int(ap_n_edge),
                        "h": float(ap_h),
                        "failures": int(failures),
                        "count_re": float(np.real(count)),
                        "count_im": float(np.imag(count)),
                        "count_abs_im": float(abs(np.imag(count))),
                    }
                )

        pd.DataFrame(rows_ap).to_csv(os.path.join(out_dir, "C_arg_principle.csv"), index=False)

    # Quick collaborator summary
    md = "# Exp C: identification-lemma checklist\n\n"
    md += "Config highlights:\n"
    md += f"- prime_support_k: {prime_support_k}\n"
    md += f"- completion_mode: {completion_mode} (dual_scale={dual_scale})\n"
    md += f"- normalize_free: {normalize_free}\n"
    md += f"- compare_modular: {compare_modular}\n"
    md += f"- s0: {s0}\n\n"

    if rows_fe:
        df_fe = pd.DataFrame(rows_fe)
        df_fe_ok = df_fe[df_fe.get("error").isna()] if "error" in df_fe.columns else df_fe
        if not df_fe_ok.empty:
            piv = (
                df_fe_ok.groupby(["N", "sigma"], as_index=False)
                .agg(
                    err_FE_med=("err_FE", "median"),
                    err_FE_p95=("err_FE", lambda x: float(np.quantile(x, 0.95))),
                    err_FE_scaled_med=("err_FE_scaled", "median"),
                    err_FE_scaled_p95=("err_FE_scaled", lambda x: float(np.quantile(x, 0.95))),
                )
                .sort_values(["N", "sigma"], ascending=True)
            )
            md += "## FE error summary (median / p95 over t-grid)\n\n"
            md += _md_table(piv, max_rows=50) + "\n"

            df_u = df_fe_ok[np.isfinite(df_fe_ok.get("unit_err_scalar", np.nan))].copy()
            if not df_u.empty:
                piv_u = (
                    df_u.groupby(["N", "sigma"], as_index=False)
                    .agg(
                        unit_err_scalar_med=("unit_err_scalar", "median"),
                        unit_err_scalar_p95=("unit_err_scalar", lambda x: float(np.quantile(x, 0.95))),
                    )
                    .sort_values(["N", "sigma"], ascending=True)
                )
                md += "## Scalar unitarity on critical line (median / p95 of ||S|-1|)\n\n"
                md += _md_table(piv_u, max_rows=50) + "\n"

            piv_sym = (
                df_fe_ok.groupby(["N", "sigma"], as_index=False)
                .agg(sym_err_med=("sym_err", "median"), sym_err_p95=("sym_err", lambda x: float(np.quantile(x, 0.95))))
                .sort_values(["N", "sigma"], ascending=True)
            )
            md += "## s↦1−s symmetry diagnostic (median / p95 of |S(s)-S(1-s)|)\n\n"
            md += _md_table(piv_sym, max_rows=50) + "\n"

    if do_quotient:
        q_path = os.path.join(out_dir, "C_quotient_Q.csv")
        if os.path.exists(q_path):
            try:
                df_q = pd.read_csv(q_path)
                if "error" in df_q.columns:
                    df_q = df_q[df_q["error"].isna()]
                if (not df_q.empty) and ("Q_FE_err" in df_q.columns):
                    piv_q = (
                        df_q.groupby(["N", "sigma"], as_index=False)
                        .agg(Q_FE_err_med=("Q_FE_err", "median"), Q_FE_err_p95=("Q_FE_err", lambda x: float(np.quantile(x, 0.95))))
                        .sort_values(["N", "sigma"], ascending=True)
                    )
                    md += "## Quotient FE (median / p95 of |Q(s)Q(1-s)-1|)\n\n"
                    md += _md_table(piv_q, max_rows=50) + "\n"
            except Exception:
                pass

    if rows_ap:
        df_ap = pd.DataFrame(rows_ap)
        md += "## Argument principle (zeros - poles) estimates\n\n"
        md += _md_table(df_ap.sort_values(["N", "sigma_min", "t_max"], ascending=True), max_rows=50)

    with open(os.path.join(out_dir, "C_summary.md"), "w", encoding="utf-8") as f:
        f.write(md)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
