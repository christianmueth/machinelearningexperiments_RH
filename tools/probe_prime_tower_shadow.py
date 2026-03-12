from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Import simulator module from sibling file (tools/ is not a package).
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
import six_by_six_prime_tower_sim as sim  # type: ignore


def _parse_p_modes(s: str) -> list[str]:
    out: list[str] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(part)
    return out


def _p_eff(p: int, *, p_mode: str) -> float:
    p = int(p)
    p_mode = str(p_mode).strip().lower()
    p_f = float(p)
    if p_mode in {"p", "prime"}:
        v = p_f
    elif p_mode in {"logp", "log(p)", "lnp"}:
        v = float(math.log(p_f))
    elif p_mode in {"p1_over_p", "(p+1)/p", "one_plus_invp", "1+1/p"}:
        v = (p_f + 1.0) / p_f
    elif p_mode in {"p_over_p1", "p/(p+1)"}:
        v = p_f / (p_f + 1.0)
    elif p_mode in {"invp", "1/p"}:
        v = 1.0 / p_f
    elif p_mode in {"p_minus1_over_p", "(p-1)/p", "1-1/p"}:
        v = (p_f - 1.0) / p_f
    else:
        raise ValueError(
            "p_mode must be one of: p, logp, p1_over_p, p_over_p1, invp, p_minus1_over_p"
        )

    if not (v > 0.0) or not math.isfinite(v):
        raise ValueError(f"invalid p_eff={v} from p={p} p_mode={p_mode}")
    return float(v)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size or x.size < 2:
        return float("nan")
    x0 = x - float(np.mean(x))
    y0 = y - float(np.mean(y))
    den = float(np.linalg.norm(x0) * np.linalg.norm(y0))
    if den <= 0 or not math.isfinite(den):
        return float("nan")
    return float(np.dot(x0, y0) / den)


@dataclass(frozen=True)
class FitLine:
    slope: float
    intercept: float
    r2: float


def _fit_loglog(p: np.ndarray, y: np.ndarray) -> FitLine:
    p = np.asarray(p, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = np.isfinite(p) & np.isfinite(y) & (p > 0) & (y > 0)
    p = p[m]
    y = y[m]
    if p.size < 2:
        return FitLine(float("nan"), float("nan"), float("nan"))

    X = np.log(p)
    Y = np.log(y)
    slope, intercept = np.polyfit(X, Y, 1)
    Yhat = slope * X + intercept
    ss_res = float(np.sum((Y - Yhat) ** 2))
    ss_tot = float(np.sum((Y - float(np.mean(Y))) ** 2))
    r2 = 1.0 - (ss_res / (ss_tot + 1e-300))
    return FitLine(float(slope), float(intercept), float(r2))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Probe how the local 6x6->Schur->Cayley reduction maps an upstream exponential tower "
            "(p_eff^{k}) into downstream p-scaling observables. This is meant to test whether an observed "
            "1/p-like effective law can arise as a shadow of an underlying k*log(p) tower."
        )
    )

    ap.add_argument("--primes", default="2,3,5,7,11,13,17,19,23,29,31,37")
    ap.add_argument("--p_modes", default="p,logp,invp")
    ap.add_argument("--k_max", type=int, default=1)
    ap.add_argument("--u", type=float, default=0.2, help="shear u; lower is set to v=-u to match FE tools")

    ap.add_argument("--boundary", default="0,5", help="Boundary indices i,j in 0..5")
    ap.add_argument("--schur_sign", choices=["-", "+"], default="+")
    ap.add_argument("--sharp", choices=["transpose", "conj_transpose"], default="conj_transpose")
    ap.add_argument("--x_mode", choices=["raw", "det1"], default="det1")
    ap.add_argument("--x_gamma", type=float, default=1.0)
    ap.add_argument("--scattering", choices=["lambda_pm_i", "i_pm_lambda"], default="i_pm_lambda")

    ap.add_argument("--out_csv", required=True)

    args = ap.parse_args()

    primes = sim._parse_int_list(args.primes)
    if not primes:
        raise SystemExit("--primes must be non-empty")

    p_modes = _parse_p_modes(args.p_modes)
    if not p_modes:
        raise SystemExit("--p_modes must be non-empty")

    parts = sim._parse_int_list(str(args.boundary))
    if len(parts) != 2:
        raise SystemExit("--boundary must be i,j")
    boundary = (int(parts[0]), int(parts[1]))

    u = float(args.u)
    v = -float(u)

    rows: list[dict] = []

    for p_mode in p_modes:
        for p in primes:
            S1: np.ndarray | None = None
            for k in range(1, int(args.k_max) + 1):
                X, A, Ash = sim._local_blocks_for_prime_power(
                    int(p),
                    int(k),
                    sharp_mode=str(args.sharp),
                    x_mode=str(args.x_mode),
                    x_gamma=float(args.x_gamma),
                    x_shear=float(u),
                    x_lower=float(v),
                    p_mode=str(p_mode),
                )
                B = sim._bulk_B_from_A(A, Ash)
                Lam = sim._schur_complement_Lambda(B, boundary=boundary, sign=str(args.schur_sign))
                S = sim._scattering_from_Lambda(Lam, mode=str(args.scattering))

                if k == 1:
                    S1 = np.asarray(S, dtype=np.complex128)

                rel_pow_closure = float("nan")
                if k >= 2 and S1 is not None:
                    Spow = np.eye(2, dtype=np.complex128)
                    for _ in range(int(k)):
                        Spow = (Spow @ S1).astype(np.complex128)
                    num = float(np.linalg.norm(S - Spow, ord="fro"))
                    den = float(np.linalg.norm(S, ord="fro"))
                    rel_pow_closure = float(num / (den + 1e-300))

                I2 = np.eye(2, dtype=np.complex128)
                dev_A = float(
                    min(
                        np.linalg.norm(A - I2, ord="fro"),
                        np.linalg.norm(A + I2, ord="fro"),
                    )
                )
                dev_S = float(
                    min(
                        np.linalg.norm(S - I2, ord="fro"),
                        np.linalg.norm(S + I2, ord="fro"),
                    )
                )

                peff = _p_eff(int(p), p_mode=str(p_mode))

                rows.append(
                    {
                        "p_mode": str(p_mode),
                        "p": int(p),
                        "k": int(k),
                        "u": float(u),
                        "boundary": f"{boundary[0]},{boundary[1]}",
                        "schur_sign": str(args.schur_sign),
                        "sharp": str(args.sharp),
                        "x_mode": str(args.x_mode),
                        "x_gamma": float(args.x_gamma),
                        "scattering": str(args.scattering),
                        "p_eff": float(peff),
                        "logp": float(math.log(float(p))),
                        "invp": float(1.0 / float(p)),
                        "k_logp": float(k) * float(math.log(float(p))),
                        "k_logp_eff": float(k) * float(math.log(float(peff))),
                        "herm_def_Lambda": float(sim._hermitian_defect(Lam)),
                        "unit_def_S": float(sim._unitarity_defect(S)),
                        "fro_Lambda": float(np.linalg.norm(Lam, ord="fro")),
                        "dev_A_pm_I": float(dev_A),
                        "dev_S_pm_I": float(dev_S),
                        "rel_pow_closure_S_vs_S1k": float(rel_pow_closure),
                    }
                )

    df = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")

    # Print a compact scaling summary by k.
    if not df.empty:
        print("\nScaling probe (dev_S_pm_I ~ p^slope):")
        for p_mode in p_modes:
            dpm_all = df[df["p_mode"] == str(p_mode)]
            if dpm_all.empty:
                continue
            print(f"  p_mode={p_mode}")
            for k in range(1, int(args.k_max) + 1):
                dpm = dpm_all[dpm_all["k"] == int(k)]
                if dpm.empty:
                    continue
                p_arr = dpm["p"].to_numpy(dtype=float)
                y = dpm["dev_S_pm_I"].to_numpy(dtype=float)
                fit = _fit_loglog(p_arr, y)
                corr_invp = _pearson(y, 1.0 / p_arr)
                corr_logp = _pearson(y, np.log(p_arr))
                print(
                    f"    k={k:2d}: slope={fit.slope:+.3f} (R^2={fit.r2:.3f}); "
                    f"corr(dev_S,1/p)={corr_invp:+.3f}; corr(dev_S,log p)={corr_logp:+.3f}"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
