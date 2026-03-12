from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _angle(z: complex) -> float:
    z = complex(z)
    return float(math.atan2(float(np.imag(z)), float(np.real(z))))


def _unit_circle(z: complex) -> complex:
    z = complex(z)
    a = abs(z)
    if not math.isfinite(a) or a <= 0:
        return complex(1.0)
    return z / a


def _wrap_pi(theta: float) -> float:
    # map to (-pi, pi]
    theta = float(theta)
    twopi = 2.0 * math.pi
    theta = (theta + math.pi) % twopi - math.pi
    # push -pi to +pi for consistency
    if theta <= -math.pi:
        theta += twopi
    return float(theta)


def _design_matrix(primes: np.ndarray, *, model: str) -> np.ndarray:
    p = primes.astype(float)
    if model == "a_over_p":
        return (1.0 / p).reshape(-1, 1)
    if model == "a_over_p_plus_b_logp":
        return np.stack([1.0 / p, np.log(p)], axis=1)
    if model == "a_over_p_plus_b_logp_plus_c":
        return np.stack([1.0 / p, np.log(p), np.ones_like(p)], axis=1)
    raise ValueError("unknown model")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Fit a simple law for inferred Satake eigen-angles theta_p from infer_satake_from_sixby6 CSVs, "
            "and export a per-prime Satake table (alpha_p,beta_p) usable by six_by_six_prime_tower_sim.py --satake_family table."
        )
    )
    ap.add_argument("--infer_csv", required=True, help="CSV produced by tools/infer_satake_from_sixby6.py")
    ap.add_argument("--out_table_csv", required=True, help="Output Satake table CSV")
    ap.add_argument("--out_fit_csv", default="", help="Optional CSV with theta_obs/theta_fit/residual")

    ap.add_argument(
        "--theta_source",
        choices=["eig1", "eig2", "avg"],
        default="eig1",
        help="How to compute theta_p from eigenvalues (default: eig1)",
    )
    ap.add_argument(
        "--model",
        choices=["a_over_p", "a_over_p_plus_b_logp", "a_over_p_plus_b_logp_plus_c"],
        default="a_over_p",
        help="Theta fit model",
    )
    ap.add_argument("--ridge", type=float, default=0.0, help="Ridge regularization strength")
    ap.add_argument("--unitize", type=int, default=1, help="If 1, project eigenvalues onto unit circle before angles")

    args = ap.parse_args()

    infer_path = Path(args.infer_csv)
    if not infer_path.exists():
        raise SystemExit(f"missing --infer_csv: {infer_path}")

    df = pd.read_csv(infer_path)
    need = {"p", "eig1_re", "eig1_im", "eig2_re", "eig2_im"}
    missing = [c for c in sorted(need) if c not in df.columns]
    if missing:
        raise SystemExit(f"infer_csv missing columns: {missing}")

    p = df["p"].astype(int).to_numpy()
    e1 = df["eig1_re"].to_numpy() + 1j * df["eig1_im"].to_numpy()
    e2 = df["eig2_re"].to_numpy() + 1j * df["eig2_im"].to_numpy()

    if int(args.unitize) == 1:
        e1 = np.array([_unit_circle(z) for z in e1], dtype=np.complex128)
        e2 = np.array([_unit_circle(z) for z in e2], dtype=np.complex128)

    if args.theta_source == "eig1":
        theta = np.array([_wrap_pi(_angle(z)) for z in e1], dtype=float)
    elif args.theta_source == "eig2":
        theta = np.array([_wrap_pi(_angle(z)) for z in e2], dtype=float)
    else:
        th1 = np.array([_wrap_pi(_angle(z)) for z in e1], dtype=float)
        th2 = np.array([_wrap_pi(_angle(z)) for z in e2], dtype=float)
        # For unitary det~1 packets we expect th2 ~ -th1; average the antisymmetric part.
        theta = 0.5 * (th1 - th2)
        theta = np.array([_wrap_pi(t) for t in theta], dtype=float)

    X = _design_matrix(p.astype(float), model=str(args.model))
    y = theta.reshape(-1, 1)

    ridge = float(max(0.0, args.ridge))
    if ridge > 0:
        rr = math.sqrt(ridge)
        Xa = np.vstack([X, rr * np.eye(X.shape[1])])
        ya = np.vstack([y, np.zeros((X.shape[1], 1))])
        coef, *_ = np.linalg.lstsq(Xa, ya, rcond=None)
    else:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)

    coef = coef.reshape(-1)
    theta_fit = (X @ coef.reshape(-1, 1)).reshape(-1)
    resid = theta - theta_fit

    # Build alpha,beta = exp(± i theta_fit)
    alpha = np.cos(theta_fit) + 1j * np.sin(theta_fit)
    beta = np.cos(theta_fit) - 1j * np.sin(theta_fit)

    out = pd.DataFrame(
        {
            "p": p.astype(int),
            "theta_obs": theta.astype(float),
            "theta_fit": theta_fit.astype(float),
            "theta_resid": resid.astype(float),
            "alpha_re": np.real(alpha).astype(float),
            "alpha_im": np.imag(alpha).astype(float),
            "beta_re": np.real(beta).astype(float),
            "beta_im": np.imag(beta).astype(float),
        }
    ).sort_values(["p"])

    out_table = out[["p", "alpha_re", "alpha_im", "beta_re", "beta_im", "theta_fit"]].copy()

    out_table_path = Path(args.out_table_csv)
    out_table_path.parent.mkdir(parents=True, exist_ok=True)
    out_table.to_csv(out_table_path, index=False)

    if str(args.out_fit_csv).strip():
        out_fit_path = Path(args.out_fit_csv)
        out_fit_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_fit_path, index=False)

    # Print a compact fit summary
    rmse = float(np.sqrt(np.mean(resid**2)))
    med_abs = float(np.median(np.abs(resid)))
    print(f"model={args.model} theta_source={args.theta_source} ridge={ridge} rmse={rmse:.3e} median_abs_resid={med_abs:.3e}")
    for i, c in enumerate(coef.tolist()):
        print(f"coef[{i}]={float(c):.12g}")
    print(f"wrote {out_table_path}")
    if str(args.out_fit_csv).strip():
        print(f"wrote {Path(args.out_fit_csv)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
