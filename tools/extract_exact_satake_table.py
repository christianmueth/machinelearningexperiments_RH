from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _unit_circle(z: complex) -> complex:
    z = complex(z)
    a = abs(z)
    if not math.isfinite(a) or a <= 0:
        return complex(1.0)
    return z / a


def _wrap_pi(theta: float) -> float:
    theta = float(theta)
    twopi = 2.0 * math.pi
    theta = (theta + math.pi) % twopi - math.pi
    if theta <= -math.pi:
        theta += twopi
    return float(theta)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Extract an exact per-prime Satake table (alpha_p,beta_p) from infer_satake_from_sixby6 CSV by using observed eigenvalues. "
            "This bypasses any theta fit so Satake injection can reproduce the same eigenvalues exactly."
        )
    )
    ap.add_argument("--infer_csv", required=True)
    ap.add_argument("--out_table_csv", required=True)
    ap.add_argument("--primes", default="", help="Optional comma list of primes to keep")
    ap.add_argument("--use", choices=["eig1", "eig2"], default="eig1")
    ap.add_argument("--unitize", type=int, default=1)
    args = ap.parse_args()

    in_path = Path(args.infer_csv)
    if not in_path.exists():
        raise SystemExit(f"missing --infer_csv: {in_path}")

    df = pd.read_csv(in_path)
    need = {"p", "eig1_re", "eig1_im", "eig2_re", "eig2_im"}
    missing = [c for c in sorted(need) if c not in df.columns]
    if missing:
        raise SystemExit(f"infer_csv missing columns: {missing}")

    if str(args.primes).strip():
        keep = {int(x.strip()) for x in str(args.primes).split(",") if x.strip()}
        df = df[df["p"].astype(int).isin(sorted(keep))].copy()

    if df.empty:
        raise SystemExit("no rows selected")

    p = df["p"].astype(int).to_numpy()
    e1 = df["eig1_re"].to_numpy() + 1j * df["eig1_im"].to_numpy()
    e2 = df["eig2_re"].to_numpy() + 1j * df["eig2_im"].to_numpy()

    if int(args.unitize) == 1:
        e1 = np.array([_unit_circle(z) for z in e1], dtype=np.complex128)
        e2 = np.array([_unit_circle(z) for z in e2], dtype=np.complex128)

    if args.use == "eig1":
        alpha = e1
        beta = e2
    else:
        alpha = e2
        beta = e1

    theta = np.array([_wrap_pi(float(math.atan2(float(np.imag(z)), float(np.real(z))))) for z in alpha], dtype=float)

    out = pd.DataFrame(
        {
            "p": p.astype(int),
            "alpha_re": np.real(alpha).astype(float),
            "alpha_im": np.imag(alpha).astype(float),
            "beta_re": np.real(beta).astype(float),
            "beta_im": np.imag(beta).astype(float),
            "theta": theta.astype(float),
        }
    ).sort_values(["p"])

    out_path = Path(args.out_table_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
