from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FitResult:
    name: str
    intercept: float
    slope: float
    rmse: float
    r2: float


def _finite_mask(*cols: np.ndarray) -> np.ndarray:
    mask = np.ones_like(np.asarray(cols[0], dtype=float), dtype=bool)
    for c in cols:
        c = np.asarray(c, dtype=float)
        mask &= np.isfinite(c)
    return mask


def _fit_linear(x: np.ndarray, y: np.ndarray, *, name: str) -> FitResult:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = _finite_mask(x, y)
    x = x[m]
    y = y[m]
    if x.size < 2:
        return FitResult(name=name, intercept=float("nan"), slope=float("nan"), rmse=float("nan"), r2=float("nan"))

    X = np.stack([np.ones_like(x), x], axis=1)
    coef, *_ = np.linalg.lstsq(X, y.reshape(-1, 1), rcond=None)
    coef = coef.reshape(-1)
    b0 = float(coef[0])
    b1 = float(coef[1])
    yhat = (X @ coef.reshape(-1, 1)).reshape(-1)
    resid = (y - yhat).astype(float)

    rmse = float(np.sqrt(np.mean(resid**2)))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return FitResult(name=name, intercept=b0, slope=b1, rmse=rmse, r2=r2)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Summarize the u->(a,b,c) moduli from sweep_u_satake_family outputs. "
            "Fits simple linear laws in u for a(u), b(u), c(u), and reports closure stability metrics."
        )
    )
    ap.add_argument("--in_csv", required=True, help="Sweep summary CSV with theta_fit_abc_* columns")
    ap.add_argument("--out_csv", required=True, help="Write a one-row summary CSV")
    ap.add_argument("--out_txt", default="", help="Optional: write a human-readable text report")

    args = ap.parse_args()

    in_path = Path(args.in_csv)
    df = pd.read_csv(in_path)
    if df.empty:
        raise SystemExit("input CSV is empty")

    required = [
        "u",
        "theta_fit_abc_a_over_p",
        "theta_fit_abc_b_logp",
        "theta_fit_abc_c",
        "theta_fit_abc_rmse",
    ]
    for c in required:
        if c not in df.columns:
            raise SystemExit(f"missing required column: {c}")

    u = df["u"].to_numpy(dtype=float)
    a = df["theta_fit_abc_a_over_p"].to_numpy(dtype=float)
    b = df["theta_fit_abc_b_logp"].to_numpy(dtype=float)
    c0 = df["theta_fit_abc_c"].to_numpy(dtype=float)
    rmse_theta = df["theta_fit_abc_rmse"].to_numpy(dtype=float)

    fa = _fit_linear(u, a, name="a_over_p")
    fb = _fit_linear(u, b, name="b_logp")
    fc = _fit_linear(u, c0, name="c")

    out: dict[str, float] = {
        "n_u": float(df.shape[0]),
        "u_min": float(np.nanmin(u)),
        "u_max": float(np.nanmax(u)),
        "theta_abc_rmse_median": float(np.nanmedian(rmse_theta)),
        "theta_abc_rmse_max": float(np.nanmax(rmse_theta)),
        "a_intercept": float(fa.intercept),
        "a_slope": float(fa.slope),
        "a_rmse": float(fa.rmse),
        "a_r2": float(fa.r2),
        "b_intercept": float(fb.intercept),
        "b_slope": float(fb.slope),
        "b_rmse": float(fb.rmse),
        "b_r2": float(fb.r2),
        "c_intercept": float(fc.intercept),
        "c_slope": float(fc.slope),
        "c_rmse": float(fc.rmse),
        "c_r2": float(fc.r2),
    }

    if "rel_rmse_exact_injection" in df.columns:
        rel = df["rel_rmse_exact_injection"].to_numpy(dtype=float)
        if np.any(np.isfinite(rel)):
            out["closure_rel_rmse_median"] = float(np.nanmedian(rel))
            out["closure_rel_rmse_max"] = float(np.nanmax(rel))

    if "max_abs_exact_injection" in df.columns:
        mx = df["max_abs_exact_injection"].to_numpy(dtype=float)
        if np.any(np.isfinite(mx)):
            out["closure_max_abs_median"] = float(np.nanmedian(mx))
            out["closure_max_abs_max"] = float(np.nanmax(mx))

    out_df = pd.DataFrame([out])
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    if str(args.out_txt).strip():
        txt_path = Path(args.out_txt)
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        lines.append(f"input: {in_path}")
        lines.append(f"u range: [{out['u_min']:.6g}, {out['u_max']:.6g}] (n={int(out['n_u'])})")
        lines.append("")
        lines.append("theta fit (a/p + b log p + c)")
        lines.append(f"  median rmse: {out['theta_abc_rmse_median']:.6g}")
        lines.append(f"  max rmse:    {out['theta_abc_rmse_max']:.6g}")
        lines.append("")
        lines.append("linear moduli laws in u")
        lines.append(f"  a(u) = {out['a_intercept']:.6g} + {out['a_slope']:.6g} * u   (rmse={out['a_rmse']:.6g}, r2={out['a_r2']:.6g})")
        lines.append(f"  b(u) = {out['b_intercept']:.6g} + {out['b_slope']:.6g} * u   (rmse={out['b_rmse']:.6g}, r2={out['b_r2']:.6g})")
        lines.append(f"  c(u) = {out['c_intercept']:.6g} + {out['c_slope']:.6g} * u   (rmse={out['c_rmse']:.6g}, r2={out['c_r2']:.6g})")

        if "closure_rel_rmse_max" in out:
            lines.append("")
            lines.append("closure (intrinsic vs exact injected Satake)")
            lines.append(f"  rel_rmse median: {out.get('closure_rel_rmse_median', float('nan')):.3e}")
            lines.append(f"  rel_rmse max:    {out.get('closure_rel_rmse_max', float('nan')):.3e}")
            lines.append(f"  max_abs median:  {out.get('closure_max_abs_median', float('nan')):.3e}")
            lines.append(f"  max_abs max:     {out.get('closure_max_abs_max', float('nan')):.3e}")

        txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {out_path}")
    if str(args.out_txt).strip():
        print(f"wrote {args.out_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
