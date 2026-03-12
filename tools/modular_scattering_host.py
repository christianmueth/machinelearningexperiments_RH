from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import mpmath as mp
import numpy as np
import pandas as pd


def _primes_upto(n: int) -> list[int]:
    n = int(n)
    if n < 2:
        return []
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, int(n**0.5) + 1):
        if sieve[p]:
            start = p * p
            sieve[start : n + 1 : p] = b"\x00" * (((n - start) // p) + 1)
    return [i for i in range(2, n + 1) if sieve[i]]


def _xi(s: mp.mpc) -> mp.mpc:
    # Completed xi: (1/2) s(s-1) pi^{-s/2} Gamma(s/2) zeta(s)
    return mp.mpf("0.5") * s * (s - 1) * mp.power(mp.pi, -s / 2) * mp.gamma(s / 2) * mp.zeta(s)


def phi_mod(s: mp.mpc) -> mp.mpc:
    return _xi(2 * s - 1) / _xi(2 * s)


def phi_P(s: mp.mpc, primes: list[int]) -> mp.mpc:
    # Tier-1 partial Euler product from notes/proposition_identification_hinge.md:
    #   phi_P(s) = prod_{p in P} (1 - p^{-2s}) / (1 - p^{-(2s-1)}).
    out = mp.mpc(1)
    for p in primes:
        pp = mp.mpf(p)
        num = 1 - mp.power(pp, -2 * s)
        den = 1 - mp.power(pp, -(2 * s - 1))
        out *= num / den
    return out


@dataclass(frozen=True)
class ComplexCols:
    re: str
    im: str


def _infer_cols(df: pd.DataFrame, prefix: str) -> ComplexCols:
    re_c = f"{prefix}_re"
    im_c = f"{prefix}_im"
    if re_c in df.columns and im_c in df.columns:
        return ComplexCols(re=re_c, im=im_c)
    raise ValueError(f"missing complex columns for prefix={prefix!r} (expected {re_c},{im_c})")


def _to_complex(df: pd.DataFrame, cols: ComplexCols) -> np.ndarray:
    zr = pd.to_numeric(df[cols.re], errors="coerce").to_numpy(dtype=float)
    zi = pd.to_numeric(df[cols.im], errors="coerce").to_numpy(dtype=float)
    z = (zr + 1j * zi).astype(np.complex128)
    return z


def _finite_mask(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.complex128).ravel()
    return np.isfinite(np.real(z)) & np.isfinite(np.imag(z))


def _rel_dispersion(z: np.ndarray) -> float:
    z = np.asarray(z, dtype=np.complex128).ravel()
    m = _finite_mask(z)
    z = z[m]
    if z.size < 3:
        return float("nan")
    denom = float(np.sqrt(np.sum(np.abs(z) ** 2)))
    if not math.isfinite(denom) or denom <= 0:
        return float("nan")
    mu = complex(np.mean(z))
    num = float(np.sqrt(np.sum(np.abs(z - mu) ** 2)))
    return float(num / denom)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Compute the explicit modular Eisenstein scattering host and Tier-1 partial Euler products on an existing (sigma,t) grid. "
            "Emits phi_mod(s)=xi(2s-1)/xi(2s), phi_P(s)=prod_p (1-p^{-2s})/(1-p^{-(2s-1)}), and quotients Q= lambda/phi." 
        )
    )
    ap.add_argument("--in_csv", required=True, help="Input CSV containing columns sigma,t and (optionally) extracted channel columns")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument("--side", default="", help="If provided and input has a 'side' column, filter to this value")
    ap.add_argument("--sigma", type=float, default=float("nan"), help="If finite, restrict to this sigma (nearest match)")
    ap.add_argument("--sigma_tol", type=float, default=1e-12)

    ap.add_argument("--dps", type=int, default=80, help="mpmath decimal precision")

    ap.add_argument(
        "--p_max",
        type=int,
        default=0,
        help=(
            "If >0, also compute Tier-1 phi_P(s) using all primes <= p_max. "
            "(p_max=0 disables phi_P.)"
        ),
    )

    ap.add_argument(
        "--lambda_prefixes",
        default="",
        help=(
            "Comma-separated list of complex prefixes in the input to quotient against phi_mod/phi_P. "
            "Each prefix must have <prefix>_re and <prefix>_im columns (e.g. phi0_plus,phi0_minus)."
        ),
    )
    ap.add_argument(
        "--normalize_basepoint",
        type=int,
        default=1,
        help=(
            "If 1, also emit normalized quotients Q_norm(s)=Q(s)/Q(s0) where s0 is the first row after filtering. "
            "This matches the docs: only an s-independent constant ambiguity is allowed."
        ),
    )

    args = ap.parse_args()

    in_path = Path(args.in_csv)
    if not in_path.exists():
        raise SystemExit(f"missing --in_csv: {in_path}")

    df = pd.read_csv(in_path)
    if df.empty:
        raise SystemExit("input CSV empty")
    if "sigma" not in df.columns or "t" not in df.columns:
        raise SystemExit("input must include columns: sigma, t")

    if str(args.side).strip() and ("side" in df.columns):
        df = df[df["side"].astype(str) == str(args.side)].copy()

    df["sigma"] = pd.to_numeric(df["sigma"], errors="coerce")
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df = df.dropna(subset=["sigma", "t"]).copy()

    if math.isfinite(float(args.sigma)):
        target = float(args.sigma)
        sigs = pd.to_numeric(df["sigma"], errors="coerce")
        sigs = sigs[np.isfinite(sigs)]
        if sigs.empty:
            raise SystemExit("no finite sigma values")
        nearest = float(sigs.iloc[(sigs - target).abs().argmin()])
        df = df[(df["sigma"] - nearest).abs() <= float(args.sigma_tol)].copy()
        if df.empty:
            raise SystemExit("no rows matched --sigma")

    df = df.sort_values(["sigma", "t"]).reset_index(drop=True)

    mp.mp.dps = int(max(50, args.dps))

    primes: list[int] = []
    if int(args.p_max) > 0:
        primes = _primes_upto(int(args.p_max))

    # Compute hosts rowwise (unique caching keeps it fast enough for ~1e3 rows).
    cache: dict[tuple[float, float], tuple[complex, complex | None]] = {}
    phi_mod_vals: list[complex] = []
    phiP_vals: list[complex] = []

    for sigma_val, t_val in zip(df["sigma"].to_numpy(dtype=float), df["t"].to_numpy(dtype=float), strict=True):
        key = (float(sigma_val), float(t_val))
        got = cache.get(key)
        if got is None:
            s = mp.mpc(sigma_val, t_val)
            pm = phi_mod(s)
            pp = phi_P(s, primes) if primes else None
            got = (complex(pm), complex(pp) if pp is not None else None)
            cache[key] = got
        pm_c, pp_c = got
        phi_mod_vals.append(complex(pm_c))
        if primes:
            phiP_vals.append(complex(pp_c) if pp_c is not None else complex("nan"))

    df["phi_mod_re"] = np.real(np.asarray(phi_mod_vals, dtype=np.complex128))
    df["phi_mod_im"] = np.imag(np.asarray(phi_mod_vals, dtype=np.complex128))

    if primes:
        df["phi_P_re"] = np.real(np.asarray(phiP_vals, dtype=np.complex128))
        df["phi_P_im"] = np.imag(np.asarray(phiP_vals, dtype=np.complex128))
        df["phi_P_pmax"] = int(args.p_max)

    prefixes = [p.strip() for p in str(args.lambda_prefixes).split(",") if p.strip()]
    if prefixes:
        phi_mod_z = np.asarray(phi_mod_vals, dtype=np.complex128)
        phiP_z = np.asarray(phiP_vals, dtype=np.complex128) if primes else None

        for pref in prefixes:
            cols = _infer_cols(df, pref)
            lam = _to_complex(df, cols)

            Qm = lam / phi_mod_z
            df[f"Qmod_{pref}_re"] = np.real(Qm)
            df[f"Qmod_{pref}_im"] = np.imag(Qm)

            if int(args.normalize_basepoint) == 1:
                base = Qm[0]
                if not (np.isfinite(np.real(base)) and np.isfinite(np.imag(base)) and abs(base) > 0):
                    base = complex(1.0)
                Qm_n = Qm / base
                df[f"QmodN_{pref}_re"] = np.real(Qm_n)
                df[f"QmodN_{pref}_im"] = np.imag(Qm_n)

            if phiP_z is not None:
                Qp = lam / phiP_z
                df[f"QP_{pref}_re"] = np.real(Qp)
                df[f"QP_{pref}_im"] = np.imag(Qp)
                if int(args.normalize_basepoint) == 1:
                    basep = Qp[0]
                    if not (np.isfinite(np.real(basep)) and np.isfinite(np.imag(basep)) and abs(basep) > 0):
                        basep = complex(1.0)
                    Qp_n = Qp / basep
                    df[f"QPN_{pref}_re"] = np.real(Qp_n)
                    df[f"QPN_{pref}_im"] = np.imag(Qp_n)

        # Print quick dispersions as a sanity check.
        for pref in prefixes:
            Qn = (df[f"QmodN_{pref}_re"].to_numpy(dtype=float) + 1j * df[f"QmodN_{pref}_im"].to_numpy(dtype=float)).astype(
                np.complex128
            )
            print(f"QmodN dispersion {pref}: {_rel_dispersion(Qn):.6g}")
            if primes:
                Qpn = (df[f"QPN_{pref}_re"].to_numpy(dtype=float) + 1j * df[f"QPN_{pref}_im"].to_numpy(dtype=float)).astype(
                    np.complex128
                )
                print(f"QPN dispersion {pref}: {_rel_dispersion(Qpn):.6g}")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
