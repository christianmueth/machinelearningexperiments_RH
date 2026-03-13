from __future__ import annotations

import argparse
import cmath
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import six_by_six_prime_tower_sim as sim  # type: ignore


def _parse_int_csv(s: str) -> list[int]:
    out: list[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _parse_float_csv(s: str) -> list[float]:
    out: list[float] = []
    for part in str(s).split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out


def _compute_corrected_coeffs(
    *,
    u: float,
    dyadic_k_max: int,
    coeff_k_max: int,
    beta2: float,
    beta3: float,
    boundary: tuple[int, int],
    schur_sign: str,
    scattering: str,
    sharp: str,
    x_mode: str,
    x_gamma: float,
) -> list[float]:
    v = -float(u)
    I2 = np.eye(2, dtype=np.complex128)
    X_cache: dict[int, np.ndarray] = {}

    def X_p1(p: int) -> np.ndarray:
        if p not in X_cache:
            X, _, _ = sim._local_blocks_for_prime_power(
                int(p),
                1,
                sharp_mode=str(sharp),
                x_mode=str(x_mode),
                x_gamma=float(x_gamma),
                x_shear=float(u),
                x_lower=float(v),
                p_mode="p",
            )
            X_cache[p] = np.asarray(X, dtype=np.complex128)
        return X_cache[p]

    def _prime_factorization(n: int) -> dict[int, int]:
        out: dict[int, int] = {}
        m = int(n)
        e = 0
        while m % 2 == 0:
            m //= 2
            e += 1
        if e:
            out[2] = e
        p = 3
        while p * p <= m:
            e = 0
            while m % p == 0:
                m //= p
                e += 1
            if e:
                out[int(p)] = e
            p += 2
        if m > 1:
            out[int(m)] = int(out.get(int(m), 0) + 1)
        return out

    def build_X_n(n: int) -> np.ndarray:
        f = _prime_factorization(int(n))
        Xn = np.eye(2, dtype=np.complex128)
        for p in sorted(f.keys()):
            e = int(f[p])
            Xn = (Xn @ np.linalg.matrix_power(X_p1(int(p)), e)).astype(np.complex128)
        return Xn

    def V_of_n(n: int) -> float:
        Xn = build_X_n(int(n))
        A = sim._cayley(Xn)
        Ash = sim._symplectic_partner(A, mode=str(sharp))
        B = sim._bulk_B_from_A(A, Ash)
        Lam = sim._schur_complement_Lambda(B, boundary=boundary, sign=str(schur_sign))
        S = sim._scattering_from_Lambda(Lam, mode=str(scattering))
        det_val = complex(np.linalg.det(I2 - np.asarray(S, dtype=np.complex128)))
        return float(math.log(max(1e-300, abs(det_val))))

    g = [0.0] * (int(dyadic_k_max) + 1)
    for m in range(1, int(dyadic_k_max) + 1):
        g[m] = float(V_of_n(int(2**m)))
        if m == 2:
            g[m] -= float(beta2)
        elif m >= 3:
            g[m] -= float(beta3)

    Acoeff = [0.0] * (int(coeff_k_max) + 1)
    Acoeff[0] = 1.0
    for k_pow in range(1, int(coeff_k_max) + 1):
        acc = 0.0
        for j in range(1, k_pow + 1):
            acc += float(g[j]) * float(Acoeff[k_pow - j])
        Acoeff[k_pow] = float(acc) / float(k_pow)
    return Acoeff


def _satake_from_A1_A2(A1: float, A2: float) -> tuple[complex, complex]:
    s1 = complex(float(A1), 0.0)
    s2 = complex(float(A1 * A1 - A2), 0.0)
    disc = s1 * s1 - 4.0 * s2
    root = cmath.sqrt(disc)
    alpha = 0.5 * (s1 + root)
    beta = 0.5 * (s1 - root)
    return complex(alpha), complex(beta)


def _coeffs_from_satake(alpha: complex, beta: complex, k_max: int) -> list[complex]:
    out = [complex(0.0, 0.0)] * (int(k_max) + 1)
    out[0] = complex(1.0, 0.0)
    if k_max >= 1:
        out[1] = complex(alpha + beta)
    if k_max >= 2:
        s1 = complex(alpha + beta)
        s2 = complex(alpha * beta)
        for k in range(2, int(k_max) + 1):
            out[k] = (s1 * out[k - 1]) - (s2 * out[k - 2])
    return out


def _compute_det_line(
    *,
    packets: list[sim.Packet],
    sigma: float,
    t_grid: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict] = []
    for t in np.asarray(t_grid, dtype=float):
        s = complex(float(sigma), float(t))
        K, _ = sim._global_K(s, packets)
        I = np.eye(K.shape[0], dtype=np.complex128)
        D = complex(np.linalg.det(I - K))
        rows.append(
            {
                "sigma": float(sigma),
                "t": float(t),
                "q_track_re": float(np.real(D)),
                "q_track_im": float(np.imag(D)),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Bridge the corrected dyadic Euler-closure back into the original prime-local pipeline. "
            "Exports corrected A_k*, derives Satake parameters from (A1,A2), compares local packets against sixby6, "
            "and writes determinant-line CSVs for Mangoldt probing."
        )
    )
    ap.add_argument("--fit_csv", default="out/ghostlift_closure_beta23_mumix050_fit17_eval29_hold19_23.csv")
    ap.add_argument("--model", default="beta23_plus_c")
    ap.add_argument("--us", default="0.2")
    ap.add_argument("--primes_global", default="2,3,5,7,11,13,17,19,23,29")
    ap.add_argument("--dyadic_k_max", type=int, default=12)
    ap.add_argument("--coeff_k_max", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--t_min", type=float, default=10.0)
    ap.add_argument("--t_max", type=float, default=30.0)
    ap.add_argument("--n_t", type=int, default=401)
    ap.add_argument("--boundary", default="0,3")
    ap.add_argument("--schur_sign", choices=["-", "+"], default="+")
    ap.add_argument("--scattering", choices=["lambda_pm_i", "i_pm_lambda"], default="i_pm_lambda")
    ap.add_argument("--sharp", choices=["transpose", "conj_transpose"], default="conj_transpose")
    ap.add_argument("--X_mode", choices=["raw", "det1"], default="det1")
    ap.add_argument("--X_gamma", type=float, default=1.0)
    ap.add_argument("--out_prefix", default="out/corrected_factor_injection")
    args = ap.parse_args()

    fit_path = Path(str(args.fit_csv))
    if not fit_path.exists():
        raise SystemExit(f"missing --fit_csv: {fit_path}")

    df_fit = pd.read_csv(fit_path)
    df_fit = df_fit[df_fit["model"].astype(str) == str(args.model)].copy()
    if df_fit.empty:
        raise SystemExit(f"no rows with model={args.model} in {fit_path}")

    us = [float(u) for u in _parse_float_csv(str(args.us))]
    if not us:
        raise SystemExit("--us must be non-empty")

    primes_global = [int(p) for p in _parse_int_csv(str(args.primes_global))]
    if not primes_global:
        raise SystemExit("--primes_global must be non-empty")

    boundary_parts = sim._parse_int_list(str(args.boundary))
    if len(boundary_parts) != 2:
        raise SystemExit("--boundary must be i,j")
    boundary = (int(boundary_parts[0]), int(boundary_parts[1]))

    t_grid = np.linspace(float(args.t_min), float(args.t_max), int(args.n_t), dtype=float)

    coeff_rows: list[dict] = []
    packet_rows: list[dict] = []
    det_rows: list[dict] = []

    for u in us:
        dfu = df_fit[np.isclose(df_fit["u"].astype(float), float(u), atol=1e-12)].copy()
        if dfu.empty:
            raise SystemExit(f"fit_csv missing model={args.model} row at u={u}")
        row = dfu.iloc[0]
        beta2 = float(row.get("beta2", row.get("beta", float("nan"))))
        beta3 = float(row.get("beta3", row.get("beta", float("nan"))))
        c_val = float(row.get("c", float("nan")))

        Acoeff = _compute_corrected_coeffs(
            u=float(u),
            dyadic_k_max=int(args.dyadic_k_max),
            coeff_k_max=int(args.coeff_k_max),
            beta2=float(beta2),
            beta3=float(beta3),
            boundary=boundary,
            schur_sign=str(args.schur_sign),
            scattering=str(args.scattering),
            sharp=str(args.sharp),
            x_mode=str(args.X_mode),
            x_gamma=float(args.X_gamma),
        )

        alpha, beta = _satake_from_A1_A2(float(Acoeff[1]), float(Acoeff[2]))
        Ac_sat = _coeffs_from_satake(alpha, beta, int(args.coeff_k_max))

        coeff_rows.append(
            {
                "u": float(u),
                "beta2": float(beta2),
                "beta3": float(beta3),
                "c": float(c_val),
                "A1_star": float(Acoeff[1]),
                "A2_star": float(Acoeff[2]),
                "A3_star": float(Acoeff[3]) if int(args.coeff_k_max) >= 3 else float("nan"),
                "alpha_re": float(np.real(alpha)),
                "alpha_im": float(np.imag(alpha)),
                "beta_re": float(np.real(beta)),
                "beta_im": float(np.imag(beta)),
                "A1_satake": float(np.real(Ac_sat[1])),
                "A2_satake": float(np.real(Ac_sat[2])),
                "A3_satake": float(np.real(Ac_sat[3])) if int(args.coeff_k_max) >= 3 else float("nan"),
                "A3_abs_err": float(abs(complex(Ac_sat[3]) - complex(float(Acoeff[3]), 0.0))) if int(args.coeff_k_max) >= 3 else float("nan"),
            }
        )

        satake_table = {int(p): (complex(alpha), complex(beta)) for p in primes_global}
        sat_packets = sim._build_packets(
            primes_global,
            1,
            local_model="satake",
            boundary=(0, 1),
            sign="-",
            sharp_mode=str(args.sharp),
            x_mode=str(args.X_mode),
            x_gamma=float(args.X_gamma),
            x_shear=float(u),
            x_lower=float(-float(u)),
            p_mode="p",
            scattering_mode=str(args.scattering),
            satake_family="table",
            satake_matrix="diag",
            theta_scale=1.0,
            seed=0,
            satake_table=satake_table,
        )
        six_packets = sim._build_packets(
            primes_global,
            1,
            local_model="sixby6",
            boundary=boundary,
            sign=str(args.schur_sign),
            sharp_mode=str(args.sharp),
            x_mode=str(args.X_mode),
            x_gamma=float(args.X_gamma),
            x_shear=float(u),
            x_lower=float(-float(u)),
            p_mode="p",
            scattering_mode=str(args.scattering),
            satake_family="trivial",
            satake_matrix="diag",
            theta_scale=1.0,
            seed=0,
        )

        by_p_sat = {int(pkt.p): pkt for pkt in sat_packets}
        by_p_six = {int(pkt.p): pkt for pkt in six_packets}
        for p in primes_global:
            pkt_sat = by_p_sat.get(int(p))
            pkt_six = by_p_six.get(int(p))
            if pkt_sat is None or pkt_six is None:
                continue
            Ssat = np.asarray(pkt_sat.S, dtype=np.complex128)
            Ssix = np.asarray(pkt_six.S, dtype=np.complex128)
            packet_rows.append(
                {
                    "u": float(u),
                    "p": int(p),
                    "trace_sat_re": float(np.real(np.trace(Ssat))),
                    "trace_sat_im": float(np.imag(np.trace(Ssat))),
                    "trace_six_re": float(np.real(np.trace(Ssix))),
                    "trace_six_im": float(np.imag(np.trace(Ssix))),
                    "trace_abs_diff": float(abs(complex(np.trace(Ssat)) - complex(np.trace(Ssix)))),
                    "det_sat_abs": float(abs(np.linalg.det(Ssat))),
                    "det_six_abs": float(abs(np.linalg.det(Ssix))),
                    "det_abs_diff": float(abs(complex(np.linalg.det(Ssat)) - complex(np.linalg.det(Ssix)))),
                    "fro_diff_S": float(np.linalg.norm(Ssat - Ssix, ord="fro")),
                }
            )

        det_sat = _compute_det_line(packets=sat_packets, sigma=float(args.sigma), t_grid=t_grid)
        det_sat["u"] = float(u)
        det_sat["model"] = "corrected_satake"
        det_rows.append(det_sat)

        det_six = _compute_det_line(packets=six_packets, sigma=float(args.sigma), t_grid=t_grid)
        det_six["u"] = float(u)
        det_six["model"] = "sixby6_p"
        det_rows.append(det_six)

    out_prefix = Path(str(args.out_prefix))
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    coeff_path = Path(str(out_prefix) + "_coefficients.csv")
    packet_path = Path(str(out_prefix) + "_packet_compare.csv")
    det_path = Path(str(out_prefix) + "_det_lines.csv")

    pd.DataFrame(coeff_rows).to_csv(coeff_path, index=False)
    pd.DataFrame(packet_rows).to_csv(packet_path, index=False)
    pd.concat(det_rows, axis=0, ignore_index=True).to_csv(det_path, index=False)

    print(f"wrote {coeff_path}")
    print(f"wrote {packet_path}")
    print(f"wrote {det_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())