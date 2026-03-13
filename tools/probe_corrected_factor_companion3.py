from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True)
class GenericPacket:
    p: int
    ell: float
    S: np.ndarray


def _parse_int_csv(s: str) -> list[int]:
    out: list[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
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

    def V_of_n(n: int) -> float:
        X, _, _ = sim._local_blocks_for_prime_power(
            2,
            int(round(math.log2(int(n)))),
            sharp_mode=str(sharp),
            x_mode=str(x_mode),
            x_gamma=float(x_gamma),
            x_shear=float(u),
            x_lower=float(v),
            p_mode="p",
        )
        A = sim._cayley(np.asarray(X, dtype=np.complex128))
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


def _companion3_from_coeffs(A1: float, A2: float, A3: float) -> np.ndarray:
    # Characteristic polynomial lambda^3 + A1 lambda^2 + A2 lambda + A3.
    # Then det(I - z C) = 1 + A1 z + A2 z^2 + A3 z^3.
    return np.array(
        [
            [0.0, 0.0, -float(A3)],
            [1.0, 0.0, -float(A2)],
            [0.0, 1.0, -float(A1)],
        ],
        dtype=np.complex128,
    )


def _global_det_line_generic(*, packets: list[GenericPacket], sigma: float, t_grid: np.ndarray) -> pd.DataFrame:
    rows: list[dict] = []
    for t in np.asarray(t_grid, dtype=float):
        s = complex(float(sigma), float(t))
        dim = int(sum(pkt.S.shape[0] for pkt in packets))
        K = np.zeros((dim, dim), dtype=np.complex128)
        ofs = 0
        for pkt in packets:
            d = int(pkt.S.shape[0])
            a = complex(np.exp(-s * float(pkt.ell)))
            K[ofs : ofs + d, ofs : ofs + d] = (a * np.asarray(pkt.S, dtype=np.complex128)).astype(np.complex128)
            ofs += d
        D = complex(np.linalg.det(np.eye(dim, dtype=np.complex128) - K))
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
            "Rebuild the prime-local frontend as a 3x3 companion packet that realizes the corrected A1,A2,A3 exactly, "
            "and export determinant lines for comparison against the existing frontends."
        )
    )
    ap.add_argument("--fit_csv", default="out/ghostlift_closure_beta23_mumix050_fit17_eval29_hold19_23.csv")
    ap.add_argument("--model", default="beta23_plus_c")
    ap.add_argument("--u", type=float, default=0.2)
    ap.add_argument("--primes_global", default="2,3,5,7,11,13,17,19,23,29")
    ap.add_argument("--dyadic_k_max", type=int, default=12)
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
    ap.add_argument("--out_prefix", default="out/corrected_factor_companion3_u020")
    args = ap.parse_args()

    fit_path = Path(str(args.fit_csv))
    if not fit_path.exists():
        raise SystemExit(f"missing --fit_csv: {fit_path}")
    df_fit = pd.read_csv(fit_path)
    df_fit = df_fit[(df_fit["model"].astype(str) == str(args.model)) & np.isclose(df_fit["u"].astype(float), float(args.u), atol=1e-12)].copy()
    if df_fit.empty:
        raise SystemExit(f"no row with model={args.model} at u={args.u}")
    fit_row = df_fit.iloc[0]

    boundary_parts = sim._parse_int_list(str(args.boundary))
    if len(boundary_parts) != 2:
        raise SystemExit("--boundary must be i,j")
    boundary = (int(boundary_parts[0]), int(boundary_parts[1]))

    Acoeff = _compute_corrected_coeffs(
        u=float(args.u),
        dyadic_k_max=int(args.dyadic_k_max),
        coeff_k_max=3,
        beta2=float(fit_row.get("beta2", fit_row.get("beta", 0.0))),
        beta3=float(fit_row.get("beta3", fit_row.get("beta", 0.0))),
        boundary=boundary,
        schur_sign=str(args.schur_sign),
        scattering=str(args.scattering),
        sharp=str(args.sharp),
        x_mode=str(args.X_mode),
        x_gamma=float(args.X_gamma),
    )

    S_comp = _companion3_from_coeffs(float(Acoeff[1]), float(Acoeff[2]), float(Acoeff[3]))
    coeff_exact = {
        "A1_star": float(Acoeff[1]),
        "A2_star": float(Acoeff[2]),
        "A3_star": float(Acoeff[3]),
        "trace_comp_re": float(np.real(np.trace(S_comp))),
        "trace_comp_im": float(np.imag(np.trace(S_comp))),
        "det_comp_re": float(np.real(np.linalg.det(S_comp))),
        "det_comp_im": float(np.imag(np.linalg.det(S_comp))),
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(S_comp)))),
    }

    primes_global = [int(p) for p in _parse_int_csv(str(args.primes_global))]
    packets_comp = [
        GenericPacket(p=int(p), ell=float(math.log(float(p))), S=np.asarray(S_comp, dtype=np.complex128))
        for p in primes_global
    ]

    t_grid = np.linspace(float(args.t_min), float(args.t_max), int(args.n_t), dtype=float)
    det_comp = _global_det_line_generic(packets=packets_comp, sigma=float(args.sigma), t_grid=t_grid)
    det_comp["u"] = float(args.u)
    det_comp["model"] = "companion3_exact"

    out_prefix = Path(str(args.out_prefix))
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    coeff_path = Path(str(out_prefix) + "_coefficients.csv")
    det_path = Path(str(out_prefix) + "_det.csv")

    pd.DataFrame([coeff_exact]).to_csv(coeff_path, index=False)
    det_comp.to_csv(det_path, index=False)

    print(f"wrote {coeff_path}")
    print(f"wrote {det_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())