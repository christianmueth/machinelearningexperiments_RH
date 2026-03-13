from __future__ import annotations

import argparse
import cmath
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


def _compute_corrected_coeffs(
    *,
    fit_csv: Path,
    model: str,
    u: float,
    dyadic_k_max: int,
    boundary: tuple[int, int],
    schur_sign: str,
    scattering: str,
    sharp: str,
    x_mode: str,
    x_gamma: float,
) -> tuple[float, float, float, float, float]:
    df = pd.read_csv(fit_csv)
    df = df[(df["model"].astype(str) == str(model)) & np.isclose(df["u"].astype(float), float(u), atol=1e-12)].copy()
    if df.empty:
        raise SystemExit(f"no row with model={model} at u={u}")
    row = df.iloc[0]
    beta2 = float(row.get("beta2", row.get("beta", 0.0)))
    beta3 = float(row.get("beta3", row.get("beta", 0.0)))
    c_val = float(row.get("c", 0.0))

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

    Acoeff = [0.0] * 4
    Acoeff[0] = 1.0
    for k_pow in range(1, 4):
        acc = 0.0
        for j in range(1, k_pow + 1):
            acc += float(g[j]) * float(Acoeff[k_pow - j])
        Acoeff[k_pow] = float(acc) / float(k_pow)
    return float(Acoeff[1]), float(Acoeff[2]), float(Acoeff[3]), float(c_val), float(beta3)


def _poly_coeffs_from_eigs(eigs: list[complex]) -> tuple[complex, complex, complex]:
    e1 = eigs[0] + eigs[1] + eigs[2]
    e2 = eigs[0] * eigs[1] + eigs[0] * eigs[2] + eigs[1] * eigs[2]
    e3 = eigs[0] * eigs[1] * eigs[2]
    # det(I - zS) = 1 - e1 z + e2 z^2 - e3 z^3
    return -e1, e2, -e3


def _fit_damped_unitary3(A1: float, A2: float, A3: float, *, theta_steps: int) -> tuple[np.ndarray, dict[str, float]]:
    target3 = complex(float(A3), 0.0)
    r = float(max(1e-8, abs(target3) ** (1.0 / 3.0)))
    arg3 = float(cmath.phase(-target3))

    best: dict[str, float] | None = None
    best_S: np.ndarray | None = None

    theta_grid = np.linspace(-math.pi, math.pi, int(theta_steps), dtype=float)
    for t1 in theta_grid:
        for t2 in theta_grid:
            t3 = float(arg3 - t1 - t2)
            eigs = [
                complex(r * np.exp(1j * t1)),
                complex(r * np.exp(1j * t2)),
                complex(r * np.exp(1j * t3)),
            ]
            B1, B2, B3 = _poly_coeffs_from_eigs(eigs)
            err1 = float(abs(B1 - complex(float(A1), 0.0)))
            err2 = float(abs(B2 - complex(float(A2), 0.0)))
            err3 = float(abs(B3 - target3))
            score = float(err1 + err2 + 0.5 * err3)
            if (best is None) or (score < float(best["score"])):
                best = {
                    "score": float(score),
                    "theta1": float(t1),
                    "theta2": float(t2),
                    "theta3": float(t3),
                    "r": float(r),
                    "A1_fit_re": float(np.real(B1)),
                    "A1_fit_im": float(np.imag(B1)),
                    "A2_fit_re": float(np.real(B2)),
                    "A2_fit_im": float(np.imag(B2)),
                    "A3_fit_re": float(np.real(B3)),
                    "A3_fit_im": float(np.imag(B3)),
                    "err1": float(err1),
                    "err2": float(err2),
                    "err3": float(err3),
                }
                best_S = np.diag(np.asarray(eigs, dtype=np.complex128))

    if best is None or best_S is None:
        raise RuntimeError("failed to fit damped unitary3")
    return np.asarray(best_S, dtype=np.complex128), best


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
            "Build a geometry-constrained damped-unitary 3x3 local packet from corrected A1,A2,A3, "
            "then export its determinant line for comparison against other frontends."
        )
    )
    ap.add_argument("--fit_csv", default="out/ghostlift_closure_beta23_mumix050_fit17_eval29_hold19_23.csv")
    ap.add_argument("--model", default="beta23_plus_c")
    ap.add_argument("--u", type=float, default=0.2)
    ap.add_argument("--dyadic_k_max", type=int, default=12)
    ap.add_argument("--theta_steps", type=int, default=181)
    ap.add_argument("--primes_global", default="2,3,5,7,11,13,17,19,23,29")
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
    ap.add_argument("--out_prefix", default="out/corrected_factor_damped_unitary3_u020")
    args = ap.parse_args()

    fit_path = Path(str(args.fit_csv))
    if not fit_path.exists():
        raise SystemExit(f"missing --fit_csv: {fit_path}")

    boundary_parts = sim._parse_int_list(str(args.boundary))
    if len(boundary_parts) != 2:
        raise SystemExit("--boundary must be i,j")
    boundary = (int(boundary_parts[0]), int(boundary_parts[1]))

    A1, A2, A3, c_val, beta3 = _compute_corrected_coeffs(
        fit_csv=fit_path,
        model=str(args.model),
        u=float(args.u),
        dyadic_k_max=int(args.dyadic_k_max),
        boundary=boundary,
        schur_sign=str(args.schur_sign),
        scattering=str(args.scattering),
        sharp=str(args.sharp),
        x_mode=str(args.X_mode),
        x_gamma=float(args.X_gamma),
    )

    S_fit, meta = _fit_damped_unitary3(float(A1), float(A2), float(A3), theta_steps=int(args.theta_steps))

    primes_global = [int(p) for p in str(args.primes_global).split(",") if str(p).strip()]
    packets = [
        GenericPacket(p=int(p), ell=float(math.log(float(p))), S=np.asarray(S_fit, dtype=np.complex128))
        for p in primes_global
    ]

    t_grid = np.linspace(float(args.t_min), float(args.t_max), int(args.n_t), dtype=float)
    det_df = _global_det_line_generic(packets=packets, sigma=float(args.sigma), t_grid=t_grid)

    out_prefix = Path(str(args.out_prefix))
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    coeff_path = Path(str(out_prefix) + "_coefficients.csv")
    det_path = Path(str(out_prefix) + "_det.csv")

    coeff_row = {
        "u": float(args.u),
        "A1_star": float(A1),
        "A2_star": float(A2),
        "A3_star": float(A3),
        "c": float(c_val),
        "beta3": float(beta3),
        **meta,
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(S_fit)))),
    }
    pd.DataFrame([coeff_row]).to_csv(coeff_path, index=False)
    det_df.to_csv(det_path, index=False)

    print(f"wrote {coeff_path}")
    print(f"wrote {det_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())