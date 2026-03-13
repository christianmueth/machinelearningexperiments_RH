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
) -> tuple[float, float, float, float, float, float]:
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
    return float(Acoeff[1]), float(Acoeff[2]), float(Acoeff[3]), float(c_val), float(beta2), float(beta3)


def _coeffs_from_eigs(eigs: np.ndarray) -> tuple[complex, complex, complex]:
    eigs = np.asarray(eigs, dtype=np.complex128)
    e1 = complex(np.sum(eigs))
    e2 = complex(eigs[0] * eigs[1] + eigs[0] * eigs[2] + eigs[1] * eigs[2])
    e3 = complex(eigs[0] * eigs[1] * eigs[2])
    return -e1, e2, -e3


def _to_disk(z: complex, radius_max: float) -> complex:
    r = abs(z)
    if r <= float(radius_max):
        return complex(z)
    return complex(float(radius_max) * np.exp(1j * np.angle(z)))


def _eigs_from_chart(chart: str, z1: complex, z2: complex, *, A1: float, A2: float, A3: float, radius_max: float) -> np.ndarray | None:
    z1 = complex(z1)
    z2 = complex(z2)
    eps = 1e-10
    if chart == "A1_exact":
        z3 = complex(-float(A1), 0.0) - z1 - z2
    elif chart == "A2_exact":
        denom = z1 + z2
        if abs(denom) <= eps:
            return None
        z3 = (complex(float(A2), 0.0) - z1 * z2) / denom
    elif chart == "A3_exact":
        denom = z1 * z2
        if abs(denom) <= eps:
            return None
        z3 = -complex(float(A3), 0.0) / denom
    else:
        raise ValueError(f"unknown chart: {chart}")

    eigs = np.asarray([
        _to_disk(z1, float(radius_max)),
        _to_disk(z2, float(radius_max)),
        _to_disk(z3, float(radius_max)),
    ], dtype=np.complex128)
    return eigs


def _score_eigs(eigs: np.ndarray, *, A1: float, A2: float, A3: float, chart: str) -> tuple[float, dict[str, float]]:
    B1, B2, B3 = _coeffs_from_eigs(np.asarray(eigs, dtype=np.complex128))
    t1 = complex(float(A1), 0.0)
    t2 = complex(float(A2), 0.0)
    t3 = complex(float(A3), 0.0)
    err1 = float(abs(B1 - t1))
    err2 = float(abs(B2 - t2))
    err3 = float(abs(B3 - t3))

    # Exactness chart gets the corresponding error prioritized, but still score remaining mismatch.
    if chart == "A1_exact":
        score = (0.25 * err1) + (1.25 * err2) + (1.0 * err3)
    elif chart == "A2_exact":
        score = (1.25 * err1) + (0.25 * err2) + (1.0 * err3)
    elif chart == "A3_exact":
        score = (1.25 * err1) + (1.0 * err2) + (0.25 * err3)
    else:
        score = (1.0 * err1) + (1.0 * err2) + (1.0 * err3)

    meta = {
        "A1_fit_re": float(np.real(B1)),
        "A1_fit_im": float(np.imag(B1)),
        "A2_fit_re": float(np.real(B2)),
        "A2_fit_im": float(np.imag(B2)),
        "A3_fit_re": float(np.real(B3)),
        "A3_fit_im": float(np.imag(B3)),
        "err1": float(err1),
        "err2": float(err2),
        "err3": float(err3),
        "spectral_radius": float(np.max(np.abs(eigs))),
        "sum_abs_eigs": float(np.sum(np.abs(eigs))),
    }
    return float(score), meta


def _search_chart(
    *,
    chart: str,
    A1: float,
    A2: float,
    A3: float,
    radius_max: float,
    n_random: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(int(seed))
    best_eigs: np.ndarray | None = None
    best_meta: dict[str, float] | None = None
    best_score = float("inf")

    def sample_z() -> complex:
        r = float(radius_max) * math.sqrt(float(rng.random()))
        theta = float(rng.uniform(-math.pi, math.pi))
        return complex(r * math.cos(theta), r * math.sin(theta))

    for _ in range(int(n_random)):
        z1 = sample_z()
        z2 = sample_z()
        eigs = _eigs_from_chart(str(chart), z1, z2, A1=float(A1), A2=float(A2), A3=float(A3), radius_max=float(radius_max))
        if eigs is None:
            continue
        score, meta = _score_eigs(eigs, A1=float(A1), A2=float(A2), A3=float(A3), chart=str(chart))
        if score < best_score:
            best_score = float(score)
            best_eigs = np.asarray(eigs, dtype=np.complex128)
            best_meta = dict(meta)

    if best_eigs is None or best_meta is None:
        raise RuntimeError(f"failed to fit chart={chart}")

    best_meta = dict(best_meta)
    best_meta["score"] = float(best_score)
    best_meta["chart"] = str(chart)
    for i, z in enumerate(best_eigs.tolist(), start=1):
        best_meta[f"eig{i}_re"] = float(np.real(z))
        best_meta[f"eig{i}_im"] = float(np.imag(z))
    return np.asarray(best_eigs, dtype=np.complex128), best_meta


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
            "Fit three exactness charts (A1 exact, A2 exact, A3 exact) in a normal 3x3 spectral host, "
            "and select a best superstructure chart for corrected local coefficients."
        )
    )
    ap.add_argument("--fit_csv", default="out/ghostlift_closure_beta23_mumix050_fit17_eval29_hold19_23.csv")
    ap.add_argument("--model", default="beta23_plus_c")
    ap.add_argument("--u", type=float, default=0.2)
    ap.add_argument("--dyadic_k_max", type=int, default=12)
    ap.add_argument("--radius_max", type=float, default=0.98)
    ap.add_argument("--n_random", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
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
    ap.add_argument("--out_prefix", default="out/exact_Aj_superstructure_u020")
    args = ap.parse_args()

    fit_path = Path(str(args.fit_csv))
    if not fit_path.exists():
        raise SystemExit(f"missing --fit_csv: {fit_path}")

    boundary_parts = sim._parse_int_list(str(args.boundary))
    if len(boundary_parts) != 2:
        raise SystemExit("--boundary must be i,j")
    boundary = (int(boundary_parts[0]), int(boundary_parts[1]))

    A1, A2, A3, c_val, beta2, beta3 = _compute_corrected_coeffs(
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

    chart_rows: list[dict] = []
    det_rows: list[pd.DataFrame] = []
    t_grid = np.linspace(float(args.t_min), float(args.t_max), int(args.n_t), dtype=float)
    primes_global = [int(p) for p in _parse_int_csv(str(args.primes_global))]

    best_super_score = float("inf")
    best_super_chart = ""
    best_super_eigs: np.ndarray | None = None
    best_super_meta: dict[str, float] | None = None

    for idx, chart in enumerate(["A1_exact", "A2_exact", "A3_exact"]):
        eigs, meta = _search_chart(
            chart=str(chart),
            A1=float(A1),
            A2=float(A2),
            A3=float(A3),
            radius_max=float(args.radius_max),
            n_random=int(args.n_random),
            seed=int(args.seed) + idx,
        )
        meta = dict(meta)
        meta.update(
            {
                "u": float(args.u),
                "A1_star": float(A1),
                "A2_star": float(A2),
                "A3_star": float(A3),
                "c": float(c_val),
                "beta2": float(beta2),
                "beta3": float(beta3),
            }
        )
        chart_rows.append(meta)

        packets = [
            GenericPacket(p=int(p), ell=float(math.log(float(p))), S=np.diag(np.asarray(eigs, dtype=np.complex128)))
            for p in primes_global
        ]
        det_df = _global_det_line_generic(packets=packets, sigma=float(args.sigma), t_grid=t_grid)
        det_df["chart"] = str(chart)
        det_rows.append(det_df)

        # Superstructure selector: choose the chart with best balanced sum of coefficient errors.
        super_score = float(meta["err1"] + meta["err2"] + meta["err3"])
        if super_score < best_super_score:
            best_super_score = float(super_score)
            best_super_chart = str(chart)
            best_super_eigs = np.asarray(eigs, dtype=np.complex128)
            best_super_meta = dict(meta)

    if best_super_eigs is None or best_super_meta is None:
        raise RuntimeError("failed to choose superstructure chart")

    # Superstructure output: best chart selector among exactness charts.
    super_packets = [
        GenericPacket(p=int(p), ell=float(math.log(float(p))), S=np.diag(np.asarray(best_super_eigs, dtype=np.complex128)))
        for p in primes_global
    ]
    det_super = _global_det_line_generic(packets=super_packets, sigma=float(args.sigma), t_grid=t_grid)
    det_super["chart"] = "superstructure"
    det_rows.append(det_super)

    super_row = dict(best_super_meta)
    super_row["chart"] = "superstructure"
    super_row["selected_chart"] = str(best_super_chart)
    super_row["selection_score"] = float(best_super_score)
    chart_rows.append(super_row)

    out_prefix = Path(str(args.out_prefix))
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    coeff_path = Path(str(out_prefix) + "_coefficients.csv")
    det_path = Path(str(out_prefix) + "_det.csv")

    pd.DataFrame(chart_rows).to_csv(coeff_path, index=False)
    pd.concat(det_rows, axis=0, ignore_index=True).to_csv(det_path, index=False)

    print(f"wrote {coeff_path}")
    print(f"wrote {det_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
