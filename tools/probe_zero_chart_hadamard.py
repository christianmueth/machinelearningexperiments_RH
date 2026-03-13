from __future__ import annotations

import argparse
import itertools
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

import probe_exact_Aj_superstructure as charts  # type: ignore
import six_by_six_prime_tower_sim as sim  # type: ignore


def _zero_targets(chart: str, *, A1: float, A2: float, A3: float) -> tuple[float, float, float]:
    if chart == "A1_exact":
        return 0.0, float(A2), float(A3)
    if chart == "A2_exact":
        return float(A1), 0.0, float(A3)
    if chart == "A3_exact":
        return float(A1), float(A2), 0.0
    raise ValueError(f"unknown chart: {chart}")


def _canonicalize_eigs(eigs: np.ndarray) -> np.ndarray:
    eigs = np.asarray(eigs, dtype=np.complex128)
    order = sorted(
        range(eigs.size),
        key=lambda idx: (
            -float(abs(eigs[idx])),
            float(np.angle(eigs[idx])),
            float(np.real(eigs[idx])),
            float(np.imag(eigs[idx])),
        ),
    )
    return np.asarray([eigs[idx] for idx in order], dtype=np.complex128)


def _coeff_meta(eigs: np.ndarray, *, A1: float, A2: float, A3: float) -> dict[str, float]:
    B1, B2, B3 = charts._coeffs_from_eigs(np.asarray(eigs, dtype=np.complex128))
    t1 = complex(float(A1), 0.0)
    t2 = complex(float(A2), 0.0)
    t3 = complex(float(A3), 0.0)
    err1 = float(abs(B1 - t1))
    err2 = float(abs(B2 - t2))
    err3 = float(abs(B3 - t3))
    return {
        "A1_fit_re": float(np.real(B1)),
        "A1_fit_im": float(np.imag(B1)),
        "A2_fit_re": float(np.real(B2)),
        "A2_fit_im": float(np.imag(B2)),
        "A3_fit_re": float(np.real(B3)),
        "A3_fit_im": float(np.imag(B3)),
        "err1": float(err1),
        "err2": float(err2),
        "err3": float(err3),
        "balanced_score": float(err1 + err2 + err3),
        "spectral_radius": float(np.max(np.abs(eigs))),
        "sum_abs_eigs": float(np.sum(np.abs(eigs))),
    }


def _best_hadamard_product(
    *,
    eigs_A1: np.ndarray,
    eigs_A2: np.ndarray,
    eigs_A3: np.ndarray,
    A1: float,
    A2: float,
    A3: float,
) -> tuple[np.ndarray, dict[str, float]]:
    base = _canonicalize_eigs(np.asarray(eigs_A1, dtype=np.complex128))
    solve2 = _canonicalize_eigs(np.asarray(eigs_A2, dtype=np.complex128))
    solve3 = _canonicalize_eigs(np.asarray(eigs_A3, dtype=np.complex128))

    best_score = float("inf")
    best_prod: np.ndarray | None = None
    best_meta: dict[str, float] | None = None

    for perm2 in itertools.permutations(range(3)):
        for perm3 in itertools.permutations(range(3)):
            prod = base * solve2[list(perm2)] * solve3[list(perm3)]
            meta = _coeff_meta(prod, A1=float(A1), A2=float(A2), A3=float(A3))
            score = float(meta["balanced_score"])
            if score < best_score:
                best_score = float(score)
                best_prod = np.asarray(prod, dtype=np.complex128)
                best_meta = dict(meta)
                best_meta["perm2"] = "".join(str(int(i) + 1) for i in perm2)
                best_meta["perm3"] = "".join(str(int(i) + 1) for i in perm3)

    if best_prod is None or best_meta is None:
        raise RuntimeError("failed to form Hadamard product")

    for i, z in enumerate(best_prod.tolist(), start=1):
        best_meta[f"eig{i}_re"] = float(np.real(z))
        best_meta[f"eig{i}_im"] = float(np.imag(z))
    return np.asarray(best_prod, dtype=np.complex128), best_meta


def _write_det(
    *,
    eigs: np.ndarray,
    out_path: Path,
    sigma: float,
    t_grid: np.ndarray,
    primes_global: list[int],
) -> None:
    packets = [
        charts.GenericPacket(p=int(p), ell=float(math.log(float(p))), S=np.diag(np.asarray(eigs, dtype=np.complex128)))
        for p in primes_global
    ]
    det_df = charts._global_det_line_generic(packets=packets, sigma=float(sigma), t_grid=t_grid)
    det_df.to_csv(out_path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Solve the A1/A2/A3 exactness charts with zeroed target coordinates, then form the best-aligned Hadamard product of the three solves."
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
    ap.add_argument("--out_prefix", default="out/zero_chart_hadamard_u020")
    args = ap.parse_args()

    fit_path = Path(str(args.fit_csv))
    if not fit_path.exists():
        raise SystemExit(f"missing --fit_csv: {fit_path}")

    boundary_parts = sim._parse_int_list(str(args.boundary))
    if len(boundary_parts) != 2:
        raise SystemExit("--boundary must be i,j")
    boundary = (int(boundary_parts[0]), int(boundary_parts[1]))

    A1, A2, A3, c_val, beta2, beta3 = charts._compute_corrected_coeffs(
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

    out_prefix = Path(str(args.out_prefix))
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    t_grid = np.linspace(float(args.t_min), float(args.t_max), int(args.n_t), dtype=float)
    primes_global = [int(p) for p in charts._parse_int_csv(str(args.primes_global))]

    solve_rows: list[dict] = []
    solve_eigs: dict[str, np.ndarray] = {}
    for idx, chart in enumerate(["A1_exact", "A2_exact", "A3_exact"]):
        target_A1, target_A2, target_A3 = _zero_targets(str(chart), A1=float(A1), A2=float(A2), A3=float(A3))
        eigs, meta = charts._search_chart(
            chart=str(chart),
            A1=float(target_A1),
            A2=float(target_A2),
            A3=float(target_A3),
            radius_max=float(args.radius_max),
            n_random=int(args.n_random),
            seed=int(args.seed) + idx,
        )
        eigs = _canonicalize_eigs(np.asarray(eigs, dtype=np.complex128))
        solve_eigs[str(chart)] = np.asarray(eigs, dtype=np.complex128)
        row = {
            "kind": str(chart),
            "u": float(args.u),
            "A1_star": float(A1),
            "A2_star": float(A2),
            "A3_star": float(A3),
            "target_A1": float(target_A1),
            "target_A2": float(target_A2),
            "target_A3": float(target_A3),
            "c": float(c_val),
            "beta2": float(beta2),
            "beta3": float(beta3),
            **meta,
        }
        for i, z in enumerate(eigs.tolist(), start=1):
            row[f"eig{i}_re"] = float(np.real(z))
            row[f"eig{i}_im"] = float(np.imag(z))
        solve_rows.append(row)
        det_path = Path(str(out_prefix) + f"_{chart}_det.csv")
        _write_det(eigs=eigs, out_path=det_path, sigma=float(args.sigma), t_grid=t_grid, primes_global=primes_global)

    hadamard_eigs, hadamard_meta = _best_hadamard_product(
        eigs_A1=solve_eigs["A1_exact"],
        eigs_A2=solve_eigs["A2_exact"],
        eigs_A3=solve_eigs["A3_exact"],
        A1=float(A1),
        A2=float(A2),
        A3=float(A3),
    )
    hadamard_row = {
        "kind": "hadamard_product",
        "u": float(args.u),
        "A1_star": float(A1),
        "A2_star": float(A2),
        "A3_star": float(A3),
        "target_A1": float("nan"),
        "target_A2": float("nan"),
        "target_A3": float("nan"),
        "c": float(c_val),
        "beta2": float(beta2),
        "beta3": float(beta3),
        **hadamard_meta,
    }
    solve_rows.append(hadamard_row)
    hadamard_det_path = Path(str(out_prefix) + "_hadamard_det.csv")
    _write_det(eigs=hadamard_eigs, out_path=hadamard_det_path, sigma=float(args.sigma), t_grid=t_grid, primes_global=primes_global)

    coeff_path = Path(str(out_prefix) + "_coefficients.csv")
    pd.DataFrame(solve_rows).to_csv(coeff_path, index=False)

    print(f"wrote {coeff_path}")
    print(f"wrote {hadamard_det_path}")
    for chart in ["A1_exact", "A2_exact", "A3_exact"]:
        print(f"wrote {Path(str(out_prefix) + f'_{chart}_det.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())