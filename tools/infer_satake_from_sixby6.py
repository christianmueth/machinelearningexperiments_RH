from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Import simulator module from sibling file (tools/ is not a package).
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
import six_by_six_prime_tower_sim as sim  # type: ignore


def _complex_cols(prefix: str, z: complex) -> dict[str, float]:
    z = complex(z)
    return {
        f"{prefix}_re": float(np.real(z)),
        f"{prefix}_im": float(np.imag(z)),
    }


def _eigvals_2x2(M: np.ndarray) -> tuple[complex, complex]:
    w = np.linalg.eigvals(np.asarray(M, dtype=np.complex128))
    if w.size != 2:
        raise ValueError("expected 2 eigenvalues")
    a, b = complex(w[0]), complex(w[1])

    # Stable ordering: by argument then magnitude.
    def key(z: complex) -> tuple[float, float]:
        ang = float(math.atan2(float(np.imag(z)), float(np.real(z))))
        return (ang, float(abs(z)))

    if key(b) < key(a):
        a, b = b, a
    return a, b


def _trace_powers(S: np.ndarray, r_max: int) -> list[complex]:
    S = np.asarray(S, dtype=np.complex128)
    out: list[complex] = []
    P = np.eye(2, dtype=np.complex128)
    for _r in range(1, int(r_max) + 1):
        P = (P @ S).astype(np.complex128)
        out.append(complex(np.trace(P)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Infer effective Satake parameters (alpha_p,beta_p)=eig(S_p) from the sixby6 local model, "
            "optionally sweeping a deformation gamma in X_p, and emit a per-prime CSV with invariants and ladders."
        )
    )
    ap.add_argument("--primes", default="2,3,5,7,11,13,17,19,23,29,31,37,41,43,47")
    ap.add_argument("--out_csv", required=True)

    ap.add_argument("--boundary", default="auto", help="Boundary indices i,j in 0..5 or auto")
    ap.add_argument("--schur_sign", choices=["-", "+"], default="-")
    ap.add_argument("--sharp", choices=["transpose", "conj_transpose"], default="conj_transpose")
    ap.add_argument("--X_mode", choices=["raw", "det1"], default="det1")
    ap.add_argument("--X_shear", type=float, default=0.0)
    ap.add_argument("--X_lower", type=float, default=0.0)
    ap.add_argument("--scattering", choices=["lambda_pm_i", "i_pm_lambda"], default="i_pm_lambda")

    ap.add_argument("--gamma", type=float, default=float("nan"), help="If finite, run only this gamma")
    ap.add_argument(
        "--gamma_list",
        default="",
        help="Optional comma list of gammas to sweep (overrides --gamma if provided)",
    )

    ap.add_argument("--r_max", type=int, default=8)

    args = ap.parse_args()

    primes = sim._parse_int_list(args.primes)
    if not primes:
        raise SystemExit("--primes must be non-empty")

    gammas: list[float] = []
    if str(args.gamma_list).strip():
        gammas = [float(x) for x in str(args.gamma_list).split(",") if str(x).strip()]
    elif math.isfinite(float(args.gamma)):
        gammas = [float(args.gamma)]
    else:
        gammas = [1.0]

    rows: list[dict] = []

    for gamma in gammas:
        boundary_arg = str(args.boundary).strip().lower()
        if boundary_arg == "auto":
            p_seed = [int(primes[0]), int(primes[len(primes) // 2]), int(primes[-1])] if len(primes) >= 3 else [int(primes[0])]
            k_seed = [1]
            boundary, schur_sign, hd0, ud0 = sim._boundary_search(
                p_seed,
                k_seed,
                sharp_mode=str(args.sharp),
                x_mode=str(args.X_mode),
                x_gamma=float(gamma),
                x_shear=float(args.X_shear),
                x_lower=float(args.X_lower),
                scattering_mode=str(args.scattering),
            )
        else:
            parts = sim._parse_int_list(str(args.boundary))
            if len(parts) != 2:
                raise SystemExit("--boundary must be auto or i,j")
            boundary = (int(parts[0]), int(parts[1]))
            schur_sign = str(args.schur_sign)
            hd0, ud0 = float("nan"), float("nan")

        packets = sim._build_packets(
            primes,
            1,
            local_model="sixby6",
            boundary=boundary,
            sign=schur_sign,
            sharp_mode=str(args.sharp),
            x_mode=str(args.X_mode),
            x_gamma=float(gamma),
            x_shear=float(args.X_shear),
            x_lower=float(args.X_lower),
            scattering_mode=str(args.scattering),
            satake_family="trivial",
            satake_matrix="diag",
            theta_scale=1.0,
            seed=0,
        )

        by_p = {pkt.p: pkt for pkt in packets}
        for p in primes:
            pkt = by_p.get(int(p))
            if pkt is None:
                continue
            S = np.asarray(pkt.S, dtype=np.complex128)
            tr = complex(np.trace(S))
            det = complex(np.linalg.det(S))
            a, b = _eigvals_2x2(S)

            ladder = _trace_powers(S, int(args.r_max))

            row: dict = {
                "p": int(p),
                "gamma": float(gamma),
                "X_shear": float(args.X_shear),
                "X_lower": float(args.X_lower),
                "boundary": f"{boundary[0]},{boundary[1]}",
                "schur_sign": str(schur_sign),
                "X_mode": str(args.X_mode),
                "scattering": str(args.scattering),
                "seed_max_herm_def": float(hd0),
                "seed_max_unit_def": float(ud0),
                "unitarity_defect": float(sim._unitarity_defect(S)),
                "abs_eig1_minus1": float(abs(a - 1.0)),
                "abs_eig2_minus1": float(abs(b - 1.0)),
                "abs_trace_minus2": float(abs(tr - 2.0)),
                "abs_det_minus1": float(abs(det - 1.0)),
            }
            row.update(_complex_cols("eig1", a))
            row.update(_complex_cols("eig2", b))
            row.update(_complex_cols("trace", tr))
            row.update(_complex_cols("det", det))

            for r, tpow in enumerate(ladder, start=1):
                row.update(_complex_cols(f"trpow{r}", tpow))
                row[f"abs_trpow{r}_minus2"] = float(abs(tpow - 2.0))

            rows.append(row)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["gamma", "p"]).to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
