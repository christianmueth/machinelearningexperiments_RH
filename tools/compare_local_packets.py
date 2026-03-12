from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Import the simulator module from the sibling file.
# (We avoid `from tools...` because this repo's `tools/` folder is not a package.)
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

    # Deterministic ordering: increasing argument, then magnitude.
    def key(z: complex) -> tuple[float, float]:
        ang = float(math.atan2(float(np.imag(z)), float(np.real(z))))
        return (ang, float(abs(z)))

    if key(b) < key(a):
        a, b = b, a
    return a, b


def _pairing_distance(a1: complex, a2: complex, b1: complex, b2: complex) -> float:
    a1, a2, b1, b2 = complex(a1), complex(a2), complex(b1), complex(b2)
    d1 = abs(a1 - b1) + abs(a2 - b2)
    d2 = abs(a1 - b2) + abs(a2 - b1)
    return float(min(d1, d2))


def _trace_powers(S: np.ndarray, r_max: int) -> list[complex]:
    S = np.asarray(S, dtype=np.complex128)
    out: list[complex] = []
    P = np.eye(2, dtype=np.complex128)
    for r in range(1, int(r_max) + 1):
        P = (P @ S).astype(np.complex128)
        out.append(complex(np.trace(P)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Quantitatively compare per-prime local packets S_p from the sixby6 construction "
            "against an explicit Satake-injected family. Outputs per-prime eigenvalues/trace/det "
            "and ladder coefficients Tr(S_p^r)."
        )
    )
    ap.add_argument("--primes", default="2,3,5,7,11,13,17,19,23,29,31,37,41,43,47")
    ap.add_argument("--out_csv", required=True)

    # sixby6 conventions
    ap.add_argument("--six_boundary", default="auto", help="Boundary indices i,j in 0..5 or auto")
    ap.add_argument("--six_schur_sign", choices=["-", "+"], default="-")
    ap.add_argument("--six_sharp", choices=["transpose", "conj_transpose"], default="conj_transpose")
    ap.add_argument("--six_X_mode", choices=["raw", "det1"], default="det1")
    ap.add_argument("--six_X_gamma", type=float, default=1.0)
    ap.add_argument("--six_X_shear", type=float, default=0.0)
    ap.add_argument("--six_X_lower", type=float, default=0.0)
    ap.add_argument("--six_scattering", choices=["lambda_pm_i", "i_pm_lambda"], default="i_pm_lambda")

    # satake family
    ap.add_argument("--satake_family", choices=["trivial", "phase_hash", "phase_logp"], default="trivial")
    ap.add_argument("--satake_matrix", choices=["diag", "companion"], default="diag")
    ap.add_argument("--theta_scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--r_max", type=int, default=8, help="Max power r for Tr(S^r) ladder")

    args = ap.parse_args()

    primes = sim._parse_int_list(args.primes)
    if not primes:
        raise SystemExit("--primes must be non-empty")

    # Pick sixby6 boundary/sign if requested.
    six_boundary_arg = str(args.six_boundary).strip().lower()
    if six_boundary_arg == "auto":
        p_seed = [int(primes[0]), int(primes[len(primes) // 2]), int(primes[-1])] if len(primes) >= 3 else [int(primes[0])]
        k_seed = [1]
        boundary, schur_sign, *_ = sim._boundary_search(
            p_seed,
            k_seed,
            sharp_mode=str(args.six_sharp),
            x_mode=str(args.six_X_mode),
            x_gamma=float(args.six_X_gamma),
            x_shear=float(args.six_X_shear),
            x_lower=float(args.six_X_lower),
            scattering_mode=str(args.six_scattering),
        )
    else:
        parts = sim._parse_int_list(str(args.six_boundary))
        if len(parts) != 2:
            raise SystemExit("--six_boundary must be auto or i,j")
        boundary = (int(parts[0]), int(parts[1]))
        schur_sign = str(args.six_schur_sign)

    # Build per-prime packets (k_max=1) for both models.
    six_packets = sim._build_packets(
        primes,
        1,
        local_model="sixby6",
        boundary=boundary,
        sign=schur_sign,
        sharp_mode=str(args.six_sharp),
        x_mode=str(args.six_X_mode),
        x_gamma=float(args.six_X_gamma),
        x_shear=float(args.six_X_shear),
        x_lower=float(args.six_X_lower),
        scattering_mode=str(args.six_scattering),
        satake_family=str(args.satake_family),
        satake_matrix=str(args.satake_matrix),
        theta_scale=float(args.theta_scale),
        seed=int(args.seed),
    )
    sat_packets = sim._build_packets(
        primes,
        1,
        local_model="satake",
        boundary=(0, 1),
        sign="-",
        sharp_mode=str(args.six_sharp),
        x_mode=str(args.six_X_mode),
        x_gamma=float(args.six_X_gamma),
        x_shear=float(args.six_X_shear),
        x_lower=float(args.six_X_lower),
        scattering_mode=str(args.six_scattering),
        satake_family=str(args.satake_family),
        satake_matrix=str(args.satake_matrix),
        theta_scale=float(args.theta_scale),
        seed=int(args.seed),
    )

    # Index by p.
    six_by_p = {pkt.p: pkt for pkt in six_packets}
    sat_by_p = {pkt.p: pkt for pkt in sat_packets}

    rows: list[dict] = []
    for p in primes:
        pkt6 = six_by_p.get(int(p))
        pkts = sat_by_p.get(int(p))
        if pkt6 is None or pkts is None:
            continue

        S6 = np.asarray(pkt6.S, dtype=np.complex128)
        Ss = np.asarray(pkts.S, dtype=np.complex128)

        tr6 = complex(np.trace(S6))
        trs = complex(np.trace(Ss))
        det6 = complex(np.linalg.det(S6))
        dets = complex(np.linalg.det(Ss))

        a6, b6 = _eigvals_2x2(S6)
        as1, bs1 = _eigvals_2x2(Ss)

        ladder6 = _trace_powers(S6, int(args.r_max))
        ladders = _trace_powers(Ss, int(args.r_max))

        row: dict = {
            "p": int(p),
            "six_boundary": f"{boundary[0]},{boundary[1]}",
            "six_schur_sign": str(schur_sign),
            "six_X_mode": str(args.six_X_mode),
            "six_scattering": str(args.six_scattering),
            "satake_family": str(args.satake_family),
            "satake_matrix": str(args.satake_matrix),
            "theta_scale": float(args.theta_scale),
            "seed": int(args.seed),
            "unitarity_defect_six": float(sim._unitarity_defect(S6)),
            "unitarity_defect_satake": float(sim._unitarity_defect(Ss)),
            "eig_pair_dist": float(_pairing_distance(a6, b6, as1, bs1)),
            "abs_trace_diff": float(abs(tr6 - trs)),
            "abs_det_diff": float(abs(det6 - dets)),
        }
        row.update(_complex_cols("tr_six", tr6))
        row.update(_complex_cols("tr_sat", trs))
        row.update(_complex_cols("det_six", det6))
        row.update(_complex_cols("det_sat", dets))
        row.update(_complex_cols("eig1_six", a6))
        row.update(_complex_cols("eig2_six", b6))
        row.update(_complex_cols("eig1_sat", as1))
        row.update(_complex_cols("eig2_sat", bs1))

        for r, (t6, ts) in enumerate(zip(ladder6, ladders, strict=True), start=1):
            row.update(_complex_cols(f"trpow{r}_six", t6))
            row.update(_complex_cols(f"trpow{r}_sat", ts))
            row[f"abs_trpow{r}_diff"] = float(abs(t6 - ts))

        rows.append(row)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["p"]).to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
