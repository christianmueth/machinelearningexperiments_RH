"""Scan boundary/sign for a non-degenerate k-observable regime.

Motivation
- Our Möbius-vs-raw comparison is only meaningful if the chosen k-observable
  has enough dynamic range above floating-point noise.

This tool scans all (i,j) boundary pairs with Schur sign ±, evaluates a chosen
packet observable at u=0 and u=±u_probe over a small seed set of (p,k), and
reports slices where the observable is comfortably above noise while still
remaining reasonably Hermitian/unitary.

It does NOT claim arithmetic structure; it is an observability diagnostics tool.
"""

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

# Import simulator from sibling file (tools/ is not a package).
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import six_by_six_prime_tower_sim as sim  # type: ignore


def _parse_int_list(s: str) -> list[int]:
    out: list[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _pairs_6() -> list[tuple[int, int]]:
    return [(i, j) for i in range(6) for j in range(i + 1, 6)]


def _k_list(k_seed_max: int) -> list[int]:
    k_seed_max = int(k_seed_max)
    if k_seed_max <= 0:
        raise ValueError("k_seed_max must be >= 1")
    return list(range(1, k_seed_max + 1))


def _hermitian_defect_clamped(A: np.ndarray, *, den_floor: float) -> float:
    A = np.asarray(A, dtype=np.complex128)
    num = float(np.linalg.norm(A - A.conj().T, ord="fro"))
    den = float(np.linalg.norm(A, ord="fro"))
    return float(num / max(float(den_floor), den))


def _devS_pm_I_and_choice(S: np.ndarray) -> tuple[float, int]:
    S = np.asarray(S, dtype=np.complex128)
    I2 = np.eye(2, dtype=np.complex128)
    d_pos = float(np.linalg.norm(S - I2, ord="fro"))
    d_neg = float(np.linalg.norm(S + I2, ord="fro"))
    if d_pos <= d_neg:
        return float(d_pos), +1
    return float(d_neg), -1


def _packet_observable(pkt: sim.Packet, *, mode: str) -> tuple[float, int | None]:
    """Return (value, aux) where aux is optional sign-choice for devS_pm_I."""

    mode = str(mode).strip().lower()
    if mode == "devs_pm_i":
        v, sgn = _devS_pm_I_and_choice(np.asarray(pkt.S, dtype=np.complex128))
        return float(v), int(sgn)
    if mode == "frolam":
        Lam = np.asarray(pkt.Lam, dtype=np.complex128)
        return float(np.linalg.norm(Lam, ord="fro")), None
    if mode == "trlam":
        Lam = np.asarray(pkt.Lam, dtype=np.complex128)
        return float(abs(complex(np.trace(Lam)))), None
    raise ValueError("mode must be one of: devS_pm_I, froLam, trLam")


@dataclass(frozen=True)
class ScanSpec:
    primes: list[int]
    k_seed: list[int]
    boundary: tuple[int, int]
    schur_sign: str
    p_mode: str
    u_list: list[float]
    sharp_mode: str
    x_mode: str
    x_gamma: float
    prime_power_mode: str
    scattering_mode: str
    den_floor: float


def _packets_for_u(spec: ScanSpec, *, u: float, p: int, k: int) -> sim.Packet:
    # Mirror fe_defect_perturbation_u0.py: x_shear=u and x_lower=-u.
    _, A, Ash = sim._local_blocks_for_prime_power(
        int(p),
        int(k),
        sharp_mode=str(spec.sharp_mode),
        x_mode=str(spec.x_mode),
        x_gamma=float(spec.x_gamma),
        x_shear=float(u),
        x_lower=float(-float(u)),
        p_mode=str(spec.p_mode),
    )
    B = sim._bulk_B_from_A(A, Ash)
    Lam = sim._schur_complement_Lambda(B, boundary=spec.boundary, sign=str(spec.schur_sign))
    S = sim._scattering_from_Lambda(Lam, mode=str(spec.scattering_mode))
    ell = float(k) * float(math.log(float(p)))
    return sim.Packet(p=int(p), k=int(k), ell=float(ell), S=S, Lam=Lam)


def _scan_one(spec: ScanSpec, *, k_obs_mode: str) -> dict:
    # For each u, compute obs across packets; also track defects and failures.
    rows_u: dict[float, list[float]] = {float(u): [] for u in spec.u_list}
    pm_choice_u: dict[float, list[int]] = {float(u): [] for u in spec.u_list}

    max_hd_u: dict[float, float] = {float(u): 0.0 for u in spec.u_list}
    max_ud_u: dict[float, float] = {float(u): 0.0 for u in spec.u_list}

    failures = 0
    total = 0

    for p in spec.primes:
        for k in spec.k_seed:
            total += 1
            pkt_u: dict[float, sim.Packet] = {}
            try:
                for u in spec.u_list:
                    pkt_u[float(u)] = _packets_for_u(spec, u=float(u), p=int(p), k=int(k))
            except Exception:
                failures += 1
                continue

            for u, pkt in pkt_u.items():
                Lam = np.asarray(pkt.Lam, dtype=np.complex128)
                S = np.asarray(pkt.S, dtype=np.complex128)
                max_hd_u[u] = max(max_hd_u[u], _hermitian_defect_clamped(Lam, den_floor=float(spec.den_floor)))
                max_ud_u[u] = max(max_ud_u[u], float(sim._unitarity_defect(S)))

                v, aux = _packet_observable(pkt, mode=str(k_obs_mode))
                rows_u[u].append(float(v))
                if aux is not None:
                    pm_choice_u[u].append(int(aux))

    def _safe_stats(xs: list[float]) -> tuple[float, float, float]:
        if not xs:
            return float("nan"), float("nan"), float("nan")
        arr = np.asarray(xs, dtype=float)
        return float(np.mean(arr)), float(np.max(arr)), float(np.median(arr))

    # Compute u-stability stats relative to the first u in spec.u_list (treated as base).
    base_u = float(spec.u_list[0]) if spec.u_list else 0.0
    base = np.asarray(rows_u.get(base_u, []), dtype=float)

    mean_abs_delta = float("nan")
    max_abs_delta = float("nan")
    flip_rate = float("nan")

    if base.size > 0 and (base_u in rows_u) and (len(spec.u_list) > 1):
        deltas: list[float] = []
        for u in spec.u_list:
            if float(u) == float(base_u):
                continue
            arr = np.asarray(rows_u.get(float(u), []), dtype=float)
            if arr.size != base.size or arr.size == 0:
                continue
            deltas.append(float(np.mean(np.abs(arr - base))))
            deltas.append(float(np.max(np.abs(arr - base))))
        if deltas:
            mean_abs_delta = float(np.mean(deltas))
            max_abs_delta = float(np.max(deltas))

    if str(k_obs_mode).strip().lower() == "devs_pm_i":
        # Flip-rate of ±I-branch choice between base_u and each other u; report the worst.
        a = pm_choice_u.get(float(base_u), [])
        if a:
            aa = np.asarray(a, dtype=int)
            worst = 0.0
            for u in spec.u_list:
                if float(u) == float(base_u):
                    continue
                b = pm_choice_u.get(float(u), [])
                if len(b) != len(a) or not b:
                    continue
                bb = np.asarray(b, dtype=int)
                worst = max(worst, float(np.mean((aa != bb).astype(float))))
            flip_rate = float(worst) if worst > 0.0 else 0.0

    out: dict[str, object] = {
        "boundary": f"{spec.boundary[0]},{spec.boundary[1]}",
        "boundary_i": int(spec.boundary[0]),
        "boundary_j": int(spec.boundary[1]),
        "schur_sign": str(spec.schur_sign),
        "p_mode": str(spec.p_mode),
        "k_obs_mode": str(k_obs_mode),
        "n_packets_ok": int(total - failures),
        "n_packets_fail": int(failures),
        "fail_rate": float(failures / max(1, total)),
        "base_u": float(base_u),
        "mean_abs_delta_u": float(mean_abs_delta),
        "max_abs_delta_u": float(max_abs_delta),
        "flip_rate_pmI": float(flip_rate),
    }

    for u in spec.u_list:
        mu, mx, med = _safe_stats(rows_u[float(u)])
        out[f"obs_mean_u{u:g}"] = float(mu)
        out[f"obs_max_u{u:g}"] = float(mx)
        out[f"obs_med_u{u:g}"] = float(med)
        out[f"max_hd_u{u:g}"] = float(max_hd_u[float(u)])
        out[f"max_ud_u{u:g}"] = float(max_ud_u[float(u)])

    # Convenience: a single "scale" number for ranking at the base u.
    # Prefer median at base_u; fall back to max if median is nan.
    base_key = f"obs_med_u{base_u:g}"
    base_max_key = f"obs_max_u{base_u:g}"
    scale = float(out.get(base_key, float("nan")))
    if not math.isfinite(scale):
        scale = float(out.get(base_max_key, float("nan")))
    out["scale_base_u"] = float(scale)
    if math.isfinite(scale) and scale > 0:
        out["log10_scale_base_u"] = float(math.log10(scale))
    else:
        out["log10_scale_base_u"] = float("nan")

    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Scan boundary/sign slices for a non-degenerate k-observable regime.")

    ap.add_argument("--primes_seed", default="2,3,5,7,11,13")
    ap.add_argument("--k_seed_max", type=int, default=3)
    ap.add_argument("--p_modes", default="p,invp", help="Comma list of p_mode values")

    ap.add_argument("--u_probe", type=float, default=0.2)
    ap.add_argument(
        "--u_list",
        default="",
        help="Optional comma list of u values. Default uses 0,+u_probe,-u_probe.",
    )
    ap.add_argument(
        "--base_u",
        type=float,
        default=float("nan"),
        help=(
            "Base u used for ranking and delta computations. Default: the first entry in --u_list, "
            "or 0 if --u_list is not provided."
        ),
    )

    ap.add_argument("--k_obs_mode", choices=["devS_pm_I", "froLam", "trLam"], default="devS_pm_I")

    ap.add_argument("--sharp", choices=["transpose", "conj_transpose"], default="conj_transpose")
    ap.add_argument("--X_mode", choices=["raw", "det1"], default="det1")
    ap.add_argument("--X_gamma", type=float, default=1.0)

    ap.add_argument("--prime_power_mode", choices=["direct", "x_power", "bulk_power"], default="direct")
    ap.add_argument("--scattering", choices=["lambda_pm_i", "i_pm_lambda"], default="i_pm_lambda")

    ap.add_argument("--den_floor", type=float, default=1e-9)

    ap.add_argument("--max_fail_rate", type=float, default=0.2)
    ap.add_argument("--max_hd", type=float, default=1e-5)
    ap.add_argument("--max_ud", type=float, default=1e-5)

    ap.add_argument("--top_n", type=int, default=15)
    ap.add_argument("--out_csv", default="out/boundary_kobs_scan.csv")

    args = ap.parse_args(argv)

    primes = _parse_int_list(str(args.primes_seed))
    if not primes:
        raise ValueError("primes_seed must be non-empty")
    k_seed = _k_list(int(args.k_seed_max))

    p_modes = [p.strip() for p in str(args.p_modes).split(",") if p.strip()]
    if not p_modes:
        raise ValueError("p_modes must be non-empty")

    if str(args.u_list).strip():
        u_list = [float(u) for u in str(args.u_list).split(",") if str(u).strip()]
        if not u_list:
            raise ValueError("u_list must be non-empty if provided")
    else:
        u = float(args.u_probe)
        u_list = [0.0, u, -u]

    if math.isfinite(float(args.base_u)):
        base_u = float(args.base_u)
        # Ensure base_u is first so stability deltas are measured from it.
        u_list = [base_u] + [float(u) for u in u_list if float(u) != float(base_u)]

    all_rows: list[dict] = []
    for p_mode in p_modes:
        for boundary in _pairs_6():
            for sgn in ["-", "+"]:
                spec = ScanSpec(
                    primes=list(primes),
                    k_seed=list(k_seed),
                    boundary=tuple(boundary),
                    schur_sign=str(sgn),
                    p_mode=str(p_mode),
                    u_list=[float(u) for u in u_list],
                    sharp_mode=str(args.sharp),
                    x_mode=str(args.X_mode),
                    x_gamma=float(args.X_gamma),
                    prime_power_mode=str(args.prime_power_mode),
                    scattering_mode=str(args.scattering),
                    den_floor=float(args.den_floor),
                )
                all_rows.append(_scan_one(spec, k_obs_mode=str(args.k_obs_mode)))

    df = pd.DataFrame(all_rows)
    out_path = Path(str(args.out_csv))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # Filter and rank.
    max_fail_rate = float(args.max_fail_rate)
    max_hd = float(args.max_hd)
    max_ud = float(args.max_ud)

    filt = df.copy()
    filt = filt[filt["fail_rate"] <= max_fail_rate]

    for u in u_list:
        col_hd = f"max_hd_u{u:g}"
        col_ud = f"max_ud_u{u:g}"
        if col_hd in filt.columns:
            filt = filt[filt[col_hd] <= max_hd]
        if col_ud in filt.columns:
            filt = filt[filt[col_ud] <= max_ud]

    # Primary rank: scale at base u descending. Secondary: low flip rate (if present), low mean_abs_delta.
    if "flip_rate_pmI" in filt.columns:
        filt = filt.sort_values(
            by=["scale_base_u", "flip_rate_pmI", "mean_abs_delta_u"],
            ascending=[False, True, True],
            kind="mergesort",
        )
    else:
        filt = filt.sort_values(by=["scale_base_u", "mean_abs_delta_u"], ascending=[False, True], kind="mergesort")

    top_n = int(args.top_n)
    show = filt.head(top_n)

    # Print a compact view.
    cols = [
        "p_mode",
        "boundary",
        "schur_sign",
        "base_u",
        "scale_base_u",
        "log10_scale_base_u",
        "mean_abs_delta_u",
        "max_abs_delta_u",
        "flip_rate_pmI",
        "fail_rate",
    ]
    # Add defect columns for u=0 and +u_probe if present.
    cols += [c for c in ["max_hd_u0", "max_ud_u0"] if c in show.columns]
    u_pos = sorted([float(u) for u in u_list if float(u) > 0])
    if u_pos:
        up = u_pos[0]
        cols += [c for c in [f"max_hd_u{up:g}", f"max_ud_u{up:g}"] if c in show.columns]

    cols = [c for c in cols if c in show.columns]

    print(f"wrote {out_path}")
    if show.empty:
        print("no candidates passed filters; loosen --max_hd/--max_ud/--max_fail_rate")
        return 0

    with pd.option_context("display.max_columns", 200, "display.width", 140):
        print("\nTop candidates (filtered+ranked):")
        print(show[cols].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
