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

import corrected_backend_interface as backend  # type: ignore
import probe_exact_A3_chart_constrained as a3_chart  # type: ignore


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


def _coeffs_from_eigs(eigs: np.ndarray) -> tuple[complex, complex, complex]:
    eigs = np.asarray(eigs, dtype=np.complex128)
    e1 = complex(np.sum(eigs))
    e2 = complex(eigs[0] * eigs[1] + eigs[0] * eigs[2] + eigs[1] * eigs[2])
    e3 = complex(eigs[0] * eigs[1] * eigs[2])
    return -e1, e2, -e3


def _score_with_thresholds(eigs: np.ndarray, *, A1: float, A2: float, A3: float, radius_max: float) -> tuple[float, dict[str, float]]:
    eigs = np.asarray(eigs, dtype=np.complex128)
    rad = np.abs(eigs)
    if np.any(rad <= 0.0) or np.any(rad >= float(radius_max) + 1e-12):
        return float("inf"), {}
    B1, B2, B3 = _coeffs_from_eigs(eigs)
    t1 = complex(float(A1), 0.0)
    t2 = complex(float(A2), 0.0)
    t3 = complex(float(A3), 0.0)
    err1 = float(abs(B1 - t1))
    err2 = float(abs(B2 - t2))
    err3 = float(abs(B3 - t3))
    score = float((err1 / 0.01) ** 2 + (err2 / 0.05) ** 2 + (err3 / 0.05) ** 2)
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
        "spectral_radius": float(np.max(rad)),
        "sum_abs_eigs": float(np.sum(rad)),
    }
    return float(score), meta


def _random_eigs(rng: np.random.Generator, *, radius_max: float) -> np.ndarray:
    radii = float(radius_max) * np.sqrt(rng.random(3, dtype=float))
    thetas = rng.uniform(-math.pi, math.pi, size=3)
    return (radii * np.exp(1j * thetas)).astype(np.complex128)


def _fit_normal3_flexible(
    *,
    A1: float,
    A2: float,
    A3: float,
    radius_max: float,
    n_random: int,
    local_steps: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(int(seed))
    roots = np.roots([1.0, float(A1), float(A2), float(A3)]).astype(np.complex128)

    seed_eigs: list[np.ndarray] = []
    if roots.size == 3:
        seed_eigs.append(np.asarray(roots, dtype=np.complex128))
        clipped = np.asarray(
            [complex(min(abs(z), float(radius_max) - 1e-6) * np.exp(1j * np.angle(z))) for z in roots.tolist()],
            dtype=np.complex128,
        )
        seed_eigs.append(clipped)
        nudged = clipped.copy()
        hi = int(np.argmax(np.abs(nudged)))
        nudged[hi] = complex((float(radius_max) - 1e-6) * np.exp(1j * np.angle(nudged[hi] + 1e-15j)))
        seed_eigs.append(nudged)

    best_score = float("inf")
    best_eigs: np.ndarray | None = None
    best_meta: dict[str, float] | None = None

    def consider(cand: np.ndarray) -> None:
        nonlocal best_score, best_eigs, best_meta
        score, meta = _score_with_thresholds(cand, A1=float(A1), A2=float(A2), A3=float(A3), radius_max=float(radius_max))
        if score < best_score:
            best_score = float(score)
            best_eigs = np.asarray(cand, dtype=np.complex128)
            best_meta = dict(meta)

    for cand in seed_eigs:
        consider(np.asarray(cand, dtype=np.complex128))

    for _ in range(int(n_random)):
        cand = _random_eigs(rng, radius_max=float(radius_max))
        consider(cand)
        if roots.size == 3 and rng.random() < 0.5:
            mix = np.asarray(roots, dtype=np.complex128).copy()
            mix *= np.exp(1j * rng.normal(loc=0.0, scale=0.18, size=3))
            mix *= rng.uniform(0.88, 1.0, size=3)
            mix = np.asarray(
                [complex(min(abs(z), float(radius_max) - 1e-6) * np.exp(1j * np.angle(z))) for z in mix.tolist()],
                dtype=np.complex128,
            )
            consider(mix)

    if best_eigs is None or best_meta is None:
        raise RuntimeError("failed to initialize normal3 flexible fit")

    radii = np.abs(best_eigs).astype(float)
    thetas = np.angle(best_eigs).astype(float)
    r_step = 0.08 * float(radius_max)
    th_step = 0.35
    for _ in range(int(local_steps)):
        improved = False
        for i in range(3):
            for delta in [-r_step, r_step]:
                rr = radii.copy()
                rr[i] = float(np.clip(rr[i] + delta, 1e-6, float(radius_max) - 1e-6))
                cand = (rr * np.exp(1j * thetas)).astype(np.complex128)
                score, meta = _score_with_thresholds(cand, A1=float(A1), A2=float(A2), A3=float(A3), radius_max=float(radius_max))
                if score < best_score:
                    best_score = float(score)
                    best_eigs = np.asarray(cand, dtype=np.complex128)
                    best_meta = dict(meta)
                    radii = rr
                    improved = True
            for delta in [-th_step, th_step]:
                tt = thetas.copy()
                tt[i] = float(tt[i] + delta)
                cand = (radii * np.exp(1j * tt)).astype(np.complex128)
                score, meta = _score_with_thresholds(cand, A1=float(A1), A2=float(A2), A3=float(A3), radius_max=float(radius_max))
                if score < best_score:
                    best_score = float(score)
                    best_eigs = np.asarray(cand, dtype=np.complex128)
                    best_meta = dict(meta)
                    thetas = tt
                    improved = True
        if not improved:
            r_step *= 0.65
            th_step *= 0.65

    best_meta = dict(best_meta)
    best_meta["score"] = float(best_score)
    for i, z in enumerate(best_eigs.tolist(), start=1):
        best_meta[f"eig{i}_re"] = float(np.real(z))
        best_meta[f"eig{i}_im"] = float(np.imag(z))
    return np.asarray(best_eigs, dtype=np.complex128), best_meta


def _fit_a3_exact_frontend(
    *,
    A1: float,
    A2: float,
    A3: float,
    radius_max: float,
    n_random: int,
    local_steps: int,
    seed: int,
    w_A1: float,
    w_A2: float,
) -> tuple[np.ndarray, dict[str, float]]:
    eigs, meta = a3_chart._search_A3_exact(
        A1=float(A1),
        A2=float(A2),
        A3=float(A3),
        radius_max=float(radius_max),
        n_random=int(n_random),
        local_steps=int(local_steps),
        w1=float(w_A1),
        w2=float(w_A2),
        seed=int(seed),
    )
    meta = dict(meta)
    meta["score_thresholded"], _ = _score_with_thresholds(
        np.asarray(eigs, dtype=np.complex128), A1=float(A1), A2=float(A2), A3=float(A3), radius_max=float(radius_max)
    )
    return np.asarray(eigs, dtype=np.complex128), meta


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
        rows.append({"sigma": float(sigma), "t": float(t), "q_track_re": float(np.real(D)), "q_track_im": float(np.imag(D))})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Canonical frontend realization driver. Reads frozen corrected coefficients from the locked backend interface, "
            "defaults to the A3_exact_frontend, and exports determinant lines for Mangoldt probing."
        )
    )
    ap.add_argument("--coeff_csv", default=str(backend.DEFAULT_COEFF_CSV))
    ap.add_argument("--u", type=float, default=0.2)
    ap.add_argument("--frontend_family", choices=["A3_exact_frontend", "normal3_flexible"], default="A3_exact_frontend")
    ap.add_argument("--radius_max", type=float, default=0.98)
    ap.add_argument("--n_random", type=int, default=30000)
    ap.add_argument("--local_steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w_A1", type=float, default=1.35)
    ap.add_argument("--w_A2", type=float, default=1.15)
    ap.add_argument("--primes_global", default="2,3,5,7,11,13,17,19,23,29")
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--t_min", type=float, default=10.0)
    ap.add_argument("--t_max", type=float, default=30.0)
    ap.add_argument("--n_t", type=int, default=401)
    ap.add_argument("--out_prefix", default="out/frontend_realization_u020")
    args = ap.parse_args()

    factors = backend.load_corrected_factors(coeff_csv=str(args.coeff_csv), u=float(args.u))
    primes_global = _parse_int_csv(str(args.primes_global))
    t_grid = np.linspace(float(args.t_min), float(args.t_max), int(args.n_t), dtype=float)
    out_prefix = Path(str(args.out_prefix))
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    if str(args.frontend_family) == "A3_exact_frontend":
        eigs, meta = _fit_a3_exact_frontend(
            A1=float(factors.A1_star),
            A2=float(factors.A2_star),
            A3=float(factors.A3_star),
            radius_max=float(args.radius_max),
            n_random=int(args.n_random),
            local_steps=int(args.local_steps),
            seed=int(args.seed),
            w_A1=float(args.w_A1),
            w_A2=float(args.w_A2),
        )
    else:
        eigs, meta = _fit_normal3_flexible(
            A1=float(factors.A1_star),
            A2=float(factors.A2_star),
            A3=float(factors.A3_star),
            radius_max=float(args.radius_max),
            n_random=int(args.n_random),
            local_steps=int(args.local_steps),
            seed=int(args.seed),
        )

    S = np.diag(np.asarray(eigs, dtype=np.complex128))
    packets = [GenericPacket(p=int(p), ell=float(math.log(float(p))), S=np.asarray(S, dtype=np.complex128)) for p in primes_global]
    det_df = _global_det_line_generic(packets=packets, sigma=float(args.sigma), t_grid=t_grid)
    feed_df = backend.build_prime_local_backend_feed(primes=primes_global, factors=factors)

    coeff_row = {
        "frontend_family": str(args.frontend_family),
        "coeff_csv": str(args.coeff_csv),
        "coeff_err_pass": bool(float(meta["err1"]) < 0.01 and float(meta["err2"]) < 0.05 and float(meta["err3"]) < 0.05),
        "spectral_radius_pass": bool(float(meta["spectral_radius"]) < 1.0),
        **feed_df.iloc[0].to_dict(),
        **meta,
    }

    coeff_path = Path(str(out_prefix) + f"_{args.frontend_family}_coefficients.csv")
    det_path = Path(str(out_prefix) + f"_{args.frontend_family}_det.csv")
    feed_path = Path(str(out_prefix) + f"_{args.frontend_family}_backend_feed.csv")
    pd.DataFrame([coeff_row]).to_csv(coeff_path, index=False)
    det_df.to_csv(det_path, index=False)
    feed_df.to_csv(feed_path, index=False)

    print(f"wrote {coeff_path}")
    print(f"wrote {det_path}")
    print(f"wrote {feed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())