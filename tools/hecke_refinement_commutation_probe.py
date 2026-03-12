from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# Ensure repository root is importable so `import src.*` works when executing from tools/.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass(frozen=True)
class RunCfg:
    primes: list[int]
    bulk_mode: str
    completion_mode: str
    dual_scale: float
    eta: float
    schur_jitter: float
    cayley_eps: float
    boundary_frac: float
    boundary_rounding: str
    weight_k: int


def _b_from_frac(N: int, *, boundary_frac: float, boundary_rounding: str) -> int:
    N0 = int(N)
    x = float(boundary_frac) * float(N0)
    r = str(boundary_rounding).strip().lower()
    if r == "floor":
        b = int(math.floor(x))
    elif r == "ceil":
        b = int(math.ceil(x))
    else:
        b = int(round(x))
    if b <= 0:
        b = 1
    if b >= N0 - 1:
        b = max(1, N0 // 4)
    return int(b)


def _selector_J_two_channel(b0_small: int, b0_large: int) -> np.ndarray:
    """Return J mapping large-boundary coords -> small-boundary coords.

    Boundary ordering is [ch1(0..b0-1), ch2(0..b0-1)] within the DN-restricted basis.
    In the large system, that means indices [0..b0L-1] (ch1) then [b0L..2*b0L-1] (ch2).
    We select the first b0S indices in each channel.
    """

    b0_small = int(b0_small)
    b0_large = int(b0_large)
    if b0_small <= 0 or b0_large <= 0:
        raise ValueError("invalid b0")
    if b0_small > b0_large:
        raise ValueError("small boundary cannot exceed large boundary")

    out = np.zeros((2 * b0_small, 2 * b0_large), dtype=np.complex128)

    # ch1 block
    for i in range(b0_small):
        out[i, i] = 1.0

    # ch2 block (offset by b0_large)
    for i in range(b0_small):
        out[b0_small + i, b0_large + i] = 1.0

    return out


def _fro_norm(M: np.ndarray) -> float:
    M = np.asarray(M)
    return float(np.sqrt(np.sum(np.abs(M) ** 2)))


def _comm_defect(A: np.ndarray, T: np.ndarray, B: np.ndarray) -> float:
    """Return normalized Frobenius defect ||A T - T B|| / (||A|| ||T|| + ||T|| ||B||)."""

    A = np.asarray(A, dtype=np.complex128)
    B = np.asarray(B, dtype=np.complex128)
    T = np.asarray(T, dtype=np.complex128)
    C = (A @ T) - (T @ B)
    num = _fro_norm(C)
    den = _fro_norm(T) * (_fro_norm(A) + _fro_norm(B)) + 1e-300
    return float(num / den)


def _load_run_cfg(config_json: Path) -> RunCfg:
    cfg = json.loads(config_json.read_text(encoding="utf-8"))
    return RunCfg(
        primes=[int(x) for x in cfg.get("primes", [])],
        bulk_mode=str(cfg.get("bulk_mode", "two_channel_symmetric")),
        completion_mode=str(cfg.get("completion_mode", "dual_1_minus_s")),
        dual_scale=float(cfg.get("dual_scale", 1.0)),
        eta=float(cfg.get("eta", 1.0)),
        schur_jitter=float(cfg.get("schur_jitter", 1e-10)),
        cayley_eps=float(cfg.get("cayley_eps", 0.0)),
        boundary_frac=float(cfg.get("boundary_frac", 0.25)),
        boundary_rounding=str(cfg.get("boundary_rounding", "round")),
        weight_k=int(cfg.get("weight_k", 0)),
    )


def _sample_t_from_boundary_csv(path: Path, *, sigma: float, max_points: int) -> np.ndarray:
    df = pd.read_csv(path)
    if "side" in df.columns:
        df = df[df["side"].astype(str) == "right"].copy()
    df["sigma"] = pd.to_numeric(df["sigma"], errors="coerce")
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df = df.dropna(subset=["sigma", "t"]).copy()
    df = df[(df["sigma"] - float(sigma)).abs() <= 1e-9].copy()
    t = np.asarray(sorted(df["t"].unique().tolist()), dtype=float)
    if t.size == 0:
        raise SystemExit("no t values found for requested sigma/right side")
    if int(max_points) > 0 and t.size > int(max_points):
        idx = np.linspace(0, t.size - 1, int(max_points)).round().astype(int)
        t = t[idx]
    return t


def _build_Srel_and_Tin(s: complex, *, N_base: int, cfg: RunCfg):
    if str(cfg.bulk_mode).lower().strip() != "two_channel_symmetric":
        raise SystemExit("this probe currently supports only bulk_mode=two_channel_symmetric")

    from src.bulk import BulkParams, build_two_channel_UW
    from src.dn import dn_map_two_channel_boundary
    from src.scattering import cayley_from_lambda

    # Experiment E uses base_N and then doubles to get the 2-channel size; boundary uses b0 ~ boundary_frac*base_N.
    base_N = int(N_base)
    b0 = _b_from_frac(base_N, boundary_frac=cfg.boundary_frac, boundary_rounding=cfg.boundary_rounding)

    bulk_params = BulkParams(N=int(base_N), weight_k=int(cfg.weight_k))

    # model
    U, W = build_two_channel_UW(
        s=s,
        primes=list(cfg.primes),
        comps=None,
        params=bulk_params,
        generator_norm="fro",
        generator_norm_target=1.0,
        completion_mode=str(cfg.completion_mode),
        dual_scale=float(cfg.dual_scale),
    )
    Lam = dn_map_two_channel_boundary(U, W, b0=int(b0), jitter=float(cfg.schur_jitter))
    Sm = cayley_from_lambda(Lam, eta=float(cfg.eta), eps=float(cfg.cayley_eps))

    # free
    U0, W0 = build_two_channel_UW(
        s=s,
        primes=[],
        comps=None,
        params=bulk_params,
        generator_norm="fro",
        generator_norm_target=1.0,
        completion_mode=str(cfg.completion_mode),
        dual_scale=float(cfg.dual_scale),
    )
    Lam0 = dn_map_two_channel_boundary(U0, W0, b0=int(b0), jitter=float(cfg.schur_jitter))
    S0 = cayley_from_lambda(Lam0, eta=float(cfg.eta), eps=float(cfg.cayley_eps))

    Srel = (Sm @ np.linalg.inv(S0)).astype(np.complex128)

    nb = int(Lam.shape[0])
    Tin = (Lam + 1j * float(cfg.eta) * np.eye(nb, dtype=np.complex128)).astype(np.complex128)
    if float(cfg.cayley_eps) != 0.0:
        Tin = (Tin + float(cfg.cayley_eps) * np.eye(nb, dtype=np.complex128)).astype(np.complex128)

    return Srel, Tin, int(b0)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Refinement commutation probe for Experiment-E-style two-channel boundary DN/scattering. "
            "Builds the correct non-contiguous selector J_{N<-pN} in the two-channel boundary ordering and "
            "tests commutation after incoming-basis conjugation: M_N T - T M_{pN} with T=Tin_N^{-1} J Tin_{pN}."
        )
    )

    ap.add_argument("--run_dir", required=True, help="Run directory containing config.json and boundary sample CSV")
    ap.add_argument("--boundary_csv", default="E_Qlambda_convergence_boundary_samples.csv")
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--t_max_points", type=int, default=40, help="Subsample to at most this many t points")

    ap.add_argument("--N", type=int, default=64, help="Base N (one-channel size); refinement uses p*N")
    ap.add_argument("--ps", default="2,3", help="Comma-separated refinement primes p")

    ap.add_argument("--out_csv", default="", help="Optional output CSV")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    config_json = run_dir / "config.json"
    if not config_json.exists():
        raise SystemExit(f"missing {config_json}")

    cfg = _load_run_cfg(config_json)

    boundary_csv = run_dir / str(args.boundary_csv)
    if not boundary_csv.exists():
        raise SystemExit(f"missing boundary csv: {boundary_csv}")

    t_list = _sample_t_from_boundary_csv(boundary_csv, sigma=float(args.sigma), max_points=int(args.t_max_points))

    ps = [int(x) for x in str(args.ps).split(",") if str(x).strip()]
    ps = [p for p in ps if p >= 2]
    if not ps:
        raise SystemExit("no valid --ps")

    rows: list[dict] = []

    for p in ps:
        N_small = int(args.N)
        N_large = int(p) * int(args.N)

        # Boundary sizes (b0) determine the non-contiguous selector.
        b0_s = _b_from_frac(N_small, boundary_frac=cfg.boundary_frac, boundary_rounding=cfg.boundary_rounding)
        b0_l = _b_from_frac(N_large, boundary_frac=cfg.boundary_frac, boundary_rounding=cfg.boundary_rounding)
        if b0_s > b0_l:
            raise SystemExit(f"boundary b0 is not monotone: b0({N_small})={b0_s} > b0({N_large})={b0_l}")

        J = _selector_J_two_channel(b0_small=int(b0_s), b0_large=int(b0_l))

        for t in t_list.tolist():
            s = complex(float(args.sigma), float(t))

            Srel_s, Tin_s, b0_s_chk = _build_Srel_and_Tin(s, N_base=int(N_small), cfg=cfg)
            Srel_l, Tin_l, b0_l_chk = _build_Srel_and_Tin(s, N_base=int(N_large), cfg=cfg)

            if int(b0_s_chk) != int(b0_s) or int(b0_l_chk) != int(b0_l):
                raise SystemExit("internal b0 mismatch")

            # Rel-basis commutation: Srel_N J - J Srel_{pN}
            defect_rel = _comm_defect(Srel_s, J, Srel_l)

            # Incoming-basis commutation
            # M = Tin^{-1} Srel Tin, T = Tin_s^{-1} J Tin_l
            try:
                M_s = np.linalg.solve(Tin_s, (Srel_s @ Tin_s)).astype(np.complex128)
                M_l = np.linalg.solve(Tin_l, (Srel_l @ Tin_l)).astype(np.complex128)
                T = np.linalg.solve(Tin_s, (J @ Tin_l)).astype(np.complex128)
                defect_incoming = _comm_defect(M_s, T, M_l)
                cond_Tin_s = float(np.linalg.cond(Tin_s))
                cond_Tin_l = float(np.linalg.cond(Tin_l))
            except Exception:
                defect_incoming = float("nan")
                cond_Tin_s = float("nan")
                cond_Tin_l = float("nan")

            rows.append(
                {
                    "sigma": float(args.sigma),
                    "t": float(t),
                    "p": int(p),
                    "N_small": int(N_small),
                    "N_large": int(N_large),
                    "b0_small": int(b0_s),
                    "b0_large": int(b0_l),
                    "defect_rel": float(defect_rel),
                    "defect_incoming": float(defect_incoming),
                    "cond_Tin_small": float(cond_Tin_s),
                    "cond_Tin_large": float(cond_Tin_l),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("no rows")

    # Summary
    def _summ(df: pd.DataFrame, col: str) -> str:
        x = pd.to_numeric(df[col], errors="coerce")
        x = x[np.isfinite(x)]
        if x.empty:
            return "nan"
        return f"mean={float(x.mean()):.4g}  p50={float(x.quantile(0.5)):.4g}  p95={float(x.quantile(0.95)):.4g}  max={float(x.max()):.4g}"

    print("Summary by p:")
    for p, g in out.groupby("p"):
        print(f"  p={int(p)}  defect_rel: { _summ(g, 'defect_rel') }")
        print(f"        defect_incoming: { _summ(g, 'defect_incoming') }")

    out_csv = str(args.out_csv).strip()
    if out_csv:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
