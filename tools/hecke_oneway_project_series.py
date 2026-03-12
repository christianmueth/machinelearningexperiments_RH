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
    generator_norm: str
    generator_norm_target: float


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
    b0_small = int(b0_small)
    b0_large = int(b0_large)
    if b0_small <= 0 or b0_large <= 0:
        raise ValueError("invalid b0")
    if b0_small > b0_large:
        raise ValueError("small boundary cannot exceed large boundary")

    out = np.zeros((2 * b0_small, 2 * b0_large), dtype=np.complex128)
    for i in range(b0_small):
        out[i, i] = 1.0
        out[b0_small + i, b0_large + i] = 1.0
    return out


def _fro_norm(M: np.ndarray) -> float:
    M = np.asarray(M)
    return float(np.sqrt(np.sum(np.abs(M) ** 2)))


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
        generator_norm=str(cfg.get("generator_norm", "fro")),
        generator_norm_target=float(cfg.get("generator_norm_target", 1.0)),
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
        raise SystemExit("this tool currently supports only bulk_mode=two_channel_symmetric")

    from src.bulk import BulkParams, build_two_channel_UW
    from src.dn import dn_map_two_channel_boundary
    from src.scattering import cayley_from_lambda

    base_N = int(N_base)
    b0 = _b_from_frac(base_N, boundary_frac=cfg.boundary_frac, boundary_rounding=cfg.boundary_rounding)

    bulk_params = BulkParams(N=int(base_N), weight_k=int(cfg.weight_k))

    # model
    U, W = build_two_channel_UW(
        s=s,
        primes=list(cfg.primes),
        comps=None,
        params=bulk_params,
        generator_norm=str(cfg.generator_norm),
        generator_norm_target=float(cfg.generator_norm_target),
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
        generator_norm=str(cfg.generator_norm),
        generator_norm_target=float(cfg.generator_norm_target),
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


def _incoming_basis_matrix(Srel: np.ndarray, Tin: np.ndarray) -> np.ndarray:
    return np.linalg.solve(Tin, (Srel @ Tin)).astype(np.complex128)


def _safe_pinv(T: np.ndarray, *, rcond: float) -> np.ndarray:
    T = np.asarray(T, dtype=np.complex128)
    return np.linalg.pinv(T, rcond=float(rcond)).astype(np.complex128)


def _tikh_right_inverse(T: np.ndarray, *, rel: float) -> np.ndarray:
    """Return R ≈ T^+ using Tikhonov on TT*: R = T^H (TT^H + λ I)^{-1}.

    rel sets λ = rel * smax(T)^2.
    Shapes: T is (m×n), R is (n×m).
    """
    T = np.asarray(T, dtype=np.complex128)
    if T.ndim != 2:
        raise ValueError("T must be 2D")
    m, _n = T.shape
    if m == 0:
        return np.zeros((T.shape[1], 0), dtype=np.complex128)
    s = np.linalg.svd(T, compute_uv=False)
    smax = float(s[0]) if s.size else 0.0
    lam = float(rel) * (smax * smax)
    if not math.isfinite(lam) or lam < 0:
        lam = 0.0
    G = (T @ T.conj().T).astype(np.complex128)
    G = (G + lam * np.eye(m, dtype=np.complex128)).astype(np.complex128)
    X = np.linalg.solve(G, np.eye(m, dtype=np.complex128)).astype(np.complex128)
    return (T.conj().T @ X).astype(np.complex128)


def _op_norm_est(M: np.ndarray) -> float:
    """Cheap-ish spectral norm estimate via SVD for small matrices."""
    M = np.asarray(M, dtype=np.complex128)
    if M.size == 0:
        return 0.0
    try:
        s = np.linalg.svd(M, compute_uv=False)
        return float(s[0]) if s.size else 0.0
    except Exception:
        return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "One-way refinement projection series generator. Builds M_N and M_{pN} in the incoming basis, "
            "forms T=Tin_N^{-1} J Tin_{pN}, and defines a projected small operator M_from_big = T M_{pN} pinv(T). "
            "Emits det/logdet series for both the original small operator and the projected-from-big operator."
        )
    )

    ap.add_argument("--run_dir", required=True, help="Run directory containing config.json and boundary sample CSV")
    ap.add_argument("--boundary_csv", default="E_Qlambda_convergence_boundary_samples.csv")
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--t_max_points", type=int, default=220)

    ap.add_argument("--N", type=int, default=64, help="Base N (one-channel size)")
    ap.add_argument("--p", type=int, default=2, help="Refinement factor p (uses p*N)")

    ap.add_argument("--pinv_rcond", type=float, default=1e-10)
    ap.add_argument(
        "--tikh_rel",
        type=float,
        default=1e-8,
        help="If >0, use Tikhonov right-inverse R=T^H(TT^H+λI)^{-1} with λ=tikh_rel*smax(T)^2 instead of pinv",
    )
    ap.add_argument("--out_csv", default="", help="Output CSV path")

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = _load_run_cfg(run_dir / "config.json")
    boundary_csv = run_dir / str(args.boundary_csv)

    t_list = _sample_t_from_boundary_csv(boundary_csv, sigma=float(args.sigma), max_points=int(args.t_max_points))

    N_small = int(args.N)
    N_large = int(args.p) * int(args.N)

    b0_s = _b_from_frac(N_small, boundary_frac=cfg.boundary_frac, boundary_rounding=cfg.boundary_rounding)
    b0_l = _b_from_frac(N_large, boundary_frac=cfg.boundary_frac, boundary_rounding=cfg.boundary_rounding)
    J = _selector_J_two_channel(b0_small=int(b0_s), b0_large=int(b0_l))

    rows: list[dict] = []

    # Diagnostics at the first point.
    diag_done = False

    for t in t_list.tolist():
        s = complex(float(args.sigma), float(t))

        Srel_s, Tin_s, b0_s_chk = _build_Srel_and_Tin(s, N_base=int(N_small), cfg=cfg)
        Srel_l, Tin_l, b0_l_chk = _build_Srel_and_Tin(s, N_base=int(N_large), cfg=cfg)
        if int(b0_s_chk) != int(b0_s) or int(b0_l_chk) != int(b0_l):
            raise SystemExit("internal b0 mismatch")

        # Incoming-basis operators
        try:
            M_s = _incoming_basis_matrix(Srel_s, Tin_s)
            M_l = _incoming_basis_matrix(Srel_l, Tin_l)
            T = np.linalg.solve(Tin_s, (J @ Tin_l)).astype(np.complex128)
            if float(args.tikh_rel) > 0:
                T_pinv = _tikh_right_inverse(T, rel=float(args.tikh_rel))
            else:
                T_pinv = _safe_pinv(T, rcond=float(args.pinv_rcond))
            M_from_big = (T @ M_l @ T_pinv).astype(np.complex128)

            # One-way commutation residual: how well does M_from_big satisfy X T ≈ T M_l?
            comm_res = float(_fro_norm((M_from_big @ T) - (T @ M_l)) / (_fro_norm(T @ M_l) + 1e-300))

            # Projection quality: does T T^+ approximate I on the small space?
            TTp = (T @ T_pinv).astype(np.complex128)
            proj_def = float(_fro_norm(TTp - np.eye(TTp.shape[0], dtype=np.complex128)) / (_fro_norm(np.eye(TTp.shape[0])) + 1e-300))
            match_def = float(_fro_norm(M_s - M_from_big) / (_fro_norm(M_s) + 1e-300))

            # Conditioning/stability info
            pinv_op = _op_norm_est(T_pinv)
            svals = None
            if not diag_done:
                try:
                    svals = np.linalg.svd(T, compute_uv=False)
                except Exception:
                    svals = None
                if svals is not None and svals.size:
                    print(
                        f"T singular values: max={float(svals[0]):.4g}  min={float(svals[-1]):.4g}  cond~={float(svals[0]/(svals[-1]+1e-300)):.4g}"
                    )
                print(f"||pinv(T)||_2 est: {float(pinv_op):.4g}")
                diag_done = True

            det_small = complex(np.linalg.det(M_s))
            det_proj = complex(np.linalg.det(M_from_big))

            I_s = np.eye(M_s.shape[0], dtype=np.complex128)
            sign_s, logabs_s = np.linalg.slogdet(I_s - M_s)
            sign_p, logabs_p = np.linalg.slogdet(I_s - M_from_big)
            logdetI_small = complex(np.log(sign_s) + logabs_s)
            logdetI_proj = complex(np.log(sign_p) + logabs_p)

            cond_Tin_s = float(np.linalg.cond(Tin_s))
            cond_Tin_l = float(np.linalg.cond(Tin_l))
        except Exception:
            det_small = complex("nan")
            det_proj = complex("nan")
            logdetI_small = complex("nan")
            logdetI_proj = complex("nan")
            proj_def = float("nan")
            match_def = float("nan")
            comm_res = float("nan")
            pinv_op = float("nan")
            cond_Tin_s = float("nan")
            cond_Tin_l = float("nan")

        rows.append(
            {
                "sigma": float(args.sigma),
                "t": float(t),
                "N": int(N_small),
                "p": int(args.p),
                "N_large": int(N_large),
                "b0_small": int(b0_s),
                "b0_large": int(b0_l),
                "proj_def_TTpinv": float(proj_def),
                "match_def_M": float(match_def),
                "comm_res_XT_minus_TMl": float(comm_res),
                "pinv_opnorm_est": float(pinv_op),
                "cond_Tin_small": float(cond_Tin_s),
                "cond_Tin_large": float(cond_Tin_l),
                "det_small_re": float(np.real(det_small)) if np.isfinite(det_small.real) and np.isfinite(det_small.imag) else float("nan"),
                "det_small_im": float(np.imag(det_small)) if np.isfinite(det_small.real) and np.isfinite(det_small.imag) else float("nan"),
                "det_proj_re": float(np.real(det_proj)) if np.isfinite(det_proj.real) and np.isfinite(det_proj.imag) else float("nan"),
                "det_proj_im": float(np.imag(det_proj)) if np.isfinite(det_proj.real) and np.isfinite(det_proj.imag) else float("nan"),
                "logdetI_small_re": float(np.real(logdetI_small)) if np.isfinite(logdetI_small.real) and np.isfinite(logdetI_small.imag) else float("nan"),
                "logdetI_small_im": float(np.imag(logdetI_small)) if np.isfinite(logdetI_small.real) and np.isfinite(logdetI_small.imag) else float("nan"),
                "logdetI_proj_re": float(np.real(logdetI_proj)) if np.isfinite(logdetI_proj.real) and np.isfinite(logdetI_proj.imag) else float("nan"),
                "logdetI_proj_im": float(np.imag(logdetI_proj)) if np.isfinite(logdetI_proj.real) and np.isfinite(logdetI_proj.imag) else float("nan"),
            }
        )

    out = pd.DataFrame(rows).sort_values(["t"]).reset_index(drop=True)
    if out.empty:
        raise SystemExit("no rows")

    out_csv = str(args.out_csv).strip()
    if not out_csv:
        out_csv = str(run_dir / f"hecke_oneway_project_series_p{int(args.p)}.csv")

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}")

    # Quick summary
    for col in ["proj_def_TTpinv", "match_def_M"]:
        x = pd.to_numeric(out[col], errors="coerce")
        x = x[np.isfinite(x)]
        if not x.empty:
            print(f"{col}: mean={float(x.mean()):.4g} p50={float(x.quantile(0.5)):.4g} p95={float(x.quantile(0.95)):.4g} max={float(x.max()):.4g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
