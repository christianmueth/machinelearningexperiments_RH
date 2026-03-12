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

    # scattering and free normalization
    Sm = cayley_from_lambda(Lam, eta=float(cfg.eta), eps=float(cfg.cayley_eps))

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


def _packet_partition_by_eigvals(evals: np.ndarray, n_packets: int) -> list[np.ndarray]:
    """Partition indices into n_packets contiguous bins by sorted evals."""
    evals = np.asarray(evals, dtype=float).ravel()
    n = int(evals.size)
    n_packets = int(max(1, n_packets))
    order = np.argsort(evals)
    # Split order into near-equal sized chunks
    splits: list[np.ndarray] = []
    edges = np.linspace(0, n, n_packets + 1).round().astype(int)
    for i in range(n_packets):
        a, b = int(edges[i]), int(edges[i + 1])
        if b <= a:
            continue
        splits.append(order[a:b])
    return splits


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Rebuild a packetized (intrinsic-channel) determinant using the near-commuting refinement family. "
            "Constructs Gram operators G_p = T_p T_p^* on the small incoming-basis boundary space, where "
            "T_p = Tin_N^{-1} J_{N<-pN} Tin_{pN}. Uses eigenstructure of avg(G_2) to define canonical packets, "
            "then emits per-packet det and logdet(I-M) series."
        )
    )

    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--boundary_csv", default="E_Qlambda_convergence_boundary_samples.csv")
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--N", type=int, default=64, help="Base N (one-channel size)")
    ap.add_argument("--ps", default="2,3", help="Comma-separated refinement primes")
    ap.add_argument("--t_max_points", type=int, default=220)

    ap.add_argument("--n_packets", type=int, default=4)
    ap.add_argument(
        "--basis_mode",
        default="sum_norm",
        choices=["p0", "sum", "sum_norm"],
        help=(
            "How to choose the packet basis U from the refinement Gram operators. "
            "p0: eigenbasis of G_{ps[0]}; sum: eigenbasis of sum_p G_p; "
            "sum_norm: eigenbasis of sum_p (G_p / tr(G_p))."
        ),
    )
    ap.add_argument("--out_dir", default="")

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = _load_run_cfg(run_dir / "config.json")

    boundary_csv = run_dir / str(args.boundary_csv)
    if not boundary_csv.exists():
        raise SystemExit(f"missing boundary csv: {boundary_csv}")

    t_list = _sample_t_from_boundary_csv(boundary_csv, sigma=float(args.sigma), max_points=int(args.t_max_points))

    ps = [int(x) for x in str(args.ps).split(",") if str(x).strip()]
    ps = [p for p in ps if p >= 2]
    if not ps:
        raise SystemExit("no valid refinement primes")

    N_small = int(args.N)
    b0_s = _b_from_frac(N_small, boundary_frac=cfg.boundary_frac, boundary_rounding=cfg.boundary_rounding)
    m = 2 * int(b0_s)

    # Accumulate Gram operators across t
    G_sum = {p: np.zeros((m, m), dtype=np.complex128) for p in ps}
    n_used = 0

    # Also keep per-t M(s) on the small space
    Ms: list[np.ndarray] = []

    for t in t_list.tolist():
        s = complex(float(args.sigma), float(t))
        Srel_s, Tin_s, b0_chk = _build_Srel_and_Tin(s, N_base=int(N_small), cfg=cfg)
        if int(b0_chk) != int(b0_s):
            raise SystemExit("b0 mismatch")
        M_s = _incoming_basis_matrix(Srel_s, Tin_s)
        Ms.append(M_s)

        for p in ps:
            N_large = int(p) * int(N_small)
            b0_l = _b_from_frac(N_large, boundary_frac=cfg.boundary_frac, boundary_rounding=cfg.boundary_rounding)
            J = _selector_J_two_channel(b0_small=int(b0_s), b0_large=int(b0_l))

            Srel_l, Tin_l, b0_l_chk = _build_Srel_and_Tin(s, N_base=int(N_large), cfg=cfg)
            if int(b0_l_chk) != int(b0_l):
                raise SystemExit("b0 large mismatch")

            # T_p = Tin_s^{-1} J Tin_l
            T = np.linalg.solve(Tin_s, (J @ Tin_l)).astype(np.complex128)
            G = (T @ T.conj().T).astype(np.complex128)
            G_sum[int(p)] += G

        n_used += 1

    if n_used <= 0:
        raise SystemExit("no points used")

    G_avg = {p: (G_sum[p] / float(n_used)).astype(np.complex128) for p in ps}

    # Choose packet basis U from the (nearly) commuting Gram family.
    basis_mode = str(args.basis_mode).strip().lower()
    if basis_mode == "p0":
        p0 = int(ps[0])
        evals0, U = np.linalg.eigh(G_avg[p0])
    elif basis_mode == "sum":
        Gs = np.zeros_like(next(iter(G_avg.values())))
        for p in ps:
            Gs = (Gs + G_avg[p]).astype(np.complex128)
        evals0, U = np.linalg.eigh(Gs)
    else:  # sum_norm
        Gs = np.zeros_like(next(iter(G_avg.values())))
        for p in ps:
            tr = complex(np.trace(G_avg[p]))
            scale = 1.0 / (float(np.real(tr)) + 1e-300)
            Gs = (Gs + scale * G_avg[p]).astype(np.complex128)
        evals0, U = np.linalg.eigh(Gs)

    evals0 = np.asarray(evals0, dtype=float)
    U = np.asarray(U, dtype=np.complex128)

    # Diagnostics: how diagonal is G_p in this basis?
    diag_rows: list[dict] = []
    for p in ps:
        GpU = (U.conj().T @ G_avg[p] @ U).astype(np.complex128)
        off = GpU - np.diag(np.diag(GpU))
        off_ratio = float(_fro_norm(off) / (_fro_norm(GpU) + 1e-300))
        diag_rows.append({"p": int(p), "offdiag_ratio_in_U": float(off_ratio)})

    # Define packets by binning eigenvalues of G_{p0}.
    packets = _packet_partition_by_eigvals(evals0, n_packets=int(args.n_packets))
    if not packets:
        raise SystemExit("no packets")

    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else (run_dir / "hecke_packets")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save basis metadata
    np.save(out_dir / "packet_basis_U.npy", U)
    pd.DataFrame(diag_rows).to_csv(out_dir / "packet_diag_summary.csv", index=False)

    meta_rows: list[dict] = []

    # Emit per-packet series (det and logdet(I-M) restricted to packet subspace)
    for k, idx in enumerate(packets):
        idx = np.asarray(idx, dtype=int).ravel()
        if idx.size <= 0:
            continue

        # Packet basis vectors (orthonormal) in the small space
        Uk = U[:, idx]

        rows: list[dict] = []
        for j, t in enumerate(t_list.tolist()):
            M_s = Ms[j]
            I = np.eye(M_s.shape[0], dtype=np.complex128)

            A_det = (Uk.conj().T @ M_s @ Uk).astype(np.complex128)
            A_Iminus = (Uk.conj().T @ (I - M_s) @ Uk).astype(np.complex128)

            try:
                det_val = complex(np.linalg.det(A_det))
            except Exception:
                det_val = complex("nan")

            try:
                sign, logabs = np.linalg.slogdet(A_Iminus)
                logdet_val = complex(np.log(sign) + logabs)
            except Exception:
                logdet_val = complex("nan")

            rows.append(
                {
                    "sigma": float(args.sigma),
                    "t": float(t),
                    "packet": int(k),
                    "packet_dim": int(idx.size),
                    "det_re": float(np.real(det_val)) if np.isfinite(det_val.real) and np.isfinite(det_val.imag) else float("nan"),
                    "det_im": float(np.imag(det_val)) if np.isfinite(det_val.real) and np.isfinite(det_val.imag) else float("nan"),
                    "logdetI_re": float(np.real(logdet_val))
                    if np.isfinite(logdet_val.real) and np.isfinite(logdet_val.imag)
                    else float("nan"),
                    "logdetI_im": float(np.imag(logdet_val))
                    if np.isfinite(logdet_val.real) and np.isfinite(logdet_val.imag)
                    else float("nan"),
                }
            )

        out_path = out_dir / f"packet_{int(k)}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)

        meta_rows.append(
            {
                "packet": int(k),
                "packet_dim": int(idx.size),
                "eigval_mean": float(np.mean(evals0[idx])) if idx.size else float("nan"),
                "eigval_min": float(np.min(evals0[idx])) if idx.size else float("nan"),
                "eigval_max": float(np.max(evals0[idx])) if idx.size else float("nan"),
            }
        )

    pd.DataFrame(meta_rows).to_csv(out_dir / "packets_meta.csv", index=False)

    print(f"wrote packet series to {out_dir}")
    print("packet_diag_summary (avg Gram operators in packet basis):")
    for r in diag_rows:
        print(f"  p={int(r['p'])}: offdiag_ratio_in_U={float(r['offdiag_ratio_in_U']):.4g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
