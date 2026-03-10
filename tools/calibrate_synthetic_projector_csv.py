import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _to_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return None
        v = float(s)
        if not math.isfinite(v):
            return None
        return float(v)
    except Exception:
        return None


def _wrap_pi(x: float) -> float:
    y = float(x)
    while y <= -math.pi:
        y += 2.0 * math.pi
    while y > math.pi:
        y -= 2.0 * math.pi
    return float(y)


def _run(cmd: list[str], *, cwd: Path) -> None:
    print("\n$", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


@dataclass
class SynthParams:
    level: int
    N: int
    N2: int
    sigma: float
    t0: float
    dt: float
    n_corridor: int
    n_swap: int
    n_cond: int
    seed: int
    c_star: complex
    p_err: float
    eps0: float
    swap_strength: float
    swap_q_strength: float
    cond_sep2: float
    cond_tinv: float
    cond_q_noise: float

    force_adversaries: bool
    gauge_phase_wobble: float
    gauge_amp_wobble: float

    emit_fit_side: bool
    fit_n: int
    fit_contam_swap_frac: float
    fit_contam_cond_frac: float


def _q_star(t: np.ndarray) -> np.ndarray:
    """Nontrivial complex quotient field with winding and magnitude modulation."""

    # Positive amplitude with gentle modulation.
    amp = 1.2 + 0.25 * np.cos(2.0 * math.pi * t) + 0.12 * np.sin(5.0 * math.pi * t)
    amp = np.maximum(amp, 1e-6)

    # Phase with winding + localized shear.
    phi = (2.0 * math.pi * 3.0 * t) + 0.6 * np.sin(2.0 * math.pi * t) + 0.2 * np.sin(7.0 * math.pi * t)

    return amp * np.exp(1j * phi)


def _complex_noise(rng: np.random.Generator, n: int, scale: float) -> np.ndarray:
    return scale * (rng.normal(size=n) + 1j * rng.normal(size=n))


def _lam_pair_corridor(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A smoothly varying 2-eigenvalue 'island' pair."""

    # Two separated eigenvalues with mild t-variation.
    lam0 = (1.0 + 0.15 * np.cos(2.0 * math.pi * t)) + 1j * (0.35 + 0.05 * np.sin(2.0 * math.pi * t))
    lam1 = (0.55 + 0.12 * np.sin(2.0 * math.pi * t)) + 1j * (0.10 + 0.03 * np.cos(4.0 * math.pi * t))
    return lam0, lam1


def _lam_pair_alt(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A different 'island' pair used to simulate identity flip/mismatch."""

    lam0 = (-1.1 + 0.05 * np.cos(2.0 * math.pi * t)) + 1j * (0.20 + 0.04 * np.sin(2.0 * math.pi * t))
    lam1 = (-0.65 + 0.04 * np.sin(2.0 * math.pi * t)) + 1j * (0.05 + 0.02 * np.cos(4.0 * math.pi * t))
    return lam0, lam1


def _emit_side_rows(
    *,
    side_label: str,
    behavior: str,
    t_vals: np.ndarray,
    params: SynthParams,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n = len(t_vals)

    enable_adversaries = (params.level >= 3) or (bool(params.force_adversaries) and params.level >= 2)

    Qs = _q_star(t_vals)

    # Base noise model: shrinking with N^{-p}.
    # eps0 acts like a "typical" absolute complex error scale.
    def eps_for(N: int) -> float:
        if params.level <= 1:
            return 0.0
        return float(params.eps0) * float(N) ** (-float(params.p_err))

    epsN = eps_for(params.N)
    eps2 = eps_for(params.N2)

    # Level 1: exact global gauge relationship.
    if params.level == 1:
        QN = Qs
        Q2 = params.c_star * QN
    else:
        # Level 2+: underlying truth + N-dependent error, plus removable global gauge.
        QN = Qs + _complex_noise(rng, n, epsN)
        Q2 = params.c_star * Qs + _complex_noise(rng, n, eps2)

        # Optional non-global gauge drift (cannot be removed by a single c*).
        if float(params.gauge_phase_wobble) != 0.0 or float(params.gauge_amp_wobble) != 0.0:
            wob_phi = float(params.gauge_phase_wobble) * np.sin(2.0 * math.pi * t_vals)
            wob_amp = 1.0 + float(params.gauge_amp_wobble) * np.cos(2.0 * math.pi * t_vals)
            wob_amp = np.maximum(wob_amp, 1e-6)
            Q2 = (wob_amp * np.exp(1j * wob_phi)) * Q2

        # Adversarial behavior: make swap-like behavior disagree in Q.
        if enable_adversaries and str(behavior).strip().lower() == "swap":
            # Alternate "truth" with a phase offset and amplitude shift; allow tunable severity.
            sQ = float(np.clip(float(params.swap_q_strength), 0.0, 1.0))
            Q_alt = 0.85 * Qs * np.exp(1j * (0.9 + 0.3 * np.sin(2.0 * math.pi * t_vals)))
            Q_mix = (1.0 - sQ) * Qs + sQ * Q_alt
            Q2 = params.c_star * Q_mix + _complex_noise(rng, n, max(eps2, 0.02))

            # Re-apply wobble to the swapped quotient as well.
            if float(params.gauge_phase_wobble) != 0.0 or float(params.gauge_amp_wobble) != 0.0:
                wob_phi = float(params.gauge_phase_wobble) * np.sin(2.0 * math.pi * t_vals)
                wob_amp = 1.0 + float(params.gauge_amp_wobble) * np.cos(2.0 * math.pi * t_vals)
                wob_amp = np.maximum(wob_amp, 1e-6)
                Q2 = (wob_amp * np.exp(1j * wob_phi)) * Q2

    # Conditioning proxies.
    beh = str(behavior).strip().lower()
    if beh == "cond":
        sep2 = np.full(n, float(params.cond_sep2), dtype=float)
        tinv = np.full(n, float(params.cond_tinv), dtype=float)
        # Optional extra quotient noise when conditioning is bad.
        if enable_adversaries and float(params.cond_q_noise) > 0:
            QN = QN + _complex_noise(rng, n, float(params.cond_q_noise))
            Q2 = Q2 + _complex_noise(rng, n, float(params.cond_q_noise))
    else:
        sep2 = np.full(n, 0.12, dtype=float)
        tinv = np.full(n, 250.0, dtype=float)

    # Island invariants: corridor/cond consistent; swap mismatched between N and N2.
    lam0A, lam1A = _lam_pair_corridor(t_vals)
    lam0B, lam1B = _lam_pair_alt(t_vals)

    # Small, N-dependent invariant noise (kept tiny so corridor identity stays good).
    inv_epsN = 0.0 if params.level <= 1 else 2e-4 * float(params.N) ** (-0.25)
    inv_eps2 = 0.0 if params.level <= 1 else 2e-4 * float(params.N2) ** (-0.25)

    def inv_noise(scale: float) -> tuple[np.ndarray, np.ndarray]:
        if scale <= 0:
            return (np.zeros(n, dtype=complex), np.zeros(n, dtype=complex))
        return (
            _complex_noise(rng, n, scale),
            _complex_noise(rng, n, scale),
        )

    dn0, dn1 = inv_noise(inv_epsN)
    d20, d21 = inv_noise(inv_eps2)

    # For swap-like zone, simulate an island *selection mismatch* with tunable severity.
    lam0_N = lam0A + dn0
    lam1_N = lam1A + dn1
    if beh == "swap" and enable_adversaries:
        s = float(np.clip(float(params.swap_strength), 0.0, 1.0))
        lam0_N2 = (1.0 - s) * (lam0A + d20) + s * (lam0B + d20)
        lam1_N2 = (1.0 - s) * (lam1A + d21) + s * (lam1B + d21)
    else:
        lam0_N2 = lam0A + d20
        lam1_N2 = lam1A + d21

    def trace_det(l0: np.ndarray, l1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tr = l0 + l1
        det = l0 * l1
        return tr, det

    trN, detN = trace_det(lam0_N, lam1_N)
    tr2, det2 = trace_det(lam0_N2, lam1_N2)

    # Build rows for both N and N2.
    basis = "incoming"
    block = "plus"
    method = "schur"
    n_edge = 80

    idx = np.arange(n, dtype=int)
    sigma = np.full(n, float(params.sigma), dtype=float)

    def rows_for(N: int, Q: np.ndarray, tr: np.ndarray, det: np.ndarray, eps_sep2: np.ndarray, eps_tinv: np.ndarray) -> pd.DataFrame:
        # Provide all three quotient variants so downstream tools can switch --quotient
        # without regenerating the synthetic CSV. For now, we keep them identical.
        Q2 = Q
        Q2_phi2 = Q
        Q2_tr = Q

        out = pd.DataFrame(
            {
                "N": np.full(n, int(N), dtype=int),
                "basis": basis,
                "block": block,
                "side": side_label,
                "idx": idx,
                "n_edge": np.full(n, int(n_edge), dtype=int),
                "sigma": sigma,
                "t": np.round(t_vals.astype(float), 12),
                "method": method,
                "Q2_re": np.real(Q2).astype(float),
                "Q2_im": np.imag(Q2).astype(float),
                "absQ2": np.abs(Q2).astype(float),
                "argQ2": np.vectorize(_wrap_pi)(np.angle(Q2)).astype(float),
                "Q2_phi2_re": np.real(Q2_phi2).astype(float),
                "Q2_phi2_im": np.imag(Q2_phi2).astype(float),
                "absQ2_phi2": np.abs(Q2_phi2).astype(float),
                "argQ2_phi2": np.vectorize(_wrap_pi)(np.angle(Q2_phi2)).astype(float),
                "Q2_tr_re": np.real(Q2_tr).astype(float),
                "Q2_tr_im": np.imag(Q2_tr).astype(float),
                "absQ2_tr": np.abs(Q2_tr).astype(float),
                "argQ2_tr": np.vectorize(_wrap_pi)(np.angle(Q2_tr)).astype(float),
                "trace2_island_re": np.real(tr).astype(float),
                "trace2_island_im": np.imag(tr).astype(float),
                "det2_island_re": np.real(det).astype(float),
                "det2_island_im": np.imag(det).astype(float),
                "sep2_track": eps_sep2.astype(float),
                "Tin_inv_norm1_est": eps_tinv.astype(float),
                "status": np.array(["ok"] * n, dtype=object),
            }
        )
        return out

    a = rows_for(params.N, QN, trN, detN, sep2, tinv)
    b = rows_for(params.N2, Q2, tr2, det2, sep2, tinv)

    return pd.concat([a, b], axis=0, ignore_index=True)


def _build_projector_combined(params: SynthParams) -> pd.DataFrame:
    rng = np.random.default_rng(int(params.seed))

    enable_adversaries = (params.level >= 3) or (bool(params.force_adversaries) and params.level >= 2)

    # Define t grids per side.
    t_corr = params.t0 + params.dt * np.arange(int(params.n_corridor), dtype=float)
    t_swap = params.t0 + params.dt * np.arange(int(params.n_swap), dtype=float)
    t_cond = params.t0 + params.dt * np.arange(int(params.n_cond), dtype=float)

    frames: list[pd.DataFrame] = []

    frames.append(_emit_side_rows(side_label="corridor", behavior="corridor", t_vals=t_corr, params=params, rng=rng))

    if enable_adversaries:
        frames.append(_emit_side_rows(side_label="swap", behavior="swap", t_vals=t_swap, params=params, rng=rng))
        frames.append(_emit_side_rows(side_label="cond", behavior="cond", t_vals=t_cond, params=params, rng=rng))

    # Optional: add a dedicated lsfit fit-side that is mostly corridor-like, but can be contaminated.
    if bool(params.emit_fit_side):
        n_fit = int(params.fit_n)
        if n_fit <= 0:
            n_fit = int(params.n_corridor)
        # Deterministic selection of behaviors within the fit side.
        swap_k = int(round(float(params.fit_contam_swap_frac) * float(n_fit)))
        cond_k = int(round(float(params.fit_contam_cond_frac) * float(n_fit)))
        swap_k = max(0, min(n_fit, swap_k))
        cond_k = max(0, min(n_fit - swap_k, cond_k))
        corr_k = max(0, n_fit - swap_k - cond_k)

        # Use the corridor t-grid for the fit side (so it's geometrically co-located),
        # but apply alternate behaviors to chosen subsets to simulate label contamination.
        t_fit = params.t0 + params.dt * np.arange(int(n_fit), dtype=float)
        behs = np.array(["corridor"] * corr_k + ["swap"] * swap_k + ["cond"] * cond_k, dtype=object)
        rng.shuffle(behs)

        pieces: list[pd.DataFrame] = []
        for beh in ["corridor", "swap", "cond"]:
            m = (behs == beh)
            if not bool(np.any(m)):
                continue
            pieces.append(
                _emit_side_rows(
                    side_label="fit",
                    behavior=str(beh),
                    t_vals=t_fit[m],
                    params=params,
                    rng=rng,
                )
            )
        if pieces:
            fit_df = pd.concat(pieces, axis=0, ignore_index=True)
            # Reindex within the fit side so join keys remain unique on (side,idx,t) AND
            # keep N and N2 paired consistently (same idx for the same t).
            t_key = pd.to_numeric(fit_df["t"], errors="coerce").round(12)
            uniq = np.unique(t_key[np.isfinite(t_key.to_numpy(dtype=float))].to_numpy(dtype=float))
            uniq = np.sort(uniq)
            t_to_idx = {float(t): int(i) for i, t in enumerate(uniq.tolist())}
            fit_df["idx"] = [t_to_idx.get(float(v), 0) for v in t_key.to_numpy(dtype=float)]
            frames.append(fit_df)

    out = pd.concat(frames, axis=0, ignore_index=True)

    # Shuffle rows so nothing implicitly assumes ordering.
    out = out.sample(frac=1.0, random_state=int(params.seed)).reset_index(drop=True)
    return out


def _read_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


def _extract_cstar(summary: dict) -> complex | None:
    norm = (summary or {}).get("normalization") or {}
    lf = (norm or {}).get("lsfit") or {}
    cr = _to_float(lf.get("c_star_re"))
    ci = _to_float(lf.get("c_star_im"))
    if cr is None or ci is None:
        return None
    return complex(float(cr), float(ci))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "End-to-end synthetic calibration: emit a schema-compatible combined projector CSV with known truth, "
            "then run make_rank2_Q2_boundary_delta.py + report_rank2_island_swap_diagnostics.py + report_boundary_delta_dashboard.py. "
            "Designed to stress gauge semantics, identity continuity, and conditioning gates."
        )
    )

    ap.add_argument("--out_dir", required=True, help="Output directory for synthetic run artifacts")
    ap.add_argument("--overwrite", type=int, default=0)

    ap.add_argument("--level", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--N", type=int, default=2048)
    ap.add_argument("--N2", type=int, default=0, help="Second N (default 2N)")

    ap.add_argument("--sigma", type=float, default=1.005)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--dt", type=float, default=0.01)

    ap.add_argument("--n_corridor", type=int, default=40)
    ap.add_argument("--n_swap", type=int, default=18)
    ap.add_argument("--n_cond", type=int, default=18)

    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--c_star_re", type=float, default=0.96)
    ap.add_argument("--c_star_im", type=float, default=-0.02)

    ap.add_argument("--p_err", type=float, default=1.0, help="Error scaling exponent: eps_N ~ eps0 * N^{-p_err}")
    ap.add_argument("--eps0", type=float, default=2.0, help="Base absolute complex error scale")

    ap.add_argument(
        "--force_adversaries",
        type=int,
        default=0,
        help="If 1, include swap/cond sides even at level=2 (for detectability/ROC sweeps).",
    )
    ap.add_argument(
        "--gauge_phase_wobble",
        type=float,
        default=0.0,
        help="Amplitude (radians) of a non-global phase drift applied to N2 quotient (level>=2).",
    )
    ap.add_argument(
        "--gauge_amp_wobble",
        type=float,
        default=0.0,
        help="Amplitude (relative) of a non-global amplitude drift applied to N2 quotient (level>=2).",
    )

    ap.add_argument(
        "--swap_strength",
        type=float,
        default=1.0,
        help="Level-3 swap severity in island invariants: 0=corridor-like, 1=full mismatch",
    )
    ap.add_argument(
        "--swap_q_strength",
        type=float,
        default=1.0,
        help="Level-3 swap severity in quotient field: 0=corridor-like, 1=full alternate Q",
    )
    ap.add_argument("--cond_sep2", type=float, default=5e-4, help="Level-3 conditioning sep2 value (cond side)")
    ap.add_argument("--cond_tinv", type=float, default=5e4, help="Level-3 conditioning Tin_inv value (cond side)")
    ap.add_argument(
        "--cond_q_noise",
        type=float,
        default=0.0,
        help="Optional extra absolute complex noise added to Q on the cond side",
    )

    ap.add_argument(
        "--emit_fit_side",
        type=int,
        default=0,
        help="If 1, add a 'fit' side used to fit c* (can be contaminated).",
    )
    ap.add_argument("--fit_n", type=int, default=0, help="Number of points on the fit side (default n_corridor)")
    ap.add_argument(
        "--fit_contam_swap_frac",
        type=float,
        default=0.0,
        help="Fraction of fit-side points with swap-like behavior",
    )
    ap.add_argument(
        "--fit_contam_cond_frac",
        type=float,
        default=0.0,
        help="Fraction of fit-side points with cond-like behavior",
    )

    # Delta+dashboard knobs.
    ap.add_argument("--quotient", default="Q2_tr", choices=["Q2", "Q2_phi2", "Q2_tr"])
    ap.add_argument("--lsfit_include_sides", default="corridor", help="Comma-separated sides used to fit c*")
    ap.add_argument("--taus", default="0.3,0.5")
    ap.add_argument("--taus_logabs", default="0.2,0.5")
    ap.add_argument("--taus_arg", default="0.5,1.0")

    ap.add_argument("--gate_tin_inv_max", type=float, default=500.0)
    ap.add_argument("--gate_sep2_min", type=float, default=0.01)
    ap.add_argument("--gate_lam_cost_max", type=float, default=2.0)

    ap.add_argument(
        "--headline_accept",
        default="tier01_logboth",
        choices=["tier01_logboth", "tier01_only", "tier01_rel", "tier01_rel_logboth"],
        help="Which acceptance definition is treated as headline for operating metrics.",
    )

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        if int(args.overwrite) != 1:
            raise SystemExit(f"out_dir exists (pass --overwrite=1 to reuse): {out_dir}")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    N = int(args.N)
    N2 = int(args.N2) if int(args.N2) else int(2 * N)

    params = SynthParams(
        level=int(args.level),
        N=N,
        N2=N2,
        sigma=float(args.sigma),
        t0=float(args.t0),
        dt=float(args.dt),
        n_corridor=int(args.n_corridor),
        n_swap=int(args.n_swap),
        n_cond=int(args.n_cond),
        seed=int(args.seed),
        c_star=complex(float(args.c_star_re), float(args.c_star_im)),
        p_err=float(args.p_err),
        eps0=float(args.eps0),
        force_adversaries=bool(int(args.force_adversaries)),
        gauge_phase_wobble=float(args.gauge_phase_wobble),
        gauge_amp_wobble=float(args.gauge_amp_wobble),
        swap_strength=float(args.swap_strength),
        swap_q_strength=float(args.swap_q_strength),
        cond_sep2=float(args.cond_sep2),
        cond_tinv=float(args.cond_tinv),
        cond_q_noise=float(args.cond_q_noise),
        emit_fit_side=bool(int(args.emit_fit_side)),
        fit_n=int(args.fit_n),
        fit_contam_swap_frac=float(args.fit_contam_swap_frac),
        fit_contam_cond_frac=float(args.fit_contam_cond_frac),
    )

    # Emit combined projector CSV.
    proj = _build_projector_combined(params)
    proj_name = "synthetic_proj_combined.csv"
    proj_path = out_dir / proj_name
    proj.to_csv(proj_path, index=False)
    print(f"wrote projector_csv: {proj_path} rows={len(proj)}")

    # Use repo root as subprocess cwd so relative paths are stable.
    repo_root = Path(__file__).resolve().parents[1]

    py = sys.executable
    delta_tool = Path(__file__).parent / "make_rank2_Q2_boundary_delta.py"
    aug_tool = Path(__file__).parent / "report_rank2_island_swap_diagnostics.py"
    dash_tool = Path(__file__).parent / "report_boundary_delta_dashboard.py"

    # Build delta with lsfit gauge (this is the main calibration artifact).
    delta_csv = out_dir / "synthetic_delta_lsfit.csv"
    delta_js = out_dir / "synthetic_delta_lsfit_summary.json"

    cmd_delta = [
        py,
        "-u",
        str(delta_tool),
        "--run_dir",
        str(out_dir.resolve()),
        "--projector_csv",
        proj_name,
        "--basis",
        "incoming",
        "--block",
        "plus",
        "--n_edge",
        "80",
        "--method",
        "schur",
        "--N",
        str(int(N)),
        "--N2",
        str(int(N2)),
        "--quotient",
        str(args.quotient),
        "--lsfit",
        "1",
        "--lsfit_include_sides",
        str(args.lsfit_include_sides),
        "--out_csv",
        str(delta_csv.name),
        "--out_json",
        str(delta_js.name),
    ]
    _run(cmd_delta, cwd=repo_root)

    # Also build a raw delta (optional, useful for debugging gauge removal).
    raw_csv = out_dir / "synthetic_delta_raw.csv"
    raw_js = out_dir / "synthetic_delta_raw_summary.json"
    cmd_raw = [
        py,
        "-u",
        str(delta_tool),
        "--run_dir",
        str(out_dir.resolve()),
        "--projector_csv",
        proj_name,
        "--basis",
        "incoming",
        "--block",
        "plus",
        "--n_edge",
        "80",
        "--method",
        "schur",
        "--N",
        str(int(N)),
        "--N2",
        str(int(N2)),
        "--quotient",
        str(args.quotient),
        "--lsfit",
        "0",
        "--out_csv",
        str(raw_csv.name),
        "--out_json",
        str(raw_js.name),
    ]
    _run(cmd_raw, cwd=repo_root)

    # Augment lsfit delta with identity + log metrics under summary convention.
    aug_csv = out_dir / "synthetic_delta_lsfit__aug_lam.csv"
    cmd_aug = [
        py,
        "-u",
        str(aug_tool),
        "--delta_csv",
        str(delta_csv.resolve()),
        "--summary_json",
        str(delta_js.resolve()),
        "--projector_combined_csv",
        str(proj_path.resolve()),
        "--out_csv",
        str(aug_csv.resolve()),
        "--log_metric_mode",
        "summary",
    ]
    _run(cmd_aug, cwd=repo_root)

    # Dashboard on augmented CSV (uses precomputed dlog/darg and lam_match_cost).
    dash_js = out_dir / "synthetic_dashboard_tiered.json"
    cmd_dash = [
        py,
        "-u",
        str(dash_tool),
        "--delta_csv",
        str(aug_csv.resolve()),
        "--summary_json",
        str(delta_js.resolve()),
        "--taus",
        str(args.taus),
        "--taus_logabs",
        str(args.taus_logabs),
        "--taus_arg",
        str(args.taus_arg),
        "--k",
        "5",
        "--min_points",
        "10",
        "--gate_tin_inv_max",
        str(float(args.gate_tin_inv_max)),
        "--gate_sep2_min",
        str(float(args.gate_sep2_min)),
        "--gate_lam_cost_max",
        str(float(args.gate_lam_cost_max)),
        "--out_json",
        str(dash_js.resolve()),
    ]
    _run(cmd_dash, cwd=repo_root)

    # Compact report.
    summ = _read_json(delta_js)
    dash = _read_json(dash_js)

    c_hat = _extract_cstar(summ)

    # Parse decision thresholds for quantitative false-accept / false-reject reporting.
    taus_rel = [
        float(x)
        for x in [s.strip() for s in str(args.taus).split(",")]
        if _to_float(x) is not None
    ]
    taus_L = [
        float(x)
        for x in [s.strip() for s in str(args.taus_logabs).split(",")]
        if _to_float(x) is not None
    ]
    taus_A = [
        float(x)
        for x in [s.strip() for s in str(args.taus_arg).split(",")]
        if _to_float(x) is not None
    ]
    tau_rel0 = float(taus_rel[0]) if taus_rel else None
    tau_L0 = float(taus_L[0]) if taus_L else None
    tau_A0 = float(taus_A[0]) if taus_A else None

    # c* errors (if recovered).
    c_err_abs = None
    c_err_angle = None
    if c_hat is not None:
        c_err_abs = float(abs(c_hat - params.c_star))
        ratio = complex(c_hat) / complex(params.c_star) if abs(params.c_star) else 1.0 + 0.0j
        c_err_angle = float(_wrap_pi(float(np.angle(ratio))))

    report: dict[str, Any] = {
        "out_dir": str(out_dir),
        "level": int(params.level),
        "N": int(params.N),
        "N2": int(params.N2),
        "quotient": str(args.quotient),
        "synth_knobs": {
            "swap_strength": float(params.swap_strength),
            "swap_q_strength": float(params.swap_q_strength),
            "cond_sep2": float(params.cond_sep2),
            "cond_tinv": float(params.cond_tinv),
            "cond_q_noise": float(params.cond_q_noise),
            "force_adversaries": bool(params.force_adversaries),
            "gauge_phase_wobble": float(params.gauge_phase_wobble),
            "gauge_amp_wobble": float(params.gauge_amp_wobble),
            "emit_fit_side": bool(params.emit_fit_side),
            "fit_n": int(params.fit_n) if int(params.fit_n) else int(params.n_corridor),
            "fit_contam_swap_frac": float(params.fit_contam_swap_frac),
            "fit_contam_cond_frac": float(params.fit_contam_cond_frac),
        },
        "truth": {
            "c_star_re": float(params.c_star.real),
            "c_star_im": float(params.c_star.imag),
            "c_star_abs": float(abs(params.c_star)),
        },
        "recovered": {
            "c_star_re": float(c_hat.real) if c_hat is not None else None,
            "c_star_im": float(c_hat.imag) if c_hat is not None else None,
            "c_star_abs": float(abs(c_hat)) if c_hat is not None else None,
            "c_star_error_abs": c_err_abs,
            "c_star_error_angle": c_err_angle,
        },
        "decision": {
            "gate_tin_inv_max": float(args.gate_tin_inv_max),
            "gate_sep2_min": float(args.gate_sep2_min),
            "gate_lam_cost_max": float(args.gate_lam_cost_max),
            "tau_rel": tau_rel0,
            "tau_logabs": tau_L0,
            "tau_arg": tau_A0,
            "positive_sides": ["corridor"],
            "ignored_sides": ["fit"],
            "headline_accept_mode": str(args.headline_accept),
            "accept_defs": {
                "tier01_only": "admissible & tier0_gate & identity_gate",
                "tier01_rel": "admissible & tier0_gate & identity_gate & (dQ_rel<=tau_rel)",
                "tier01_logboth": "admissible & tier0_gate & identity_gate & (dlog_abs<=tau_logabs) & (darg<=tau_arg)",
                "tier01_rel_logboth": "admissible & tier0_gate & identity_gate & (dQ_rel<=tau_rel) & (dlog_abs<=tau_logabs) & (darg<=tau_arg)",
            },
        },
        "artifacts": {
            "projector_csv": str(proj_path),
            "delta_lsfit_csv": str(delta_csv),
            "delta_lsfit_summary": str(delta_js),
            "delta_raw_csv": str(raw_csv),
            "delta_raw_summary": str(raw_js),
            "aug_csv": str(aug_csv),
            "dashboard_json": str(dash_js),
        },
        "by_side": {},
    }

    # Compute planted-truth operating characteristics directly from the augmented CSV.
    # This keeps the report consistent with the same gate thresholds used in the dashboard.
    try:
        df_aug = pd.read_csv(aug_csv)
        # Admissible mask.
        if "admissible_pair" in df_aug.columns:
            adm = df_aug["admissible_pair"].astype(str).str.lower().isin(["true", "1", "yes", "ok"])
        elif "admissible" in df_aug.columns:
            adm = df_aug["admissible"].astype(str).str.lower().isin(["true", "1", "yes", "ok"])
        elif "status_pair_ok" in df_aug.columns:
            adm = df_aug["status_pair_ok"].astype(bool)
        elif "status_N" in df_aug.columns and "status_N2" in df_aug.columns:
            adm = (df_aug["status_N"].astype(str) == "ok") & (df_aug["status_N2"].astype(str) == "ok")
        else:
            adm = pd.Series([True] * len(df_aug), index=df_aug.index, dtype=bool)

        # Tier-0 gate.
        gate = pd.Series([True] * len(df_aug), index=df_aug.index, dtype=bool)
        if math.isfinite(float(args.gate_tin_inv_max)) and "Tin_inv_norm1_est_N2" in df_aug.columns:
            tin = pd.to_numeric(df_aug["Tin_inv_norm1_est_N2"], errors="coerce")
            gate &= tin <= float(args.gate_tin_inv_max)
        if math.isfinite(float(args.gate_sep2_min)):
            sep_col = None
            for cand in ["sep2_N2", "sep2_track_N2"]:
                if cand in df_aug.columns:
                    sep_col = cand
                    break
            if sep_col is not None:
                sep = pd.to_numeric(df_aug[sep_col], errors="coerce")
                gate &= sep >= float(args.gate_sep2_min)

        # Tier-1 identity.
        ident = pd.Series([True] * len(df_aug), index=df_aug.index, dtype=bool)
        if math.isfinite(float(args.gate_lam_cost_max)) and "lam_match_cost" in df_aug.columns:
            lamc = pd.to_numeric(df_aug["lam_match_cost"], errors="coerce")
            ident &= lamc <= float(args.gate_lam_cost_max)

        # Rel delta column name.
        rel_col = "dQ2_rel" if "dQ2_rel" in df_aug.columns else ("dQ_rel" if "dQ_rel" in df_aug.columns else None)
        dlog_col = "dlog_abs" if "dlog_abs" in df_aug.columns else None
        darg_col = "darg" if "darg" in df_aug.columns else None

        rel_good = pd.Series([True] * len(df_aug), index=df_aug.index, dtype=bool)
        if rel_col is not None and tau_rel0 is not None:
            relv = pd.to_numeric(df_aug[rel_col], errors="coerce")
            rel_good &= relv <= float(tau_rel0)

        log_good = pd.Series([True] * len(df_aug), index=df_aug.index, dtype=bool)
        if dlog_col is not None and tau_L0 is not None:
            xv = pd.to_numeric(df_aug[dlog_col], errors="coerce")
            log_good &= xv <= float(tau_L0)
        if darg_col is not None and tau_A0 is not None:
            yv = pd.to_numeric(df_aug[darg_col], errors="coerce")
            log_good &= yv <= float(tau_A0)

        accept_tier01 = adm & gate & ident
        accept_modes: dict[str, pd.Series] = {
            "tier01_only": accept_tier01,
            "tier01_rel": (accept_tier01 & rel_good),
            "tier01_logboth": (accept_tier01 & log_good),
            "tier01_rel_logboth": (accept_tier01 & rel_good & log_good),
        }

        headline_mode = str(args.headline_accept)
        accept = accept_modes.get(headline_mode, accept_modes["tier01_logboth"])
        side_series = df_aug["side"].astype(str) if "side" in df_aug.columns else pd.Series(["(all)"] * len(df_aug))

        pos = set(["corridor"])
        ign = set(["fit"])

        def _count(mask: pd.Series) -> int:
            try:
                return int(mask.sum())
            except Exception:
                return 0

        def _metrics_for(mask_accept: pd.Series) -> dict[str, Any]:
            # Global precision/recall (excluding ignored sides).
            use_mask = ~side_series.isin(list(ign))
            y_true = side_series.isin(list(pos)) & use_mask
            y_pred = mask_accept & use_mask

            tp = _count(y_true & y_pred)
            fp = _count((~y_true) & y_pred)
            fn = _count(y_true & (~y_pred))
            tn = _count((~y_true) & (~y_pred) & use_mask)

            precision = float(tp / (tp + fp)) if (tp + fp) else None
            recall = float(tp / (tp + fn)) if (tp + fn) else None

            # By-side false accept / reject rates.
            by_side_oper: dict[str, Any] = {}
            for side in sorted(side_series.unique().tolist()):
                if side in ign:
                    continue
                m = (side_series == side) & use_mask
                n = _count(m)
                n_acc = _count(mask_accept & m)
                is_pos = side in pos
                by_side_oper[side] = {
                    "n": n,
                    "n_accept": n_acc,
                    "accept_frac": float(n_acc / n) if n else None,
                    "false_accept_frac": (float(n_acc / n) if (n and (not is_pos)) else None),
                    "false_reject_frac": (float((n - n_acc) / n) if (n and is_pos) else None),
                }

            return {
                "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
                "precision": precision,
                "recall": recall,
                "by_side": by_side_oper,
            }

        headline_payload = _metrics_for(accept)
        report["operating"] = {
            "headline_mode": headline_mode,
            **headline_payload,
            "modes": {k: _metrics_for(v) for k, v in accept_modes.items()},
        }
    except Exception as e:
        report["operating"] = {"error": str(e)}

    by_side = (dash.get("by_side") or {}) if isinstance(dash, dict) else {}
    for side, payload in by_side.items():
        if not isinstance(payload, dict):
            continue
        gate = payload.get("gate") or {}
        ident = payload.get("identity") or {}
        log = payload.get("log") or {}
        log_good = payload.get("log_good") or {}

        rel = payload.get("rel") or {}
        ab = payload.get("abs") or {}
        good_tier01 = payload.get("good_tier01") or {}

        tier01 = (log_good.get("tier01") or {}) if isinstance(log_good, dict) else {}
        tier01_fracs = tier01.get("fracs") or {}

        report["by_side"][str(side)] = {
            "n_total": payload.get("n_total"),
            "n_admissible": payload.get("n_admissible"),
            "gate_frac_of_admissible": (gate.get("gate_frac_of_admissible") if isinstance(gate, dict) else None),
            "tier01_n": (tier01.get("n") if isinstance(tier01, dict) else None),

            "dQ_rel_p50": (rel.get("p50") if isinstance(rel, dict) else None),
            "dQ_rel_p95": (rel.get("p95") if isinstance(rel, dict) else None),
            "dQ_rel_sup": (rel.get("sup") if isinstance(rel, dict) else None),
            "dQ_abs_p50": (ab.get("p50") if isinstance(ab, dict) else None),
            "dQ_abs_p95": (ab.get("p95") if isinstance(ab, dict) else None),
            "dQ_abs_sup": (ab.get("sup") if isinstance(ab, dict) else None),

            "lam_match_cost_p95": (ident.get("lam_match_cost_p95") if isinstance(ident, dict) else None),
            "dlog_abs_p50": (log.get("dlog_abs_p50") if isinstance(log, dict) else None),
            "dlog_abs_p95": (log.get("dlog_abs_p95") if isinstance(log, dict) else None),
            "darg_p50": (log.get("darg_p50") if isinstance(log, dict) else None),
            "darg_p95": (log.get("darg_p95") if isinstance(log, dict) else None),

            "good_tier01": (good_tier01 if isinstance(good_tier01, dict) else None),
            "tier01_phase_fracs": (tier01_fracs.get("phase") if isinstance(tier01_fracs, dict) else None),
            "tier01_both_fracs": (tier01_fracs.get("both") if isinstance(tier01_fracs, dict) else None),
        }

    rep_path = out_dir / "synthetic_calibration_report.json"
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== synthetic calibration report ===")
    print(f"level={params.level} N={N} N2={N2} quotient={args.quotient}")
    print(f"true c*={params.c_star}  recovered c*={c_hat}")
    print(f"report_json={rep_path}")

    for side, r in report["by_side"].items():
        print(
            f"side={side} n_total={r.get('n_total')} n_adm={r.get('n_admissible')} gate_frac_adm={r.get('gate_frac_of_admissible')} "
            f"tier01_n={r.get('tier01_n')} lam_p95={r.get('lam_match_cost_p95')} darg_p50={r.get('darg_p50')} darg_p95={r.get('darg_p95')}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
