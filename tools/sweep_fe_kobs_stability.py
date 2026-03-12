"""Sweep k_max / completion / p_mode and summarize k-indexed observables.

This is a thin experiment harness around tools/fe_defect_perturbation_u0.py.
It exists to answer the next honest question:

- Does Möbius primitive extraction on a real packet observable (over k) produce a
  more stable object as k_max varies and as completion changes?

It does NOT claim to extract primes or an Euler product.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CompletionCase:
    mode: str  # fit_u0 | fixed
    basis: str
    fixed: str
    label: str


def _parse_int_list(s: str) -> list[int]:
    out: list[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _parse_kmax_spec(spec: str) -> list[int]:
    spec = str(spec).strip()
    if not spec:
        raise ValueError("k_max spec must be non-empty")
    if ":" in spec:
        toks = [t.strip() for t in spec.split(":") if t.strip()]
        if len(toks) != 2:
            raise ValueError("k_max spec range must be 'a:b'")
        a = int(toks[0])
        b = int(toks[1])
        if a <= 0 or b < a:
            raise ValueError("bad k_max range")
        return list(range(a, b + 1))
    return sorted(set(_parse_int_list(spec)))


def _parse_completion_cases(s: str) -> list[CompletionCase]:
    """Parse semicolon-separated completion specs.

    Supported forms:
      - fit_u0:poly2
      - fit_u0:poly2_gamma
      - fixed:zeta
      - fixed:gl2
      - fixed:zeta2

    A label is auto-generated.
    """

    s = str(s).strip()
    if not s:
        return [CompletionCase(mode="fit_u0", basis="poly2", fixed="zeta", label="fit_u0:poly2")]

    out: list[CompletionCase] = []
    for part in [p.strip() for p in s.split(";") if p.strip()]:
        if ":" not in part:
            raise ValueError("completion case must be like 'fit_u0:poly2' or 'fixed:zeta'")
        mode, tail = [t.strip() for t in part.split(":", 1)]
        mode = mode.lower()
        if mode == "fit_u0":
            basis = tail
            if basis not in {"none", "poly2", "poly2_gamma"}:
                raise ValueError("fit_u0 basis must be one of: none, poly2, poly2_gamma")
            out.append(CompletionCase(mode="fit_u0", basis=basis, fixed="zeta", label=f"fit_u0:{basis}"))
        elif mode == "fixed":
            fixed = tail
            if fixed not in {"none", "zeta", "gl2", "zeta2"}:
                raise ValueError("fixed completion must be one of: none, zeta, gl2, zeta2")
            out.append(CompletionCase(mode="fixed", basis="poly2", fixed=fixed, label=f"fixed:{fixed}"))
        else:
            raise ValueError("completion mode must be fit_u0 or fixed")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Sweep FE defect runs across k_max, p_mode, and completion settings, "
            "recording raw k-tower observable summaries (F_k) and Möbius-extracted summaries (P_k)."
        )
    )

    ap.add_argument("--primes_global", default="2,3,5,7,11,13")
    ap.add_argument("--k_max", default="3:8", help="k_max sweep spec: 'a:b' or '3,4,5'")
    ap.add_argument("--p_modes", default="p,logp,invp", help="Comma list of p_mode values")

    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--t_min", type=float, default=10.0)
    ap.add_argument("--t_max", type=float, default=12.0)
    ap.add_argument("--n_t", type=int, default=51)
    ap.add_argument("--h", type=float, default=0.05)
    ap.add_argument(
        "--u_list",
        default="",
        help=(
            "Optional comma list of additional u values to evaluate in fe_defect_perturbation_u0.py. "
            "Needed if you want to summarize the 'Fu' row at a specific u."
        ),
    )

    ap.add_argument(
        "--boundary",
        default="auto",
        help=(
            "Boundary indices as 'i,j' in 0..5, or 'auto'. "
            "Use a fixed boundary when comparing p_mode choices to avoid gauge-search confounds."
        ),
    )
    ap.add_argument(
        "--schur_sign",
        choices=["-", "+"],
        default="+",
        help="Schur sign to use when --boundary is not auto.",
    )

    ap.add_argument(
        "--completion_cases",
        default="fit_u0:poly2;fixed:zeta",
        help="Semicolon list like 'fit_u0:poly2;fixed:zeta;fixed:gl2'",
    )

    ap.add_argument("--prime_power_mode", choices=["direct", "x_power", "bulk_power"], default="direct")

    ap.add_argument(
        "--k_obs_mode",
        choices=["none", "trS", "trLam", "detS", "devS_pm_I", "froLam"],
        default="trS",
    )
    ap.add_argument("--k_obs_aggregate", choices=["sum", "mean"], default="sum")
    ap.add_argument("--k_obs_apply_mobius", choices=["0", "1"], default="1")

    ap.add_argument(
        "--target_label",
        default="F0",
        help="Which fe_defect_perturbation_u0 output row label to summarize (default: F0).",
    )
    ap.add_argument(
        "--target_u",
        type=float,
        default=float("nan"),
        help=(
            "If target_label is 'Fu', select the row with u approximately equal to this value. "
            "Ignored otherwise."
        ),
    )

    ap.add_argument("--out_csv", required=True, help="Output summary CSV")
    ap.add_argument(
        "--out_runs_json",
        default="",
        help="Optional JSON log with the exact command-lines executed.",
    )

    ap.add_argument(
        "--keep_tmp",
        choices=["0", "1"],
        default="0",
        help="If 1, keep intermediate per-run CSVs; default deletes them after summarizing.",
    )

    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    fe_tool = repo_root / "tools" / "fe_defect_perturbation_u0.py"
    if not fe_tool.exists():
        raise FileNotFoundError(f"missing {fe_tool}")

    k_max_list = _parse_kmax_spec(str(args.k_max))
    p_modes = [p.strip() for p in str(args.p_modes).split(",") if p.strip()]
    if not p_modes:
        raise ValueError("p_modes must be non-empty")

    cases = _parse_completion_cases(str(args.completion_cases))

    out_path = Path(str(args.out_csv))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_logs: list[dict] = []
    rows: list[dict] = []

    keep_tmp = str(args.keep_tmp).strip() == "1"

    for case in cases:
        for p_mode in p_modes:
            for k_max in k_max_list:
                tmp_out = out_path.parent / f"_tmp_fe_kobs_{case.label.replace(':','_')}_pm_{p_mode}_k{k_max}.csv"

                cmd = [
                    sys.executable,
                    str(fe_tool),
                    "--primes_global",
                    str(args.primes_global),
                    "--k_max",
                    str(int(k_max)),
                    "--sigma",
                    str(float(args.sigma)),
                    "--t_min",
                    str(float(args.t_min)),
                    "--t_max",
                    str(float(args.t_max)),
                    "--n_t",
                    str(int(args.n_t)),
                    "--h",
                    str(float(args.h)),
                    "--boundary",
                    str(args.boundary),
                    "--schur_sign",
                    str(args.schur_sign),
                    "--completion_mode",
                    str(case.mode),
                    "--completion_basis",
                    str(case.basis),
                    "--fixed_completion",
                    str(case.fixed),
                    "--p_mode",
                    str(p_mode),
                    "--prime_power_mode",
                    str(args.prime_power_mode),
                    "--k_obs_mode",
                    str(args.k_obs_mode),
                    "--k_obs_aggregate",
                    str(args.k_obs_aggregate),
                    "--k_obs_apply_mobius",
                    str(args.k_obs_apply_mobius),
                    "--u_list",
                    str(args.u_list),
                    "--out_csv",
                    str(tmp_out),
                ]

                run_logs.append({"case": case.label, "p_mode": str(p_mode), "k_max": int(k_max), "cmd": cmd})

                proc = subprocess.run(cmd)
                if proc.returncode != 0:
                    raise RuntimeError(f"run failed: case={case.label} p_mode={p_mode} k_max={k_max}")

                df = pd.read_csv(tmp_out)
                if df.empty:
                    continue
                target_label = str(args.target_label).strip()
                d0 = df[df["label"] == target_label]
                if d0.empty:
                    continue
                if target_label == "Fu" and math.isfinite(float(args.target_u)):
                    target_u = float(args.target_u)
                    d0 = d0[np.abs(d0["u"].astype(float) - target_u) <= 1e-12]
                    if d0.empty:
                        continue
                r0 = dict(d0.iloc[0].to_dict())

                rows.append(
                    {
                        "case": case.label,
                        "completion_mode": str(case.mode),
                        "completion_basis": str(case.basis),
                        "fixed_completion": str(case.fixed),
                        "p_mode": str(p_mode),
                        "k_max": int(k_max),
                        "target_label": str(target_label),
                        "target_u": float(r0.get("u", float("nan"))),
                        "rel_l2_F0": float(r0.get("rel_l2", float("nan"))),
                        "max_abs_F0": float(r0.get("max_abs", float("nan"))),
                        "kobs_F_l2": float(r0.get("kobs_F_l2", float("nan"))),
                        "kobs_F_maxabs": float(r0.get("kobs_F_maxabs", float("nan"))),
                        "kobs_P_l2": float(r0.get("kobs_P_l2", float("nan"))),
                        "kobs_P_maxabs": float(r0.get("kobs_P_maxabs", float("nan"))),
                    }
                )

                if (not keep_tmp) and tmp_out.exists():
                    try:
                        tmp_out.unlink()
                    except Exception:
                        # Best-effort cleanup only.
                        pass

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")

    out_runs_json = str(args.out_runs_json).strip()
    if out_runs_json:
        p = Path(out_runs_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(run_logs, indent=2), encoding="utf-8")
        print(f"wrote {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
