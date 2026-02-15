import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GroupKey:
    seed: int
    anchor_seed: int
    wlo: float
    whi: float
    mu: float
    block: int
    dim: int


def _safe_float(x: object) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _unique_sorted(xs: Iterable[float]) -> List[float]:
    out = sorted({float(x) for x in xs if np.isfinite(float(x))})
    return out


def _refine_interval(a: float, b: float, n_sub: int) -> List[float]:
    a = float(a)
    b = float(b)
    n = int(n_sub)
    if n <= 1:
        return [a, b]
    lo, hi = (a, b) if a <= b else (b, a)
    return [float(x) for x in np.linspace(lo, hi, num=n + 1, endpoint=True).tolist()]


def _format_list(xs: Sequence[float], *, ndp: int = 6) -> str:
    fmt = f"{{:.{int(ndp)}g}}"
    return ",".join(fmt.format(float(x)) for x in xs)


def load_summary(out_root: str) -> Dict:
    p = os.path.join(out_root, "phase3e_elambda_suite_summary.json")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_df(out_root: str, name: str) -> pd.DataFrame:
    p = os.path.join(out_root, name)
    if not os.path.exists(p):
        raise SystemExit(f"Missing required file: {p}")
    return pd.read_csv(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True, help="Phase-3E run folder")
    ap.add_argument("--top_k", type=int, default=8, help="How many candidate groups to keep")
    ap.add_argument("--min_commutator", type=float, default=0.5, help="Minimum ||[A,B]|| to consider")
    ap.add_argument(
        "--max_flip_overlap",
        type=float,
        default=0.995,
        help="Keep only pairs with flip_overlap <= this (avoid trivially identical gens)",
    )
    ap.add_argument(
        "--lambda_subdivide",
        type=int,
        default=4,
        help="When refining a lambda cell [l0,l1], subdivide into this many sub-intervals (default 4)",
    )
    ap.add_argument(
        "--mu_halfwidth",
        type=float,
        default=0.25,
        help="Suggested mu neighborhood half-width around the mu slice (clamped to [0,1])",
    )
    ap.add_argument(
        "--mu_step",
        type=float,
        default=0.125,
        help="Suggested mu step in neighborhood refinement (default 0.125)",
    )
    args = ap.parse_args()

    out_root = str(args.out_root)
    summ = load_summary(out_root)

    # Use the parameter-space lambdas (in [0,1]) if available.
    lambdas = summ.get("lambdas", [])
    if not lambdas:
        lambdas = summ.get("lambdas_eff", [])
    lambdas = [float(x) for x in lambdas]

    comp = load_df(out_root, "phase3f_batch_composition.csv")

    # Filter to genuine pairs.
    comp = comp[(comp["kind"] == "pair")].copy()
    comp = comp[np.isfinite(comp["norm_commutator"].astype(float, errors="ignore"))]
    comp = comp[comp["genA"].astype(str) != comp["genB"].astype(str)]
    comp["norm_commutator"] = comp["norm_commutator"].astype(float)
    if "flip_overlap" in comp.columns:
        comp["flip_overlap"] = comp["flip_overlap"].astype(float)
        comp = comp[np.isfinite(comp["flip_overlap"])]
        comp = comp[comp["flip_overlap"] <= float(args.max_flip_overlap)]

    comp = comp[comp["norm_commutator"] >= float(args.min_commutator)]
    if comp.empty:
        raise SystemExit("No candidate generator pairs passed filters; try lowering --min_commutator or raising --max_flip_overlap")

    group_cols = ["seed", "anchor_seed", "wlo", "whi", "mu", "block", "dim"]
    for c in group_cols:
        if c not in comp.columns:
            raise SystemExit(f"composition CSV missing column: {c}")

    # For each group, pick the strongest commutator.
    comp_sorted = comp.sort_values("norm_commutator", ascending=False)
    best = comp_sorted.groupby(group_cols, as_index=False).first()
    best = best.sort_values("norm_commutator", ascending=False).head(int(args.top_k)).reset_index(drop=True)

    cells = load_df(out_root, "phase3f_event_cells_with_gen.csv")

    # Build recommendations per selected group.
    rows: List[Dict] = []
    cmd_lines: List[str] = []

    outA = str(summ.get("out_root_a", "")).strip()
    outB = str(summ.get("out_root_b", "")).strip()
    outB2 = str(summ.get("out_root_b2", "")).strip()
    deform_alpha = _safe_float(summ.get("deform_alpha", 1.0))

    python_exe = sys.executable

    for _, r in best.iterrows():
        gk = GroupKey(
            seed=int(r["seed"]),
            anchor_seed=int(r["anchor_seed"]),
            wlo=float(r["wlo"]),
            whi=float(r["whi"]),
            mu=float(r["mu"]),
            block=int(r["block"]),
            dim=int(r["dim"]),
        )
        genA = str(r.get("genA", "")).strip()
        genB = str(r.get("genB", "")).strip()
        comm = float(r["norm_commutator"])
        flip = _safe_float(r.get("flip_overlap", float("nan")))

        # Collect lambda cells involved in these two generators within this group.
        sub = cells[
            (cells["seed"].astype(int) == gk.seed)
            & (cells["anchor_seed"].astype(int) == gk.anchor_seed)
            & (cells["wlo"].astype(float) == gk.wlo)
            & (cells["whi"].astype(float) == gk.whi)
            & (cells["mu"].astype(float) == gk.mu)
            & (cells["block"].astype(int) == gk.block)
            & (cells["dim"].astype(int) == gk.dim)
            & (cells["gen_id"].astype(str).isin([genA, genB]))
        ].copy()

        lam_pts: List[float] = []
        if not sub.empty and ("lambda0" in sub.columns) and ("lambda1" in sub.columns):
            for a, b in zip(sub["lambda0"].tolist(), sub["lambda1"].tolist()):
                lam_pts.extend(_refine_interval(float(a), float(b), int(args.lambda_subdivide)))

        # Always include the global endpoints for safety.
        lam_pts.extend([0.0, 1.0])
        if lambdas:
            lam_pts.extend(lambdas)
        lambdas_suggest = _unique_sorted(lam_pts)

        # Suggested mu neighborhood.
        mu0 = float(gk.mu)
        hw = float(args.mu_halfwidth)
        step = float(args.mu_step)
        mu_pts: List[float] = [mu0]
        if np.isfinite(hw) and hw > 0 and np.isfinite(step) and step > 0:
            lo = max(0.0, mu0 - hw)
            hi = min(1.0, mu0 + hw)
            n = int(round((hi - lo) / step))
            n = max(n, 1)
            mu_pts.extend([float(x) for x in np.linspace(lo, hi, num=n + 1, endpoint=True).tolist()])
        mus_suggest = _unique_sorted(mu_pts)

        out_ref = os.path.join(out_root, f"refine_prime_seed{gk.seed}_a{gk.anchor_seed}_w{gk.wlo:g}_{gk.whi:g}_mu{gk.mu:g}_b{gk.block}_d{gk.dim}")

        rows.append(
            {
                **gk.__dict__,
                "genA": genA,
                "genB": genB,
                "norm_commutator": comm,
                "flip_overlap": flip,
                "suggest_lambdas": _format_list(lambdas_suggest, ndp=8),
                "suggest_mus": _format_list(mus_suggest, ndp=8),
                "suggest_out_root": out_ref,
            }
        )

        # The Phase-3E runner only supports seed/anchor/window filtering after our patch.
        cmd = (
            f"{python_exe} phase3e_elambda_loop_suite.py "
            f"--out_root_a {outA} --out_root_b {outB} "
            f"--out_root_b2 {outB2} "
            f"--out_root {out_ref} "
            f"--only_seed {gk.seed} --only_anchor {gk.anchor_seed} --only_window {gk.wlo}:{gk.whi} "
            f"--suite_mode fwd_rect --dump_holonomy pi_only --dump_holonomy_max_dim 16 "
            f"--lambdas {rows[-1]['suggest_lambdas']} --mus {rows[-1]['suggest_mus']} "
            f"--blocks {gk.block + 1} --refine_steps 2 --deform_alpha {deform_alpha}"
        )
        cmd_lines.append(cmd)

    out_csv = os.path.join(out_root, "phase3f_prime_candidates.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    out_txt = os.path.join(out_root, "phase3f_prime_candidates_commands.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("# Phase-3E refinement commands for prime-candidate groups\n")
        f.write("# Generated by tools/phase3f_pick_prime_candidates.py\n\n")
        for line in cmd_lines:
            f.write(line)
            f.write("\n\n")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_txt}")
    print("Top candidates:")
    show_cols = [
        "seed",
        "anchor_seed",
        "wlo",
        "whi",
        "mu",
        "block",
        "dim",
        "norm_commutator",
        "flip_overlap",
        "genA",
        "genB",
    ]
    print(pd.DataFrame(rows)[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
