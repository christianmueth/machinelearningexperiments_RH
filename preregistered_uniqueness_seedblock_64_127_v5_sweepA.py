"""Preregistered Phase-3B-legal scan: v5 knob sweep (A).

Goal:
- Keep the Phase-3B measurement spine frozen (dnmap_only, label-scramble null).
- Explore a small, preregistered family of v5 admissible backends where the
  self-dual harmonic modulation changes coupling strength, while eps auto-tune
  keeps overall Tp delta in the same target band.

This script mirrors the structure of preregistered_uniqueness_seedblock_64_127_v5.py
but changes the backend list.

Outputs under out_root:
- preregistration.json + preregistration_digest.txt
- raw.csv
- kpi_by_seed.csv
- kpi_fraction.csv
- uniqueness.csv

Usage (PowerShell):
  & .\.venv\Scripts\python.exe preregistered_uniqueness_seedblock_64_127_v5_sweepA.py \
      --out_root out_phase3B_preregistered_uniqueness_64_127_v5_sweepA \
      --N_null 1024 --dnmap_null_batch 8 --resume
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date

import numpy as np
import pandas as pd

import machinelearning_rh_colab_cells as h


def _digest(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return h.sha16(payload)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, default="out_phase3B_preregistered_uniqueness_64_127_v5_sweepA")
    ap.add_argument("--N_null", type=int, default=1024)
    ap.add_argument("--dnmap_null_batch", type=int, default=8)
    ap.add_argument("--amp", type=float, default=0.03)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out_root = args.out_root
    os.makedirs(out_root, exist_ok=True)

    seeds = list(range(64, 128))
    anchors = [2, 9, 14, 44, 46, 51, 60]

    prereg = {
        "date": str(date.today()),
        "phase": "3B->3C",
        "mode": "dnmap_only",
        "null_family": "label_scramble",
        "alpha": 0.01,
        "seeds": [min(seeds), max(seeds) + 1],
        "anchors": anchors,
        "k": 13,
        "N_null": int(args.N_null),
        "windows": [(2.0, 5.0), (0.6, 7.5)],
        "dnmap_stride": 2,
        "dnmap_null_batch": int(args.dnmap_null_batch),
        "amp": float(args.amp),
        "collision_tol": 0.0,
        # Keep the same “geometry policy” defaults used previously.
        "warp_mode": "ratio",
        "warp_unwrap": True,
        "warp_y_scale": 1.0,
        # Sweep family: v5 only. eps auto-tune keeps delta near 0.02.
        "backends": [],
        "measurement_frozen": True,
        "geometry_policy_frozen": True,
    }

    # Small preregistered sweep grid.
    # We vary harmonic_strength and gamma; keep theta_scale fixed to prior (0.125)
    # and keep eps auto-tune target delta at 0.02.
    theta = 0.125
    eps_target_delta = 0.02
    strengths = [0.5, 1.0, 2.0]
    gammas = [0.5, 1.0]

    for g in gammas:
        for s in strengths:
            prereg["backends"].append(
                {
                    "backend_label": f"geom_v5_t{theta}_g{g}_s{s}_autotune",
                    "tp_backend": "geom_warp_dirac_v5",
                    "geom_v5_theta_scale": theta,
                    "geom_v5_harmonic_gamma": g,
                    "geom_v5_harmonic_strength": s,
                    "eps_auto_tune": True,
                    "eps_target_delta": eps_target_delta,
                }
            )

    # Optional control backend (legacy) for sanity comparisons.
    prereg["backends"].insert(0, {"backend_label": "legacy", "tp_backend": "legacy"})

    digest = _digest(prereg)
    with open(os.path.join(out_root, "preregistration.json"), "w", encoding="utf-8") as f:
        json.dump(prereg, f, indent=2)
    with open(os.path.join(out_root, "preregistration_digest.txt"), "w", encoding="utf-8") as f:
        f.write(digest + "\n")

    # Base config: measurement spine is frozen in h.CFG_BASE and h.Phase3Harness.
    base = dict(h.CFG_BASE)
    base["phase3_mode"] = "dnmap_only"
    base["k"] = int(prereg["k"])
    base["windows"] = prereg["windows"]
    base["dnmap_stride"] = int(prereg["dnmap_stride"])
    base["dnmap_null_batch"] = int(prereg["dnmap_null_batch"])

    rows = []

    for b in prereg["backends"]:
        backend_label = b["backend_label"]
        for seed in seeds:
            for aseed in anchors:
                cfg = dict(base)
                cfg["tp_backend"] = str(b["tp_backend"])

                # v5 knobs (if present)
                if "geom_v5_theta_scale" in b:
                    cfg["geom_v5_theta_scale"] = float(b["geom_v5_theta_scale"])
                if "geom_v5_harmonic_gamma" in b:
                    cfg["geom_v5_harmonic_gamma"] = float(b["geom_v5_harmonic_gamma"])
                if "geom_v5_harmonic_strength" in b:
                    cfg["geom_v5_harmonic_strength"] = float(b["geom_v5_harmonic_strength"])

                # autotune knobs
                if "eps_auto_tune" in b:
                    cfg["eps_auto_tune"] = bool(b["eps_auto_tune"])
                if "eps_target_delta" in b:
                    cfg["eps_target_delta"] = float(b["eps_target_delta"])

                out_dir = os.path.join(out_root, f"backend_{backend_label}", f"N{int(args.N_null)}", f"seed{seed}_anchor{aseed}")
                rows_csv = os.path.join(out_dir, "rows.csv")

                if args.resume and os.path.exists(rows_csv):
                    df = pd.read_csv(rows_csv)
                    print(f"\n=== prereg SKIP (resume) backend={backend_label} seed={seed} anchor={aseed} ===", flush=True)
                else:
                    print(f"\n=== prereg backend={backend_label} seed={seed} anchor={aseed} ===", flush=True)
                    df = h.run_one(cfg, seed=int(seed), amp=float(args.amp), out_dir=out_dir)

                df["backend_label"] = str(backend_label)
                df["tp_backend"] = str(b["tp_backend"])
                df["seed"] = int(seed)
                df["anchor_seed"] = int(aseed)
                df["N_null"] = int(args.N_null)

                rows.append(df)

    big = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    big.to_csv(os.path.join(out_root, "raw.csv"), index=False)

    # KPI: per-seed and fraction
    if len(big) and "reject" in big.columns:
        by_seed = (
            big.groupby(["backend_label", "wlo", "whi", "seed"], dropna=False)
            .agg(
                any_reject=("reject", "max"),
                anchor_reject_fraction=("reject", "mean"),
                anchor_reject_count=("reject", "sum"),
                best_p=("p_family", "min"),
                worst_p=("p_family", "max"),
                median_p=("p_family", "median"),
                anchor_min_p=("p_family", "min"),
                anchor_median_p=("p_family", "median"),
                median_dn_identity_rank=("dn_identity_rank", "median") if "dn_identity_rank" in big.columns else ("p_family", lambda x: np.nan),
                rows=("reject", "size"),
            )
            .reset_index()
        )
        by_seed["any_reject"] = by_seed["any_reject"].astype(int)

        frac = (
            by_seed.groupby(["backend_label", "wlo", "whi"], dropna=False)
            .agg(
                n_seeds=("seed", "nunique"),
                n_seeds_any_reject=("any_reject", "sum"),
                frac_seeds_any_reject=("any_reject", "mean"),
                median_anchor_reject_fraction=("anchor_reject_fraction", "median"),
                best_p_overall=("best_p", "min"),
                median_best_p=("best_p", "median"),
                median_rank=("median_dn_identity_rank", "median"),
            )
            .reset_index()
        )

        by_seed.to_csv(os.path.join(out_root, "kpi_by_seed.csv"), index=False)
        frac.to_csv(os.path.join(out_root, "kpi_fraction.csv"), index=False)

    # Uniqueness/collision table (per run) - derived-only columns emitted by the harness
    uniq_cols = [
        "backend_label",
        "tp_backend",
        "seed",
        "anchor_seed",
        "wlo",
        "whi",
        "N_null_target",
        "p_dnmap",
        "dn_obs",
        "dn_null_mean",
        "dn_null_std",
        "dn_null_cv",
        "dn_delta",
        "dn_snr",
        "dn_null_min",
        "dn_null_argmin",
        "dn_identity_rank",
        "dn_collision_tol",
        "dn_collision_count",
        "dn_collision_rate",
        "dn_gap_min",
        "meas_params_digest",
        "meas_boundary_digest",
        "meas_E_digest",
        "geom_policy_digest",
        "tp_delta_mean",
        "tp_delta_max",
        "tp_comm_mean",
        "tp_comm_max",
        "geom_v4_eps_global",
        "geom_v4_eps_global_auto",
        "geom_v5_eps_global",
        "geom_v5_eps_global_auto",
        "eps_auto_tune",
        "eps_target_delta",
    ]
    uniq_cols = [c for c in uniq_cols if c in big.columns]
    uniq = big[uniq_cols].copy() if len(big) else pd.DataFrame(columns=uniq_cols)
    uniq.to_csv(os.path.join(out_root, "uniqueness.csv"), index=False)

    # Measurement + policy digest invariance summaries
    dig_cols = [
        c
        for c in [
            "meas_params_digest",
            "meas_boundary_digest",
            "meas_E_digest",
            "geom_policy_digest",
        ]
        if c in big.columns
    ]
    if len(big) and dig_cols:
        inv = big.groupby(["backend_label", "wlo", "whi"], dropna=False)[dig_cols].nunique(dropna=False).reset_index()
    else:
        inv = pd.DataFrame()
    inv.to_csv(os.path.join(out_root, "digest_invariance_summary.csv"), index=False)

    print("\nDONE.")


if __name__ == "__main__":
    main()
