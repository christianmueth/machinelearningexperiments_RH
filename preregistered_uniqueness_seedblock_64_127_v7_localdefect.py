"""Preregistered Phase-3B-legal scan: legacy + v4-autotune + v7 localized-defect (block + bernoulli) autotune.

Purpose:
- Keep the Phase-3B measurement spine frozen (dnmap_only, label-scramble null).
- Test the next admissible direction suggested by the v4→v6 pattern: localization.
- v7 introduces a bounded *localized* modulation of the chiral coupling on a subset of primes.

Usage (PowerShell):
  & .\.venv\Scripts\python.exe preregistered_uniqueness_seedblock_64_127_v7_localdefect.py \
      --out_root out_phase3B_preregistered_uniqueness_64_127_v7_localdefect \
      --N_null 1024 --dnmap_null_batch 2 \
      --geom_v7_defect_rho 0.25 --geom_v7_defect_strength 1.0 --geom_v7_defect_clip 0.75

Notes:
- This script is Phase-3B legal: it changes only Tp construction backends.
- The default anchors match the existing robustness pack.
"""

import os
import time
import json
import argparse
import numpy as np
import pandas as pd

import machinelearning_rh_colab_cells as h


def parse_int_list(s: str):
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_windows(s: str):
    # format: "2.0,5.0;0.6,7.5" (semicolon-separated pairs)
    wins = []
    for part in str(s).split(";"):
        part = part.strip()
        if not part:
            continue
        a, b = [float(x.strip()) for x in part.split(",")]
        wins.append((float(a), float(b)))
    return wins


def prereg_digest(spec: dict) -> str:
    return h.sha16(repr(spec).encode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="out_phase3B_preregistered_uniqueness_64_127_v7_localdefect")
    ap.add_argument("--seed_start", type=int, default=64)
    ap.add_argument("--seed_end_exclusive", type=int, default=128)
    ap.add_argument("--anchors", default="2,9,14,44,46,51,60")
    ap.add_argument("--N_null", type=int, default=1024)
    ap.add_argument("--k", type=int, default=13)
    ap.add_argument("--windows", default="2.0,5.0;0.6,7.5")
    ap.add_argument("--dnmap_stride", type=int, default=2)
    ap.add_argument("--dnmap_null_batch", type=int, default=2)
    ap.add_argument("--amp", type=float, default=0.03)
    ap.add_argument("--collision_tol", type=float, default=0.0)

    # v7 knobs
    ap.add_argument("--geom_v7_defect_rho", type=float, default=0.25)
    ap.add_argument("--geom_v7_defect_strength", type=float, default=1.0)
    ap.add_argument("--geom_v7_defect_clip", type=float, default=0.75)
    ap.add_argument("--geom_v7_defect_salt", type=int, default=1)

    ap.add_argument("--resume", action="store_true", help="Skip runs whose out_dir already contains rows.csv")
    args = ap.parse_args()

    out_root = args.out_root
    seeds = list(range(int(args.seed_start), int(args.seed_end_exclusive)))
    anchors = parse_int_list(args.anchors)
    windows = parse_windows(args.windows)

    v7_common = {
        "tp_backend": "geom_warp_dirac_v7_localdefect",
        "geom_v7_theta_scale": 0.125,
        "geom_v7_defect_rho": float(args.geom_v7_defect_rho),
        "geom_v7_defect_strength": float(args.geom_v7_defect_strength),
        "geom_v7_defect_clip": float(args.geom_v7_defect_clip),
        "geom_v7_defect_salt": int(args.geom_v7_defect_salt),
        # IMPORTANT: defect pattern is geometry-class-defined; do not key off observation seed.
        "geom_v7_defect_seed": 0,
        "eps_auto_tune": True,
        "eps_target_delta": 0.02,
    }

    backends = [
        {"backend_label": "legacy", "tp_backend": "legacy"},
        {
            "backend_label": "geom_v4_theta0p125_autotune",
            "tp_backend": "geom_warp_dirac_v4",
            "geom_v4_theta_scale": 0.125,
            "eps_auto_tune": True,
            "eps_target_delta": 0.02,
        },
        {
            "backend_label": f"geom_v7_block_r{float(args.geom_v7_defect_rho)}_s{float(args.geom_v7_defect_strength)}_c{float(args.geom_v7_defect_clip)}_autotune",
            **v7_common,
            "geom_v7_defect_mode": "block",
        },
        {
            "backend_label": f"geom_v7_bern_r{float(args.geom_v7_defect_rho)}_s{float(args.geom_v7_defect_strength)}_c{float(args.geom_v7_defect_clip)}_autotune",
            **v7_common,
            "geom_v7_defect_mode": "bernoulli",
        },
    ]

    prereg_spec = {
        "date": time.strftime("%Y-%m-%d"),
        "phase": "3B->3C",
        "mode": "dnmap_only",
        "null_family": "label_scramble",
        "alpha": float(h.CFG_BASE.get("alpha", 0.01)),
        "seeds": [int(args.seed_start), int(args.seed_end_exclusive)],
        "anchors": list(map(int, anchors)),
        "k": int(args.k),
        "N_null": int(args.N_null),
        "windows": [list(map(float, w)) for w in windows],
        "dnmap_stride": int(args.dnmap_stride),
        "dnmap_null_batch": int(args.dnmap_null_batch),
        "amp": float(args.amp),
        "collision_tol": float(args.collision_tol),
        "backends": backends,
        "measurement_frozen": True,
        "geometry_policy_frozen": True,
        "warp_mode": "ratio",
        "warp_unwrap": True,
        "warp_y_scale": 1.0,
    }

    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "preregistration.json"), "w") as f:
        json.dump(prereg_spec, f, indent=2)
    with open(os.path.join(out_root, "preregistration_digest.txt"), "w") as f:
        f.write(prereg_digest(prereg_spec) + "\n")

    rows = []
    for b in backends:
        backend_label = b["backend_label"]
        for seed in seeds:
            for aseed in anchors:
                cfg = dict(h.CFG_BASE)
                cfg.update(
                    {
                        "tp_backend": str(b["tp_backend"]),
                        "n_ops_list": [int(args.k)],
                        "windows": [tuple(w) for w in windows],
                        "N_null": int(args.N_null),
                        "dnmap_stride": int(args.dnmap_stride),
                        "dnmap_null_batch": int(args.dnmap_null_batch),
                        "anchor_seed": int(aseed),
                        "phase3_mode": "dnmap_only",
                        "freeze_measurement": True,
                        "freeze_geometry_policy": True,
                        "gauge_policy": "none",
                        "warp_mode": "ratio",
                        "warp_unwrap": True,
                        "warp_y_scale": 1.0,
                        "dnmap_uniqueness": True,
                        "dn_collision_tol": float(args.collision_tol),
                        # closure knobs OFF
                        "pq_anchor_M": 1,
                        "pqr_anchor_M": 0,
                        "p_pow_kmax": 0,
                        "use_pq": True,
                        "use_pqr": False,
                        "use_pow": False,
                        "use_tower": False,
                    }
                )

                for k, v in b.items():
                    if k in {"backend_label", "tp_backend"}:
                        continue
                    cfg[k] = v

                out_dir = os.path.join(
                    out_root,
                    f"backend_{backend_label}",
                    f"N{int(args.N_null)}",
                    f"seed{int(seed)}_anchor{int(aseed)}",
                )
                rows_csv = os.path.join(out_dir, "rows.csv")
                if args.resume and os.path.exists(rows_csv):
                    df = pd.read_csv(rows_csv)
                    print(f"\n=== prereg SKIP (resume) backend={backend_label} seed={seed} anchor={aseed} ===", flush=True)
                else:
                    print(f"\n=== prereg backend={backend_label} seed={seed} anchor={aseed} ===", flush=True)
                    df = h.run_one(cfg, seed=int(seed), amp=float(args.amp), out_dir=out_dir)

                df = df.assign(
                    backend_label=str(backend_label),
                    tp_backend=str(b["tp_backend"]),
                    seed=int(seed),
                    anchor_seed=int(aseed),
                    N_null=int(args.N_null),
                )
                rows.append(df)

    big = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    big.to_csv(os.path.join(out_root, "raw.csv"), index=False)

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
    else:
        by_seed = pd.DataFrame()
        frac = pd.DataFrame()

    by_seed.to_csv(os.path.join(out_root, "kpi_by_seed.csv"), index=False)
    frac.to_csv(os.path.join(out_root, "kpi_fraction.csv"), index=False)

    # Uniqueness/collision table (per run)
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
        "geom_policy_summary",
        "tp_delta_mean",
        "tp_delta_max",
        "tp_comm_mean",
        "tp_comm_max",
        "geom_v4_eps_global",
        "geom_v4_eps_global_auto",
        "geom_v5_eps_global",
        "geom_v5_eps_global_auto",
        "geom_v6_eps_global",
        "geom_v6_eps_global_auto",
        "geom_v7_eps_global",
        "geom_v7_eps_global_auto",
        "defect_support_size",
        "defect_support_rule",
        "defect_primes_marked_digest",
        "defect_amp_before_autotune",
        "defect_amp_after_autotune",
        "eps_auto_tune",
        "eps_target_delta",
    ]
    uniq_cols = [c for c in uniq_cols if c in big.columns]
    uniq = big[uniq_cols].copy() if len(big) else pd.DataFrame(columns=uniq_cols)
    uniq.to_csv(os.path.join(out_root, "uniqueness.csv"), index=False)

    # Digest invariance
    dig_cols = [c for c in ["meas_params_digest", "meas_boundary_digest", "meas_E_digest", "geom_policy_digest"] if c in big.columns]
    if len(big) and dig_cols:
        inv = big.groupby(["backend_label", "wlo", "whi"], dropna=False)[dig_cols].nunique(dropna=False).reset_index()
    else:
        inv = pd.DataFrame()
    inv.to_csv(os.path.join(out_root, "digest_invariance_summary.csv"), index=False)

    # Windowpair KPIs (primary vs secondary) + seed ranking (derived-only)
    try:
        if len(by_seed):
            # prefer canonical window ordering
            win_primary = (2.0, 5.0)
            win_secondary = (0.6, 7.5)
            have_primary = bool(((by_seed["wlo"] == win_primary[0]) & (by_seed["whi"] == win_primary[1])).any())
            have_secondary = bool(((by_seed["wlo"] == win_secondary[0]) & (by_seed["whi"] == win_secondary[1])).any())
            if not (have_primary and have_secondary):
                wins = sorted({(float(r.wlo), float(r.whi)) for r in by_seed.itertuples(index=False)})
                if len(wins) >= 2:
                    win_primary, win_secondary = wins[0], wins[1]

            def _slice(w):
                return by_seed[(by_seed["wlo"] == float(w[0])) & (by_seed["whi"] == float(w[1]))].copy()

            prim = _slice(win_primary).rename(
                columns={
                    "anchor_reject_fraction": "anchor_reject_fraction_primary",
                    "anchor_reject_count": "anchor_reject_count_primary",
                    "anchor_min_p": "anchor_min_p_primary",
                    "anchor_median_p": "anchor_median_p_primary",
                    "best_p": "best_p_primary",
                    "median_p": "median_p_primary",
                }
            )
            sec = _slice(win_secondary).rename(
                columns={
                    "anchor_reject_fraction": "anchor_reject_fraction_secondary",
                    "anchor_reject_count": "anchor_reject_count_secondary",
                    "anchor_min_p": "anchor_min_p_secondary",
                    "anchor_median_p": "anchor_median_p_secondary",
                    "best_p": "best_p_secondary",
                    "median_p": "median_p_secondary",
                }
            )

            key_cols = ["backend_label", "seed"]
            keep_prim = key_cols + [
                "anchor_reject_fraction_primary",
                "anchor_reject_count_primary",
                "anchor_min_p_primary",
                "anchor_median_p_primary",
                "best_p_primary",
                "median_p_primary",
            ]
            keep_sec = key_cols + [
                "anchor_reject_fraction_secondary",
                "anchor_reject_count_secondary",
                "anchor_min_p_secondary",
                "anchor_median_p_secondary",
                "best_p_secondary",
                "median_p_secondary",
            ]
            prim = prim[[c for c in keep_prim if c in prim.columns]]
            sec = sec[[c for c in keep_sec if c in sec.columns]]
            wp = prim.merge(sec, on=key_cols, how="outer")
            wp.to_csv(os.path.join(out_root, "kpi_by_seed_windowpair.csv"), index=False)

            # aggregate fractions across seeds per backend
            if len(wp):
                frac_wp = (
                    wp.groupby(["backend_label"], dropna=False)
                    .agg(
                        n_seeds=("seed", "nunique"),
                        median_primary=("anchor_reject_fraction_primary", "median"),
                        median_secondary=("anchor_reject_fraction_secondary", "median"),
                        best_primary_min_p=("anchor_min_p_primary", "min"),
                        best_secondary_min_p=("anchor_min_p_secondary", "min"),
                        n_seeds_any_primary_reject=("anchor_reject_count_primary", lambda s: int((s.fillna(0) > 0).sum())),
                        n_seeds_any_secondary_reject=("anchor_reject_count_secondary", lambda s: int((s.fillna(0) > 0).sum())),
                    )
                    .reset_index()
                )
            else:
                frac_wp = pd.DataFrame()
            frac_wp.to_csv(os.path.join(out_root, "kpi_fraction_windowpair.csv"), index=False)

            # seed ranking: primary robustness first
            if len(wp):
                rank_df = wp.copy()
                rank_df = rank_df.sort_values(
                    ["anchor_reject_fraction_primary", "anchor_min_p_primary"],
                    ascending=[False, True],
                    na_position="last",
                )
                rank_df.to_csv(os.path.join(out_root, "seed_rankings.csv"), index=False)
    except Exception as e:
        # Do not fail the scan if postprocessing cannot be produced.
        with open(os.path.join(out_root, "postprocess_error.txt"), "w") as f:
            f.write(repr(e) + "\n")

    print("\nDONE.")


if __name__ == "__main__":
    main()
