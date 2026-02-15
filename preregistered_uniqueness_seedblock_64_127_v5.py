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
    ap.add_argument("--out_root", default="out_phase3B_preregistered_uniqueness_64_127_v5")
    ap.add_argument("--seed_start", type=int, default=64)
    ap.add_argument("--seed_end_exclusive", type=int, default=128)
    ap.add_argument("--anchors", default="2,9,14,44,46,51,60")
    ap.add_argument("--N_null", type=int, default=1024)
    ap.add_argument("--k", type=int, default=13)
    ap.add_argument("--windows", default="2.0,5.0;0.6,7.5")
    ap.add_argument("--dnmap_stride", type=int, default=2)
    ap.add_argument("--dnmap_null_batch", type=int, default=16)
    ap.add_argument("--amp", type=float, default=0.03)
    ap.add_argument("--collision_tol", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true", help="Skip runs whose out_dir already contains rows.csv")
    args = ap.parse_args()

    out_root = args.out_root
    seeds = list(range(int(args.seed_start), int(args.seed_end_exclusive)))
    anchors = parse_int_list(args.anchors)
    windows = parse_windows(args.windows)

    # Preregistered backend set (3 only): legacy + v4-autotune + v5-autotune
    backends = [
        {"backend_label": "legacy", "tp_backend": "legacy"},
        {

            "geom_v4_theta_scale": 0.125,
            "eps_auto_tune": True,
            "eps_target_delta": 0.02,
        },
        {
            "backend_label": "geom_v5_theta0p125_autotune",
            "tp_backend": "geom_warp_dirac_v5",
            "geom_v5_theta_scale": 0.125,
            "geom_v5_harmonic_gamma": 1.0,
            "geom_v5_harmonic_strength": 1.0,
            "eps_auto_tune": True,
            "eps_target_delta": 0.02,
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

    t0 = time.time()
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

                        # closure knobs (OFF means OFF)
                        "pq_anchor_M": 1,
                        "pqr_anchor_M": 0,
                        "p_pow_kmax": 0,
                        "use_pq": True,
                        "use_pqr": False,
                        "use_pow": False,
                        "use_tower": False,
                    }
                )

                # Backend-specific knobs (v4)
                if "geom_v4_theta_scale" in b:
                    cfg["geom_v4_theta_scale"] = float(b["geom_v4_theta_scale"])

                # Backend-specific knobs (v5)
                if "geom_v5_theta_scale" in b:
                    cfg["geom_v5_theta_scale"] = float(b["geom_v5_theta_scale"])
                if "geom_v5_harmonic_gamma" in b:
                    cfg["geom_v5_harmonic_gamma"] = float(b["geom_v5_harmonic_gamma"])
                if "geom_v5_harmonic_strength" in b:
                    cfg["geom_v5_harmonic_strength"] = float(b["geom_v5_harmonic_strength"])

                # Shared autotune knobs
                if "eps_auto_tune" in b:
                    cfg["eps_auto_tune"] = bool(b["eps_auto_tune"])
                if "eps_target_delta" in b:
                    cfg["eps_target_delta"] = float(b["eps_target_delta"])

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
    group_cols = [c for c in ["backend_label", "wlo", "whi"] if c in big.columns]
    dig_cols = [
        c
        for c in ["meas_params_digest", "meas_boundary_digest", "meas_E_digest", "geom_policy_digest"]
        if c in big.columns
    ]
    try:
        if len(big) and group_cols and dig_cols:
            inv = big.groupby(group_cols, dropna=False)[dig_cols].nunique(dropna=False).reset_index()
        else:
            inv = pd.DataFrame()
        inv.to_csv(os.path.join(out_root, "digest_invariance_summary.csv"), index=False)
    except Exception as e:
        with open(os.path.join(out_root, "postprocess_error.txt"), "w") as f:
            f.write("digest_invariance_summary failed:\n")
            f.write(repr(e) + "\n")

    print(f"DONE prereg v5-scan: rows={len(big)} elapsed={time.time()-t0:.1f}s out_root={out_root}")


if __name__ == "__main__":
    main()
