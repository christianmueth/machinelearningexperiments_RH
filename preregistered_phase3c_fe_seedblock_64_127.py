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
    ap.add_argument("--out_root", default="out_phase3C_preregistered_fe_64_127")
    ap.add_argument("--seed_start", type=int, default=64)
    ap.add_argument("--seed_end_exclusive", type=int, default=128)
    ap.add_argument("--anchors", default="2,9,14,44,46,51,60")
    ap.add_argument("--N_null", type=int, default=1024)
    ap.add_argument("--k", type=int, default=13)
    ap.add_argument("--windows", default="2.0,5.0;0.6,7.5")
    ap.add_argument("--dnmap_stride", type=int, default=2)
    ap.add_argument("--dnmap_null_batch", type=int, default=16)
    ap.add_argument("--amp", type=float, default=0.03)
    ap.add_argument("--resume", action="store_true", help="Skip runs whose out_dir already contains rows.csv")

    # Phase-3C knobs
    ap.add_argument(
        "--fe_gate",
        default="fe_and_rigid",
        choices=["fe_only", "fe_and_rigid"],
        help="Gating policy: fe_only uses p_fe; fe_and_rigid requires BOTH FE-consistency and DN-map rigidity.",
    )
    ap.add_argument(
        "--fe_completion_mode",
        default="none",
        choices=["none", "analytic_harmonic"],
        help="Null-free analytic completion G(E) applied before FE differencing.",
    )

    ap.add_argument(
        "--fe_calibration_mode",
        default="none",
        choices=["none", "center_null_mean", "center_null_median", "zscore_null"],
        help="Null-based per-energy calibration (stabilizer), not analytic completion.",
    )
    ap.add_argument("--fe_alpha_E", type=float, default=1.0)
    ap.add_argument("--fe_beta", type=float, default=1.0)
    ap.add_argument("--fe_agg", default="mean", choices=["mean", "median"])

    # optional v9 knobs (only used by v9 backend)
    ap.add_argument("--v9_transport_rule", default="delta_raw_wrapped")
    ap.add_argument("--v9_transport_lambda", type=float, default=1.0)
    ap.add_argument("--v9_transport_clip", type=float, default=0.25)

    args = ap.parse_args()

    out_root = args.out_root
    seeds = list(range(int(args.seed_start), int(args.seed_end_exclusive)))
    anchors = parse_int_list(args.anchors)
    windows = parse_windows(args.windows)

    # Phase-3C prereg: observable changes (p_fe), DN-map mechanics unchanged.
    # Keep a baseline and one nontrivial Tp backend for robustness checks.
    backends = [
        {"backend_label": "legacy", "tp_backend": "legacy"},
        {
            "backend_label": "geom_v9_phasetransport_autotune",
            "tp_backend": "geom_warp_dirac_v9_phasetransport",
            "geom_v9_transport_rule": str(args.v9_transport_rule),
            "geom_v9_transport_lambda": float(args.v9_transport_lambda),
            "geom_v9_transport_clip": float(args.v9_transport_clip),
            "eps_auto_tune": True,
            "eps_target_delta": 0.02,
        },
    ]

    phase3_mode = "dnmap_fe" if str(args.fe_gate) == "fe_only" else "dnmap_fe_both"

    prereg_spec = {
        "date": time.strftime("%Y-%m-%d"),
        "phase": "3C",
        "mode": str(phase3_mode),
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
        "fe_gate": str(args.fe_gate),
        "fe_completion_mode": str(args.fe_completion_mode),
        "fe_calibration_mode": str(args.fe_calibration_mode),
        "fe_alpha_E": float(args.fe_alpha_E),
        "fe_beta": float(args.fe_beta),
        "fe_agg": str(args.fe_agg),
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
                        "phase3_mode": str(phase3_mode),
                        "fe_completion_mode": str(args.fe_completion_mode),
                        "fe_calibration_mode": str(args.fe_calibration_mode),
                        "fe_alpha_E": float(args.fe_alpha_E),
                        "fe_beta": float(args.fe_beta),
                        "fe_agg": str(args.fe_agg),
                        "freeze_measurement": True,
                        "freeze_geometry_policy": True,
                        "gauge_policy": "none",
                        "warp_mode": "ratio",
                        "warp_unwrap": True,
                        "warp_y_scale": 1.0,

                        # closure knobs: OFF means OFF (avoid accidental closure-null usage)
                        "pq_anchor_M": 0,
                        "pqr_anchor_M": 0,
                        "p_pow_kmax": 0,
                        "use_pq": False,
                        "use_pqr": False,
                        "use_pow": False,
                        "use_tower": False,
                    }
                )

                # Backend-specific knobs (v9)
                if "geom_v9_transport_rule" in b:
                    cfg["geom_v9_transport_rule"] = str(b["geom_v9_transport_rule"])
                if "geom_v9_transport_lambda" in b:
                    cfg["geom_v9_transport_lambda"] = float(b["geom_v9_transport_lambda"])
                if "geom_v9_transport_clip" in b:
                    cfg["geom_v9_transport_clip"] = float(b["geom_v9_transport_clip"])

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
                    print(
                        f"\n=== prereg SKIP (resume) backend={backend_label} seed={seed} anchor={aseed} ===",
                        flush=True,
                    )
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

    # Digest invariance summary (audit): measurement + geometry + FE observable policy
    try:
        dig_cols = [
            c
            for c in [
                "meas_params_digest",
                "meas_boundary_digest",
                "meas_E_digest",
                "geom_policy_digest",
                "fe_policy_digest",
            ]
            if c in big.columns
        ]
        if len(big) and dig_cols:
            inv = (
                big.groupby(["backend_label", "wlo", "whi"], dropna=False)
                .agg(**{f"nuniq_{c}": (c, "nunique") for c in dig_cols})
                .reset_index()
            )
        else:
            inv = pd.DataFrame()
        inv.to_csv(os.path.join(out_root, "digest_invariance_summary.csv"), index=False)
    except Exception as e:
        with open(os.path.join(out_root, "digest_invariance_summary_error.txt"), "w") as f:
            f.write(repr(e) + "\n")

    pcol = "p_family" if "p_family" in big.columns else ("p_fe" if "p_fe" in big.columns else None)
    if len(big) and "reject" in big.columns and pcol is not None:
        by_seed = (
            big.groupby(["backend_label", "wlo", "whi", "seed"], dropna=False)
            .agg(
                any_reject=("reject", "max"),
                anchor_reject_fraction=("reject", "mean"),
                anchor_reject_count=("reject", "sum"),
                best_p=(pcol, "min"),
                worst_p=(pcol, "max"),
                median_p=(pcol, "median"),
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
            )
            .reset_index()
        )
    else:
        by_seed = pd.DataFrame()
        frac = pd.DataFrame()

    by_seed.to_csv(os.path.join(out_root, "kpi_by_seed.csv"), index=False)
    frac.to_csv(os.path.join(out_root, "kpi_fraction.csv"), index=False)

    print(f"\n[phase3c_fe] DONE rows={len(big)} out_root={out_root}")


if __name__ == "__main__":
    main()
