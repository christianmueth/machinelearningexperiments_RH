import os
import time
import json
import argparse
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
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--anchors", required=True, help="Comma-separated anchor seeds")

    ap.add_argument("--backend_label", default="legacy", choices=["legacy", "geom_v9_phasetransport_autotune"])
    ap.add_argument(
        "--tp_backend",
        default=None,
        help="Override tp_backend string (otherwise inferred from backend_label)",
    )

    ap.add_argument("--N_null", type=int, default=16384)
    ap.add_argument("--k", type=int, default=13)
    ap.add_argument("--windows", default="2.0,5.0;0.6,7.5")
    ap.add_argument("--dnmap_stride", type=int, default=2)
    ap.add_argument("--dnmap_null_batch", type=int, default=64)
    ap.add_argument("--amp", type=float, default=0.03)

    # FE knobs
    ap.add_argument("--fe_completion_mode", default="analytic_harmonic", choices=["none", "analytic_harmonic"])
    ap.add_argument(
        "--fe_calibration_mode",
        default="none",
        choices=["none", "center_null_mean", "center_null_median", "zscore_null"],
    )
    ap.add_argument("--fe_alpha_E", type=float, default=1.0)
    ap.add_argument("--fe_beta", type=float, default=1.0)
    ap.add_argument("--fe_agg", default="mean", choices=["mean", "median"])

    # optional v9 knobs
    ap.add_argument("--v9_transport_rule", default="delta_raw_wrapped")
    ap.add_argument("--v9_transport_lambda", type=float, default=1.0)
    ap.add_argument("--v9_transport_clip", type=float, default=0.25)

    args = ap.parse_args()

    anchors = parse_int_list(args.anchors)
    windows = parse_windows(args.windows)

    if args.tp_backend is None:
        if args.backend_label == "legacy":
            tp_backend = "legacy"
        else:
            tp_backend = "geom_warp_dirac_v9_phasetransport"
    else:
        tp_backend = str(args.tp_backend)

    prereg_spec = {
        "date": time.strftime("%Y-%m-%d"),
        "phase": "3C_confirm",
        "mode": "dnmap_fe",
        "null_family": "label_scramble",
        "alpha": float(h.CFG_BASE.get("alpha", 0.01)),
        "seed": int(args.seed),
        "anchors": list(map(int, anchors)),
        "backend_label": str(args.backend_label),
        "tp_backend": str(tp_backend),
        "k": int(args.k),
        "N_null": int(args.N_null),
        "windows": [list(map(float, w)) for w in windows],
        "dnmap_stride": int(args.dnmap_stride),
        "dnmap_null_batch": int(args.dnmap_null_batch),
        "amp": float(args.amp),
        "fe_completion_mode": str(args.fe_completion_mode),
        "fe_calibration_mode": str(args.fe_calibration_mode),
        "fe_alpha_E": float(args.fe_alpha_E),
        "fe_beta": float(args.fe_beta),
        "fe_agg": str(args.fe_agg),
        "measurement_frozen": True,
        "geometry_policy_frozen": True,
        "warp_mode": "ratio",
        "warp_unwrap": True,
        "warp_y_scale": 1.0,
    }

    os.makedirs(args.out_root, exist_ok=True)
    with open(os.path.join(args.out_root, "preregistration.json"), "w") as f:
        json.dump(prereg_spec, f, indent=2)
    with open(os.path.join(args.out_root, "preregistration_digest.txt"), "w") as f:
        f.write(prereg_digest(prereg_spec) + "\n")

    rows = []

    for aseed in anchors:
        cfg = dict(h.CFG_BASE)
        cfg.update(
            {
                "tp_backend": str(tp_backend),
                "n_ops_list": [int(args.k)],
                "windows": [tuple(w) for w in windows],
                "N_null": int(args.N_null),
                "dnmap_stride": int(args.dnmap_stride),
                "dnmap_null_batch": int(args.dnmap_null_batch),
                "anchor_seed": int(aseed),
                "phase3_mode": "dnmap_fe",
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

                # closure knobs OFF
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
        if args.backend_label == "geom_v9_phasetransport_autotune":
            cfg["geom_v9_transport_rule"] = str(args.v9_transport_rule)
            cfg["geom_v9_transport_lambda"] = float(args.v9_transport_lambda)
            cfg["geom_v9_transport_clip"] = float(args.v9_transport_clip)
            cfg["eps_auto_tune"] = True
            cfg["eps_target_delta"] = 0.02

        out_dir = os.path.join(args.out_root, f"backend_{args.backend_label}", f"N{int(args.N_null)}", f"seed{int(args.seed)}_anchor{int(aseed)}")
        print(f"\n=== confirm backend={args.backend_label} seed={args.seed} anchor={aseed} ===", flush=True)
        df = h.run_one(cfg, seed=int(args.seed), amp=float(args.amp), out_dir=out_dir)

        df["backend_label"] = str(args.backend_label)
        df["tp_backend"] = str(tp_backend)
        df["seed"] = int(args.seed)
        df["anchor_seed"] = int(aseed)
        df["N_null"] = int(args.N_null)
        rows.append(df)

    big = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    big.to_csv(os.path.join(args.out_root, "raw.csv"), index=False)

    # simple KPI by window
    if len(big):
        gb = big.groupby(["backend_label", "wlo", "whi"], dropna=False)
        kpi = (
            gb.agg(
                n_anchors=("anchor_seed", "nunique"),
                reject_count=("reject", "sum"),
                best_p=("p_family", "min"),
                median_p=("p_family", "median"),
            )
            .reset_index()
        )
        kpi["reject_fraction"] = kpi["reject_count"] / kpi["n_anchors"].clip(lower=1)
    else:
        kpi = pd.DataFrame()
    kpi.to_csv(os.path.join(args.out_root, "kpi_by_window.csv"), index=False)

    # digest invariance summary
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
        inv.to_csv(os.path.join(args.out_root, "digest_invariance_summary.csv"), index=False)
    except Exception as e:
        with open(os.path.join(args.out_root, "digest_invariance_summary_error.txt"), "w") as f:
            f.write(repr(e) + "\n")

    # top 20 smallest p_family rows
    try:
        cols = [c for c in ["backend_label", "seed", "anchor_seed", "wlo", "whi", "p_family", "p_fe", "p_fe_window", "fe_obs_raw", "fe_obs"] if c in big.columns]
        top = big.sort_values("p_family").head(20)[cols]
        top.to_csv(os.path.join(args.out_root, "top20_smallest_p_family.csv"), index=False)
    except Exception:
        pass


if __name__ == "__main__":
    main()
