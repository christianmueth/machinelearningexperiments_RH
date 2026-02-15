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
    ap.add_argument("--out_root", default="out_phase3C_fe_confirm_seed76")
    ap.add_argument("--seed", type=int, default=76)
    ap.add_argument("--anchors", default="46,9,14,44,2")
    ap.add_argument("--N_null", type=int, default=16384)
    ap.add_argument("--k", type=int, default=13)
    ap.add_argument("--windows", default="2.0,5.0;0.6,7.5")
    ap.add_argument("--dnmap_stride", type=int, default=2)
    ap.add_argument("--dnmap_null_batch", type=int, default=64)
    ap.add_argument("--amp", type=float, default=0.03)
    ap.add_argument("--resume", action="store_true", help="Skip runs whose out_dir already contains rows.csv")

    # Confirm run is FE-only by design.
    ap.add_argument(
        "--fe_completion_mode",
        default="analytic_harmonic",
        choices=["none", "analytic_harmonic"],
    )
    ap.add_argument(
        "--fe_calibration_mode",
        default="none",
        choices=["none", "center_null_mean", "center_null_median", "zscore_null"],
    )
    ap.add_argument("--fe_alpha_E", type=float, default=1.0)
    ap.add_argument("--fe_beta", type=float, default=1.0)
    ap.add_argument("--fe_agg", default="mean", choices=["mean", "median"])

    args = ap.parse_args()

    out_root = str(args.out_root)
    seed = int(args.seed)
    anchors = parse_int_list(args.anchors)
    windows = parse_windows(args.windows)

    # Confirm-stage spec: FE-only gate, frozen DN-map mechanics.
    phase3_mode = "dnmap_fe"

    # Confirm focuses on the backend that produced the smallest p_fe in scan.
    backends = [
        {"backend_label": "legacy", "tp_backend": "legacy"},
    ]

    prereg_spec = {
        "date": time.strftime("%Y-%m-%d"),
        "phase": "3C_confirm",
        "mode": phase3_mode,
        "null_family": "label_scramble",
        "alpha": float(h.CFG_BASE.get("alpha", 0.01)),
        "seed": seed,
        "anchors": list(map(int, anchors)),
        "k": int(args.k),
        "N_null": int(args.N_null),
        "windows": [list(map(float, w)) for w in windows],
        "dnmap_stride": int(args.dnmap_stride),
        "dnmap_null_batch": int(args.dnmap_null_batch),
        "amp": float(args.amp),
        "fe_gate": "fe_only",
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
                    "phase3_mode": phase3_mode,
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

                    # closure knobs: OFF means OFF
                    "pq_anchor_M": 0,
                    "pqr_anchor_M": 0,
                    "p_pow_kmax": 0,
                    "use_pq": False,
                    "use_pqr": False,
                    "use_pow": False,
                    "use_tower": False,
                }
            )

            out_dir = os.path.join(
                out_root,
                f"backend_{backend_label}",
                f"N{int(args.N_null)}",
                f"seed{int(seed)}_anchor{int(aseed)}",
            )

            rows_csv = os.path.join(out_dir, "rows.csv")
            if args.resume and os.path.exists(rows_csv):
                df = pd.read_csv(rows_csv)
                print(f"\n=== confirm SKIP (resume) backend={backend_label} seed={seed} anchor={aseed} ===", flush=True)
            else:
                print(f"\n=== confirm backend={backend_label} seed={seed} anchor={aseed} ===", flush=True)
                df = h.run_one(cfg, seed=int(seed), amp=float(args.amp), out_dir=out_dir)

            df["backend_label"] = str(backend_label)
            df["tp_backend"] = str(b["tp_backend"])
            df["seed"] = int(seed)
            df["anchor_seed"] = int(aseed)
            df["N_null"] = int(args.N_null)

            rows.append(df)

    big = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    big.to_csv(os.path.join(out_root, "raw.csv"), index=False)

    # Digest invariance summary
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

    # KPI summary
    try:
        if len(big):
            alpha = float(h.CFG_BASE.get("alpha", 0.01))
            if "reject" not in big.columns:
                big["reject"] = big.get("p_family", 1.0) <= alpha

            # by-window and overall
            frac = (
                big.groupby(["backend_label", "wlo", "whi"], dropna=False)
                .agg(
                    n_anchors=("anchor_seed", "nunique"),
                    reject_count=("reject", "sum"),
                    best_p=("p_family", "min"),
                    median_p=("p_family", "median"),
                )
                .reset_index()
            )
            frac["reject_fraction"] = frac["reject_count"] / frac["n_anchors"].clip(lower=1)
            frac.to_csv(os.path.join(out_root, "kpi_by_window.csv"), index=False)

            top_cols = [c for c in ["backend_label", "seed", "anchor_seed", "wlo", "whi", "p_family", "p_fe", "p_fe_window", "fe_obs_raw", "fe_obs"] if c in big.columns]
            best_rows = big.sort_values("p_family").head(20)[top_cols]
            best_rows.to_csv(os.path.join(out_root, "top20_smallest_p_family.csv"), index=False)
    except Exception as e:
        with open(os.path.join(out_root, "kpi_error.txt"), "w") as f:
            f.write(repr(e) + "\n")

    print(f"\n[phase3c_fe_confirm] DONE rows={len(big)} out_root={out_root}", flush=True)


if __name__ == "__main__":
    main()
