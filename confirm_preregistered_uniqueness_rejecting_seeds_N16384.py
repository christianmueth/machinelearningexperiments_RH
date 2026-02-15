import os
import time
import json
import argparse
from datetime import datetime
import numpy as np
import pandas as pd

import machinelearning_rh_colab_cells as h


def parse_int_list(s: str):
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_windows(s: str):
    wins = []
    for part in str(s).split(";"):
        part = part.strip()
        if not part:
            continue
        a, b = [float(x.strip()) for x in part.split(",")]
        wins.append((float(a), float(b)))
    return wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_root", default="out_phase3B_preregistered_uniqueness_64_127")
    ap.add_argument("--out_root", default="out_phase3B_confirm_preregistered_uniqueness_rejecting_seeds_N16384")
    ap.add_argument("--anchors", default="2,9,14,44,46,51,60")
    ap.add_argument("--N_null", type=int, default=16384)
    ap.add_argument("--k", type=int, default=13)
    ap.add_argument("--windows", default="2.0,5.0;0.6,7.5")
    ap.add_argument("--dnmap_stride", type=int, default=2)
    ap.add_argument("--dnmap_null_batch", type=int, default=16)
    ap.add_argument("--amp", type=float, default=0.03)
    ap.add_argument("--collision_tol", type=float, default=0.0)
    ap.add_argument("--backend_label", default="geom_v4_theta0p125_autotune")
    ap.add_argument(
        "--min_anchor_reject_fraction_scan",
        type=float,
        default=0.0,
        help="Optional prereg seed-selection gate: require scan-stage anchor_reject_fraction >= this value (primary window).",
    )
    ap.add_argument(
        "--min_anchor_reject_fraction_confirm",
        type=float,
        default=0.0,
        help="Optional report gate: require confirm-stage anchor_reject_fraction >= this value (per seed/window) for a seed to be considered robust.",
    )
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    scan_root = args.scan_root
    out_root = args.out_root
    anchors = parse_int_list(args.anchors)
    windows = parse_windows(args.windows)

    scan_by_seed = os.path.join(scan_root, "kpi_by_seed.csv")
    if not os.path.exists(scan_by_seed):
        raise FileNotFoundError(f"missing scan file: {scan_by_seed}")

    scan_prereg = os.path.join(scan_root, "preregistration.json")
    if not os.path.exists(scan_prereg):
        raise FileNotFoundError(
            f"missing scan preregistration file: {scan_prereg}. "
            "This confirm script intentionally requires the scan preregistration to prevent config drift."
        )

    with open(scan_prereg, "r") as f:
        prereg = json.load(f)

    backend_specs = prereg.get("backends", [])
    wanted = str(args.backend_label)
    matches = [b for b in backend_specs if str(b.get("backend_label")) == wanted]
    if not matches:
        raise ValueError(
            f"backend_label={wanted!r} not found in scan preregistration backends. "
            f"Available: {[str(b.get('backend_label')) for b in backend_specs]}"
        )
    backend = dict(matches[0])

    by_seed = pd.read_csv(scan_by_seed)
    # Preregistered rule: choose rejecting seeds from the scan block only.
    # Use primary window (2,5) by default.
    primary = by_seed[(by_seed["backend_label"] == str(args.backend_label)) & (by_seed["wlo"] == 2.0) & (by_seed["whi"] == 5.0)]

    seeds_reject = primary.loc[primary["any_reject"] == 1, "seed"].unique().tolist()
    min_frac = float(args.min_anchor_reject_fraction_scan)
    if ("anchor_reject_fraction" in primary.columns) and np.isfinite(min_frac) and (min_frac > 0):
        primary2 = primary.loc[primary["anchor_reject_fraction"] >= min_frac]
        seeds_reject = primary2.loc[primary2["any_reject"] == 1, "seed"].unique().tolist()
    seeds_reject = sorted(map(int, seeds_reject))

    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "confirm_plan.json"), "w") as f:
        json.dump(
            {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "scan_root": scan_root,
                "scan_preregistration": scan_prereg,
                "backend_label": str(args.backend_label),
                "backend_spec": backend,
                "selected_seeds": seeds_reject,
                "min_anchor_reject_fraction_scan": float(args.min_anchor_reject_fraction_scan),
                "min_anchor_reject_fraction_confirm": float(args.min_anchor_reject_fraction_confirm),
                "anchors": anchors,
                "N_null": int(args.N_null),
                "k": int(args.k),
                "windows": [list(map(float, w)) for w in windows],
                "dnmap_stride": int(args.dnmap_stride),
                "amp": float(args.amp),
                "collision_tol": float(args.collision_tol),
            },
            f,
            indent=2,
        )

    if not seeds_reject:
        print("No preregistered rejecting seeds found in scan; nothing to confirm.")
        return

    # Confirm backend: exactly the preregistered scan backend spec (no drift)
    backend_label = str(backend.get("backend_label"))
    tp_backend = str(backend.get("tp_backend"))

    t0 = time.time()
    rows = []

    for seed in seeds_reject:
        for aseed in anchors:
            cfg = dict(h.CFG_BASE)
            cfg.update(
                {
                    "tp_backend": tp_backend,
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

            # Apply any preregistered backend knobs onto cfg, excluding labels.
            for k, v in backend.items():
                if k in {"backend_label", "tp_backend"}:
                    continue
                cfg[k] = v

            out_dir = os.path.join(out_root, f"seed{int(seed)}_anchor{int(aseed)}")
            rows_csv = os.path.join(out_dir, "rows.csv")
            if args.resume and os.path.exists(rows_csv):
                df = pd.read_csv(rows_csv)
                print(f"\n=== confirm SKIP (resume) seed={seed} anchor={aseed} ===", flush=True)
            else:
                print(f"\n=== confirm seed={seed} anchor={aseed} ===", flush=True)
                df = h.run_one(cfg, seed=int(seed), amp=float(args.amp), out_dir=out_dir)

            df = df.assign(
                backend_label=backend_label,
                tp_backend=tp_backend,
                seed=int(seed),
                anchor_seed=int(aseed),
                N_null=int(args.N_null),
            )
            rows.append(df)

    big = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    big.to_csv(os.path.join(out_root, "raw.csv"), index=False)

    # Confirm-stage per-seed anchor robustness KPIs (derived-only)
    if len(big) and ("reject" in big.columns) and ("p_family" in big.columns):
        by_seed_anchor = (
            big.groupby(["backend_label", "wlo", "whi", "seed"], dropna=False)
            .agg(
                anchor_reject_fraction=("reject", "mean"),
                anchor_reject_count=("reject", "sum"),
                anchor_min_p=("p_family", "min"),
                anchor_median_p=("p_family", "median"),
                n_anchors=("anchor_seed", "nunique"),
                rows=("reject", "size"),
            )
            .reset_index()
        )
        by_seed_anchor.to_csv(os.path.join(out_root, "kpi_by_seed_anchor.csv"), index=False)

        frac_anchor = (
            by_seed_anchor.groupby(["backend_label", "wlo", "whi"], dropna=False)
            .agg(
                n_seeds=("seed", "nunique"),
                median_anchor_reject_fraction=("anchor_reject_fraction", "median"),
                best_anchor_min_p=("anchor_min_p", "min"),
            )
            .reset_index()
        )
        frac_anchor.to_csv(os.path.join(out_root, "kpi_fraction_anchor.csv"), index=False)

        min_confirm = float(args.min_anchor_reject_fraction_confirm)
        if np.isfinite(min_confirm) and (min_confirm > 0):
            robust = by_seed_anchor.loc[by_seed_anchor["anchor_reject_fraction"] >= min_confirm, ["seed", "wlo", "whi", "anchor_reject_fraction", "anchor_min_p"]]
            robust = robust.sort_values(["wlo", "whi", "anchor_reject_fraction", "anchor_min_p"], ascending=[True, True, False, True])
            robust.to_csv(os.path.join(out_root, "robust_seeds_by_window.csv"), index=False)

    # Aggregate confirm KPIs
    if len(big) and "reject" in big.columns:
        agg = (
            big.groupby(["wlo", "whi"], dropna=False)
            .agg(
                rows=("reject", "size"),
                reject_rate=("reject", "mean"),
                best_p=("p_family", "min"),
                median_rank=("dn_identity_rank", "median") if "dn_identity_rank" in big.columns else ("p_family", lambda x: np.nan),
                mean_collision_rate=("dn_collision_rate", "mean") if "dn_collision_rate" in big.columns else ("p_family", lambda x: np.nan),
            )
            .reset_index()
        )
    else:
        agg = pd.DataFrame()
    agg.to_csv(os.path.join(out_root, "aggregate.csv"), index=False)

    # Digest invariance
    digest_cols = [c for c in ["meas_params_digest", "meas_boundary_digest", "meas_E_digest", "geom_policy_digest"] if c in big.columns]
    if len(big) and digest_cols:
        digest_summary = big.groupby(["wlo", "whi"], dropna=False)[digest_cols].nunique(dropna=False).reset_index()
    else:
        digest_summary = pd.DataFrame()
    digest_summary.to_csv(os.path.join(out_root, "digest_invariance_summary.csv"), index=False)

    print("\n=== DONE confirm preregistered rejecting seeds ===")
    print(f"selected seeds: {seeds_reject}")
    if float(args.min_anchor_reject_fraction_scan) > 0:
        print(f"scan gate: anchor_reject_fraction >= {float(args.min_anchor_reject_fraction_scan):.6g} (primary window)")
    if float(args.min_anchor_reject_fraction_confirm) > 0:
        print(f"confirm report gate: anchor_reject_fraction >= {float(args.min_anchor_reject_fraction_confirm):.6g}")
    print(f"out_root: {out_root}")
    print(f"runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
