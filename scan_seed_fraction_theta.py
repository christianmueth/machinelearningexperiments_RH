import os
import time
import argparse
import numpy as np
import pandas as pd

import machinelearning_rh_colab_cells as h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="out_phase3B_scan_seed_fraction_theta")
    ap.add_argument("--n_seeds", type=int, default=32)
    ap.add_argument("--anchors", default="2,9,14,44,46,51,60")
    ap.add_argument("--k", type=int, default=13)
    ap.add_argument("--window", default="2.0,5.0")
    ap.add_argument("--dnmap_stride", type=int, default=2)
    ap.add_argument("--amp", type=float, default=0.03)
    ap.add_argument("--N_null", type=int, default=256)
    args = ap.parse_args()

    out_root = args.out_root

    # Cheap scan protocol (per user guidance)
    seeds = list(range(int(args.n_seeds)))
    anchors = [int(x.strip()) for x in str(args.anchors).split(",") if x.strip()]

    k = int(args.k)
    wlo, whi = [float(x.strip()) for x in str(args.window).split(",")]
    windows = [(float(wlo), float(whi))]
    dnmap_stride = int(args.dnmap_stride)
    amp = float(args.amp)
    N_target = int(args.N_null)

    backends = [
        {"backend_label": "legacy", "tp_backend": "legacy"},
        {"backend_label": "geom_theta0p25", "tp_backend": "geom_v3_shared_axis", "geom_v3_theta_scale": 0.25},
        {"backend_label": "geom_theta0p125", "tp_backend": "geom_v3_shared_axis", "geom_v3_theta_scale": 0.125},
    ]

    rows = []
    t0 = time.time()

    for b in backends:
        backend_label = b["backend_label"]
        for seed in seeds:
            for aseed in anchors:
                cfg = dict(h.CFG_BASE) if hasattr(h, "CFG_BASE") else {}
                cfg.update(
                    {
                        "tp_backend": str(b["tp_backend"]),
                        "n_ops_list": [int(k)],
                        "windows": [tuple(w) for w in windows],
                        "N_null": int(N_target),
                        "dnmap_stride": int(dnmap_stride),
                        "anchor_seed": int(aseed),
                        "phase3_mode": "dnmap_only",
                        "warp_y_scale": 1.0,

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

                if "geom_v3_theta_scale" in b:
                    cfg["geom_v3_theta_scale"] = float(b["geom_v3_theta_scale"])

                out_dir = os.path.join(
                    out_root,
                    f"backend_{backend_label}",
                    f"N{int(N_target)}",
                    f"seed{int(seed)}_anchor{int(aseed)}",
                )

                print(f"\n=== scan backend={backend_label} seed={seed} anchor={aseed} ===")
                df = h.run_one(cfg, seed=int(seed), amp=float(amp), out_dir=out_dir)

                df["backend_label"] = backend_label
                df["tp_backend"] = str(b["tp_backend"])
                df["seed"] = int(seed)
                df["anchor_seed"] = int(aseed)
                df["N_target"] = int(N_target)

                rows.append(df)

    big = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    os.makedirs(out_root, exist_ok=True)

    raw_path = os.path.join(out_root, "raw.csv")
    big.to_csv(raw_path, index=False)

    # KPI: fraction of observation seeds with ANY reject (across anchors)
    if "reject" in big.columns and len(big):
        seed_kpi = (
            big.groupby(["backend_label", "seed"], dropna=False)
            .agg(
                any_reject=("reject", "max"),
                best_p=("p_family", "min"),
                worst_p=("p_family", "max"),
                rows=("reject", "size"),
            )
            .reset_index()
        )
        seed_kpi["any_reject"] = seed_kpi["any_reject"].astype(int)

        frac = (
            seed_kpi.groupby(["backend_label"], dropna=False)
            .agg(
                n_seeds=("seed", "nunique"),
                n_seeds_any_reject=("any_reject", "sum"),
                frac_seeds_any_reject=("any_reject", "mean"),
                best_p_overall=("best_p", "min"),
            )
            .reset_index()
        )
    else:
        seed_kpi = pd.DataFrame()
        frac = pd.DataFrame()

    seed_kpi_path = os.path.join(out_root, "kpi_by_seed.csv")
    frac_path = os.path.join(out_root, "kpi_fraction.csv")
    seed_kpi.to_csv(seed_kpi_path, index=False)
    frac.to_csv(frac_path, index=False)

    # Measurement fingerprint invariance audit
    digest_cols = [c for c in ["meas_params_digest", "meas_boundary_digest", "meas_E_digest"] if c in big.columns]
    if digest_cols:
        digest_summary = (
            big.groupby(["backend_label"], dropna=False)[digest_cols]
            .nunique(dropna=False)
            .reset_index()
        )
    else:
        digest_summary = pd.DataFrame()

    digest_path = os.path.join(out_root, "measurement_digest_summary.csv")
    digest_summary.to_csv(digest_path, index=False)

    print("\n=== DONE scan ===")
    print(f"wrote: {raw_path}")
    print(f"wrote: {seed_kpi_path}")
    print(f"wrote: {frac_path}")
    print(f"wrote: {digest_path}")
    if len(frac):
        print("\nKPI fraction of seeds with any reject:")
        print(frac)
    print(f"runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
