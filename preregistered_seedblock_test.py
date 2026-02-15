import os
import time
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="out_phase3B_preregistered_seedblock_32_63")
    ap.add_argument("--seed_start", type=int, default=32)
    ap.add_argument("--seed_end_exclusive", type=int, default=64)
    ap.add_argument("--anchors", default="2,9,14,44,46,51,60")
    ap.add_argument("--N_null", type=int, default=4096)
    ap.add_argument("--k", type=int, default=13)
    ap.add_argument("--windows", default="2.0,5.0;0.6,7.5")
    ap.add_argument("--dnmap_stride", type=int, default=2)
    ap.add_argument("--amp", type=float, default=0.03)
    ap.add_argument("--resume", action="store_true", help="Skip runs whose out_dir already contains rows.csv")
    args = ap.parse_args()

    out_root = args.out_root
    seeds = list(range(int(args.seed_start), int(args.seed_end_exclusive)))
    anchors = parse_int_list(args.anchors)
    windows = parse_windows(args.windows)

    # Preregistered backends: lock theta=0.125, do not tune anything else.
    backends = [
        {"backend_label": "legacy", "tp_backend": "legacy"},
        {"backend_label": "geom_theta0p125", "tp_backend": "geom_v3_shared_axis", "geom_v3_theta_scale": 0.125},
    ]

    t0 = time.time()
    rows = []

    for b in backends:
        backend_label = b["backend_label"]
        for seed in seeds:
            for aseed in anchors:
                cfg = dict(h.CFG_BASE) if hasattr(h, "CFG_BASE") else {}
                cfg.update(
                    {
                        "tp_backend": str(b["tp_backend"]),
                        "n_ops_list": [int(args.k)],
                        "windows": [tuple(w) for w in windows],
                        "N_null": int(args.N_null),
                        "dnmap_stride": int(args.dnmap_stride),
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

                df["backend_label"] = backend_label
                df["tp_backend"] = str(b["tp_backend"])
                df["seed"] = int(seed)
                df["anchor_seed"] = int(aseed)
                df["N_null"] = int(args.N_null)

                # Derived KPIs
                if "dn_obs" in df.columns and "dn_null_mean" in df.columns and "dn_delta" not in df.columns:
                    df["dn_delta"] = df["dn_obs"] - df["dn_null_mean"]

                rows.append(df)

    big = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    os.makedirs(out_root, exist_ok=True)

    raw_path = os.path.join(out_root, "raw.csv")
    big.to_csv(raw_path, index=False)

    # KPI tables
    if len(big) and "reject" in big.columns:
        by_seed = (
            big.groupby(["backend_label", "wlo", "whi", "seed"], dropna=False)
            .agg(
                any_reject=("reject", "max"),
                best_p=("p_family", "min"),
                worst_p=("p_family", "max"),
                best_dn_delta=("dn_delta", "max") if "dn_delta" in big.columns else ("p_family", lambda x: np.nan),
                median_dn_delta=("dn_delta", "median") if "dn_delta" in big.columns else ("p_family", lambda x: np.nan),
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
                best_p_overall=("best_p", "min"),
                median_best_p=("best_p", "median"),
                median_best_dn_delta=("best_dn_delta", "median"),
            )
            .reset_index()
        )

        # Optional: distributions (per-run), useful for plots
        dist_cols = [
            c
            for c in [
                "backend_label",
                "wlo",
                "whi",
                "seed",
                "anchor_seed",
                "p_family",
                "p_dnmap",
                "dn_obs",
                "dn_null_mean",
                "dn_null_std",
                "dn_delta",
                "reject",
            ]
            if c in big.columns
        ]
        dist = big[dist_cols].copy()
    else:
        by_seed = pd.DataFrame()
        frac = pd.DataFrame()
        dist = pd.DataFrame()

    by_seed_path = os.path.join(out_root, "kpi_by_seed.csv")
    frac_path = os.path.join(out_root, "kpi_fraction.csv")
    dist_path = os.path.join(out_root, "distributions.csv")

    by_seed.to_csv(by_seed_path, index=False)
    frac.to_csv(frac_path, index=False)
    dist.to_csv(dist_path, index=False)

    # Measurement fingerprint invariance audit
    # Note: meas_E_digest can legitimately differ across windows; so audit by (backend, window).
    digest_cols = [c for c in ["meas_params_digest", "meas_boundary_digest", "meas_E_digest"] if c in big.columns]
    if digest_cols and {"wlo", "whi"}.issubset(set(big.columns)):
        digest_summary = (
            big.groupby(["backend_label", "wlo", "whi"], dropna=False)[digest_cols]
            .nunique(dropna=False)
            .reset_index()
        )
    elif digest_cols:
        digest_summary = (
            big.groupby(["backend_label"], dropna=False)[digest_cols]
            .nunique(dropna=False)
            .reset_index()
        )
    else:
        digest_summary = pd.DataFrame()

    digest_path = os.path.join(out_root, "measurement_digest_summary.csv")
    digest_summary.to_csv(digest_path, index=False)

    print("\n=== DONE preregistered test ===")
    print(f"wrote: {raw_path}")
    print(f"wrote: {by_seed_path}")
    print(f"wrote: {frac_path}")
    print(f"wrote: {dist_path}")
    print(f"wrote: {digest_path}")
    if len(frac):
        print("\nKPI fraction of seeds with any reject:")
        print(frac)
    print(f"runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
