import os
import time
import argparse
import pandas as pd

import machinelearning_rh_colab_cells as h


def parse_int_list(s: str):
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="out_phase3B_confirm_top_seeds_highnull")
    ap.add_argument("--seeds", default="1,8,14,20")
    ap.add_argument("--anchors", default="2,9,14,44,46,51,60")
    ap.add_argument("--N_null", type=int, default=4096)
    ap.add_argument("--k", type=int, default=13)
    ap.add_argument("--window", default="2.0,5.0")
    ap.add_argument("--dnmap_stride", type=int, default=2)
    ap.add_argument("--amp", type=float, default=0.03)
    args = ap.parse_args()

    out_root = args.out_root
    seeds = parse_int_list(args.seeds)
    anchors = parse_int_list(args.anchors)

    wlo, whi = [float(x.strip()) for x in str(args.window).split(",")]
    windows = [(float(wlo), float(whi))]

    backends = [
        {"backend_label": "legacy", "tp_backend": "legacy"},
        {"backend_label": "geom_theta0p25", "tp_backend": "geom_v3_shared_axis", "geom_v3_theta_scale": 0.25},
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

                print(f"\n=== confirm backend={backend_label} seed={seed} anchor={aseed} ===")
                df = h.run_one(cfg, seed=int(seed), amp=float(args.amp), out_dir=out_dir)
                df["backend_label"] = backend_label
                df["tp_backend"] = str(b["tp_backend"])
                df["seed"] = int(seed)
                df["anchor_seed"] = int(aseed)
                df["N_null"] = int(args.N_null)
                rows.append(df)

    big = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    os.makedirs(out_root, exist_ok=True)

    raw_path = os.path.join(out_root, "raw.csv")
    big.to_csv(raw_path, index=False)

    # Aggregate by backend
    if "reject" in big.columns and len(big):
        agg = (
            big.groupby(["backend_label"], dropna=False)
            .agg(
                rows=("reject", "size"),
                reject_rate=("reject", "mean"),
                best_p=("p_family", "min"),
                worst_p=("p_family", "max"),
            )
            .reset_index()
            .sort_values(["reject_rate", "best_p"], ascending=[False, True])
        )

        by_seed = (
            big.groupby(["backend_label", "seed"], dropna=False)
            .agg(
                any_reject=("reject", "max"),
                best_p=("p_family", "min"),
                worst_p=("p_family", "max"),
                rows=("reject", "size"),
            )
            .reset_index()
            .sort_values(["backend_label", "any_reject", "best_p"], ascending=[True, False, True])
        )
        by_seed["any_reject"] = by_seed["any_reject"].astype(int)
    else:
        agg = pd.DataFrame()
        by_seed = pd.DataFrame()

    agg_path = os.path.join(out_root, "aggregate.csv")
    by_seed_path = os.path.join(out_root, "aggregate_by_seed.csv")
    agg.to_csv(agg_path, index=False)
    by_seed.to_csv(by_seed_path, index=False)

    print("\n=== DONE confirm ===")
    print(f"wrote: {raw_path}")
    print(f"wrote: {agg_path}")
    print(f"wrote: {by_seed_path}")
    if len(agg):
        print("\nAggregate:")
        print(agg)
    print(f"runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
