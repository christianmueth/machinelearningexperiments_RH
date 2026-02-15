import importlib.util
import os

import pandas as pd


def load_harness(path: str):
    spec = importlib.util.spec_from_file_location("mlrh", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    harness_path = os.path.join(here, "machinelearning_rh_colab_cells.py")
    m = load_harness(harness_path)

    out_root = os.path.join(here, "out_phase3B_dnmap_only_confirm_theta0125_vs_baselines")
    os.makedirs(out_root, exist_ok=True)

    # Extended confirm protocol
    seeds = tuple(range(8))  # observation seeds
    anchors = (2, 9, 14, 44, 46, 51, 60)  # scramble/anchor seeds
    k = 13
    N_null = 1024
    dnmap_stride = 2
    amp = 0.03

    base_cfg = dict(m.CFG_BASE)
    base_cfg.update(
        {
            "phase3_mode": "dnmap_only",
            "n_ops_list": [k],
            "N_null": N_null,
            "windows": [(0.6, 7.5), (2.0, 5.0)],
            "dnmap_stride": dnmap_stride,
            "freeze_measurement": True,
            "use_pq": False,
            "use_pqr": False,
            "use_pow": False,
            "use_dnmap_gate": True,
        }
    )

    candidates = [
        {"label": "legacy", "tp_backend": "legacy"},
        {
            "label": "geom_theta0.25",
            "tp_backend": "geom_v3_shared_axis",
            "warp_y_scale": 1.0,
            "geom_v3_rot_scale": 0.25,
            "geom_v3_theta_scale": 0.25,
        },
        {
            "label": "geom_theta0.125",
            "tp_backend": "geom_v3_shared_axis",
            "warp_y_scale": 1.0,
            "geom_v3_rot_scale": 0.25,
            "geom_v3_theta_scale": 0.125,
        },
    ]

    rows = []
    for cand in candidates:
        for seed in seeds:
            for aseed in anchors:
                cfg = dict(base_cfg)
                cfg["tp_backend"] = cand["tp_backend"]
                cfg["anchor_seed"] = aseed
                for kset, v in cand.items():
                    if kset in ("label", "tp_backend"):
                        continue
                    cfg[kset] = v

                out_dir = os.path.join(out_root, cand["label"], f"s{seed}_a{aseed}")
                df = m.run_one(cfg, seed=seed, amp=amp, out_dir=out_dir)
                df["backend_label"] = cand["label"]
                rows.append(df)

    raw = pd.concat(rows, ignore_index=True)
    raw_path = os.path.join(out_root, "raw.csv")
    raw.to_csv(raw_path, index=False)

    raw["dn_delta"] = raw["dn_null_median"] - raw["dn_obs"]
    raw["dn_snr"] = raw["dn_delta"] / (raw["dn_null_std"] + 1e-12)
    raw["dn_null_cv"] = raw["dn_null_std"] / (raw["dn_null_mean"].abs() + 1e-12)

    agg = (
        raw.groupby(["backend_label", "tp_backend", "wlo", "whi"])
        .agg(
            rows=("p_dnmap", "size"),
            rejects=("reject", "sum"),
            reject_rate=("reject", "mean"),
            p_med=("p_dnmap", "median"),
            p_best=("p_dnmap", "min"),
            p_worst=("p_dnmap", "max"),
            dn_snr_med=("dn_snr", "median"),
            dn_null_cv_med=("dn_null_cv", "median"),
        )
        .reset_index()
    )
    agg_path = os.path.join(out_root, "aggregate.csv")
    agg.to_csv(agg_path, index=False)

    agg_seed = (
        raw.groupby(["backend_label", "tp_backend", "seed", "wlo", "whi"])
        .agg(
            rows=("p_dnmap", "size"),
            rejects=("reject", "sum"),
            reject_rate=("reject", "mean"),
            p_med=("p_dnmap", "median"),
            p_best=("p_dnmap", "min"),
            dn_snr_med=("dn_snr", "median"),
            dn_null_cv_med=("dn_null_cv", "median"),
        )
        .reset_index()
    )
    agg_seed_path = os.path.join(out_root, "aggregate_by_seed.csv")
    agg_seed.to_csv(agg_seed_path, index=False)

    print("\nWROTE")
    print(raw_path)
    print(agg_path)
    print(agg_seed_path)

    print("\n=== OVERALL aggregate ===")
    print(agg.sort_values(["wlo", "backend_label"]).to_string(index=False))

    print("\n=== Seed stability (window 2.0-5.0) ===")
    sub = agg_seed[(agg_seed["wlo"] == 2.0) & (agg_seed["whi"] == 5.0)].copy()
    print(sub.sort_values(["backend_label", "seed"]).to_string(index=False))


if __name__ == "__main__":
    main()
