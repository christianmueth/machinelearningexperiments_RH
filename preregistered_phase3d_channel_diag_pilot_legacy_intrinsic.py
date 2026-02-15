import os
import time
import json
import argparse
import numpy as np
import pandas as pd

import machinelearning_rh_colab_cells as h


def _stringify_window_dict(d: dict) -> dict:
    out = {}
    for k, v in dict(d).items():
        if isinstance(k, (tuple, list)) and len(k) == 2:
            ks = f"{float(k[0])},{float(k[1])}"
        else:
            ks = str(k)
        out[ks] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="out_phase3D_channel_diag_pilot_legacy_intrinsic_v4")
    ap.add_argument("--amp", type=float, default=0.03)
    ap.add_argument("--k", type=int, default=13)
    ap.add_argument("--dnmap_stride", type=int, default=2)

    ap.add_argument("--backend", default="legacy", choices=["legacy", "geom_v9"], help="Which Tp backend to use for the pilot")

    # Pilot scope (pre-registered defaults)
    ap.add_argument("--seeds", default="64,72,80,88,96,104,112,120")
    ap.add_argument("--anchors", default="2,14,60")

    # Orbit cutoff
    ap.add_argument("--trace_K", type=int, default=8)

    # Cluster count for KPI postpass (written into prereg only; postpass script uses its own arg)
    ap.add_argument("--clusters", type=int, default=8)

    # Deterministic eta selection (reference run only)
    ap.add_argument("--eta_seed_ref", type=int, default=64)
    ap.add_argument("--eta_anchor_ref", type=int, default=2)
    ap.add_argument("--eta_tau_grid", default="0.25,0.5,1,2,4")
    ap.add_argument("--eta_lambda_normality", type=float, default=0.25)

    args = ap.parse_args()

    seeds = [int(x) for x in str(args.seeds).split(",") if str(x).strip()]
    anchors = [int(x) for x in str(args.anchors).split(",") if str(x).strip()]

    windows = [(0.60, 7.50), (2.0, 5.0)]

    # Fixed E snapshots per directive
    energies_snap_by_window = {
        (2.0, 5.0): [2.2, 3.2, 4.2],
        (0.60, 7.50): [1.0, 3.0, 6.5],
    }

    tau_grid = [float(x) for x in str(args.eta_tau_grid).split(",") if str(x).strip()]

    backend_label = str(args.backend)
    tp_backend = "legacy" if backend_label == "legacy" else "geom_warp_dirac_v9_phasetransport"

    prereg_spec = {
        "date": time.strftime("%Y-%m-%d"),
        "phase": "3D",
        "objective": "pilot_intrinsic_channel_anchor_stability",
        "backend": str(backend_label),
        "seeds": list(map(int, seeds)),
        "anchors": list(map(int, anchors)),
        "k": int(args.k),
        "windows": [list(map(float, w)) for w in windows],
        "dnmap_stride": int(args.dnmap_stride),
        "amp": float(args.amp),
        "E_snapshots_by_window": {str(k): list(map(float, v)) for k, v in energies_snap_by_window.items()},
        "trace_K": int(args.trace_K),
        "eta_selection": {
            "method": "tau_grid_minimize_unitarity_plus_normality",
            "reference_run": {"backend": "legacy", "seed": int(args.eta_seed_ref), "anchor": int(args.eta_anchor_ref)},
            "tau_grid": list(map(float, tau_grid)),
            "lambda_normality": float(args.eta_lambda_normality),
            "frozen_per_window": True,
        },
        "notes": "Pilot answers: intrinsic channels + anchor stability up to one gauge vs gauge-sliding.",
    }

    out_root = str(args.out_root)
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "preregistration.json"), "w", encoding="utf-8") as f:
        json.dump(prereg_spec, f, indent=2)
    with open(os.path.join(out_root, "preregistration_digest.txt"), "w", encoding="utf-8") as f:
        f.write(h.sha16(repr(prereg_spec).encode("utf-8")) + "\n")

    # Build a minimal cfg template
    cfg_template = dict(h.CFG_BASE)
    cfg_template.update(
        {
            "tp_backend": str(tp_backend),
            "k": int(args.k),
            "windows": [tuple(w) for w in windows],
            "dnmap_stride": int(args.dnmap_stride),
            "freeze_measurement": True,
            "freeze_geometry_policy": True,
            "gauge_policy": "none",
            "warp_mode": "ratio",
            "warp_unwrap": True,
            "warp_y_scale": 1.0,
            # channel diag knobs
            "chan_trace_K": int(args.trace_K),
            "chan_n_snap": 3,
            "chan_energy_snap_by_window": energies_snap_by_window,
            # Convergent orbit residuals use fixed damping |z|<1
            "chan_orbit_z": 0.5,
            # explicitly no closure injection
            "pq_anchor_M": 0,
            "pqr_anchor_M": 0,
            "p_pow_kmax": 0,
            "use_pq": False,
            "use_pqr": False,
            "use_pow": False,
            "use_tower": False,
        }
    )

    # Backend-specific prereg knobs
    if backend_label == "geom_v9":
        cfg_template.update(
            {
                "eps_auto_tune": True,
                "eps_target_delta": 0.02,
                "geom_v9_transport_rule": "delta_raw_wrapped",
                "geom_v9_transport_lambda": 1.0,
                "geom_v9_transport_clip": 0.25,
            }
        )

    # Deterministic per-window eta selection from fixed reference run
    eta_sel = h.select_cayley_eta_per_window(
        cfg=cfg_template,
        seed_ref=int(args.eta_seed_ref),
        anchor_ref=int(args.eta_anchor_ref),
        amp=float(args.amp),
        k_used=int(args.k),
        windows=[tuple(w) for w in windows],
        energies_snap_by_window=energies_snap_by_window,
        tau_grid=tau_grid,
        lambda_normality=float(args.eta_lambda_normality),
    )
    eta_by_window = eta_sel["eta_by_window"]

    with open(os.path.join(out_root, "eta_selection.json"), "w", encoding="utf-8") as f:
        eta_sel_json = dict(eta_sel)
        eta_sel_json["eta_by_window"] = _stringify_window_dict(eta_sel_json.get("eta_by_window", {}))
        eta_sel_json["selection_details"] = _stringify_window_dict(eta_sel_json.get("selection_details", {}))
        json.dump(eta_sel_json, f, indent=2, default=str)

    rows = []
    for seed in seeds:
        for aseed in anchors:
            cfg = dict(cfg_template)
            cfg["anchor_seed"] = int(aseed)
            cfg["chan_cayley_eta_by_window"] = eta_by_window

            out_dir = os.path.join(out_root, f"backend_{backend_label}", f"seed{int(seed)}_anchor{int(aseed)}")
            print(f"\n=== phase3D pilot {backend_label} seed={seed} anchor={aseed} ===", flush=True)
            df = h.run_one_channel_diag(cfg, seed=int(seed), amp=float(args.amp), out_dir=out_dir)

            df["backend_label"] = str(backend_label)
            df["seed"] = int(seed)
            df["anchor_seed"] = int(aseed)
            rows.append(df)

    big = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    big.to_csv(os.path.join(out_root, "raw.csv"), index=False)
    print(f"\n[phase3d_pilot_legacy_intrinsic] DONE rows={len(big)} out_root={out_root}")


if __name__ == "__main__":
    main()
