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
    ap.add_argument("--out_root", default="out_phase3D_channel_diag_64_127")
    ap.add_argument("--seed_start", type=int, default=64)
    ap.add_argument("--seed_end_exclusive", type=int, default=128)
    ap.add_argument("--anchors", default="2,9,14,44,46,51,60")

    ap.add_argument("--k", type=int, default=13)
    ap.add_argument("--windows", default="2.0,5.0;0.6,7.5")
    ap.add_argument("--dnmap_stride", type=int, default=2)
    ap.add_argument("--amp", type=float, default=0.03)

    # Channel diagnostics knobs
    ap.add_argument("--chan_n_snap", type=int, default=5, help="Number of energies per window to snapshot")
    ap.add_argument("--chan_trace_K", type=int, default=8, help="Max K for trace orbit consistency")
    ap.add_argument("--chan_cayley_eta", type=float, default=1.0, help="Eta for Cayley transform S=(Λ-iηI)(Λ+iηI)^{-1}")

    ap.add_argument("--resume", action="store_true", help="Skip runs whose out_dir already contains rows.csv")

    # optional v9 knobs (only used by v9 backend)
    ap.add_argument("--v9_transport_rule", default="delta_raw_wrapped")
    ap.add_argument("--v9_transport_lambda", type=float, default=1.0)
    ap.add_argument("--v9_transport_clip", type=float, default=0.25)

    args = ap.parse_args()

    out_root = args.out_root
    seeds = list(range(int(args.seed_start), int(args.seed_end_exclusive)))
    anchors = parse_int_list(args.anchors)
    windows = parse_windows(args.windows)

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

    prereg_spec = {
        "date": time.strftime("%Y-%m-%d"),
        "phase": "3D",
        "objective": "channel_diagnostics_pack",
        "seeds": [int(args.seed_start), int(args.seed_end_exclusive)],
        "anchors": list(map(int, anchors)),
        "k": int(args.k),
        "windows": [list(map(float, w)) for w in windows],
        "dnmap_stride": int(args.dnmap_stride),
        "amp": float(args.amp),
        "chan_n_snap": int(args.chan_n_snap),
        "chan_trace_K": int(args.chan_trace_K),
        "chan_cayley_eta": float(args.chan_cayley_eta),
        "backends": backends,
        "measurement_frozen": True,
        "geometry_policy_frozen": True,
        "warp_mode": "ratio",
        "warp_unwrap": True,
        "warp_y_scale": 1.0,
        "no_tp_injection": True,
    }

    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "preregistration.json"), "w", encoding="utf-8") as f:
        json.dump(prereg_spec, f, indent=2)
    with open(os.path.join(out_root, "preregistration_digest.txt"), "w", encoding="utf-8") as f:
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
                        "k": int(args.k),
                        "windows": [tuple(w) for w in windows],
                        "dnmap_stride": int(args.dnmap_stride),
                        "anchor_seed": int(aseed),
                        "freeze_measurement": True,
                        "freeze_geometry_policy": True,
                        "gauge_policy": "none",
                        "warp_mode": "ratio",
                        "warp_unwrap": True,
                        "warp_y_scale": 1.0,
                        # channel diag knobs
                        "chan_n_snap": int(args.chan_n_snap),
                        "chan_trace_K": int(args.chan_trace_K),
                        "chan_cayley_eta": float(args.chan_cayley_eta),
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
                    f"seed{int(seed)}_anchor{int(aseed)}",
                )

                rows_csv = os.path.join(out_dir, "rows.csv")
                if args.resume and os.path.exists(rows_csv):
                    df = pd.read_csv(rows_csv)
                    print(f"\n=== prereg SKIP (resume) backend={backend_label} seed={seed} anchor={aseed} ===", flush=True)
                else:
                    print(f"\n=== prereg backend={backend_label} seed={seed} anchor={aseed} ===", flush=True)
                    df = h.run_one_channel_diag(cfg, seed=int(seed), amp=float(args.amp), out_dir=out_dir)

                df["backend_label"] = str(backend_label)
                df["tp_backend"] = str(b["tp_backend"])
                df["seed"] = int(seed)
                df["anchor_seed"] = int(aseed)

                rows.append(df)

    big = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    big.to_csv(os.path.join(out_root, "raw.csv"), index=False)

    print(f"\n[phase3d_channel_diag] DONE rows={len(big)} out_root={out_root}")


if __name__ == "__main__":
    main()
