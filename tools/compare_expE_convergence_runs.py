from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


def _load_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def pick_runs(runs_root: Path) -> tuple[Path, Path]:
    run_dirs = sorted([p for p in runs_root.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
    if len(run_dirs) < 2:
        raise SystemExit("Need at least two run directories.")

    autopick_dir: Path | None = None
    baseline_dir: Path | None = None

    # Find most recent baseline and autopick runs by reading config.json.
    for rd in run_dirs:
        cfg = _load_config(rd)
        do_conv = bool(cfg.get("do_convergence_sweep_only", False))
        if not do_conv:
            continue
        has_conv_csv = (rd / "E_Qlambda_convergence_sweep.csv").exists()
        if not has_conv_csv:
            continue

        ap = bool(cfg.get("boundary_autopick", False))
        if ap and autopick_dir is None:
            autopick_dir = rd
        if (not ap) and baseline_dir is None:
            baseline_dir = rd
        if autopick_dir is not None and baseline_dir is not None:
            break

    # Fallback: just take the newest two.
    if baseline_dir is None or autopick_dir is None:
        baseline_dir = run_dirs[0]
        autopick_dir = run_dirs[1]

    return baseline_dir, autopick_dir


def _load_autopick_acceptance(autopick_dir: Path) -> pd.DataFrame:
    """Return per-N acceptance stats from E_boundary_autopick_candidates.csv if present.

    Computes both:
      - Independent failure counts per constraint (not disjoint)
      - Ordered-pipeline rejections (disjoint by stage)
    """

    cand_path = autopick_dir / "E_boundary_autopick_candidates.csv"
    if not cand_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(cand_path)
    if "N" not in df.columns:
        return pd.DataFrame()

    for col, default in [
        ("metric_ok", False),
        ("gap_ok", False),
        ("wind_ok_stage1", False),
        ("top5_guard_ok", True),
    ]:
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].astype(bool)

    out_rows: list[dict] = []
    for N, g in df.groupby("N", dropna=False):
        n_total = int(len(g))

        # Independent failures (not disjoint).
        indep_fail_metric = int((~g["metric_ok"]).sum())
        indep_fail_gap = int((~g["gap_ok"]).sum())
        indep_fail_wind = int((~g["wind_ok_stage1"]).sum())
        indep_fail_guard = int((~g["top5_guard_ok"]).sum())

        # Ordered pipeline counts (disjoint stage rejections).
        s0 = g
        s1 = s0[s0["metric_ok"]]
        s2 = s1[s1["gap_ok"]]
        s3 = s2[s2["wind_ok_stage1"]]
        s4 = s3[s3["top5_guard_ok"]]

        n_after_metric = int(len(s1))
        n_after_gap = int(len(s2))
        n_after_wind = int(len(s3))
        n_after_guard = int(len(s4))

        seq_rej_metric = int(n_total - n_after_metric)
        seq_rej_gap = int(n_after_metric - n_after_gap)
        seq_rej_wind = int(n_after_gap - n_after_wind)
        seq_rej_guard = int(n_after_wind - n_after_guard)

        out_rows.append(
            {
                "N": int(N),
                "ap_n_candidates": n_total,
                "ap_indep_fail_metric": indep_fail_metric,
                "ap_indep_fail_gap": indep_fail_gap,
                "ap_indep_fail_wind": indep_fail_wind,
                "ap_indep_fail_guard": indep_fail_guard,
                "ap_seq_rej_metric": seq_rej_metric,
                "ap_seq_rej_gap": seq_rej_gap,
                "ap_seq_rej_wind": seq_rej_wind,
                "ap_seq_rej_guard": seq_rej_guard,
                "ap_seq_pass_all": n_after_guard,
            }
        )

    out = pd.DataFrame(out_rows).sort_values("N")
    return out


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runs_root = repo_root / "runs"

    baseline_dir, autopick_dir = pick_runs(runs_root)

    b = pd.read_csv(baseline_dir / "E_Qlambda_convergence_sweep.csv")
    a = pd.read_csv(autopick_dir / "E_Qlambda_convergence_sweep.csv")

    key_cols = ["N", "basis", "block"]
    metrics = [
        "boundary_b",
        "boundary_b0",
        "err_phi_norm_median",
        "err_phi_norm_p95",
        "err_phi_norm_top5_mean",
        "err_phi_norm_max",
        "gap_phi_min",
    ]

    m = (
        b[key_cols + metrics]
        .merge(a[key_cols + metrics], on=key_cols, suffixes=("_baseline", "_autopick"), how="inner")
        .copy()
    )

    print(f"BASELINE_RUN={baseline_dir.name}")
    print(f"AUTOPICK_RUN={autopick_dir.name}")

    # Focus channel: incoming/+.
    view = m[(m["basis"] == "incoming") & (m["block"] == "plus")].copy().sort_values("N")
    for col in [
        "err_phi_norm_median",
        "err_phi_norm_p95",
        "err_phi_norm_top5_mean",
        "err_phi_norm_max",
        "gap_phi_min",
    ]:
        view[col + "_delta"] = view[col + "_autopick"] - view[col + "_baseline"]

    cols = [
        "N",
        "boundary_b0_baseline",
        "boundary_b0_autopick",
        "err_phi_norm_median_baseline",
        "err_phi_norm_median_autopick",
        "err_phi_norm_median_delta",
        "err_phi_norm_p95_baseline",
        "err_phi_norm_p95_autopick",
        "err_phi_norm_p95_delta",
        "err_phi_norm_top5_mean_baseline",
        "err_phi_norm_top5_mean_autopick",
        "err_phi_norm_top5_mean_delta",
        "gap_phi_min_baseline",
        "gap_phi_min_autopick",
        "gap_phi_min_delta",
    ]

    print("\nIncoming/+ comparison (negative deltas are improvements):")
    print(view[cols].to_string(index=False))

    # Aggregate per N across 4 channels.
    agg_rows: list[dict] = []
    for N, g in m.groupby("N", dropna=False):
        row = {"N": int(N)}

        def _ch_id(df: pd.DataFrame, col: str) -> str:
            dd = df.copy()
            dd[col] = pd.to_numeric(dd[col], errors="coerce")
            dd = dd[dd[col].notna()]
            if dd.empty:
                return "(na)"
            j = int(dd[col].astype(float).values.argmax())
            r = dd.iloc[j]
            return f"{r['basis']}/{r['block']}"

        row["p95_baseline_max_over_channels"] = float(g["err_phi_norm_p95_baseline"].max())
        row["p95_autopick_max_over_channels"] = float(g["err_phi_norm_p95_autopick"].max())
        row["p95_delta_max_over_channels"] = row["p95_autopick_max_over_channels"] - row["p95_baseline_max_over_channels"]

        row["p95_baseline_argmax_channel"] = _ch_id(g, "err_phi_norm_p95_baseline")
        row["p95_autopick_argmax_channel"] = _ch_id(g, "err_phi_norm_p95_autopick")

        row["p95_baseline_median_over_channels"] = float(g["err_phi_norm_p95_baseline"].median())
        row["p95_autopick_median_over_channels"] = float(g["err_phi_norm_p95_autopick"].median())
        row["p95_delta_median_over_channels"] = row["p95_autopick_median_over_channels"] - row["p95_baseline_median_over_channels"]

        row["gapmin_baseline_min_over_channels"] = float(g["gap_phi_min_baseline"].min())
        row["gapmin_autopick_min_over_channels"] = float(g["gap_phi_min_autopick"].min())
        row["gapmin_delta_min_over_channels"] = row["gapmin_autopick_min_over_channels"] - row["gapmin_baseline_min_over_channels"]

        row["top5_baseline_max_over_channels"] = float(g["err_phi_norm_top5_mean_baseline"].max())
        row["top5_autopick_max_over_channels"] = float(g["err_phi_norm_top5_mean_autopick"].max())
        row["top5_delta_max_over_channels"] = row["top5_autopick_max_over_channels"] - row["top5_baseline_max_over_channels"]

        row["top5_baseline_argmax_channel"] = _ch_id(g, "err_phi_norm_top5_mean_baseline")
        row["top5_autopick_argmax_channel"] = _ch_id(g, "err_phi_norm_top5_mean_autopick")
        agg_rows.append(row)

    agg = pd.DataFrame(agg_rows).sort_values("N")

    # Attach autopick acceptance stats if available.
    ap_acc = _load_autopick_acceptance(autopick_dir)
    if not ap_acc.empty:
        agg = agg.merge(ap_acc, on="N", how="left")

    print("\nPer-N WORST-CHANNEL curve (max over 4 channels):")
    compact_cols = [
        "N",
        "p95_baseline_max_over_channels",
        "p95_autopick_max_over_channels",
        "p95_delta_max_over_channels",
        "p95_baseline_argmax_channel",
        "p95_autopick_argmax_channel",
        "top5_baseline_max_over_channels",
        "top5_autopick_max_over_channels",
        "top5_delta_max_over_channels",
        "top5_baseline_argmax_channel",
        "top5_autopick_argmax_channel",
        "gapmin_baseline_min_over_channels",
        "gapmin_autopick_min_over_channels",
        "gapmin_delta_min_over_channels",
    ]

    if not ap_acc.empty:
        # Add acceptance-rate columns (numeric) + a compact "k/total" presentation.
        agg = agg.copy()
        total = pd.to_numeric(agg.get("ap_n_candidates"), errors="coerce")
        for kcol in [
            "ap_indep_fail_metric",
            "ap_indep_fail_wind",
            "ap_indep_fail_gap",
            "ap_indep_fail_guard",
            "ap_seq_rej_metric",
            "ap_seq_rej_wind",
            "ap_seq_rej_gap",
            "ap_seq_rej_guard",
            "ap_seq_pass_all",
        ]:
            if kcol in agg.columns:
                kk = pd.to_numeric(agg[kcol], errors="coerce")
                agg[kcol + "_of_total"] = (kk.fillna(0).astype(int).astype(str) + "/" + total.fillna(0).astype(int).astype(str))

        compact_cols = compact_cols + [
            "ap_indep_fail_metric_of_total",
            "ap_indep_fail_wind_of_total",
            "ap_indep_fail_gap_of_total",
            "ap_indep_fail_guard_of_total",
            "ap_seq_rej_metric_of_total",
            "ap_seq_rej_wind_of_total",
            "ap_seq_rej_gap_of_total",
            "ap_seq_rej_guard_of_total",
            "ap_seq_pass_all_of_total",
        ]
        print("\n(acceptance stats shown as k/total; independent fails may overlap)")

    print(agg[compact_cols].to_string(index=False))

    out_csv = repo_root / "runs" / "_compare_latest_baseline_vs_autopick.csv"
    # Save full numeric columns to CSV (drop the pretty k/total strings if present).
    save_df = agg.copy()
    for c in list(save_df.columns):
        if c.endswith("_of_total"):
            pass
    save_df.to_csv(out_csv, index=False)
    print(f"\nWrote summary CSV: {out_csv}")


if __name__ == "__main__":
    main()
