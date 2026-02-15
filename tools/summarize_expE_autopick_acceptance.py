import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _find_latest_run_with_candidates(runs_dir: Path) -> Path | None:
    if not runs_dir.exists():
        return None
    candidates = []
    for p in runs_dir.iterdir():
        if not p.is_dir():
            continue
        cand = p / "E_boundary_autopick_candidates.csv"
        if cand.exists():
            candidates.append((cand.stat().st_mtime, p))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[-1][1]


def summarize(run_dir: Path) -> pd.DataFrame:
    cand_path = run_dir / "E_boundary_autopick_candidates.csv"
    if not cand_path.exists():
        raise FileNotFoundError(f"Missing {cand_path}")

    df = pd.read_csv(cand_path)

    # Coerce expected boolean fields.
    for col, default in [
        ("metric_ok", False),
        ("gap_ok", False),
        ("wind_ok_stage1", False),
        ("top5_guard_ok", True),
    ]:
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].astype(bool)

    if "N" not in df.columns:
        raise ValueError("Candidates CSV missing column 'N'")

    out_rows: list[dict] = []
    for N0, g in df.groupby("N", sort=True):
        n_total = int(len(g))
        n_pass_metric = int(g["metric_ok"].sum())
        n_pass_gap = int(g["gap_ok"].sum())
        n_pass_wind = int(g["wind_ok_stage1"].sum())
        n_pass_guard = int(g["top5_guard_ok"].sum())
        n_pass_all = int((g["metric_ok"] & g["gap_ok"] & g["wind_ok_stage1"] & g["top5_guard_ok"]).sum())

        # Ordered pipeline counts (disjoint stage rejections): metric → gap → wind → guard.
        s1 = g[g["metric_ok"]]
        s2 = s1[s1["gap_ok"]]
        s3 = s2[s2["wind_ok_stage1"]]
        s4 = s3[s3["top5_guard_ok"]]

        n_after_metric = int(len(s1))
        n_after_gap = int(len(s2))
        n_after_wind = int(len(s3))
        n_after_guard = int(len(s4))

        n_rej_metric = int(n_total - n_after_metric)
        n_rej_gap = int(n_after_metric - n_after_gap)
        n_rej_wind = int(n_after_gap - n_after_wind)
        n_rej_guard = int(n_after_wind - n_after_guard)

        out_rows.append(
            {
                "N": int(N0),
                "n_candidates": n_total,
                "n_pass_metric": n_pass_metric,
                "n_pass_gap": n_pass_gap,
                "n_pass_wind_stage1": n_pass_wind,
                "n_pass_top5_guard": n_pass_guard,
                "n_pass_all": n_pass_all,
                "n_fail_metric": int((~g["metric_ok"]).sum()),
                "n_fail_gap": int((~g["gap_ok"]).sum()),
                "n_fail_wind_stage1": int((~g["wind_ok_stage1"]).sum()),
                "n_fail_top5_guard": int((~g["top5_guard_ok"]).sum()),
                "pipeline_order": "metric->gap->wind->guard",
                "n_after_metric": n_after_metric,
                "n_after_gap": n_after_gap,
                "n_after_wind_stage1": n_after_wind,
                "n_after_top5_guard": n_after_guard,
                "n_rej_metric": n_rej_metric,
                "n_rej_gap": n_rej_gap,
                "n_rej_wind_stage1": n_rej_wind,
                "n_rej_top5_guard": n_rej_guard,
            }
        )

    out = pd.DataFrame(out_rows)

    # Add guard cap diagnostics if present.
    if "top5_star" in df.columns:
        star = df.groupby("N")["top5_star"].first().reset_index().rename(columns={"top5_star": "top5_star"})
        out = out.merge(star, on="N", how="left")
    if "top5_guard_ok" in df.columns and "top5_star" in df.columns:
        pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="", help="Run directory under runs/ (or absolute path).")
    ap.add_argument("--runs_dir", type=str, default="runs", help="Runs directory (default: runs)")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if args.run:
        run_dir = Path(args.run)
        if not run_dir.is_absolute():
            # Allow either a bare run name (e.g. "2026...__expE...") or a relative path
            # like "runs/2026...__expE..." without duplicating the prefix.
            if run_dir.parts and run_dir.parts[0] == runs_dir.name:
                # Treat as already relative to repo root.
                run_dir = run_dir
            else:
                run_dir = runs_dir / run_dir
    else:
        run_dir = _find_latest_run_with_candidates(runs_dir)
        if run_dir is None:
            raise FileNotFoundError(f"No runs found with E_boundary_autopick_candidates.csv under {runs_dir}")

    out = summarize(run_dir)
    out_path = run_dir / "E_boundary_autopick_acceptance.csv"
    out.to_csv(out_path, index=False)

    print(f"RUN={run_dir}")
    print(f"WROTE={out_path}")
    with pd.option_context("display.max_rows", 200, "display.width", 200):
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
