import argparse
import json
import os
import runpy
import sys
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


def _fmt(x: float) -> str:
    return f"{float(x):.10g}"


def _round10(xs: Iterable[float]) -> List[float]:
    return [float(np.round(float(x), 10)) for x in xs]


def _parse_pair_csv(s: str) -> Tuple[float, float]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected 'a,b' got: {s!r}")
    return float(parts[0]), float(parts[1])


def _build_dense_points(
    intervals: Sequence[Tuple[float, float]], *, pad: float, fine_step: float, max_points: int
) -> List[float]:
    pts: List[float] = []
    for l0, l1 in intervals:
        lo = max(0.0, float(l0) - float(pad))
        hi = min(1.0, float(l1) + float(pad))
        if hi <= lo:
            continue

        step = float(fine_step)
        if step <= 0.0:
            raise ValueError("fine_step must be > 0")

        n = int(np.ceil((hi - lo) / step))
        n = max(1, n)
        # Avoid pathological blow-ups.
        if n + 1 > int(max_points):
            n = int(max_points) - 1

        grid = np.linspace(lo, hi, n + 1)
        pts.extend([float(x) for x in grid.tolist()])

    return _round10(pts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source_out_root",
        default="out_phase3E_elambda_scale8_legacy_geomv9_localrefine_0p25_0p50",
        help="Existing run directory with wall tables (expects wall_cells_top20.csv + suite_summary.json).",
    )
    ap.add_argument(
        "--lambda_band",
        default="0.25,0.50",
        help="Only consider hot cells whose [lambda0,lambda1] lies within this band (a,b).",
    )
    ap.add_argument("--top_k", type=int, default=6, help="How many hottest intervals to refine around")
    ap.add_argument(
        "--pad",
        type=float,
        default=0.0125,
        help="Pad (in lambda units) around each selected [lambda0,lambda1] interval",
    )
    ap.add_argument(
        "--fine_step",
        type=float,
        default=0.00625,
        help="Fine lambda step size inside padded regions (default is 0.0125/2; decrease for more resolution)",
    )
    ap.add_argument(
        "--max_fine_points",
        type=int,
        default=200,
        help="Hard cap on dense points generated across all intervals",
    )
    ap.add_argument(
        "--use_source_lambdas",
        action="store_true",
        help="If set, include all lambdas from the source run as the coarse backbone (can be much denser).",
    )
    ap.add_argument(
        "--out_root",
        default="",
        help="Output directory for the refined rerun; default derives from source_out_root",
    )
    ap.add_argument(
        "--loop_pairs",
        default="",
        help="Optional override for --loop_pairs passed to phase3e_elambda_loop_suite.py (e.g. '0:1' or '0:1,0:2').",
    )
    ap.add_argument("--no_postprocess", action="store_true", help="Skip wall/threshold postprocess")
    args = ap.parse_args()

    src = str(args.source_out_root)
    band_lo, band_hi = _parse_pair_csv(args.lambda_band)
    top_k = int(args.top_k)

    summary_path = os.path.join(src, "phase3e_elambda_suite_summary.json")
    top20_path = os.path.join(src, "phase3e_elambda_wall_cells_top20.csv")
    if not os.path.exists(summary_path):
        raise SystemExit(f"Missing summary: {summary_path}")
    if not os.path.exists(top20_path):
        raise SystemExit(f"Missing wall-cells top20: {top20_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    df = pd.read_csv(top20_path)
    if df.empty:
        raise SystemExit(f"No rows in: {top20_path}")

    df = df[df.get("pi_rate", 0.0) > 0.0].copy()
    df = df[(df["lambda0"] >= float(band_lo)) & (df["lambda1"] <= float(band_hi))].copy()
    if df.empty:
        raise SystemExit(f"No pi-hot cells inside lambda_band=[{band_lo},{band_hi}] in: {top20_path}")

    # Prefer higher pi_rate, then worse conditioning (smaller min_step_smin_min).
    sort_cols = ["pi_rate"]
    asc = [False]
    if "min_step_smin_min" in df.columns:
        sort_cols.append("min_step_smin_min")
        asc.append(True)

    df = df.sort_values(sort_cols, ascending=asc).head(top_k)

    intervals = sorted({(float(r.lambda0), float(r.lambda1)) for r in df.itertuples(index=False)})
    intervals = intervals[:top_k]

    # Coarse backbone for context; default is minimal to keep the rerun cheap.
    if bool(args.use_source_lambdas):
        coarse = summary.get("lambdas") or [0.0, 0.25, 0.5, 0.75, 1.0]
    else:
        coarse = [0.0, 0.25, 0.5, 0.75, 1.0]

    dense = _build_dense_points(
        intervals,
        pad=float(args.pad),
        fine_step=float(args.fine_step),
        max_points=int(args.max_fine_points),
    )

    lambdas = sorted(set(_round10(list(coarse) + dense)))
    lambdas_str = ",".join(_fmt(x) for x in lambdas)

    out_root = str(args.out_root).strip() or f"{src}_autorefine2_band{_fmt(band_lo).replace('.', 'p')}_{_fmt(band_hi).replace('.', 'p')}"
    os.makedirs(out_root, exist_ok=True)

    print("[run_phase3e_elambda_autorefine2] source_out_root:", src)
    print("[run_phase3e_elambda_autorefine2] selected intervals:")
    for (l0, l1) in intervals:
        print("  - [", _fmt(l0), ",", _fmt(l1), "]")
    print("[run_phase3e_elambda_autorefine2] out_root:", out_root)
    print("[run_phase3e_elambda_autorefine2] num lambdas:", len(lambdas))
    print("[run_phase3e_elambda_autorefine2] lambdas:", lambdas_str)

    loop_pairs = str(args.loop_pairs).strip() or str(summary.get("loop_pairs", "0:1,0:2"))

    sys.argv = [
        "phase3e_elambda_loop_suite.py",
        "--out_root_a",
        str(summary["out_root_a"]),
        "--out_root_b",
        str(summary["out_root_b"]),
        "--out_root",
        out_root,
        "--blocks",
        str(int(summary.get("blocks", 4))),
        "--loop_pairs",
        loop_pairs,
        "--refine_steps",
        str(int(summary.get("refine_steps", 1))),
        "--eta",
        str(float(summary.get("eta", -1.0))),
        "--deform_alpha",
        str(float(summary.get("deform_alpha", 1.0))),
        "--lambdas",
        lambdas_str,
    ]

    runpy.run_path("phase3e_elambda_loop_suite.py", run_name="__main__")

    if not bool(args.no_postprocess):
        sys.argv = [
            "phase3e_elambda_wall_postprocess.py",
            "--out_root",
            out_root,
        ]
        runpy.run_path("phase3e_elambda_wall_postprocess.py", run_name="__main__")


if __name__ == "__main__":
    main()
