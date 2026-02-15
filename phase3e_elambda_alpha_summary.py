import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd


def _read_summary(out_root: str) -> Dict:
    path = os.path.join(out_root, "phase3e_elambda_suite_summary.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_threshold_table(out_root: str) -> pd.DataFrame:
    path = os.path.join(out_root, "phase3e_elambda_pi_rate_thresholds_step_smin_min.csv")
    return pd.read_csv(path)


def _row_for_threshold(df: pd.DataFrame, t: float) -> Optional[pd.Series]:
    hit = df[df["threshold"].astype(float) == float(t)]
    if len(hit) != 1:
        return None
    return hit.iloc[0]


def summarize_one(out_root: str, thresholds: List[float]) -> Dict:
    summ = _read_summary(out_root)
    df = _read_threshold_table(out_root)

    out: Dict = {
        "out_root": str(out_root),
        "deform_alpha": float(summ.get("deform_alpha", float("nan"))),
        "blocks": int(summ.get("blocks", -1)),
        "loop_pairs": str(summ.get("loop_pairs", "")),
        "n_rows": int(summ.get("n_rows", -1)),
    }

    # Overall pi fraction is the threshold=1.0 row's pi_rate_given_below.
    r_all = _row_for_threshold(df, 1.0)
    if r_all is not None:
        out["pi_frac_overall"] = float(r_all["pi_rate_given_below"])
    else:
        out["pi_frac_overall"] = float(summ.get("det_phase_frac_near_pi", float("nan")))

    for t in thresholds:
        r = _row_for_threshold(df, t)
        key = f"t{str(t).replace('.', 'p')}"
        if r is None:
            out[f"coverage_{key}"] = float("nan")
            out[f"pi_rate_below_{key}"] = float("nan")
            out[f"pi_rate_above_{key}"] = float("nan")
        else:
            out[f"coverage_{key}"] = float(r["coverage_frac"]) if pd.notna(r["coverage_frac"]) else float("nan")
            out[f"pi_rate_below_{key}"] = float(r["pi_rate_given_below"]) if pd.notna(r["pi_rate_given_below"]) else float("nan")
            out[f"pi_rate_above_{key}"] = float(r["pi_rate_given_above"]) if pd.notna(r["pi_rate_given_above"]) else float("nan")

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out_roots",
        default="out_phase3E_elambda_scale8_legacy_geomv9_alpha0p25_blocks4,out_phase3E_elambda_scale8_legacy_geomv9_alpha0p50_blocks4,out_phase3E_elambda_scale8_legacy_geomv9_alpha0p75_blocks4,out_phase3E_elambda_scale8_legacy_geomv9_alpha1p00_blocks4",
        help="Comma-separated list of out_root folders to summarize",
    )
    ap.add_argument("--thresholds", default="0.01,0.05", help="Comma list of thresholds to report")
    ap.add_argument(
        "--out_csv",
        default="phase3e_elambda_alpha_sweep_summary.csv",
        help="Output CSV filename (written in cwd)",
    )
    args = ap.parse_args()

    out_roots = [p.strip() for p in str(args.out_roots).split(",") if p.strip()]
    thresholds = [float(x.strip()) for x in str(args.thresholds).split(",") if x.strip()]

    rows: List[Dict] = []
    for r in out_roots:
        rows.append(summarize_one(r, thresholds))

    df = pd.DataFrame(rows)
    df = df.sort_values(["deform_alpha", "blocks"]).reset_index(drop=True)
    df.to_csv(str(args.out_csv), index=False)
    print("[phase3e_elambda_alpha_summary] wrote:", str(args.out_csv))


if __name__ == "__main__":
    main()
