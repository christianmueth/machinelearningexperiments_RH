import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RunSpec:
    mu_mix: float
    path: Path


def _parse_runs(parts: list[str]) -> list[RunSpec]:
    runs: list[RunSpec] = []
    for raw in parts:
        if "=" not in raw:
            raise SystemExit(f"Bad --run '{raw}': expected mu=path")
        mu_s, path_s = raw.split("=", 1)
        mu = float(mu_s)
        path = Path(path_s)
        if not path.exists():
            raise SystemExit(f"Missing input CSV: {path}")
        runs.append(RunSpec(mu_mix=mu, path=path))
    runs.sort(key=lambda r: r.mu_mix)
    return runs


def _finite_median(arr: pd.Series) -> float:
    x = pd.to_numeric(arr, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.median(x))


def _finite_min(arr: pd.Series) -> float:
    x = pd.to_numeric(arr, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.min(x))


def _finite_max(arr: pd.Series) -> float:
    x = pd.to_numeric(arr, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.max(x))


def summarize_beta_plus_c(df: pd.DataFrame) -> dict:
    df = df[df["model"] == "beta_plus_c"].copy()
    if df.empty:
        raise ValueError("No beta_plus_c rows found")

    out: dict[str, float] = {}
    out["n_u"] = float(df.shape[0])

    for prefix in ["train", "eval", "holdout"]:
        for metric in ["E_sf", "E_pp", "E_mix", "L"]:
            col = f"{prefix}_{metric}"
            if col not in df.columns:
                out[f"{prefix}_{metric}_med"] = float("nan")
            else:
                out[f"{prefix}_{metric}_med"] = _finite_median(df[col])

    for col in ["beta", "c"]:
        if col in df.columns:
            out[f"{col}_med"] = _finite_median(df[col])
            out[f"{col}_min"] = _finite_min(df[col])
            out[f"{col}_max"] = _finite_max(df[col])
        else:
            out[f"{col}_med"] = float("nan")
            out[f"{col}_min"] = float("nan")
            out[f"{col}_max"] = float("nan")

    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Summarize a mu_mix sweep for ghostlift closure fits.\n"
            "Reads multiple CSVs produced by tools/fit_ghostlift_closure_bias.py and\n"
            "outputs a single summary CSV focusing on the beta_plus_c model."))

    ap.add_argument(
        "--run",
        action="append",
        required=True,
        help="Repeatable. Format: mu=path/to/csv",
    )
    ap.add_argument(
        "--out_csv",
        default="out/ghostlift_closure_mumix_sweep_summary_beta_plus_c.csv",
    )

    args = ap.parse_args()
    runs = _parse_runs(list(args.run))

    rows: list[dict] = []
    for r in runs:
        df = pd.read_csv(r.path)
        row = {"mu_mix": float(r.mu_mix), "in_csv": str(r.path)}
        row.update(summarize_beta_plus_c(df))
        rows.append(row)

    out_path = Path(str(args.out_csv))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("mu_mix").to_csv(out_path, index=False)
    print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
