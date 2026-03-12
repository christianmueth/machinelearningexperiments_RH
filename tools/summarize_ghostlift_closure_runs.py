import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Run:
    label: str
    path: Path


def _parse_run(spec: str) -> Run:
    if "=" not in spec:
        raise SystemExit(f"Bad --run '{spec}': expected label=path")
    label, path_s = spec.split("=", 1)
    path = Path(path_s)
    if not path.exists():
        raise SystemExit(f"Missing CSV: {path}")
    return Run(label=label, path=path)


def _finite(x: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def _median(x: pd.Series) -> float:
    arr = _finite(x)
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def _min(x: pd.Series) -> float:
    arr = _finite(x)
    if arr.size == 0:
        return float("nan")
    return float(np.min(arr))


def _max(x: pd.Series) -> float:
    arr = _finite(x)
    if arr.size == 0:
        return float("nan")
    return float(np.max(arr))


def summarize_beta_plus_c(df: pd.DataFrame) -> dict[str, float]:
    df = df[df["model"] == "beta_plus_c"].copy()
    if df.empty:
        raise SystemExit("No beta_plus_c rows")

    out: dict[str, float] = {"n_u": float(df.shape[0])}

    for prefix in ["train", "eval", "holdout", "newprime"]:
        for metric in ["E_sf", "E_pp", "E_mix", "L"]:
            col = f"{prefix}_{metric}"
            out[f"{prefix}_{metric}_med"] = _median(df[col]) if col in df.columns else float("nan")

    for col in ["beta", "c"]:
        if col in df.columns:
            out[f"{col}_med"] = _median(df[col])
            out[f"{col}_min"] = _min(df[col])
            out[f"{col}_max"] = _max(df[col])
        else:
            out[f"{col}_med"] = float("nan")
            out[f"{col}_min"] = float("nan")
            out[f"{col}_max"] = float("nan")

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize one or more closure-fit CSVs (beta_plus_c only).")
    ap.add_argument("--run", action="append", required=True, help="Repeatable: label=path/to/csv")
    ap.add_argument("--out_csv", default="out/ghostlift_closure_runs_summary_beta_plus_c.csv")

    args = ap.parse_args()

    runs = [_parse_run(s) for s in args.run]

    rows: list[dict] = []
    for r in runs:
        df = pd.read_csv(r.path)
        row: dict[str, object] = {"label": r.label, "in_csv": str(r.path)}
        row.update(summarize_beta_plus_c(df))
        rows.append(row)

    out_path = Path(str(args.out_csv))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
