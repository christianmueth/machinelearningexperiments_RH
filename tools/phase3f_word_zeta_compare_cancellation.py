"""Compare cancellation summaries across runs.

Reads one or more *_cancellation.csv files produced by tools/phase3f_word_zeta_probe.py
(with --write_cancellation_summary) and writes:

- A stacked long-form CSV (all rows with case labels)
- A compact wide-form CSV (one row per case, one column per (metric, s))
- A simple Markdown table (one block per s)

This is intentionally lightweight and has no external deps beyond pandas/numpy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS_DEFAULT = [
    "n_words_primitive",
    "sum_contrib_real",
    "sum_contrib_abs",
    "frac_neg",
    "abs_neg_share",
    "cancellation_ratio",
]


def _read_case(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.insert(0, "case", label)
    return df


def _pivot_wide(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    use = ["case", "s"] + metrics
    missing = [c for c in use if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in cancellation CSV: {missing}")

    rows = []
    for case, g in df[use].groupby("case"):
        row: dict[str, object] = {"case": case}
        for _, r in g.iterrows():
            s = float(r["s"])
            for m in metrics:
                row[f"{m}@s={s:g}"] = float(r[m])
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values("case").reset_index(drop=True)
    return out


def _format_float(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    ax = abs(float(x))
    if ax != 0 and (ax < 1e-3 or ax >= 1e4):
        return f"{x:.3e}"
    return f"{x:.6g}"


def _write_markdown(df: pd.DataFrame, metrics: list[str], out_path: Path) -> None:
    lines: list[str] = []
    for s in sorted(df["s"].unique()):
        sub = df[df["s"] == s].copy()
        sub = sub.sort_values("case")

        lines.append(f"### s = {s:g}\n")
        header = "| case | " + " | ".join(metrics) + " |"
        sep = "|---" + "|---" * (len(metrics) + 0) + "|"
        lines.append(header)
        lines.append(sep)
        for _, r in sub.iterrows():
            cells = [str(r["case"])] + [_format_float(float(r[m])) for m in metrics]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare word-zeta cancellation summaries across cases")
    ap.add_argument(
        "--case",
        action="append",
        default=[],
        help=(
            "Case spec in the form label=path/to/_cancellation.csv . "
            "May be provided multiple times."
        ),
    )
    ap.add_argument(
        "--case_file",
        default=None,
        help=(
            "Optional text file with one case per line in the form label=path/to/_cancellation.csv. "
            "Blank lines and lines starting with # are ignored."
        ),
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Directory to write outputs into",
    )
    ap.add_argument(
        "--out_prefix",
        default="word_zeta_cancellation_compare",
        help="Output filename prefix",
    )
    ap.add_argument(
        "--metrics",
        default=",".join(METRICS_DEFAULT),
        help=f"Comma-separated metrics to include (default: {','.join(METRICS_DEFAULT)})",
    )

    args = ap.parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [m.strip() for m in str(args.metrics).split(",") if m.strip()]
    if not metrics:
        raise ValueError("No metrics requested")

    raw_specs: list[str] = list(args.case)
    if args.case_file is not None:
        case_path = Path(str(args.case_file)).resolve()
        if not case_path.exists():
            raise FileNotFoundError(case_path)
        for line in case_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            raw_specs.append(line)

    cases = []
    for spec in raw_specs:
        if "=" not in spec:
            raise ValueError(f"Bad --case spec (expected label=path): {spec}")
        label, path = spec.split("=", 1)
        p = Path(path).resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        cases.append((label.strip(), p))

    if not cases:
        raise ValueError("Provide at least one --case label=path")

    long_df = pd.concat([_read_case(p, label) for label, p in cases], ignore_index=True)

    long_out = out_dir / f"{args.out_prefix}_long.csv"
    long_df.to_csv(long_out, index=False)

    wide_df = _pivot_wide(long_df, metrics)
    wide_out = out_dir / f"{args.out_prefix}_wide.csv"
    wide_df.to_csv(wide_out, index=False)

    md_out = out_dir / f"{args.out_prefix}.md"
    _write_markdown(long_df, metrics, md_out)

    print(f"Wrote: {long_out}")
    print(f"Wrote: {wide_out}")
    print(f"Wrote: {md_out}")


if __name__ == "__main__":
    main()
