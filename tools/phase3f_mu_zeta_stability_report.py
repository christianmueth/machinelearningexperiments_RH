"""Build a normalized mu-stability report from mu_zeta_*_summary outputs.

This takes the outputs produced by tools/phase3f_mu_zeta_stability.py:
- *_scalar_summary.csv
- *_word_summary.csv

and writes:
- *_report.csv : merged table with normalized columns
- *_report.md  : easy-to-paste per-s Markdown tables (rows=mu)

Normalization rationale
- validate_mugrid runs often have varying generator counts across mu.
- So we report both raw aggregates and per-generator / per-word normalized values.

Columns added
Scalar:
- logZ_per_gen = logZ / n_gens

Word:
- sum_real_per_word = sum_contrib_real / n_words_primitive
- sum_abs_per_word  = sum_contrib_abs / n_words_primitive

We also carry cancellation metrics like abs_neg_share and cancellation_ratio.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "nan"
    ax = abs(float(x))
    if ax != 0.0 and (ax < 1e-3 or ax >= 1e4):
        return f"{x:.3e}"
    return f"{x:.6g}"


def _write_md(df: pd.DataFrame, out_path: Path) -> None:
    lines: list[str] = []

    # Keep tables compact and aligned to the narrative.
    cols = [
        "mu",
        "n_gens",
        "n_words_primitive",
        "logZ",
        "logZ_per_gen",
        "Z",
        "sum_contrib_real",
        "sum_real_per_word",
        "sum_contrib_abs",
        "sum_abs_per_word",
        "abs_neg_share",
        "cancellation_ratio",
    ]
    cols = [c for c in cols if c in df.columns]

    for s in sorted(df["s"].unique()):
        sub = df[df["s"] == s].copy().sort_values("mu")
        lines.append(f"### s = {s:g}\n")
        header = "| " + " | ".join(cols) + " |"
        sep = "|" + "---|" * len(cols)
        lines.append(header)
        lines.append(sep)
        for _, r in sub.iterrows():
            row = [_fmt(float(r[c])) if c not in {"mu"} else _fmt(float(r[c])) for c in cols]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Make a normalized mu-stability report from scalar+word summaries")
    ap.add_argument("--scalar_csv", required=True, help="Path to *_scalar_summary.csv")
    ap.add_argument("--word_csv", required=True, help="Path to *_word_summary.csv")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--out_prefix", default="mu_zeta_report", help="Output filename prefix")

    args = ap.parse_args()

    scalar = pd.read_csv(Path(args.scalar_csv))
    word = pd.read_csv(Path(args.word_csv))

    for df, name in [(scalar, "scalar"), (word, "word")]:
        for col in ["mu", "s"]:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' in {name} CSV")

    merged = pd.merge(word, scalar, on=["mu", "s"], how="outer", suffixes=("_word", "_scalar"))

    # Choose canonical n_gens and dim if duplicated.
    if "n_gens_word" in merged.columns and "n_gens_scalar" in merged.columns:
        merged["n_gens"] = merged["n_gens_word"].combine_first(merged["n_gens_scalar"])
    elif "n_gens" not in merged.columns:
        merged["n_gens"] = merged.get("n_gens_word", merged.get("n_gens_scalar"))

    if "dim_word" in merged.columns and "dim_scalar" in merged.columns:
        merged["dim"] = merged["dim_word"].combine_first(merged["dim_scalar"])
    elif "dim" not in merged.columns:
        merged["dim"] = merged.get("dim_word", merged.get("dim_scalar"))

    # Normalizations.
    if "logZ" in merged.columns and "n_gens" in merged.columns:
        merged["logZ_per_gen"] = merged["logZ"] / merged["n_gens"].replace(0, np.nan)

    if "sum_contrib_real" in merged.columns and "n_words_primitive" in merged.columns:
        merged["sum_real_per_word"] = merged["sum_contrib_real"] / merged["n_words_primitive"].replace(0, np.nan)

    if "sum_contrib_abs" in merged.columns and "n_words_primitive" in merged.columns:
        merged["sum_abs_per_word"] = merged["sum_contrib_abs"] / merged["n_words_primitive"].replace(0, np.nan)

    merged = merged.sort_values(["s", "mu"]).reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / f"{args.out_prefix}_report.csv"
    out_md = out_dir / f"{args.out_prefix}_report.md"

    merged.to_csv(out_csv, index=False)
    _write_md(merged, out_md)

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
