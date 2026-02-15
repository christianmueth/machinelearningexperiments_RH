"""Compare normalized mu-stability reports across multiple cases.

Each case provides scalar_csv and word_csv (from phase3f_mu_zeta_stability.py).
We generate normalized report tables for each case and then concatenate them with
an added 'case' column.

Outputs:
- <out_prefix>_long.csv : concatenated report rows
- <out_prefix>.md       : per-s Markdown tables with (case, mu) rows

This is designed for quick copy/paste into a results packet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _make_report(scalar_csv: Path, word_csv: Path) -> pd.DataFrame:
    scalar = pd.read_csv(scalar_csv)
    word = pd.read_csv(word_csv)

    merged = pd.merge(word, scalar, on=["mu", "s"], how="outer", suffixes=("_word", "_scalar"))

    if "n_gens_word" in merged.columns and "n_gens_scalar" in merged.columns:
        merged["n_gens"] = merged["n_gens_word"].combine_first(merged["n_gens_scalar"])
    elif "n_gens" not in merged.columns:
        merged["n_gens"] = merged.get("n_gens_word", merged.get("n_gens_scalar"))

    if "logZ" in merged.columns and "n_gens" in merged.columns:
        merged["logZ_per_gen"] = merged["logZ"] / merged["n_gens"].replace(0, np.nan)

    if "sum_contrib_real" in merged.columns and "n_words_primitive" in merged.columns:
        merged["sum_real_per_word"] = merged["sum_contrib_real"] / merged["n_words_primitive"].replace(0, np.nan)

    if "sum_contrib_abs" in merged.columns and "n_words_primitive" in merged.columns:
        merged["sum_abs_per_word"] = merged["sum_contrib_abs"] / merged["n_words_primitive"].replace(0, np.nan)

    return merged


def _fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "nan"
    ax = abs(float(x))
    if ax != 0.0 and (ax < 1e-3 or ax >= 1e4):
        return f"{x:.3e}"
    return f"{x:.6g}"


def _write_md(df: pd.DataFrame, out_path: Path) -> None:
    cols = [
        "case",
        "mu",
        "n_gens",
        "n_words_primitive",
        "logZ_per_gen",
        "sum_real_per_word",
        "abs_neg_share",
        "cancellation_ratio",
    ]
    cols = [c for c in cols if c in df.columns]

    lines: list[str] = []
    for s in sorted(df["s"].unique()):
        sub = df[df["s"] == s].copy().sort_values(["case", "mu"])
        lines.append(f"### s = {s:g}\n")
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for _, r in sub.iterrows():
            row = []
            for c in cols:
                if c == "case":
                    row.append(str(r[c]))
                else:
                    row.append(_fmt(float(r[c])))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare mu-stability across cases")
    ap.add_argument(
        "--case_file",
        required=True,
        help=(
            "Text file with one case per line: label=scalar_csv|word_csv . "
            "Paths may be relative to out_dir."
        ),
    )
    ap.add_argument("--out_dir", required=True, help="Directory to write outputs into (and base for relative paths)")
    ap.add_argument("--out_prefix", default="mu_zeta_compare", help="Output filename prefix")

    args = ap.parse_args()
    out_dir = Path(args.out_dir).resolve()

    specs: list[tuple[str, Path, Path]] = []
    case_path = Path(args.case_file).resolve()
    for line in case_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line or "|" not in line:
            raise ValueError(f"Bad case line: {line}")
        label, rest = line.split("=", 1)
        scalar_s, word_s = rest.split("|", 1)
        scalar_csv = Path(scalar_s)
        word_csv = Path(word_s)
        if not scalar_csv.is_absolute():
            scalar_csv = (out_dir / scalar_csv).resolve()
        if not word_csv.is_absolute():
            word_csv = (out_dir / word_csv).resolve()
        specs.append((label.strip(), scalar_csv, word_csv))

    frames = []
    for label, scalar_csv, word_csv in specs:
        rep = _make_report(scalar_csv, word_csv)
        rep.insert(0, "case", label)
        frames.append(rep)

    long_df = pd.concat(frames, ignore_index=True).sort_values(["s", "case", "mu"]).reset_index(drop=True)

    out_csv = out_dir / f"{args.out_prefix}_long.csv"
    out_md = out_dir / f"{args.out_prefix}.md"

    long_df.to_csv(out_csv, index=False)
    _write_md(long_df, out_md)

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
