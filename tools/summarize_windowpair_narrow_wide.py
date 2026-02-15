import argparse
from pathlib import Path

import pandas as pd


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"None of the expected columns exist: {candidates}. Available: {list(df.columns)}")


def _resolve_windowpair_cols(df: pd.DataFrame) -> dict[str, str]:
    # Newer naming (what we observed): anchor_reject_fraction_primary
    # Older naming (what some ad-hoc scripts expected): primary_anchor_reject_fraction
    cols: dict[str, str] = {}

    cols["backend"] = _pick_col(df, ["backend_label", "backend", "geom_backend", "tp_backend"])
    cols["seed"] = _pick_col(df, ["seed", "rng_seed"])

    cols["primary_frac"] = _pick_col(
        df,
        [
            "anchor_reject_fraction_primary",
            "primary_anchor_reject_fraction",
        ],
    )
    cols["secondary_frac"] = _pick_col(
        df,
        [
            "anchor_reject_fraction_secondary",
            "secondary_anchor_reject_fraction",
        ],
    )

    cols["primary_minp"] = _pick_col(
        df,
        [
            "anchor_min_p_primary",
            "primary_anchor_min_p",
        ],
    )
    cols["secondary_minp"] = _pick_col(
        df,
        [
            "anchor_min_p_secondary",
            "secondary_anchor_min_p",
        ],
    )

    cols["primary_count"] = _pick_col(
        df,
        [
            "anchor_reject_count_primary",
            "primary_anchor_reject_count",
        ],
    )
    cols["secondary_count"] = _pick_col(
        df,
        [
            "anchor_reject_count_secondary",
            "secondary_anchor_reject_count",
        ],
    )

    return cols


def _format_table(df: pd.DataFrame, max_rows: int) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    return df.to_string(index=False)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Summarize windowpair anchor KPIs (primary vs secondary) from kpi_by_seed_windowpair.csv. "
            "Prints: top-by-primary, narrow-only (primary-secondary), wide-only (secondary-primary), and best-per-backend."
        )
    )
    ap.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to kpi_by_seed_windowpair.csv",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of rows to print per table (default: 20)",
    )
    ap.add_argument(
        "--best-per-backend",
        type=int,
        default=5,
        help="Number of top rows to print per backend for primary ranking (default: 5)",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional path to write an augmented CSV including delta columns.",
    )
    ap.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory to write compact CSV reports: "
            "windowpair_top_primary.csv, windowpair_top_narrow_only.csv, "
            "windowpair_top_wide_only.csv, windowpair_best_per_backend_primary.csv"
        ),
    )
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    cols = _resolve_windowpair_cols(df)

    backend = cols["backend"]
    seed = cols["seed"]
    prim_frac = cols["primary_frac"]
    sec_frac = cols["secondary_frac"]
    prim_minp = cols["primary_minp"]
    sec_minp = cols["secondary_minp"]
    prim_count = cols["primary_count"]
    sec_count = cols["secondary_count"]

    df = df.copy()
    df["delta_primary_minus_secondary"] = df[prim_frac] - df[sec_frac]
    df["delta_secondary_minus_primary"] = df[sec_frac] - df[prim_frac]

    print(f"rows={len(df)} input={args.input}")
    print(f"Using columns: primary={prim_frac}, secondary={sec_frac}")

    top_primary = df.sort_values([prim_frac, prim_minp], ascending=[False, True])
    print("\nTop by primary robustness:")
    print(
        _format_table(
            top_primary[[backend, seed, prim_frac, prim_count, prim_minp, sec_frac, sec_count, sec_minp]],
            args.top,
        )
    )

    narrow_only = df.sort_values(["delta_primary_minus_secondary", prim_minp], ascending=[False, True])
    print("\nNarrow-only (primary - secondary) top deltas:")
    print(
        _format_table(
            narrow_only[[backend, seed, "delta_primary_minus_secondary", prim_frac, sec_frac, prim_minp, sec_minp]],
            args.top,
        )
    )

    wide_only = df.sort_values(["delta_secondary_minus_primary", sec_minp], ascending=[False, True])
    print("\nWide-only (secondary - primary) top deltas:")
    print(
        _format_table(
            wide_only[[backend, seed, "delta_secondary_minus_primary", prim_frac, sec_frac, prim_minp, sec_minp]],
            args.top,
        )
    )

    print("\nBest per backend (ranked by primary robustness):")
    for backend_label, group in df.groupby(backend, sort=True):
        g = group.sort_values([prim_frac, prim_minp], ascending=[False, True]).head(args.best_per_backend)
        print(f"\n{backend_label}")
        print(_format_table(g[[seed, prim_frac, prim_count, prim_minp, sec_frac, sec_count, sec_minp]], None))

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"\nWrote augmented CSV: {args.out_csv}")

    if args.report_dir is not None:
        args.report_dir.mkdir(parents=True, exist_ok=True)

        top_primary_path = args.report_dir / "windowpair_top_primary.csv"
        narrow_only_path = args.report_dir / "windowpair_top_narrow_only.csv"
        wide_only_path = args.report_dir / "windowpair_top_wide_only.csv"
        best_per_backend_path = args.report_dir / "windowpair_best_per_backend_primary.csv"

        top_primary.head(args.top).loc[:, [
            backend,
            seed,
            prim_frac,
            prim_count,
            prim_minp,
            sec_frac,
            sec_count,
            sec_minp,
            "delta_primary_minus_secondary",
            "delta_secondary_minus_primary",
        ]].to_csv(top_primary_path, index=False)

        narrow_only.head(args.top).loc[:, [
            backend,
            seed,
            "delta_primary_minus_secondary",
            prim_frac,
            sec_frac,
            prim_minp,
            sec_minp,
        ]].to_csv(narrow_only_path, index=False)

        wide_only.head(args.top).loc[:, [
            backend,
            seed,
            "delta_secondary_minus_primary",
            prim_frac,
            sec_frac,
            prim_minp,
            sec_minp,
        ]].to_csv(wide_only_path, index=False)

        best_rows = []
        for backend_label, group in df.groupby(backend, sort=True):
            g = group.sort_values([prim_frac, prim_minp], ascending=[False, True]).head(args.best_per_backend)
            best_rows.append(g.assign(**{backend: backend_label}))
        best_df = pd.concat(best_rows, ignore_index=True)
        best_df.loc[:, [
            backend,
            seed,
            prim_frac,
            prim_count,
            prim_minp,
            sec_frac,
            sec_count,
            sec_minp,
            "delta_primary_minus_secondary",
            "delta_secondary_minus_primary",
        ]].to_csv(best_per_backend_path, index=False)

        print("\nWrote reports:")
        print(f"- {top_primary_path}")
        print(f"- {narrow_only_path}")
        print(f"- {wide_only_path}")
        print(f"- {best_per_backend_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
