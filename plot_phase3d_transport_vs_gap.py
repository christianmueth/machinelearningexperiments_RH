import argparse
import os

import numpy as np
import pandas as pd


def _finite(df: pd.DataFrame, col: str) -> pd.Series:
    v = np.asarray(df[col].values, dtype=np.float64)
    return np.isfinite(v)


def binned_median(df: pd.DataFrame, x: str, y: str, *, bins: int = 30):
    m = _finite(df, x) & _finite(df, y)
    df = df.loc[m, [x, y]].copy()
    if len(df) == 0:
        return pd.DataFrame(columns=["bin_left", "bin_right", "count", "x_median", "y_median", "y_q25", "y_q75"])

    xvals = df[x].values
    xmin, xmax = float(np.min(xvals)), float(np.max(xvals))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        return pd.DataFrame(columns=["bin_left", "bin_right", "count", "x_median", "y_median", "y_q25", "y_q75"])

    edges = np.linspace(xmin, xmax, int(bins) + 1)
    out = []
    for i in range(int(bins)):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = (xvals >= lo) & (xvals < hi) if i < bins - 1 else (xvals >= lo) & (xvals <= hi)
        sub = df.loc[mask, :]
        if len(sub) == 0:
            out.append({"bin_left": lo, "bin_right": hi, "count": 0, "x_median": np.nan, "y_median": np.nan, "y_q25": np.nan, "y_q75": np.nan})
            continue
        yvals = sub[y].values
        out.append(
            {
                "bin_left": lo,
                "bin_right": hi,
                "count": int(len(sub)),
                "x_median": float(np.median(sub[x].values)),
                "y_median": float(np.median(yvals)),
                "y_q25": float(np.quantile(yvals, 0.25)),
                "y_q75": float(np.quantile(yvals, 0.75)),
            }
        )
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Output root containing kpi_energy_transport_pairs.csv")
    ap.add_argument("--bins", type=int, default=30)
    ap.add_argument("--max_points", type=int, default=200_000, help="Subsample scatter to keep PNGs light")
    args = ap.parse_args()

    root = str(args.root)
    pairs_path = os.path.join(root, "kpi_energy_transport_pairs.csv")
    if not os.path.exists(pairs_path):
        raise SystemExit(f"Missing {pairs_path}")

    df = pd.read_csv(pairs_path)
    needed = ["gap_center_rad", "gap_min_rad", "angle_max_rad", "angle_per_dE", "overlap"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in pairs CSV: {missing}")

    # Subsample for scatter, but compute binned curves on full data
    df_scatter = df.copy()
    if len(df_scatter) > int(args.max_points):
        df_scatter = df_scatter.sample(n=int(args.max_points), random_state=0)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = [
        ("gap_center_rad", "angle_max_rad", "transport_angle_vs_gap_center.png", "Gap (centroid) [rad]", "Angle max [rad]"),
        ("gap_center_rad", "overlap", "transport_overlap_vs_gap_center.png", "Gap (centroid) [rad]", "Overlap (Tr(PQ)/d)"),
        ("gap_center_rad", "angle_per_dE", "transport_angle_per_dE_vs_gap_center.png", "Gap (centroid) [rad]", "Angle max / |dE| [rad per E]"),
        ("gap_min_rad", "angle_max_rad", "transport_angle_vs_gap_min.png", "Gap (min boundary) [rad]", "Angle max [rad]"),
    ]

    # Write binned summaries
    binned_rows = []
    for x, y, _fn, *_labels in plots:
        b = binned_median(df, x, y, bins=int(args.bins))
        b["x"] = x
        b["y"] = y
        binned_rows.append(b)
    pd.concat(binned_rows, ignore_index=True).to_csv(os.path.join(root, "transport_gap_binned.csv"), index=False)

    for x, y, fn, xlabel, ylabel in plots:
        m = _finite(df_scatter, x) & _finite(df_scatter, y)
        d = df_scatter.loc[m, [x, y]].copy()

        fig = plt.figure(figsize=(7.5, 5.0), dpi=160)
        plt.scatter(d[x].values, d[y].values, s=4, alpha=0.15, linewidths=0)

        # Overlay binned medians
        b = binned_median(df, x, y, bins=int(args.bins))
        b = b.loc[b["count"] > 0].copy()
        if len(b):
            plt.plot(b["x_median"].values, b["y_median"].values, color="black", linewidth=2.0, label="binned median")
            plt.fill_between(b["x_median"].values, b["y_q25"].values, b["y_q75"].values, color="black", alpha=0.10, label="IQR")
            plt.legend(loc="best", frameon=False)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(os.path.basename(root))
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        out_path = os.path.join(root, fn)
        plt.savefig(out_path)
        plt.close(fig)

    # If tracked transport exists, add simple distribution comparisons (no gap proxy there).
    tracked_pairs_path = os.path.join(root, "kpi_energy_transport_pairs_tracked.csv")
    if os.path.exists(tracked_pairs_path):
        dft = pd.read_csv(tracked_pairs_path)
        # Compare angle distributions (clustering vs tracked)
        import matplotlib.pyplot as plt

        def _vals(frame, col):
            if col not in frame.columns:
                return np.asarray([], dtype=np.float64)
            v = np.asarray(frame[col].values, dtype=np.float64)
            return v[np.isfinite(v)]

        ang_c = _vals(df, "angle_max_rad")
        ang_t = _vals(dft, "angle_max_rad")
        ov_c = _vals(df, "overlap")
        ov_t = _vals(dft, "overlap")

        if ang_t.size:
            fig = plt.figure(figsize=(7.5, 5.0), dpi=160)
            bins = np.linspace(0.0, float(np.nanmax(np.concatenate([ang_c, ang_t]))), 40)
            plt.hist(ang_c, bins=bins, alpha=0.45, density=True, label="clustering")
            plt.hist(ang_t, bins=bins, alpha=0.45, density=True, label="tracked")
            plt.xlabel("Adjacent transport angle max [rad]")
            plt.ylabel("Density")
            plt.title(os.path.basename(root) + " (adjacent transport)")
            plt.legend(loc="best", frameon=False)
            plt.grid(True, alpha=0.2)
            plt.tight_layout()
            plt.savefig(os.path.join(root, "transport_angle_hist_compare.png"))
            plt.close(fig)

        if ov_t.size:
            fig = plt.figure(figsize=(7.5, 5.0), dpi=160)
            bins = np.linspace(0.0, 1.0, 40)
            plt.hist(ov_c, bins=bins, alpha=0.45, density=True, label="clustering")
            plt.hist(ov_t, bins=bins, alpha=0.45, density=True, label="tracked")
            plt.xlabel("Adjacent transport overlap (Tr(PQ)/d)")
            plt.ylabel("Density")
            plt.title(os.path.basename(root) + " (adjacent transport)")
            plt.legend(loc="best", frameon=False)
            plt.grid(True, alpha=0.2)
            plt.tight_layout()
            plt.savefig(os.path.join(root, "transport_overlap_hist_compare.png"))
            plt.close(fig)

        # Write a tiny one-row summary
        summ = {
            "n_pairs_clustering": int(ang_c.size),
            "n_pairs_tracked": int(ang_t.size),
            "angle_median_clustering": float(np.median(ang_c)) if ang_c.size else np.nan,
            "angle_median_tracked": float(np.median(ang_t)) if ang_t.size else np.nan,
            "overlap_median_clustering": float(np.median(ov_c)) if ov_c.size else np.nan,
            "overlap_median_tracked": float(np.median(ov_t)) if ov_t.size else np.nan,
        }
        pd.DataFrame([summ]).to_csv(os.path.join(root, "transport_tracked_compare_summary.csv"), index=False)

    print(f"[plot_phase3d_transport_vs_gap] wrote PNGs + transport_gap_binned.csv under {root}")


if __name__ == "__main__":
    main()
