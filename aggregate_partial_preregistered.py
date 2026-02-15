import os
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="out_phase3B_preregistered_seedblock_32_63")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    out_root = args.out_root

    rows_paths = []
    for root, _dirs, files in os.walk(out_root):
        if "rows.csv" in files:
            rows_paths.append(os.path.join(root, "rows.csv"))

    if not rows_paths:
        print("no rows.csv found yet")
        return

    parts = []
    for p in sorted(rows_paths):
        try:
            parts.append(pd.read_csv(p))
        except Exception as e:
            print(f"skip {p}: {e}")

    big = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    print(f"loaded rows.csv files: {len(rows_paths)}")
    print(f"rows: {len(big)}")

    if len(big) == 0:
        return

    # derived
    if "dn_obs" in big.columns and "dn_null_mean" in big.columns and "dn_delta" not in big.columns:
        big["dn_delta"] = big["dn_obs"] - big["dn_null_mean"]

    # KPI fraction so far
    if "reject" in big.columns:
        by_seed = (
            big.groupby(["backend_label", "wlo", "whi", "seed"], dropna=False)
            .agg(
                any_reject=("reject", "max"),
                best_p=("p_family", "min"),
                rows=("reject", "size"),
            )
            .reset_index()
        )
        by_seed["any_reject"] = by_seed["any_reject"].astype(int)

        frac = (
            by_seed.groupby(["backend_label", "wlo", "whi"], dropna=False)
            .agg(
                n_seeds=("seed", "nunique"),
                n_seeds_any_reject=("any_reject", "sum"),
                frac_seeds_any_reject=("any_reject", "mean"),
                best_p_overall=("best_p", "min"),
            )
            .reset_index()
            .sort_values(["backend_label", "wlo", "whi"])
        )

        print("\nKPI fraction so far:")
        print(frac.to_string(index=False))

        if args.write:
            os.makedirs(out_root, exist_ok=True)
            big.to_csv(os.path.join(out_root, "partial_raw.csv"), index=False)
            by_seed.to_csv(os.path.join(out_root, "partial_kpi_by_seed.csv"), index=False)
            frac.to_csv(os.path.join(out_root, "partial_kpi_fraction.csv"), index=False)
            print("wrote partial_*.csv into out_root")


if __name__ == "__main__":
    main()
