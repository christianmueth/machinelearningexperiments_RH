import os
import pandas as pd


def _must_read_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def make_stage_gate_table(
    prereg_root: str = "out_phase3B_preregistered_seedblock_32_63",
    confirm_root: str = "out_phase3B_confirm_preregistered_rejecting_seeds_N16384",
    out_path: str = "out_stage_gate_figure_table.csv",
) -> pd.DataFrame:
    prereg_path = os.path.join(prereg_root, "kpi_fraction.csv")
    confirm_agg_path = os.path.join(confirm_root, "aggregate.csv")
    confirm_by_seed_path = os.path.join(confirm_root, "aggregate_by_seed.csv")

    prereg = _must_read_csv(prereg_path).copy()
    prereg.insert(0, "stage", "preregistered_seedblock_32_63")

    # Normalize expected column names to a stable schema.
    prereg = prereg.rename(
        columns={
            "n_seeds_any_reject": "n_units_any_reject",
            "frac_seeds_any_reject": "frac_units_any_reject",
            "n_seeds": "n_units",
        }
    )
    prereg["unit"] = "seed"

    confirm = _must_read_csv(confirm_agg_path).copy()
    confirm.insert(0, "stage", "confirm_preregistered_rejecting_seeds_N16384")

    # Attach window columns (wlo/whi) from raw.csv if aggregate.csv doesn't include them.
    # This keeps the figure table self-describing even when aggregates were computed without window columns.
    confirm_raw_path = os.path.join(confirm_root, "raw.csv")
    if ("wlo" not in confirm.columns) or ("whi" not in confirm.columns):
        raw = _must_read_csv(confirm_raw_path)
        # Expect a single window for the confirm, but keep it robust.
        if "backend_label" in raw.columns and "wlo" in raw.columns and "whi" in raw.columns:
            win = raw.groupby("backend_label")[["wlo", "whi"]].agg(lambda s: float(s.iloc[0]))
            confirm = confirm.merge(win.reset_index(), on="backend_label", how="left")
        else:
            if "wlo" not in confirm.columns:
                confirm["wlo"] = float("nan")
            if "whi" not in confirm.columns:
                confirm["whi"] = float("nan")

    # Helpful: embed the confirmed seed set as a compact string.
    confirm_by_seed = _must_read_csv(confirm_by_seed_path)
    seed_list = (
        confirm_by_seed[["backend_label", "seed"]]
        .drop_duplicates()
        .groupby("backend_label")["seed"]
        .apply(lambda s: ",".join(map(str, sorted(map(int, s.tolist())))))
        .to_dict()
    )
    confirm["unit"] = "seed×anchor"
    confirm["seed_set"] = confirm["backend_label"].map(seed_list).fillna("")

    # Merge into one figure table with a superset schema.
    # We keep prereg fields and confirm fields; blanks are allowed.
    out = pd.concat([prereg, confirm], ignore_index=True, sort=False)

    # Reorder columns for readability.
    preferred = [
        "stage",
        "backend_label",
        "wlo",
        "whi",
        "unit",
        "seed_set",
        "N_null",
        "rows",
        "n_units",
        "n_units_any_reject",
        "frac_units_any_reject",
        "reject_rate",
        "best_p_overall",
        "median_best_p",
        "best_p",
        "median_p",
        "worst_p",
        "median_best_dn_delta",
        "median_dn_delta",
    ]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    out = out[cols]

    out.to_csv(out_path, index=False)
    return out


if __name__ == "__main__":
    df = make_stage_gate_table()
    print(f"wrote {len(df)} rows -> out_stage_gate_figure_table.csv")
    print(df.to_string(index=False))
