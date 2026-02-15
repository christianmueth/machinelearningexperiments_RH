import pandas as pd

SCAN_ROOT = "out_phase3B_preregistered_uniqueness_64_127"
CONF_ROOT = "out_phase3B_confirm_preregistered_uniqueness_rejecting_seeds_N16384"


def summ(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    g = (
        df.groupby(["backend_label", "wlo", "whi"], dropna=False)
        .agg(
            rank_med=("dn_identity_rank", "median"),
            coll_med=("dn_collision_rate", "median"),
            gap_med=("dn_gap_min", "median"),
            best_p=("p_dnmap", "min"),
            rows=("p_dnmap", "size"),
        )
        .reset_index()
    )
    g["tag"] = tag
    return g


def main():
    u_scan = pd.read_csv(f"{SCAN_ROOT}/uniqueness.csv")
    u_conf = pd.read_csv(f"{CONF_ROOT}/raw.csv")

    out = pd.concat([summ(u_scan, "scan"), summ(u_conf, "confirm")], ignore_index=True)
    out = out.sort_values(["backend_label", "wlo", "whi", "tag"]) 

    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
