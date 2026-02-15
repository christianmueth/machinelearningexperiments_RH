import argparse
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


@dataclass(frozen=True)
class MuSlice:
    name: str
    out_root: str


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default="", help="Python executable to use (default: python)")

    ap.add_argument("--out_base", required=True, help="Same --out_base used for run_phase3e_elambda_mu_slices.py")
    ap.add_argument(
        "--mu_names",
        required=True,
        help="Comma-separated list of mu slice names (must match suffixes written by run_phase3e_elambda_mu_slices.py)",
    )

    ap.add_argument(
        "--dump_index_csv",
        default="",
        help="Optional phase3f_pi_dump_index.csv for rectangle-loop v_minus1; not needed when --use_lasso 1.",
    )
    ap.add_argument("--use_lasso", type=int, default=1)
    ap.add_argument("--min_pi_rows_per_cell", type=int, default=1)
    ap.add_argument("--lambda_tol", type=float, default=1e-6)
    ap.add_argument("--flip_overlap_tol", type=float, default=0.9)

    ap.add_argument("--only_blocks", type=int, default=2)
    ap.add_argument("--only_dim", type=int, default=2)

    ap.add_argument("--dry_run", type=int, default=0)

    args = ap.parse_args()

    py = str(args.python).strip() or "python"

    base = Path(str(args.out_base)).resolve()
    mu_names = [s.strip() for s in str(args.mu_names).split(",") if s.strip()]
    if not mu_names:
        raise SystemExit("No --mu_names parsed")

    slices: List[MuSlice] = []
    for name in mu_names:
        out_root = str(base) + "_" + name
        if not os.path.exists(out_root):
            raise SystemExit(f"Missing slice out_root: {out_root}")
        slices.append(MuSlice(name=name, out_root=out_root))

    # Run generator registry + scan summary per slice
    summ_rows = []
    for sl in slices:
        print("\n=== Phase-3F on mu slice:", sl.name, "===")

        cmd_reg = [
            py,
            str(Path(__file__).with_name("phase3f_elambda_generator_registry.py")),
            "--out_root",
            sl.out_root,
            "--use_lasso",
            str(int(args.use_lasso)),
            "--min_pi_rows_per_cell",
            str(int(args.min_pi_rows_per_cell)),
            "--lambda_tol",
            str(float(args.lambda_tol)),
            "--flip_overlap_tol",
            str(float(args.flip_overlap_tol)),
        ]
        if str(args.dump_index_csv).strip():
            cmd_reg += ["--dump_index_csv", str(args.dump_index_csv).strip()]

        cmd_scan = [
            py,
            str(Path(__file__).with_name("phase3f_scan_multilambda_candidates.py")),
            "--root",
            sl.out_root,
            "--glob",
            "*",
            "--only_blocks",
            str(int(args.only_blocks)),
            "--only_dim",
            str(int(args.only_dim)),
            "--out_csv",
            os.path.join(sl.out_root, "phase3f_multilambda_candidates.csv"),
        ]

        print("REG:", " ".join(cmd_reg))
        print("SCAN:", " ".join(cmd_scan))

        if int(args.dry_run) == 0:
            subprocess.run(cmd_reg, check=True)
            subprocess.run(cmd_scan, check=True)

        # Load headline numbers
        gen_csv = os.path.join(sl.out_root, "phase3f_generators.csv")
        scan_csv = os.path.join(sl.out_root, "phase3f_multilambda_candidates.csv")
        n_gen = n_multi = max_cells = pi_rows = 0
        if os.path.exists(gen_csv):
            g = pd.read_csv(gen_csv)
            n_gen = int(len(g))
            if set(["seed", "anchor_seed", "wlo", "whi", "block", "dim"]).issubset(g.columns):
                per = g.groupby(["seed", "anchor_seed", "wlo", "whi", "block", "dim"], dropna=False).size().reset_index(name="n_gen")
                n_multi = int((per.n_gen > 1).sum())
                max_cells = int(per.n_gen.max()) if len(per) else 0
        if os.path.exists(scan_csv):
            s = pd.read_csv(scan_csv)
            if "pi_rows" in s.columns and len(s):
                pi_rows = int(s["pi_rows"].max())

        summ_rows.append(
            {
                "mu": sl.name,
                "out_root": sl.out_root,
                "n_generators": n_gen,
                "pairs_with_multi_gen": n_multi,
                "max_gens_per_pair": max_cells,
                "pi_rows_headline": pi_rows,
            }
        )

    out = pd.DataFrame(summ_rows).sort_values(["pairs_with_multi_gen", "max_gens_per_pair", "n_generators"], ascending=[False, False, False])
    out_path = str(base) + "_phase3f_mu_summary.csv"
    out.to_csv(out_path, index=False)
    print("\nWrote:", out_path)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
