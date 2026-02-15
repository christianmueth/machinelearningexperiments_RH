import argparse
import json
import os
import runpy
import sys
from typing import List


def _load_lambdas(source_out_root: str) -> List[float]:
    path = os.path.join(source_out_root, "phase3e_elambda_suite_summary.json")
    with open(path, "r", encoding="utf-8") as f:
        summ = json.load(f)
    lambdas = summ.get("lambdas")
    if not lambdas:
        raise SystemExit(f"No lambdas found in {path}")
    return [float(x) for x in lambdas]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_out_root", required=True, help="Existing run with suite_summary.json containing lambdas")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--loop_pairs", default="0:2", help="e.g. '0:2' or '1:2'")
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--refine_steps", type=int, default=1)
    ap.add_argument("--eta", type=float, default=-1.0)
    ap.add_argument("--deform_alpha", type=float, default=1.0)
    ap.add_argument("--out_root_a", default="out_phase3D_channel_diag_scale8_fullanchors_legacy")
    ap.add_argument("--out_root_b", default="out_phase3D_channel_diag_scale8_fullanchors_geomv9")
    args = ap.parse_args()

    lambdas = _load_lambdas(str(args.source_out_root))
    lambdas_str = ",".join(f"{x:.10g}" for x in lambdas)

    print("[run_phase3e_elambda_fixed_lambdas_from_run] source:", str(args.source_out_root))
    print("[run_phase3e_elambda_fixed_lambdas_from_run] out_root:", str(args.out_root))
    print("[run_phase3e_elambda_fixed_lambdas_from_run] loop_pairs:", str(args.loop_pairs))
    print("[run_phase3e_elambda_fixed_lambdas_from_run] num lambdas:", len(lambdas))

    sys.argv = [
        "phase3e_elambda_loop_suite.py",
        "--out_root_a",
        str(args.out_root_a),
        "--out_root_b",
        str(args.out_root_b),
        "--out_root",
        str(args.out_root),
        "--blocks",
        str(int(args.blocks)),
        "--loop_pairs",
        str(args.loop_pairs),
        "--refine_steps",
        str(int(args.refine_steps)),
        "--eta",
        str(float(args.eta)),
        "--deform_alpha",
        str(float(args.deform_alpha)),
        "--lambdas",
        lambdas_str,
    ]

    runpy.run_path("phase3e_elambda_loop_suite.py", run_name="__main__")

    sys.argv = [
        "phase3e_elambda_wall_postprocess.py",
        "--out_root",
        str(args.out_root),
    ]
    runpy.run_path("phase3e_elambda_wall_postprocess.py", run_name="__main__")


if __name__ == "__main__":
    main()
