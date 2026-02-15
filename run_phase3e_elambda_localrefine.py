import runpy
import sys


def main() -> None:
    # Dense refinement on the hottest interval [0.25, 0.50]
    dense = [0.25 + k * (0.25 / 20.0) for k in range(21)]
    lambdas = sorted(set([0.0, 0.25] + dense + [0.5, 0.75, 1.0]))
    lambdas_str = ",".join(f"{x:.10g}" for x in lambdas)

    out_root = "out_phase3E_elambda_scale8_legacy_geomv9_localrefine_0p25_0p50"

    sys.argv = [
        "phase3e_elambda_loop_suite.py",
        "--out_root_a",
        "out_phase3D_channel_diag_scale8_fullanchors_legacy",
        "--out_root_b",
        "out_phase3D_channel_diag_scale8_fullanchors_geomv9",
        "--out_root",
        out_root,
        "--blocks",
        "4",
        "--loop_pairs",
        "0:1,0:2",
        "--refine_steps",
        "1",
        "--eta",
        "-1",
        "--deform_alpha",
        "1.0",
        "--lambdas",
        lambdas_str,
    ]

    print("[run_phase3e_elambda_localrefine] out_root:", out_root)
    print("[run_phase3e_elambda_localrefine] lambdas:", lambdas_str)
    runpy.run_path("phase3e_elambda_loop_suite.py", run_name="__main__")


if __name__ == "__main__":
    main()
