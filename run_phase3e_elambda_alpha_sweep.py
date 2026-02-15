import runpy
import sys


def run_one(alpha: float) -> None:
    out_root = f"out_phase3E_elambda_scale8_legacy_geomv9_alpha{alpha:.2f}_blocks4"
    out_root = out_root.replace(".", "p")

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
        "0:1,0:2,1:2",
        "--refine_steps",
        "1",
        "--eta",
        "-1",
        "--deform_alpha",
        str(alpha),
        "--lambda_grid",
        "5",
    ]

    print("\n[run_phase3e_elambda_alpha_sweep] alpha:", alpha, "out_root:", out_root)
    runpy.run_path("phase3e_elambda_loop_suite.py", run_name="__main__")


def main() -> None:
    for alpha in [0.25, 0.50, 0.75, 1.00]:
        run_one(alpha)


if __name__ == "__main__":
    main()
