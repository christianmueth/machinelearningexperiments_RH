import runpy
import sys


def run_one(blocks: int) -> None:
    out_root = f"out_phase3E_elambda_scale8_legacy_geomv9_alpha1p00_blocks{int(blocks)}"

    sys.argv = [
        "phase3e_elambda_loop_suite.py",
        "--out_root_a",
        "out_phase3D_channel_diag_scale8_fullanchors_legacy",
        "--out_root_b",
        "out_phase3D_channel_diag_scale8_fullanchors_geomv9",
        "--out_root",
        out_root,
        "--blocks",
        str(int(blocks)),
        "--loop_pairs",
        "0:1,0:2,1:2",
        "--refine_steps",
        "1",
        "--eta",
        "-1",
        "--deform_alpha",
        "1.0",
        "--lambda_grid",
        "5",
    ]

    print("\n[run_phase3e_elambda_blocks_sweep] blocks:", blocks, "out_root:", out_root)
    runpy.run_path("phase3e_elambda_loop_suite.py", run_name="__main__")


def main() -> None:
    for blocks in [2, 8]:
        run_one(blocks)


if __name__ == "__main__":
    main()
