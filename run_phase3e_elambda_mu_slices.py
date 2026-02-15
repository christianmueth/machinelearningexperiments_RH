import argparse
import os
import subprocess
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class MuSlice:
    name: str
    out_root_b: str


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--python", default="", help="Python executable to use (default: sys.executable)")

    ap.add_argument("--out_root_a", required=True, help="Artifact root A (legacy or other baseline channel_diag tree)")
    ap.add_argument(
        "--mu_b",
        required=True,
        help=(
            "Comma-separated list of mu slices as name=path. Example: "
            "geomv9_subset=.../out_phase3D_channel_diag_scale8_fullanchors_geomv9__subset_multipi_blocks2," 
            "geomv9_intr2=.../out_phase3D_channel_diag_pilot_geomv9_intrinsic_v2"
        ),
    )

    ap.add_argument("--out_base", required=True, help="Base output folder prefix. Each slice writes to out_base + '_' + name")

    # passthrough knobs for phase3e_elambda_loop_suite.py
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--loop_pairs", default="0:1,0:2,1:2")
    ap.add_argument("--refine_steps", type=int, default=1)
    ap.add_argument("--eta", type=float, default=-1.0)
    ap.add_argument("--deform_alpha", type=float, default=1.0)
    ap.add_argument(
        "--lambdas",
        default="0,0.0125,0.025,0.0375,0.05,0.0625,0.075,0.0875,0.1,0.1125,0.125,0.1375,0.15,0.1625,0.175,0.1875,0.2,0.2125,0.225,0.2375,0.25,0.2625,0.275,0.2875,0.3,0.3125,0.325,0.3375,0.35,0.3625,0.375,0.3875,0.4,0.4125,0.425,0.4375,0.45,0.4625,0.475,0.4875,0.5",
        help="Comma-separated lambdas list to pass through",
    )

    ap.add_argument("--dry_run", type=int, default=0, help="If 1: print commands only")

    args = ap.parse_args()

    py = str(args.python).strip() or None

    def parse_mu_b(s: str) -> List[MuSlice]:
        out: List[MuSlice] = []
        for part in [p.strip() for p in str(s).split(",") if p.strip()]:
            if "=" not in part:
                raise SystemExit("--mu_b items must be name=path")
            name, path = part.split("=", 1)
            name = name.strip()
            path = path.strip().strip('"')
            if not name or not path:
                raise SystemExit("--mu_b items must be name=path")
            out.append(MuSlice(name=name, out_root_b=path))
        if not out:
            raise SystemExit("No mu slices parsed")
        return out

    slices = parse_mu_b(str(args.mu_b))

    out_root_a = os.path.abspath(str(args.out_root_a))
    if not os.path.exists(out_root_a):
        raise SystemExit(f"Missing out_root_a: {out_root_a}")

    for sl in slices:
        out_root_b = os.path.abspath(sl.out_root_b)
        if not os.path.exists(out_root_b):
            raise SystemExit(f"Missing out_root_b for slice {sl.name}: {out_root_b}")

        out_root = os.path.abspath(str(args.out_base) + "_" + sl.name)

        cmd = [
            py or "python",
            os.path.join(os.path.dirname(__file__), "phase3e_elambda_loop_suite.py"),
            "--out_root_a",
            out_root_a,
            "--out_root_b",
            out_root_b,
            "--out_root",
            out_root,
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
            str(args.lambdas),
        ]

        print("\n=== mu slice:", sl.name, "===")
        print("out_root_b:", out_root_b)
        print("out_root  :", out_root)
        print("CMD:", " ".join(cmd))

        if int(args.dry_run) != 0:
            continue

        os.makedirs(out_root, exist_ok=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
