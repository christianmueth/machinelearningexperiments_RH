import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pandas as pd


def run_one_scan(
    python_exe: str,
    base_script: str,
    out_root: str,
    fe_gate: str,
    fe_agg: str,
    fe_beta: float,
    seed_start: int,
    seed_end_exclusive: int,
    anchors: str,
    n_null: int,
    k: int,
    windows: str,
    dnmap_stride: int,
    dnmap_null_batch: int,
    amp: float,
    fe_alpha_E: float,
    log_path: str | None = None,
    heartbeat_seconds: float = 60.0,
):
    cmd = [
        python_exe,
        base_script,
        "--out_root",
        out_root,
        "--seed_start",
        str(seed_start),
        "--seed_end_exclusive",
        str(seed_end_exclusive),
        "--anchors",
        anchors,
        "--N_null",
        str(n_null),
        "--k",
        str(k),
        "--windows",
        windows,
        "--dnmap_stride",
        str(dnmap_stride),
        "--dnmap_null_batch",
        str(dnmap_null_batch),
        "--amp",
        str(amp),
        "--resume",
        "--fe_gate",
        fe_gate,
        "--fe_completion_mode",
        "analytic_harmonic",
        "--fe_calibration_mode",
        "none",
        "--fe_alpha_E",
        str(fe_alpha_E),
        "--fe_beta",
        str(fe_beta),
        "--fe_agg",
        fe_agg,
    ]

    t0 = time.time()
    print(f"\n=== RUN out_root={out_root} gate={fe_gate} agg={fe_agg} beta={fe_beta} ===", flush=True)
    if log_path is None:
        subprocess.run(cmd, check=True)
    else:
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as f:
            f.write("COMMAND:\n")
            f.write(" ".join(cmd) + "\n\n")
            f.write("OUTPUT:\n")
            f.flush()

            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            try:
                while True:
                    try:
                        rc = proc.wait(timeout=float(heartbeat_seconds))
                        break
                    except subprocess.TimeoutExpired:
                        dt_hb = time.time() - t0
                        print(
                            f"... still running out_root={out_root} elapsed={dt_hb:.1f}s (see {log_file}) ...",
                            flush=True,
                        )
                if rc != 0:
                    raise subprocess.CalledProcessError(rc, cmd)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=5)
    dt = time.time() - t0
    print(f"=== DONE out_root={out_root} elapsed={dt:.1f}s ===", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_start", type=int, default=64)
    ap.add_argument("--seed_end_exclusive", type=int, default=128)
    ap.add_argument("--anchors", default="2,9,14,44,46,51,60")

    ap.add_argument("--N_null", type=int, default=1024)
    ap.add_argument("--k", type=int, default=13)
    ap.add_argument("--windows", default="2.0,5.0;0.6,7.5")
    ap.add_argument("--dnmap_stride", type=int, default=2)
    ap.add_argument("--dnmap_null_batch", type=int, default=16)
    ap.add_argument("--amp", type=float, default=0.03)

    ap.add_argument("--fe_alpha_E", type=float, default=1.0)
    ap.add_argument("--fe_beta_list", default="1.0,2.0")
    ap.add_argument("--fe_agg_list", default="mean,median")
    ap.add_argument("--fe_gate_list", default="fe_only,fe_and_rigid")

    ap.add_argument(
        "--out_parent",
        default="out_phase3C_fe_matrix64_127",
        help="Parent folder; each configuration gets its own out_root under this.",
    )

    ap.add_argument(
        "--skip_if_done",
        action="store_true",
        help="If set, skip running a configuration when kpi_fraction.csv already exists in its out_root.",
    )
    ap.add_argument(
        "--continue_on_error",
        action="store_true",
        help="If set, keep going even if a configuration fails; errors are recorded under out_parent.",
    )

    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    base_script = str(repo_root / "preregistered_phase3c_fe_seedblock_64_127.py")

    betas = [float(x.strip()) for x in str(args.fe_beta_list).split(",") if x.strip()]
    aggs = [x.strip() for x in str(args.fe_agg_list).split(",") if x.strip()]
    gates = [x.strip() for x in str(args.fe_gate_list).split(",") if x.strip()]

    out_parent = Path(args.out_parent)
    out_parent.mkdir(parents=True, exist_ok=True)

    matrix_spec = {
        "date": time.strftime("%Y-%m-%d"),
        "phase": "3C",
        "matrix": {
            "fe_gate": gates,
            "fe_agg": aggs,
            "fe_beta": betas,
            "fe_alpha_E": float(args.fe_alpha_E),
            "fe_completion_mode": "analytic_harmonic",
            "fe_calibration_mode": "none",
        },
        "seedblock": [int(args.seed_start), int(args.seed_end_exclusive)],
        "anchors": str(args.anchors),
        "N_null": int(args.N_null),
        "k": int(args.k),
        "windows": str(args.windows),
        "dnmap_stride": int(args.dnmap_stride),
        "dnmap_null_batch": int(args.dnmap_null_batch),
        "amp": float(args.amp),
        "runner": str(Path(__file__).name),
    }
    (out_parent / "matrix_spec.json").write_text(json.dumps(matrix_spec, indent=2) + "\n", encoding="utf-8")

    summary_rows = []
    failure_rows = []

    for fe_gate in gates:
        for fe_agg in aggs:
            for fe_beta in betas:
                tag = f"gate-{fe_gate}__agg-{fe_agg}__beta-{str(fe_beta).replace('.', 'p')}"
                out_root = str(out_parent / tag)

                kpi_path = Path(out_root) / "kpi_fraction.csv"
                if args.skip_if_done and kpi_path.exists():
                    print(f"\n=== SKIP (kpi exists) out_root={out_root} ===", flush=True)
                else:
                    try:
                        run_one_scan(
                            python_exe=sys.executable,
                            base_script=base_script,
                            out_root=out_root,
                            fe_gate=fe_gate,
                            fe_agg=fe_agg,
                            fe_beta=fe_beta,
                            seed_start=int(args.seed_start),
                            seed_end_exclusive=int(args.seed_end_exclusive),
                            anchors=str(args.anchors),
                            n_null=int(args.N_null),
                            k=int(args.k),
                            windows=str(args.windows),
                            dnmap_stride=int(args.dnmap_stride),
                            dnmap_null_batch=int(args.dnmap_null_batch),
                            amp=float(args.amp),
                            fe_alpha_E=float(args.fe_alpha_E),
                            log_path=str(Path(out_root) / "subprocess.log"),
                        )
                    except Exception as e:
                        failure_rows.append(
                            {
                                "tag": tag,
                                "out_root": out_root,
                                "fe_gate": fe_gate,
                                "fe_agg": fe_agg,
                                "fe_beta": fe_beta,
                                "error": repr(e),
                                "traceback": traceback.format_exc(),
                            }
                        )

                        failures_path = out_parent / "matrix_failures.jsonl"
                        with failures_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(failure_rows[-1]) + "\n")

                        print(f"=== FAIL out_root={out_root} (see {failures_path} and {Path(out_root) / 'subprocess.log'}) ===")
                        if not args.continue_on_error:
                            raise
                if kpi_path.exists():
                    kpi = pd.read_csv(kpi_path)
                    kpi["out_root"] = out_root
                    kpi["fe_gate"] = fe_gate
                    kpi["fe_agg"] = fe_agg
                    kpi["fe_beta"] = fe_beta
                    summary_rows.append(kpi)

                summary_path = out_parent / "matrix_kpi_fraction__concat.csv"
                if summary_rows:
                    pd.concat(summary_rows, ignore_index=True).to_csv(summary_path, index=False)

    print("\n=== MATRIX DONE ===")
    if summary_rows:
        print(f"Wrote: {out_parent / 'matrix_kpi_fraction__concat.csv'}")
    if failure_rows:
        print(f"Failures recorded: {out_parent / 'matrix_failures.jsonl'}")


if __name__ == "__main__":
    main()
