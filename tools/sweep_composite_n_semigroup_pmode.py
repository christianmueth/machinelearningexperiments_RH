import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _safe_log_abs(x: float, *, floor: float = 1e-12) -> float:
    return float(math.log(max(float(floor), abs(float(x)))))


def _summarize_probe_csv(in_csv: Path, *, min_abs_P: float = 0.0) -> dict:
    df = pd.read_csv(str(in_csv))
    if not {"n", "P"}.issubset(set(df.columns)):
        raise ValueError(f"Expected columns n,P in {in_csv}")

    P = {int(r.n): float(r.P) for r in df.itertuples(index=False)}
    ns = sorted(P.keys())

    rel_mult_errs: list[float] = []
    logadd_errs: list[float] = []
    n_pairs_total = 0
    n_pairs_kept = 0

    for a in ns:
        for b in ns:
            if a <= 1 or b <= 1:
                continue
            if math.gcd(int(a), int(b)) != 1:
                continue
            ab = int(a * b)
            if ab not in P:
                continue
            n_pairs_total += 1

            Pa = float(P[a])
            Pb = float(P[b])
            Pab = float(P[ab])

            thr = float(min_abs_P)
            if thr > 0 and (abs(Pa) < thr or abs(Pb) < thr or abs(Pab) < thr):
                continue

            n_pairs_kept += 1
            denom = max(1e-12, abs(Pa) * abs(Pb))
            rel_mult_errs.append(float(abs(Pab - Pa * Pb) / denom))
            logadd_errs.append(float(abs(_safe_log_abs(Pab) - (_safe_log_abs(Pa) + _safe_log_abs(Pb)))))

    def medmax(xs: list[float]) -> tuple[float, float]:
        if not xs:
            return float("nan"), float("nan")
        arr = np.asarray(xs, dtype=float)
        return float(np.median(arr)), float(np.max(arr))

    mult_med, mult_max = medmax(rel_mult_errs)
    log_med, log_max = medmax(logadd_errs)

    return {
        "in_csv": str(in_csv).replace("\\", "/"),
        "ns": ",".join(str(n) for n in ns),
        "n_pairs_total": int(n_pairs_total),
        "n_pairs_kept": int(n_pairs_kept),
        "min_abs_P": float(min_abs_P),
        "mult_rel_median": float(mult_med),
        "mult_rel_max": float(mult_max),
        "logadd_median": float(log_med),
        "logadd_max": float(log_max),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Run tools/probe_composite_n_semigroup.py across p_mode/obs_mode settings and write a single summary CSV, "
            "ranking by median log|P| additivity (primary composite-n gate)."
        )
    )

    ap.add_argument(
        "--ns",
        default="2,3,5,6,10,15,30",
        help="Comma-list of n to evaluate (recommend squarefree-heavy if testing coprime log-additivity)",
    )
    ap.add_argument("--u", type=float, default=0.2)
    ap.add_argument("--boundary", default="0,3")
    ap.add_argument("--schur_sign", choices=["-", "+"], default="+")

    ap.add_argument("--sharp", choices=["transpose", "conj_transpose"], default="conj_transpose")
    ap.add_argument("--X_mode", choices=["raw", "det1"], default="det1")
    ap.add_argument("--X_gamma", type=float, default=1.0)
    ap.add_argument("--scattering", choices=["lambda_pm_i", "i_pm_lambda"], default="i_pm_lambda")

    ap.add_argument(
        "--p_modes",
        default="p,invp,p1_over_p",
        help="Comma-list among: p,logp,p1_over_p,p_over_p1,invp,p_minus1_over_p",
    )
    ap.add_argument(
        "--obs_modes",
        default="logdet_I_minus_S,devS_pm_I,froLam",
        help="Comma-list among: devS_pm_I,froLam,trLam,logdet_I_minus_S,logdetLam",
    )

    ap.add_argument(
        "--tag",
        default="",
        help="Optional tag for output filenames (e.g. nsbig).",
    )
    ap.add_argument(
        "--min_abs_P",
        type=float,
        default=0.0,
        help="Optional filter for scoring: skip coprime pairs if any |P| below threshold.",
    )

    ap.add_argument(
        "--out_dir",
        default="out",
        help="Directory to place probe CSVs and summary CSV.",
    )
    ap.add_argument(
        "--out_summary_csv",
        default="",
        help="Optional summary CSV path. Defaults to out_dir/composite_n_semigroup_pmode_sweep_summary{_tag}.csv",
    )

    args = ap.parse_args()

    out_dir = REPO_ROOT / str(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p_modes = [x.strip() for x in str(args.p_modes).split(",") if x.strip()]
    obs_modes = [x.strip() for x in str(args.obs_modes).split(",") if x.strip()]

    tag = str(args.tag).strip()
    tag_suffix = f"_{tag}" if tag else ""

    summary_path = Path(str(args.out_summary_csv)) if str(args.out_summary_csv).strip() else (out_dir / f"composite_n_semigroup_pmode_sweep_summary{tag_suffix}.csv")

    probe_script = REPO_ROOT / "tools" / "probe_composite_n_semigroup.py"
    if not probe_script.exists():
        raise SystemExit(f"Missing {probe_script}")

    rows: list[dict] = []

    for obs_mode in obs_modes:
        for p_mode in p_modes:
            out_csv = out_dir / f"composite_n_semigroup_{obs_mode}_{p_mode}{tag_suffix}.csv"
            cmd = [
                sys.executable,
                str(probe_script),
                "--ns",
                str(args.ns),
                "--u",
                str(float(args.u)),
                "--boundary",
                str(args.boundary),
                "--schur_sign",
                str(args.schur_sign),
                "--sharp",
                str(args.sharp),
                "--X_mode",
                str(args.X_mode),
                "--X_gamma",
                str(float(args.X_gamma)),
                "--p_mode",
                str(p_mode),
                "--scattering",
                str(args.scattering),
                "--obs_mode",
                str(obs_mode),
                "--out_csv",
                str(out_csv),
            ]

            r = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
            if r.returncode != 0:
                sys.stderr.write(r.stdout)
                sys.stderr.write(r.stderr)
                raise SystemExit(r.returncode)

            s = _summarize_probe_csv(out_csv, min_abs_P=float(args.min_abs_P))
            s.update(
                {
                    "p_mode": str(p_mode),
                    "obs_mode": str(obs_mode),
                    "u": float(args.u),
                    "boundary": str(args.boundary),
                    "schur_sign": str(args.schur_sign),
                    "sharp": str(args.sharp),
                    "X_mode": str(args.X_mode),
                    "X_gamma": float(args.X_gamma),
                    "scattering": str(args.scattering),
                    "tag": tag,
                }
            )
            rows.append(s)

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["obs_mode", "logadd_median", "p_mode"], ascending=[True, True, True])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(str(summary_path), index=False)

    # Print a compact ranking table.
    view = out_df[["obs_mode", "p_mode", "logadd_median", "logadd_max", "mult_rel_median", "n_pairs_kept"]].copy()
    pd.set_option("display.max_rows", 200)
    print("wrote", str(summary_path))
    print(view.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
