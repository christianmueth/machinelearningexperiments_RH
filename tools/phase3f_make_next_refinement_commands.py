import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


def _fmt_num(x: float) -> str:
    # Keep deterministic, shell-safe formatting.
    if float(x).is_integer():
        return str(int(x))
    s = f"{x:.10g}"
    return s


def _uniq_sorted(xs: Iterable[float], ndigits: int = 10) -> List[float]:
    seen = set()
    out: List[float] = []
    for x in xs:
        xr = round(float(x), ndigits)
        if xr in seen:
            continue
        seen.add(xr)
        out.append(float(xr))
    out.sort()
    return out


def _densify_lambdas(base: Sequence[float], j_pairs: Sequence[Tuple[int, int]]) -> List[float]:
    base = [float(x) for x in base]
    add: List[float] = []
    n = len(base)

    for j0, j1 in j_pairs:
        if not (0 <= j0 < n and 0 <= j1 < n):
            continue
        a = base[j0]
        b = base[j1]
        lo = min(a, b)
        hi = max(a, b)
        mid = 0.5 * (lo + hi)
        add.append(mid)

        # Also add one "neighbor midpoint" on each side when possible.
        # This helps if the π-cell is straddling a coarse interval.
        jlo = min(j0, j1)
        jhi = max(j0, j1)
        if jlo - 1 >= 0:
            add.append(0.5 * (base[jlo - 1] + base[jlo]))
        if jhi + 1 < n:
            add.append(0.5 * (base[jhi] + base[jhi + 1]))

    return _uniq_sorted(list(base) + add)


def _lambda_pairs_from_cells(cells: pd.DataFrame) -> List[Tuple[int, int]]:
    # Prefer confirmed π counts if present.
    if "n_pi_confirm" in cells.columns:
        filt = pd.to_numeric(cells["n_pi_confirm"], errors="coerce").fillna(0) >= 1
        use = cells[filt].copy()
        if len(use) == 0:
            use = cells.copy()
    else:
        use = cells.copy()

    j0 = pd.to_numeric(use.get("lambda_j0"), errors="coerce").dropna().astype(int)
    j1 = pd.to_numeric(use.get("lambda_j1"), errors="coerce").dropna().astype(int)
    pairs = list({(int(a), int(b)) for a, b in zip(j0.tolist(), j1.tolist())})
    pairs.sort()
    return pairs


def _infer_window_tag(wlo: float, whi: float) -> str:
    # Matches existing naming scheme-ish: 0.6_7.5, 2_5, etc.
    def t(x: float) -> str:
        s = _fmt_num(x)
        return s.replace(".", ".")

    # Use minimal formatting but keep '.'
    return f"w{t(wlo)}_{t(whi)}".replace(".", ".")


def _default_next_dir_name(seed: int, anchor: int, wlo: float, whi: float, mu_tag: str, blocks: int, dim: int, refine_steps: int) -> str:
    # Example: refine2_prime_seed96_a14_w0.6_7.5_mu1_b1_d2_rs3
    # Keep '.' in name (Windows allows it); replace ':'
    wtag = f"w{_fmt_num(wlo)}_{_fmt_num(whi)}".replace(":", "_")
    return f"refine2_prime_seed{seed}_a{anchor}_{wtag}_{mu_tag}_b1_d{dim}_rs{refine_steps}".replace(" ", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke_out_root", required=True, help="Smoke out_root containing refined_ranking.csv")
    ap.add_argument("--ranking_csv", default="refined_ranking.csv")
    ap.add_argument("--top_n", type=int, default=3)
    ap.add_argument("--refine_steps", type=int, default=3)
    ap.add_argument("--write_txt", default="next_refinement_commands_top3.txt")
    ap.add_argument("--write_csv", default="next_refinement_plan_top3.csv")
    args = ap.parse_args()

    smoke = Path(str(args.smoke_out_root)).resolve()
    ranking_path = smoke / str(args.ranking_csv)
    if not ranking_path.exists():
        raise SystemExit(f"Missing ranking CSV: {ranking_path}")

    df = pd.read_csv(ranking_path)
    if len(df) == 0:
        raise SystemExit("Empty ranking CSV")

    top = df.head(int(args.top_n)).copy()

    python_exe = str((Path.cwd() / ".venv" / "Scripts" / "python.exe").resolve())

    lines: List[str] = []
    plan_rows: List[dict] = []

    for _, r in top.iterrows():
        refined_dir = Path(str(r.get("refined_dir"))).resolve()
        suite_summary = refined_dir / "phase3e_elambda_suite_summary.json"
        cells_csv = refined_dir / "phase3f_event_cells_with_gen.csv"
        if not suite_summary.exists() or not cells_csv.exists():
            raise SystemExit(f"Missing required files under {refined_dir}")

        with open(suite_summary, "r", encoding="utf-8") as f:
            summ = json.load(f)

        base_lambdas = summ.get("lambdas", [])
        mus = summ.get("mus", [])
        blocks = int(summ.get("blocks", 2))

        # Parse identity
        seed = int(r.get("seed"))
        anchor = int(r.get("anchor_seed"))
        wlo = float(r.get("wlo"))
        whi = float(r.get("whi"))
        dim = int(r.get("dim"))

        # Build densified lambdas
        cells = pd.read_csv(cells_csv)
        pairs = _lambda_pairs_from_cells(cells)
        lambdas_dense = _densify_lambdas(base_lambdas, pairs)

        mu_tag = f"mu{_fmt_num(float(r.get('mu')))}"
        next_dir = _default_next_dir_name(seed, anchor, wlo, whi, mu_tag, blocks, dim, int(args.refine_steps))
        out_root = smoke / next_dir

        lambdas_arg = ",".join(_fmt_num(x) for x in lambdas_dense)
        mus_arg = ",".join(_fmt_num(float(x)) for x in mus)

        cmd = (
            f"{python_exe} phase3e_elambda_loop_suite.py "
            f"--out_root_a ./out_phase3D_channel_diag_pilot_legacy_intrinsic "
            f"--out_root_b ./out_phase3D_channel_diag_scale8_fullanchors_geomv9__subset_multipi_blocks2 "
            f"--out_root_b2 ./out_phase3D_channel_diag_pilot_legacy_intrinsic_v2 "
            f"--out_root {out_root.as_posix()} "
            f"--only_seed {seed} --only_anchor {anchor} --only_window {_fmt_num(wlo)}:{_fmt_num(whi)} "
            f"--suite_mode fwd_rect --dump_holonomy pi_only --dump_holonomy_max_dim 16 "
            f"--lambdas {lambdas_arg} --mus {mus_arg} "
            f"--blocks {blocks} --refine_steps {int(args.refine_steps)} --deform_alpha 1.0"
        )
        lines.append(cmd)
        plan_rows.append(
            {
                "seed": seed,
                "anchor": anchor,
                "wlo": wlo,
                "whi": whi,
                "mu": float(r.get("mu")),
                "dim": dim,
                "blocks": blocks,
                "refine_steps": int(args.refine_steps),
                "base_refined_dir": str(refined_dir),
                "next_out_root": str(out_root),
                "n_base_lambdas": len(base_lambdas),
                "n_dense_lambdas": len(lambdas_dense),
                "dense_lambdas": lambdas_arg,
            }
        )

    txt_path = smoke / str(args.write_txt)
    txt_path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")

    csv_path = smoke / str(args.write_csv)
    pd.DataFrame(plan_rows).to_csv(csv_path, index=False)

    print(f"Wrote: {txt_path}")
    print(f"Wrote: {csv_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
