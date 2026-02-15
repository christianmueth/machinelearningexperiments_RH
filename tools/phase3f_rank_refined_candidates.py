import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _parse_lens_list(s: str) -> list[str]:
    out: list[str] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(part)
    return out


def _active_metric(df: pd.DataFrame, base: str, has_refine2: bool) -> pd.Series:
    """Pick refine2 values when available, else refined."""
    if has_refine2 and f"{base}_refine2" in df.columns:
        v2 = pd.to_numeric(df.get(f"{base}_refine2"), errors="coerce")
        v1 = pd.to_numeric(df.get(f"{base}_refined"), errors="coerce")
        return v2.where(np.isfinite(v2), v1)
    return pd.to_numeric(df.get(f"{base}_refined"), errors="coerce")


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True, help="Smoke out_root containing prime_persistence_table.csv and refine_prime_* subfolders")
    ap.add_argument("--in_csv", default="prime_persistence_table.csv", help="Input CSV name (in out_root)")
    ap.add_argument("--out_csv", default="refined_ranking.csv", help="Output CSV name (written into out_root)")
    ap.add_argument(
        "--mu_gate_lenses",
        default="lambda,S",
        help=(
            "Comma-separated list of µ-direction KPI lenses to include in the worst-case gate. "
            "Examples: 'lambda,S' (default), 'lambda,S,series,resid'."
        ),
    )
    args = ap.parse_args()

    out_root = Path(str(args.out_root)).resolve()
    in_path = out_root / str(args.in_csv)
    if not in_path.exists():
        raise SystemExit(f"Missing: {in_path}")

    df = pd.read_csv(in_path)
    if "refined_dir" not in df.columns:
        raise SystemExit("prime_persistence_table.csv missing refined_dir column")

    has_refine2 = "refine2_dir" in df.columns

    # Prefer deepest available run directory per candidate.
    active_dirs = []
    active_stage = []
    for _, r in df.iterrows():
        r2 = str(r.get("refine2_dir", "")).strip() if has_refine2 else ""
        if r2:
            p2 = Path(r2)
            if p2.exists():
                active_dirs.append(str(p2))
                active_stage.append("refine2")
                continue
        r1 = str(r.get("refined_dir", "")).strip()
        active_dirs.append(r1)
        active_stage.append("refined")

    df["active_dir"] = active_dirs
    df["active_stage"] = active_stage

    # Load word-test metrics (if present)
    word_norms = []
    word_genA = []
    word_genB = []
    word_comm = []
    for _, r in df.iterrows():
        active_dir = Path(str(r.get("active_dir", "")))
        wp = active_dir / "phase3f_word_test.json"
        d = _read_json(wp) if active_dir and active_dir.exists() else None
        if not d:
            word_norms.append(np.nan)
            word_genA.append("")
            word_genB.append("")
            word_comm.append(np.nan)
            continue
        word_norms.append(float(d.get("norm_M12_minus_M21", np.nan)))
        word_genA.append(str(d.get("genA", "")))
        word_genB.append(str(d.get("genB", "")))
        word_comm.append(float(d.get("norm_commutator_reported", np.nan)))

    df["word_norm_M12_minus_M21"] = word_norms
    df["word_genA"] = word_genA
    df["word_genB"] = word_genB
    df["word_norm_commutator_reported"] = word_comm

    # Basic derived metrics
    comm_ref = pd.to_numeric(df.get("max_comm_refined"), errors="coerce")
    if has_refine2 and "max_comm_refine2" in df.columns:
        comm_ref2 = pd.to_numeric(df.get("max_comm_refine2"), errors="coerce")
        comm_best = comm_ref2.where(np.isfinite(comm_ref2), comm_ref)
    else:
        comm_best = comm_ref

    df["max_comm_active"] = comm_best
    df["comm_gain"] = comm_best - pd.to_numeric(df.get("max_comm_smoke"), errors="coerce")

    # Composite score:
    # - Start with max commutator (refined)
    # - Add a small bonus for word-level noncommutativity (log-scaled)
    # - Penalize mu-collapse risk using a worst-case µ gate across selected lenses
    # - Hard-gate: if worst-case mu_rank < 2, penalize strongly
    comm = pd.to_numeric(df.get("max_comm_active"), errors="coerce")
    word = pd.to_numeric(df.get("word_norm_M12_minus_M21"), errors="coerce")

    lenses_raw = _parse_lens_list(str(args.mu_gate_lenses))
    lenses = [x.strip() for x in lenses_raw if x.strip()]
    if not lenses:
        lenses = ["lambda", "S"]

    cos_cols: list[str] = []
    rank_cols: list[str] = []
    for lens in lenses:
        key = str(lens).strip()
        key_l = key.lower()
        norm_map = {
            "lambda": "lambda",
            "lambdasnap": "lambda",
            "lambda_snap": "lambda",
            "s": "S",
            "ssnap": "S",
            "s_snap": "S",
            "trace": "trace",
            "trace_powers": "trace",
            "logdet": "logdet",
            "logdet_iminuss": "logdet",
            "series": "series",
            "series_partial": "series",
            "resid": "resid",
            "resid_partial": "resid",
        }
        lens_norm = norm_map.get(key_l, key)
        # Keep column naming consistent: mu_cos_<lens>_refined/refine2
        cos_active = _active_metric(df, f"mu_cos_{lens_norm}", has_refine2)
        rank_active = _active_metric(df, f"mu_rank_{lens_norm}", has_refine2)
        cos_name = f"mu_cos_{lens_norm}_active"
        rank_name = f"mu_rank_{lens_norm}_active"
        df[cos_name] = cos_active
        df[rank_name] = rank_active
        cos_cols.append(cos_name)
        rank_cols.append(rank_name)

    # Gate summaries (skipna so if some lenses are missing, we fall back to the ones present).
    df["mu_cos_gate"] = df[cos_cols].max(axis=1, skipna=True)
    df["mu_rank_gate"] = df[rank_cols].min(axis=1, skipna=True)

    word_bonus = 0.25 * np.log1p(np.maximum(0.0, word.fillna(0.0)))
    mu_penalty = 0.50 * np.maximum(0.0, df["mu_cos_gate"].fillna(1.0) - 0.20)
    rank_penalty = np.where(df["mu_rank_gate"].fillna(0.0) < 2.0, 5.0, 0.0)

    df["score"] = comm + word_bonus - mu_penalty - rank_penalty

    # Sort and write
    df = df.sort_values(["score", "max_comm_active"], ascending=[False, False]).reset_index(drop=True)

    out_path = out_root / str(args.out_csv)
    df.to_csv(out_path, index=False)

    top = df.head(3)
    print(f"Wrote: {out_path}")
    print("Top-3 refined candidates by composite score:")
    for i, r in top.iterrows():
        print(
            f"#{i+1}: seed={int(r['seed'])} a={int(r['anchor_seed'])} w={r['wlo']}:{r['whi']} mu={r['mu']} "
            f"comm_active={r['max_comm_active']:.6g} word={r['word_norm_M12_minus_M21']:.6g} "
            f"mu_cos_gate={r.get('mu_cos_gate', float('nan')):.3g} stage={r.get('active_stage','')} dir={r.get('active_dir','')}"
        )


if __name__ == "__main__":
    main()
