import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _q(x: pd.Series, q: float) -> float:
    x = pd.to_numeric(x, errors="coerce")
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    return float(np.quantile(x, q))


def _load_summary_kpis(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "phase3e_elambda_suite_summary.json"
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


_REFINE_DIR_RE = re.compile(
    r"^(?P<prefix>refine2_prime|refine_prime)_seed(?P<seed>\d+)_a(?P<anchor>\d+)_w(?P<wlo>[^_]+)_(?P<whi>[^_]+)_mu(?P<mu>[^_]+)_b(?P<block>\d+)_d(?P<dim>\d+)(?:_rs(?P<rs>\d+))?$"
)


def _key(seed: int, anchor: int, wlo: float, whi: float, mu: float, block: int, dim: int) -> Tuple[int, int, float, float, float, int, int]:
    return (seed, anchor, round(float(wlo), 6), round(float(whi), 6), round(float(mu), 6), block, dim)


def _index_refine2_dirs(out_root: Path) -> Dict[Tuple[int, int, float, float, float, int, int], Path]:
    best: Dict[Tuple[int, int, float, float, float, int, int], Tuple[int, Path]] = {}
    if not out_root.exists():
        return {}
    for p in out_root.iterdir():
        if not p.is_dir():
            continue
        m = _REFINE_DIR_RE.match(p.name)
        if not m or m.group("prefix") != "refine2_prime":
            continue

        try:
            seed = int(m.group("seed"))
            anchor = int(m.group("anchor"))
            wlo = float(m.group("wlo"))
            whi = float(m.group("whi"))
            mu = float(m.group("mu"))
            block = int(m.group("block"))
            dim = int(m.group("dim"))
            rs = int(m.group("rs") or 0)
        except Exception:
            continue

        k = _key(seed, anchor, wlo, whi, mu, block, dim)
        prev = best.get(k)
        if prev is None or rs > prev[0]:
            best[k] = (rs, p)
    return {k: v for k, (_, v) in best.items()}


def _group_mask(df: pd.DataFrame, seed: int, anchor: int, wlo: float, whi: float, mu: float, block: int, dim: int) -> pd.Series:
    m = (
        (df.get("seed") == seed)
        & (df.get("anchor_seed") == anchor)
        & (df.get("wlo") == wlo)
        & (df.get("whi") == whi)
        & (df.get("mu") == mu)
        & (df.get("block") == block)
        & (df.get("dim") == dim)
    )
    return m


def _compute_metrics(run_dir: Path, seed: int, anchor: int, wlo: float, whi: float, mu: float, block: int, dim: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    gens_path = run_dir / "phase3f_generators.csv"
    comp_path = run_dir / "phase3f_batch_composition.csv"

    if gens_path.exists():
        gens = pd.read_csv(gens_path)
        gm = _group_mask(gens, seed, anchor, wlo, whi, mu, block, dim)
        out["n_gens"] = int(gm.sum())
    else:
        out["n_gens"] = float("nan")

    if comp_path.exists():
        comp = pd.read_csv(comp_path)
        pm = (comp.get("kind") == "pair") & _group_mask(comp, seed, anchor, wlo, whi, mu, block, dim)
        gm2 = (comp.get("kind") == "gen") & _group_mask(comp, seed, anchor, wlo, whi, mu, block, dim)

        if pm.any():
            out["max_comm"] = float(pd.to_numeric(comp.loc[pm, "norm_commutator"], errors="coerce").max())
            fo = pd.to_numeric(comp.loc[pm, "flip_overlap"], errors="coerce")
            out["flip_overlap_median"] = float(np.nanmedian(fo.values))
            out["flip_overlap_min"] = float(np.nanmin(fo.values))
            out["flip_overlap_p10"] = _q(fo, 0.10)

            cd = pd.to_numeric(comp.loc[pm, "norm_concat_minus_prod"], errors="coerce")
            out["concat_defect_median"] = float(np.nanmedian(cd.values))
            out["concat_defect_p95"] = _q(cd, 0.95)

        else:
            out["max_comm"] = float("nan")
            out["flip_overlap_median"] = float("nan")
            out["flip_overlap_min"] = float("nan")
            out["flip_overlap_p10"] = float("nan")
            out["concat_defect_median"] = float("nan")
            out["concat_defect_p95"] = float("nan")

        if gm2.any():
            inv = pd.to_numeric(comp.loc[gm2, "norm_M2_minus_I"], errors="coerce")
            out["involution_defect_median"] = float(np.nanmedian(inv.values))
            out["involution_defect_p95"] = _q(inv, 0.95)
        else:
            out["involution_defect_median"] = float("nan")
            out["involution_defect_p95"] = float("nan")

    else:
        out["max_comm"] = float("nan")
        out["flip_overlap_median"] = float("nan")
        out["flip_overlap_min"] = float("nan")
        out["flip_overlap_p10"] = float("nan")
        out["concat_defect_median"] = float("nan")
        out["concat_defect_p95"] = float("nan")
        out["involution_defect_median"] = float("nan")
        out["involution_defect_p95"] = float("nan")

    # rank-2 KPIs from Phase-3E summary (only meaningful for refined runs, but harmless)
    summ = _load_summary_kpis(run_dir)
    k = summ.get("mu_direction_kpis", {})
    lam = k.get("lambda_snap", {}) if isinstance(k, dict) else {}
    out["mu_rank_lambda"] = lam.get("mu_effective_rank_median", float("nan"))
    out["mu_cos_lambda"] = lam.get("mu_cos_to_lambda_abs_median", float("nan"))
    out["mu_norm_ratio_lambda"] = lam.get("mu_norm_ratio_median", float("nan"))

    s = k.get("S_snap", {}) if isinstance(k, dict) else {}
    out["mu_rank_S"] = s.get("mu_effective_rank_median", float("nan"))
    out["mu_cos_S"] = s.get("mu_cos_to_lambda_abs_median", float("nan"))
    out["mu_norm_ratio_S"] = s.get("mu_norm_ratio_median", float("nan"))

    t = k.get("trace_powers", {}) if isinstance(k, dict) else {}
    out["mu_rank_trace"] = t.get("mu_effective_rank_median", float("nan"))
    out["mu_cos_trace"] = t.get("mu_cos_to_lambda_abs_median", float("nan"))
    out["mu_norm_ratio_trace"] = t.get("mu_norm_ratio_median", float("nan"))

    ld = k.get("logdet_IminusS", {}) if isinstance(k, dict) else {}
    out["mu_rank_logdet"] = ld.get("mu_effective_rank_median", float("nan"))
    out["mu_cos_logdet"] = ld.get("mu_cos_to_lambda_abs_median", float("nan"))
    out["mu_norm_ratio_logdet"] = ld.get("mu_norm_ratio_median", float("nan"))

    sp = k.get("series_partial", {}) if isinstance(k, dict) else {}
    out["mu_rank_series"] = sp.get("mu_effective_rank_median", float("nan"))
    out["mu_cos_series"] = sp.get("mu_cos_to_lambda_abs_median", float("nan"))
    out["mu_norm_ratio_series"] = sp.get("mu_norm_ratio_median", float("nan"))

    rp = k.get("resid_partial", {}) if isinstance(k, dict) else {}
    out["mu_rank_resid"] = rp.get("mu_effective_rank_median", float("nan"))
    out["mu_cos_resid"] = rp.get("mu_cos_to_lambda_abs_median", float("nan"))
    out["mu_norm_ratio_resid"] = rp.get("mu_norm_ratio_median", float("nan"))

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True, help="Smoke out_root containing phase3f_prime_candidates.csv")
    ap.add_argument("--candidates_csv", default="", help="Override candidates CSV path")
    ap.add_argument("--out_csv", default="prime_persistence_table.csv", help="Output CSV name (written into out_root)")
    args = ap.parse_args()

    out_root = Path(str(args.out_root)).resolve()
    cand_path = Path(str(args.candidates_csv)).resolve() if str(args.candidates_csv).strip() else (out_root / "phase3f_prime_candidates.csv")
    if not cand_path.exists():
        raise SystemExit(f"Missing candidates CSV: {cand_path}")

    cand = pd.read_csv(cand_path)

    refine2_index = _index_refine2_dirs(out_root)

    rows = []
    for _, r in cand.iterrows():
        seed = int(r["seed"])
        anchor = int(r["anchor_seed"])
        wlo = float(r["wlo"])
        whi = float(r["whi"])
        mu = float(r["mu"])
        block = int(r["block"])
        dim = int(r["dim"])

        sug = str(r.get("suggest_out_root", "")).strip()
        refined_dir = Path(sug) if sug else None
        if refined_dir is not None and not refined_dir.is_absolute():
            refined_dir = (out_root / refined_dir).resolve()

        base_metrics = _compute_metrics(out_root, seed, anchor, wlo, whi, mu, block, dim)
        ref_metrics = _compute_metrics(refined_dir, seed, anchor, wlo, whi, mu, block, dim) if (refined_dir and refined_dir.exists()) else {}

        refine2_dir = refine2_index.get(_key(seed, anchor, wlo, whi, mu, block, dim))
        ref2_metrics = _compute_metrics(refine2_dir, seed, anchor, wlo, whi, mu, block, dim) if (refine2_dir and refine2_dir.exists()) else {}

        row = {
            "seed": seed,
            "anchor_seed": anchor,
            "wlo": wlo,
            "whi": whi,
            "mu": mu,
            "block": block,
            "dim": dim,
            "genA": r.get("genA"),
            "genB": r.get("genB"),
            "max_comm_smoke": base_metrics.get("max_comm"),
            "max_comm_refined": ref_metrics.get("max_comm", float("nan")),
            "max_comm_refine2": ref2_metrics.get("max_comm", float("nan")),
            "n_gens_smoke": base_metrics.get("n_gens"),
            "n_gens_refined": ref_metrics.get("n_gens", float("nan")),
            "n_gens_refine2": ref2_metrics.get("n_gens", float("nan")),
            "flip_overlap_median_smoke": base_metrics.get("flip_overlap_median"),
            "flip_overlap_median_refined": ref_metrics.get("flip_overlap_median", float("nan")),
            "flip_overlap_median_refine2": ref2_metrics.get("flip_overlap_median", float("nan")),
            "flip_overlap_min_smoke": base_metrics.get("flip_overlap_min"),
            "flip_overlap_min_refined": ref_metrics.get("flip_overlap_min", float("nan")),
            "flip_overlap_min_refine2": ref2_metrics.get("flip_overlap_min", float("nan")),
            "involution_defect_med_smoke": base_metrics.get("involution_defect_median"),
            "involution_defect_med_refined": ref_metrics.get("involution_defect_median", float("nan")),
            "involution_defect_med_refine2": ref2_metrics.get("involution_defect_median", float("nan")),
            "concat_defect_med_smoke": base_metrics.get("concat_defect_median"),
            "concat_defect_med_refined": ref_metrics.get("concat_defect_median", float("nan")),
            "concat_defect_med_refine2": ref2_metrics.get("concat_defect_median", float("nan")),
            "mu_rank_lambda_refined": ref_metrics.get("mu_rank_lambda", float("nan")),
            "mu_cos_lambda_refined": ref_metrics.get("mu_cos_lambda", float("nan")),
            "mu_norm_ratio_lambda_refined": ref_metrics.get("mu_norm_ratio_lambda", float("nan")),
            "mu_rank_S_refined": ref_metrics.get("mu_rank_S", float("nan")),
            "mu_cos_S_refined": ref_metrics.get("mu_cos_S", float("nan")),
            "mu_norm_ratio_S_refined": ref_metrics.get("mu_norm_ratio_S", float("nan")),
            "mu_rank_trace_refined": ref_metrics.get("mu_rank_trace", float("nan")),
            "mu_cos_trace_refined": ref_metrics.get("mu_cos_trace", float("nan")),
            "mu_norm_ratio_trace_refined": ref_metrics.get("mu_norm_ratio_trace", float("nan")),
            "mu_rank_logdet_refined": ref_metrics.get("mu_rank_logdet", float("nan")),
            "mu_cos_logdet_refined": ref_metrics.get("mu_cos_logdet", float("nan")),
            "mu_norm_ratio_logdet_refined": ref_metrics.get("mu_norm_ratio_logdet", float("nan")),
            "mu_rank_series_refined": ref_metrics.get("mu_rank_series", float("nan")),
            "mu_cos_series_refined": ref_metrics.get("mu_cos_series", float("nan")),
            "mu_norm_ratio_series_refined": ref_metrics.get("mu_norm_ratio_series", float("nan")),
            "mu_rank_resid_refined": ref_metrics.get("mu_rank_resid", float("nan")),
            "mu_cos_resid_refined": ref_metrics.get("mu_cos_resid", float("nan")),
            "mu_norm_ratio_resid_refined": ref_metrics.get("mu_norm_ratio_resid", float("nan")),
            "mu_rank_lambda_refine2": ref2_metrics.get("mu_rank_lambda", float("nan")),
            "mu_cos_lambda_refine2": ref2_metrics.get("mu_cos_lambda", float("nan")),
            "mu_norm_ratio_lambda_refine2": ref2_metrics.get("mu_norm_ratio_lambda", float("nan")),
            "mu_rank_S_refine2": ref2_metrics.get("mu_rank_S", float("nan")),
            "mu_cos_S_refine2": ref2_metrics.get("mu_cos_S", float("nan")),
            "mu_norm_ratio_S_refine2": ref2_metrics.get("mu_norm_ratio_S", float("nan")),
            "mu_rank_trace_refine2": ref2_metrics.get("mu_rank_trace", float("nan")),
            "mu_cos_trace_refine2": ref2_metrics.get("mu_cos_trace", float("nan")),
            "mu_norm_ratio_trace_refine2": ref2_metrics.get("mu_norm_ratio_trace", float("nan")),
            "mu_rank_logdet_refine2": ref2_metrics.get("mu_rank_logdet", float("nan")),
            "mu_cos_logdet_refine2": ref2_metrics.get("mu_cos_logdet", float("nan")),
            "mu_norm_ratio_logdet_refine2": ref2_metrics.get("mu_norm_ratio_logdet", float("nan")),
            "mu_rank_series_refine2": ref2_metrics.get("mu_rank_series", float("nan")),
            "mu_cos_series_refine2": ref2_metrics.get("mu_cos_series", float("nan")),
            "mu_norm_ratio_series_refine2": ref2_metrics.get("mu_norm_ratio_series", float("nan")),
            "mu_rank_resid_refine2": ref2_metrics.get("mu_rank_resid", float("nan")),
            "mu_cos_resid_refine2": ref2_metrics.get("mu_cos_resid", float("nan")),
            "mu_norm_ratio_resid_refine2": ref2_metrics.get("mu_norm_ratio_resid", float("nan")),
            "refined_exists": bool(refined_dir and refined_dir.exists()),
            "refined_dir": str(refined_dir) if refined_dir else "",
            "refine2_exists": bool(refine2_dir and refine2_dir.exists()),
            "refine2_dir": str(refine2_dir) if refine2_dir else "",
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = out_root / str(args.out_csv)
    df.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
