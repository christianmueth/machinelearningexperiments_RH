from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ART_PATTERNS = ("*__aug_lam.csv", "*_delta*.csv", "*_samples*.csv")


def _count_artifacts(scan_dir: Path, *, cap: int = 200) -> int:
    n = 0
    for pat in ART_PATTERNS:
        for _ in scan_dir.rglob(pat):
            n += 1
            if n >= cap:
                return n
    return n


def _pick_run_dir_in_family(
    family_dir: Path, *, min_artifacts: int, scan_family: bool
) -> tuple[Path, int] | None:
    if not family_dir.is_dir():
        return None

    # Prefer timestamped child dirs.
    kids = [p for p in family_dir.iterdir() if p.is_dir()]
    if not kids:
        return None

    if scan_family:
        n_art = _count_artifacts(family_dir)
        if n_art < int(min_artifacts):
            return None
        # Pick a stable output location: newest-looking timestamped dir.
        best_kid = sorted(kids, key=lambda p: str(p.name), reverse=True)[0]
        return best_kid, int(n_art)

    scored: list[tuple[int, str, Path]] = []
    for p in kids:
        n_art = _count_artifacts(p)
        if n_art >= int(min_artifacts):
            scored.append((int(n_art), str(p.name), p))

    if not scored:
        return None

    # Prefer most artifacts; break ties by newest-looking name.
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    best = scored[0]
    return best[2], int(best[0])


def _support_status_to_score(status: Any) -> float:
    s = str(status or "").strip().lower()
    if s == "ok":
        return 1.0
    if s in {"support-limited", "variance-limited"}:
        return 0.5
    return 0.0


def _clip01(x: float) -> float:
    if x != x:  # NaN
        return 0.0
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


def _iter_family_dirs(root: Path) -> list[Path]:
    fams: list[Path] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if p.name.startswith("runs_"):
            fams.append(p)
    return sorted(fams, key=lambda p: p.name)


def _format_label_mix(labels: list[str], topk: int = 4) -> str:
    c = Counter([str(x) for x in labels])
    parts = [f"{k}={v}" for (k, v) in c.most_common(topk)]
    return "; ".join(parts)


def _extract_ablation(payload: dict[str, Any], key: str, default: float = float("nan")) -> float:
    ab = payload.get("ablation", {}) or {}
    for block in ["zeta_structure", "spectral_identity", "structure", "robustness"]:
        try:
            v = (ab.get(block, {}) or {})[key]["spearman_rank_corr"]
            return float(v)
        except Exception:
            continue
    return float(default)


def _dominant_signal(payload: dict[str, Any]) -> str:
    # Prefer ablation-based importance (1 - spearman) when present; fall back to support counts.
    def _imp(k: str) -> float:
        v = _extract_ablation(payload, k)
        if v != v:
            return float("nan")
        return float(1.0 - float(v))

    imp_mirror = _imp("drop_mirror")
    imp_bestt = _imp("drop_best_t_align")
    imp_div = _imp("drop_divisor")
    imps = {"symmetry-dominant": imp_mirror, "best-t-dominant": imp_bestt, "divisor-supported": imp_div}
    finite = {k: v for k, v in imps.items() if (v == v)}
    if finite:
        best = max(finite.items(), key=lambda kv: kv[1])
        if float(best[1]) > 0.01:
            return str(best[0])

    counts = (payload.get("support_regime", {}) or {}).get("counts", {})
    mir = float(counts.get("mirror_available") or 0.0)
    div = float(counts.get("divisor_available") or 0.0)
    if (mir > 0) and (mir >= div):
        return "symmetry-dominant"
    if div > 0:
        return "divisor-supported"
    return "generic-only"


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-basket validation runner (selector_v1 default; can run selector_v2)")
    ap.add_argument("--root", default=".", help="Workspace root containing run families")
    ap.add_argument("--n", type=int, default=4, help="How many distinct run families to validate")
    ap.add_argument(
        "--include",
        default="",
        help="Comma-separated run family dir names to include (e.g. runs_ladder_candidate1,runs_sigma_sweep3). If empty, auto-pick.",
    )
    ap.add_argument(
        "--exclude",
        default="",
        help="Comma-separated run family dir names to exclude.",
    )
    ap.add_argument(
        "--synthetic_controls",
        type=int,
        default=1,
        help="Pass through to validator as --synthetic_smoothness_controls.",
    )
    ap.add_argument(
        "--min_artifacts",
        type=int,
        default=5,
        help="Require at least this many artifact CSVs in a picked run_dir (pre-filter).",
    )
    ap.add_argument(
        "--min_basket",
        type=int,
        default=30,
        help="Skip run_dirs whose validator basket is smaller than this (post-filter).",
    )
    ap.add_argument(
        "--selector_version",
        default="selector_v1",
        choices=["selector_v1", "selector_v2"],
        help="Selector version to request from the validator.",
    )
    ap.add_argument(
        "--out_csv",
        default="cross_basket_selector_v1_comparison.csv",
        help="Where to write the comparison CSV (relative to root).",
    )
    ap.add_argument(
        "--out_regime_csv",
        default="cross_basket_selector_v1_regime_summary.csv",
        help="Where to write the per-family support-regime summary CSV (relative to root).",
    )
    ap.add_argument(
        "--out_syn_hard_csv",
        default="cross_basket_selector_v1_synthetic_hard_controls.csv",
        help="Where to write the aggregated synthetic-hard-controls CSV (relative to root).",
    )
    ap.add_argument(
        "--out_family_digest_csv",
        default="cross_basket_selector_v1_family_digest.csv",
        help="Where to write the per-family digest CSV (relative to root).",
    )
    ap.add_argument(
        "--out_report_json",
        default="cross_basket_selector_v1_report.json",
        help="Where to write a short cross-basket JSON report (relative to root).",
    )
    ap.add_argument(
        "--out_report_txt",
        default="cross_basket_selector_v1_report.txt",
        help="Where to write a short cross-basket text report (relative to root).",
    )
    ap.add_argument(
        "--scan_family",
        type=int,
        default=1,
        help="If 1, run validator with --scan_root=<family_dir> so artifacts can be aggregated across timestamped runs within a family.",
    )

    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"root not found: {root}")

    include = [x.strip() for x in str(args.include).split(",") if x.strip()]
    exclude = {x.strip() for x in str(args.exclude).split(",") if x.strip()}

    if include:
        family_dirs = [root / x for x in include if (root / x).is_dir()]
    else:
        family_dirs = [p for p in _iter_family_dirs(root) if p.name not in exclude]

    # Prefer families with artifacts; pick one run_dir per family.
    picked: list[tuple[Path, int]] = []
    for fam in family_dirs:
        pick = _pick_run_dir_in_family(
            fam, min_artifacts=int(args.min_artifacts), scan_family=bool(int(args.scan_family))
        )
        if pick is not None:
            picked.append(pick)

    # De-dup by family.
    uniq: dict[str, tuple[Path, int]] = {}
    for rd, n_art in picked:
        fam = rd.parent.name
        if fam not in uniq:
            uniq[fam] = (rd, int(n_art))

    # Try most artifact-rich candidates first.
    run_dirs = [rd for (rd, _n) in sorted(uniq.values(), key=lambda t: t[1], reverse=True)]

    if not run_dirs:
        raise SystemExit("No candidate run_dirs found with artifacts")

    validator = root / "tools" / "build_validate_bias_scorecard_artifact_only.py"
    if not validator.exists():
        raise SystemExit(f"validator not found: {validator}")

    rows: list[dict[str, Any]] = []
    regime_rows_all: list[dict[str, Any]] = []
    family_digest_rows: list[dict[str, Any]] = []
    syn_hard_rows: list[pd.DataFrame] = []
    for rd in run_dirs:
        if len(rows) >= max(1, int(args.n)):
            break

        fam_dir = rd.parent
        cmd = [
            sys.executable,
            str(validator),
            "--run_dir",
            str(rd.relative_to(root) if rd.is_relative_to(root) else rd),
            "--out_dir",
            str(rd.relative_to(root) if rd.is_relative_to(root) else rd),
            "--synthetic_smoothness_controls",
            str(int(args.synthetic_controls)),
            "--selector_version",
            str(args.selector_version),
        ]

        if int(args.scan_family):
            cmd.extend([
                "--scan_root",
                str(fam_dir.relative_to(root) if fam_dir.is_relative_to(root) else fam_dir),
            ])
        subprocess.run(cmd, cwd=str(root), check=True)

        out_prefix = "_bias_scorecard_v2"
        report_json = rd / f"{out_prefix}_report.json"
        eval_csv = rd / f"{out_prefix}_eval.csv"

        payload = json.loads(report_json.read_text(encoding="utf-8")) if report_json.exists() else {}
        df = pd.read_csv(eval_csv) if eval_csv.exists() else pd.DataFrame()

        n_basket = payload.get("n_basket")
        try:
            n_basket_i = int(n_basket) if n_basket is not None else (int(len(df)) if not df.empty else 0)
        except Exception:
            n_basket_i = int(len(df)) if not df.empty else 0

        if n_basket_i < int(args.min_basket):
            print(f"Skip (basket too small n={n_basket_i}): {rd}")
            continue

        top10 = df.sort_values("combo", ascending=False).head(10) if (not df.empty and "combo" in df.columns) else pd.DataFrame()
        top10_labels = top10.get("hand_label", pd.Series([], dtype=str)).astype(str).tolist() if not top10.empty else []

        sel = payload.get("selector", {})
        syn = payload.get("synthetic_smoothness_controls", {})
        syn_plausible = (syn or {}).get("plausible", {}) if isinstance(syn, dict) else {}
        support_counts = (payload.get("support_regime", {}) or {}).get("counts", {})

        anchor_audit = payload.get("anchor_audit", {}) or {}
        anchor_ab = (anchor_audit.get("anchor_ablation") if isinstance(anchor_audit, dict) else {}) or {}
        syn_anchor = (anchor_audit.get("synthetic_anchor_matched") if isinstance(anchor_audit, dict) else {}) or {}
        syn_anchor_plaus = (syn_anchor.get("plausible_only") if isinstance(syn_anchor, dict) else {}) or {}
        anchor_active_n = anchor_audit.get("n_anchor_active") if isinstance(anchor_audit, dict) else None
        anchor_ab_spearman = anchor_ab.get("spearman_rank_corr") if isinstance(anchor_ab, dict) else None
        anchor_ab_top10_jacc = anchor_ab.get("top10_overlap_jacc") if isinstance(anchor_ab, dict) else None
        syn_plaus_min_rank_vs_anchor = syn_anchor_plaus.get("min_rank") if isinstance(syn_anchor_plaus, dict) else None
        syn_plaus_any_top10_vs_anchor = syn_anchor_plaus.get("any_in_top10") if isinstance(syn_anchor_plaus, dict) else None

        support_regime = payload.get("support_regime", {}) or {}
        support_score = (
            _support_status_to_score(support_regime.get("mirror_block_status"))
            + _support_status_to_score(support_regime.get("best_t_align_block_status"))
            + _support_status_to_score(support_regime.get("divisor_block_status"))
        ) / 3.0
        artifacts_seen = int(uniq.get(rd.parent.name, (rd, 0))[1])
        confidence_basket = _clip01(float(n_basket_i) / 80.0)
        confidence_artifacts = _clip01(float(artifacts_seen) / 50.0)
        confidence_support = _clip01(float(support_score))
        confidence_total = _clip01(
            0.50 * confidence_basket + 0.30 * confidence_support + 0.20 * confidence_artifacts
        )

        rows.append(
            {
                "family": rd.parent.name,
                "run_dir": str(rd.relative_to(root) if rd.is_relative_to(root) else rd),
                "selector_version": sel.get("version"),
                "selector_hash12": sel.get("hash12"),
                "artifact_csvs_seen": artifacts_seen,
                "n_groups_discovered": payload.get("n_groups"),
                "n_basket": n_basket_i,
                "confidence_basket": round(confidence_basket, 4),
                "confidence_support": round(confidence_support, 4),
                "confidence_artifacts": round(confidence_artifacts, 4),
                "confidence_total": round(confidence_total, 4),
                "conclusion": payload.get("conclusion"),
                "dead_in_top10_structure": (payload.get("confusion", {}) or {}).get("structure", {}).get("dead_in_topk"),
                "dead_in_top10_robust": (payload.get("confusion", {}) or {}).get("robustness", {}).get("dead_in_topk"),
                "dead_in_top10_zeta": (payload.get("confusion", {}) or {}).get("zeta_structure", {}).get("dead_in_topk"),
                "dead_in_top10_spectral": (payload.get("confusion", {}) or {}).get("spectral_identity", {}).get("dead_in_topk"),
                "dead_in_top10_continuation": (payload.get("confusion", {}) or {}).get("continuation", {}).get("dead_in_topk"),
                "drop_best_t_align": _extract_ablation(payload, "drop_best_t_align"),
                "drop_mirror": _extract_ablation(payload, "drop_mirror"),
                "drop_divisor": _extract_ablation(payload, "drop_divisor"),
                "drop_FE": _extract_ablation(payload, "drop_FE"),
                "drop_unitarity": _extract_ablation(payload, "drop_unitarity"),
                "drop_hermitian": _extract_ablation(payload, "drop_hermitian"),
                "drop_cont_t": _extract_ablation(payload, "drop_cont_t"),
                "drop_cont_sigma": _extract_ablation(payload, "drop_cont_sigma"),
                "support_mirror_available": support_counts.get("mirror_available"),
                "support_divisor_available": support_counts.get("divisor_available"),
                "support_both_unavailable": support_counts.get("both_unavailable"),
                "syn_any_in_top10": syn.get("any_in_top10"),
                "syn_n_in_top10": syn.get("n_in_top10"),
                "syn_min_rank": syn.get("min_rank"),
                "syn_plausible_any_in_top10": syn_plausible.get("any_in_top10"),
                "syn_plausible_n_in_top10": syn_plausible.get("n_in_top10"),
                "syn_plausible_min_rank": syn_plausible.get("min_rank"),
                "mirror_anchor_active_n": anchor_active_n,
                "mirror_anchor_ablate_spearman": anchor_ab_spearman,
                "mirror_anchor_ablate_top10_jacc": anchor_ab_top10_jacc,
                "syn_plausible_any_in_top10_vs_anchor": syn_plaus_any_top10_vs_anchor,
                "syn_plausible_min_rank_vs_anchor": syn_plaus_min_rank_vs_anchor,
                "top10_label_mix": _format_label_mix(top10_labels),
            }
        )

        # Per-family regime summary rows (already computed by validator).
        for rr in payload.get("support_regime_summary", []) or []:
            rrd = dict(rr)
            rrd["family"] = rd.parent.name
            rrd["run_dir"] = str(rd.relative_to(root) if rd.is_relative_to(root) else rd)
            regime_rows_all.append(rrd)

        # Aggregate synthetic controls (hard/impostor cases live in the validator's synthetic CSV).
        syn_csv = rd / f"{out_prefix}_synthetic_smoothness_controls.csv"
        if syn_csv.exists():
            syn_df = pd.read_csv(syn_csv)
            if not syn_df.empty:
                syn_df.insert(0, "family", rd.parent.name)
                syn_df.insert(1, "run_dir", str(rd.relative_to(root) if rd.is_relative_to(root) else rd))
                syn_hard_rows.append(syn_df)

        # Family-level digest (structured latent/operator regime framing).
        dom = _dominant_signal(payload)
        combo_mean = float(pd.to_numeric(df.get("combo", pd.Series([], dtype=float)), errors="coerce").mean()) if not df.empty else float("nan")
        topk_mean = float(pd.to_numeric(top10.get("combo", pd.Series([], dtype=float)), errors="coerce").mean()) if not top10.empty else float("nan")
        frac_mirror = float((support_counts.get("mirror_available") or 0) / max(n_basket_i, 1))
        frac_div = float((support_counts.get("divisor_available") or 0) / max(n_basket_i, 1))
        frac_unavail = float((support_counts.get("both_unavailable") or 0) / max(n_basket_i, 1))
        bench = (
            f"conf={confidence_total:.3f} syn_plaus_min_rank={syn_plausible.get('min_rank')} "
            f"support(mir={frac_mirror:.2f},div={frac_div:.2f}) dom={dom}"
        )
        family_digest_rows.append(
            {
                "family": rd.parent.name,
                "run_dir": str(rd.relative_to(root) if rd.is_relative_to(root) else rd),
                "selector_hash12": sel.get("hash12"),
                "n_basket": n_basket_i,
                "confidence_total": round(confidence_total, 4),
                "confidence_support": round(confidence_support, 4),
                "dead_in_top10_combo": (payload.get("confusion", {}) or {}).get("structure", {}).get("dead_in_topk"),
                "syn_plausible_any_in_top10": syn_plausible.get("any_in_top10"),
                "syn_plausible_min_rank": syn_plausible.get("min_rank"),
                "mirror_anchor_active_n": anchor_active_n,
                "mirror_anchor_ablate_spearman": anchor_ab_spearman,
                "mirror_anchor_ablate_top10_jacc": anchor_ab_top10_jacc,
                "syn_plausible_any_in_top10_vs_anchor": syn_plaus_any_top10_vs_anchor,
                "syn_plausible_min_rank_vs_anchor": syn_plaus_min_rank_vs_anchor,
                "support_frac_mirror_available": round(frac_mirror, 4),
                "support_frac_divisor_available": round(frac_div, 4),
                "support_frac_both_unavailable": round(frac_unavail, 4),
                "combo_mean": round(combo_mean, 6) if combo_mean == combo_mean else float("nan"),
                "combo_top10_mean": round(topk_mean, 6) if topk_mean == topk_mean else float("nan"),
                "dominant_signal_type": dom,
                "benchmark_one_liner": bench,
            }
        )

    out_csv = (root / str(args.out_csv)).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")

    out_regime_csv = (root / str(args.out_regime_csv)).resolve()
    pd.DataFrame(regime_rows_all).to_csv(out_regime_csv, index=False)
    print(f"Wrote: {out_regime_csv}")

    out_family_csv = (root / str(args.out_family_digest_csv)).resolve()
    pd.DataFrame(family_digest_rows).to_csv(out_family_csv, index=False)
    print(f"Wrote: {out_family_csv}")

    out_syn_csv = (root / str(args.out_syn_hard_csv)).resolve()
    if syn_hard_rows:
        pd.concat(syn_hard_rows, ignore_index=True).to_csv(out_syn_csv, index=False)
    else:
        pd.DataFrame([]).to_csv(out_syn_csv, index=False)
    print(f"Wrote: {out_syn_csv}")

    # Short cross-basket report.
    fam_df = pd.DataFrame(family_digest_rows)
    hi = fam_df.loc[pd.to_numeric(fam_df.get("confidence_total"), errors="coerce") >= 0.7]
    fooled = fam_df.loc[fam_df.get("syn_plausible_any_in_top10") == True]
    report = {
        "selector": "selector_v1",
        "n_families": int(len(fam_df)),
        "high_confidence_families": hi.get("family", []).tolist() if not hi.empty else [],
        "support_limited_families": fam_df.loc[pd.to_numeric(fam_df.get("confidence_support"), errors="coerce") < 0.4].get("family", []).tolist() if not fam_df.empty else [],
        "plausible_synthetic_top10_families": fooled.get("family", []).tolist() if not fooled.empty else [],
        "dominant_signal_counts": dict(pd.Series(fam_df.get("dominant_signal_type", [])).value_counts().to_dict()) if not fam_df.empty else {},
    }
    out_rj = (root / str(args.out_report_json)).resolve()
    out_rj.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote: {out_rj}")

    out_rt = (root / str(args.out_report_txt)).resolve()
    lines = []
    lines.append("Cross-basket selector_v1 report")
    lines.append(f"Families evaluated: {report['n_families']}")
    lines.append(f"High-confidence: {', '.join(report['high_confidence_families']) if report['high_confidence_families'] else '(none)'}")
    lines.append(f"Support-limited: {', '.join(report['support_limited_families']) if report['support_limited_families'] else '(none)'}")
    lines.append(f"Fooled by plausible synthetics (top10): {', '.join(report['plausible_synthetic_top10_families']) if report['plausible_synthetic_top10_families'] else '(none)'}")
    lines.append(f"Dominant signals: {report['dominant_signal_counts']}")
    out_rt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {out_rt}")


if __name__ == "__main__":
    main()
