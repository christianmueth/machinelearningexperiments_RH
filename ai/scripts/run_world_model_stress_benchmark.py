from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent


TYPE_BASE_WEIGHTS = {
    "mixed": 1.0,
    "dependency": 0.85,
    "structural": 0.7,
    "arithmetic": 0.3,
    "other": 0.2,
}


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    path = _local_path(path)
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path = _local_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _local_path(raw_path: str | Path) -> Path:
    path = Path(str(raw_path).replace("\\", "/"))
    resolved = path if path.is_absolute() else REPO_ROOT / path
    if sys.platform == "win32":
        raw = str(resolved)
        if not raw.startswith("\\\\?\\") and len(raw) >= 240:
            return Path("\\\\?\\" + raw)
    return resolved


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _parse_quota_spec(raw: str) -> dict[str, int]:
    quotas: dict[str, int] = {}
    for item in _parse_csv(raw):
        if ":" not in item:
            continue
        name, count = item.split(":", 1)
        try:
            quotas[str(name).strip()] = max(0, int(count))
        except ValueError:
            continue
    return quotas


def _run_python(script_name: str, args: list[str]) -> None:
    normalized_args = [str(arg).replace("\\", "/") for arg in args]
    command = [sys.executable, str(SCRIPT_DIR / script_name), *normalized_args]
    print(f"running: {' '.join(command)}")
    subprocess.run(command, check=True, cwd=str(REPO_ROOT))


def _hardness_score(row: dict[str, object]) -> float:
    group_type = str(row.get("group_type", "other"))
    surface_margin = float(row.get("gold_minus_best_wrong_surface_score", 0.0))
    surface_hard_count = int(row.get("surface_hard_candidate_count", 0))
    ambiguous_count = len(row.get("ambiguous_candidate_indexes", []))
    ambiguity_score = float(row.get("max_ambiguity_score", 0.0))
    tag_bonus = 0.0
    tags = set(str(tag) for tag in row.get("ambiguous_tags", []))
    if "dependency_conflict" in tags:
        tag_bonus += 0.2
    if "dependency_omission" in tags:
        tag_bonus += 0.12
    if "structurally_plausible_wrong" in tags:
        tag_bonus += 0.08
    return float(
        TYPE_BASE_WEIGHTS.get(group_type, TYPE_BASE_WEIGHTS["other"])
        + 1.35 * ambiguity_score
        + 0.2 * min(surface_hard_count, 3)
        + 0.05 * min(ambiguous_count, 5)
        + tag_bonus
        - 1.5 * max(surface_margin, 0.0)
        + 0.5 * max(-surface_margin, 0.0)
    )


def _select_rows(rows: list[dict[str, object]], *, target_groups: int, type_priority: list[str], type_quotas: dict[str, int]) -> list[dict[str, object]]:
    ranked_rows = sorted(
        rows,
        key=lambda row: (
            _hardness_score(row),
            float(row.get("max_ambiguity_score", 0.0)),
            -float(row.get("gold_minus_best_wrong_surface_score", 0.0)),
            len(row.get("ambiguous_candidate_indexes", [])),
        ),
        reverse=True,
    )
    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    per_type_counts: Counter[str] = Counter()

    for group_type in type_priority:
        quota = int(type_quotas.get(group_type, 0))
        if quota <= 0:
            continue
        candidates = [row for row in ranked_rows if str(row.get("group_type", "other")) == group_type]
        for row in candidates:
            if len(selected) >= target_groups or per_type_counts[group_type] >= quota:
                break
            problem_id = str(row.get("problem_id", ""))
            if problem_id in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(problem_id)
            per_type_counts[group_type] += 1

    for row in ranked_rows:
        if len(selected) >= target_groups:
            break
        problem_id = str(row.get("problem_id", ""))
        if problem_id in selected_ids:
            continue
        selected.append(row)
        selected_ids.add(problem_id)
        per_type_counts.update([str(row.get("group_type", "other"))])
    return selected[:target_groups]


def _suite_best_text_baseline(summary: dict[str, object]) -> dict[str, float | str]:
    candidates: list[tuple[str, float]] = []
    surface = dict(summary.get("surface_text_chooser_challenger_only", {}))
    candidates.append(("surface_text_chooser", float(surface.get("group_accuracy", 0.0))))
    for run in summary.get("reranker_runs", []):
        mode = str(run.get("feature_mode", ""))
        if mode == "oracle":
            continue
        metric = float(dict(run.get("eval_metrics_challenger_only", {})).get("group_accuracy", 0.0))
        candidates.append((f"{mode}_reranker", metric))
    best_name, best_score = max(candidates, key=lambda item: item[1]) if candidates else ("", 0.0)
    return {
        "name": str(best_name),
        "group_accuracy": float(best_score),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build and evaluate a fixed-size world-model stress benchmark from the frozen structured stack.")
    ap.add_argument("--train_dataset", required=True)
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--controller_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--artifact_prefix", default="world_model_stress")
    ap.add_argument("--target_groups", type=int, default=24)
    ap.add_argument("--type_priority", default="mixed,dependency,structural")
    ap.add_argument("--type_quotas", default="mixed:12,dependency:8,structural:4")
    ap.add_argument("--group_type_allowlist", default="mixed,dependency,structural")
    ap.add_argument("--min_ambiguity_score", type=float, default=0.62)
    ap.add_argument("--min_wrong_surface_margin", type=float, default=0.0)
    ap.add_argument("--min_surface_hard_candidates", type=int, default=1)
    ap.add_argument("--min_ambiguous_candidates", type=int, default=2)
    ap.add_argument("--run_suite", action="store_true")
    args = ap.parse_args()

    out_dir = _local_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(args.artifact_prefix).strip() or "world_model_stress"
    source_jsonl = out_dir / f"{prefix}_source.jsonl"
    source_summary_json = out_dir / f"{prefix}_source_summary.json"
    benchmark_jsonl = out_dir / f"{prefix}.jsonl"
    benchmark_summary_json = out_dir / f"{prefix}_summary.json"
    suite_json = out_dir / f"{prefix}_suite.json"

    _run_python(
        "build_structural_ambiguity_benchmark.py",
        [
            "--input_jsonl",
            str(args.input_jsonl),
            "--controller_json",
            str(args.controller_json),
            "--out_jsonl",
            str(source_jsonl),
            "--out_summary_json",
            str(source_summary_json),
            "--min_ambiguity_score",
            str(float(args.min_ambiguity_score)),
            "--group_type_allowlist",
            str(args.group_type_allowlist),
            "--min_wrong_surface_margin",
            str(float(args.min_wrong_surface_margin)),
            "--min_surface_hard_candidates",
            str(int(args.min_surface_hard_candidates)),
        ],
    )

    source_rows = [
        row
        for row in _read_jsonl(source_jsonl)
        if len(row.get("ambiguous_candidate_indexes", [])) >= int(args.min_ambiguous_candidates)
    ]
    selected_rows = _select_rows(
        source_rows,
        target_groups=max(1, int(args.target_groups)),
        type_priority=_parse_csv(str(args.type_priority)),
        type_quotas=_parse_quota_spec(str(args.type_quotas)),
    )
    _write_jsonl(benchmark_jsonl, selected_rows)

    type_counter: Counter[str] = Counter(str(row.get("group_type", "other")) for row in selected_rows)
    benchmark_summary: dict[str, object] = {
        "train_dataset": str(args.train_dataset),
        "input_jsonl": str(args.input_jsonl),
        "controller_json": str(args.controller_json),
        "source_jsonl": str(source_jsonl),
        "out_jsonl": str(benchmark_jsonl),
        "target_groups": int(args.target_groups),
        "type_priority": _parse_csv(str(args.type_priority)),
        "type_quotas": _parse_quota_spec(str(args.type_quotas)),
        "group_type_allowlist": _parse_csv(str(args.group_type_allowlist)),
        "min_ambiguity_score": float(args.min_ambiguity_score),
        "min_wrong_surface_margin": float(args.min_wrong_surface_margin),
        "min_surface_hard_candidates": int(args.min_surface_hard_candidates),
        "min_ambiguous_candidates": int(args.min_ambiguous_candidates),
        "source_groups": int(len(source_rows)),
        "selected_groups": int(len(selected_rows)),
        "selected_group_types": dict(sorted(type_counter.items())),
        "mean_max_ambiguity_score": _mean([float(row.get("max_ambiguity_score", 0.0)) for row in selected_rows]),
        "mean_gold_minus_best_wrong_surface_score": _mean([float(row.get("gold_minus_best_wrong_surface_score", 0.0)) for row in selected_rows]),
        "mean_ambiguous_candidates_per_group": _mean([float(len(row.get("ambiguous_candidate_indexes", []))) for row in selected_rows]),
        "sample_problem_ids": [str(row.get("problem_id", "")) for row in selected_rows[:12]],
    }

    if bool(args.run_suite):
        _run_python(
            "run_structural_ambiguity_suite.py",
            [
                "--train_dataset",
                str(args.train_dataset),
                "--benchmark_jsonl",
                str(benchmark_jsonl),
                "--controller_json",
                str(args.controller_json),
                "--out_json",
                str(suite_json),
                "--out_dir",
                str(out_dir),
            ],
        )
        suite_summary = json.loads(suite_json.read_text(encoding="utf-8"))
        best_text_baseline = _suite_best_text_baseline(suite_summary)
        branch_ranker_accuracy = float(dict(suite_summary.get("controller_path_challenger_only", {})).get("branch_ranker_group_accuracy", 0.0))
        oracle_accuracy = float(dict(suite_summary.get("controller_path_challenger_only", {})).get("oracle_group_accuracy", 0.0))
        benchmark_summary["proof_eval"] = {
            "surface_text_chooser_challenger_only": float(dict(suite_summary.get("surface_text_chooser_challenger_only", {})).get("group_accuracy", 0.0)),
            "best_text_side_baseline": best_text_baseline,
            "oracle_path_challenger_only": oracle_accuracy,
            "branch_ranker_path_challenger_only": branch_ranker_accuracy,
            "branch_ranker_beats_best_text_baseline": bool(branch_ranker_accuracy > float(best_text_baseline.get("group_accuracy", 0.0))),
            "suite_json": str(suite_json),
        }

    benchmark_summary_json.write_text(json.dumps(benchmark_summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(benchmark_summary, indent=2))
    print(f"wrote {benchmark_summary_json}")
    print(f"wrote {benchmark_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())