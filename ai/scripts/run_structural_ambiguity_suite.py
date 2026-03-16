from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml_oracle.datasets import load_reasoning_examples, materialize_dataset
from src.ml_oracle.frozen_oracle_client import AnchoredOracleClient
from src.ml_oracle.reranker import PairwiseMLPReranker, ranking_metrics
from src.ml_oracle.translator import HeuristicAnchoredTranslator


def _local_path(raw_path: str | Path) -> Path:
    path = Path(str(raw_path).replace("\\", "/"))
    resolved = path if path.is_absolute() else REPO_ROOT / path
    if sys.platform == "win32":
        raw = str(resolved)
        if not raw.startswith("\\\\?\\") and len(raw) >= 240:
            return Path("\\\\?\\" + raw)
    return resolved


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


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _extract_answer_line(text: str) -> str:
    match = re.search(r"(?:answer|final answer)\s*:\s*([^\n]+)", str(text), flags=re.IGNORECASE)
    return str(match.group(1)).strip() if match is not None else ""


def _extract_answer_value(text: str) -> float | None:
    answer_line = _extract_answer_line(text)
    if not answer_line:
        return None
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", answer_line)
    if not numbers:
        return None
    try:
        return float(numbers[-1].replace(",", ""))
    except ValueError:
        return None


def _reasoning_prefix(text: str) -> str:
    lowered = str(text)
    match = re.search(r"(?:answer|final answer)\s*:", lowered, flags=re.IGNORECASE)
    if match is None:
        return lowered
    return lowered[: match.start()]


def _looks_incomplete(text: str) -> bool:
    stripped = str(text).rstrip()
    if not stripped:
        return True
    if stripped.endswith((":", "$", "=", "*", "/", "+", "-", "(", ",")):
        return True
    has_answer = bool(re.search(r"(?:^|\n)\s*(?:answer|final answer)\s*:", stripped, flags=re.IGNORECASE))
    return bool(re.search(r"(?:Step\s+\d+:[^\n]*)$", stripped) and not has_answer and stripped[-1] not in ".!?")


def _equation_count(text: str) -> int:
    return len(re.findall(r"-?\d[\d,]*(?:\.\d+)?\s*[+\-*/x]\s*-?\d[\d,]*(?:\.\d+)?\s*=\s*-?\d[\d,]*(?:\.\d+)?", str(text), flags=re.IGNORECASE))


def _step_count(text: str) -> int:
    explicit = re.findall(r"(?:^|\n)\s*Step\s+\d+\s*:", str(text), flags=re.IGNORECASE)
    if explicit:
        return len(explicit)
    rough_lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    return len(rough_lines)


def _surface_text_score(text: str) -> float:
    raw = str(text)
    reasoning = _reasoning_prefix(raw)
    answer_line = _extract_answer_line(raw)
    answer_value = _extract_answer_value(raw)
    has_answer = bool(answer_line)
    incomplete = _looks_incomplete(raw)
    eq_count = _equation_count(raw)
    steps = _step_count(raw)
    reasoning_numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", reasoning)
    answer_mentioned = False
    if answer_value is not None:
        normalized_answer = str(int(answer_value)) if float(answer_value).is_integer() else str(answer_value)
        answer_mentioned = any(token.replace(",", "") == normalized_answer for token in reasoning_numbers)
    distinct_reasoning_numbers = len({token.replace(",", "") for token in reasoning_numbers})
    token_count = len(re.findall(r"\S+", raw))

    score = 0.0
    score += 1.4 if has_answer else -1.0
    score += 1.2 if not incomplete else -1.5
    score += min(float(steps), 6.0) * 0.12
    score += min(float(eq_count), 4.0) * 0.22
    score += min(float(distinct_reasoning_numbers), 8.0) * 0.05
    score += 0.45 if answer_mentioned else -0.15
    if has_answer and answer_line.endswith((".", "!", "?")):
        score += 0.05
    if token_count < 8:
        score -= 0.4
    if token_count > 120:
        score -= 0.25
    if re.search(r"\b(?:maybe|guess|probably|approximately|approx)\b", raw, flags=re.IGNORECASE):
        score -= 0.35
    return float(score)


def _dataset_matrices(
    dataset_path: str,
    *,
    client: AnchoredOracleClient,
    translator: HeuristicAnchoredTranslator,
    feature_mode: str,
    text_encoder: str,
    text_dim: int,
    hf_model: str,
    hf_max_length: int,
):
    examples = load_reasoning_examples(dataset_path)
    X, y, groups = materialize_dataset(
        examples,
        client=client,
        translator=translator,
        text_dim=int(text_dim),
        feature_mode=str(feature_mode),
        oracle_feature_groups=None,
        text_encoder_name=str(text_encoder),
        hf_model=str(hf_model),
        hf_max_length=int(hf_max_length),
    )
    return X, y, groups, examples


def _controller_subset_metrics(benchmark_rows: list[dict[str, object]], controller_report: dict[str, object]) -> dict[str, object]:
    groups_by_problem = {
        str(group.get("problem_id", "")): dict(group)
        for group in controller_report.get("group_reports", [])
    }
    benchmark_ids = [str(row.get("problem_id", "")) for row in benchmark_rows]
    subset_groups = [groups_by_problem[problem_id] for problem_id in benchmark_ids if problem_id in groups_by_problem]
    oracle_correct = [bool(group.get("baseline_top_correct", False)) for group in subset_groups]
    controller_correct = [bool(group.get("controller_top_correct", False)) for group in subset_groups]
    ranker_correct = [bool(group.get("ranker_top_correct", False)) for group in subset_groups]
    low_margin = [float(group.get("oracle_margin", 0.0)) <= float(controller_report.get("controller_config", {}).get("controller_margin_threshold", 0.0)) for group in subset_groups]

    wins_vs_controller = 0
    regressions_vs_controller = 0
    wins_vs_oracle = 0
    regressions_vs_oracle = 0
    for oracle_ok, controller_ok, ranker_ok in zip(oracle_correct, controller_correct, ranker_correct):
        if ranker_ok and not controller_ok:
            wins_vs_controller += 1
        elif controller_ok and not ranker_ok:
            regressions_vs_controller += 1
        if ranker_ok and not oracle_ok:
            wins_vs_oracle += 1
        elif oracle_ok and not ranker_ok:
            regressions_vs_oracle += 1

    low_margin_count = sum(1 for value in low_margin if value)
    return {
        "groups_evaluated": int(len(subset_groups)),
        "oracle_group_accuracy": _mean([1.0 if value else 0.0 for value in oracle_correct]),
        "controller_group_accuracy": _mean([1.0 if value else 0.0 for value in controller_correct]),
        "branch_ranker_group_accuracy": _mean([1.0 if value else 0.0 for value in ranker_correct]),
        "low_margin_groups": int(low_margin_count),
        "high_margin_groups": int(len(subset_groups) - low_margin_count),
        "branch_ranker_wins_vs_controller": int(wins_vs_controller),
        "branch_ranker_regressions_vs_controller": int(regressions_vs_controller),
        "branch_ranker_wins_vs_oracle": int(wins_vs_oracle),
        "branch_ranker_regressions_vs_oracle": int(regressions_vs_oracle),
    }


def _top_candidate_by_score(candidates: list[dict[str, object]], score_name: str) -> dict[str, object]:
    return sorted(
        candidates,
        key=lambda candidate: (
            float(candidate.get(score_name, 0.0)),
            float(candidate.get("full_score", 0.0)),
            -int(candidate.get("candidate_index", 0)),
        ),
        reverse=True,
    )[0]


def _controller_subset_metrics_challenger_only(benchmark_rows: list[dict[str, object]], controller_report: dict[str, object]) -> dict[str, object]:
    groups_by_problem = {
        str(group.get("problem_id", "")): dict(group)
        for group in controller_report.get("group_reports", [])
    }
    oracle_rows: list[tuple[str, bool]] = []
    controller_rows: list[tuple[str, bool]] = []
    ranker_rows: list[tuple[str, bool]] = []
    wins_vs_controller = 0
    regressions_vs_controller = 0
    wins_vs_oracle = 0
    regressions_vs_oracle = 0

    for row in benchmark_rows:
        problem_id = str(row.get("problem_id", ""))
        group_type = str(row.get("group_type", "other"))
        report_group = groups_by_problem.get(problem_id)
        if report_group is None:
            continue
        allowed_indexes = set(int(index) for index in row.get("ambiguous_candidate_indexes", []))
        allowed_indexes.add(int(row.get("gold_candidate_index", -1)))
        filtered_candidates = [
            dict(candidate)
            for candidate in report_group.get("candidates", [])
            if int(candidate.get("candidate_index", -1)) in allowed_indexes
        ]
        if not filtered_candidates:
            continue
        oracle_top = _top_candidate_by_score(filtered_candidates, "full_score")
        controller_top = _top_candidate_by_score(filtered_candidates, "adjusted_score")
        ranker_top = _top_candidate_by_score(filtered_candidates, "ranker_adjusted_score")
        oracle_correct = float(oracle_top.get("label", 0.0)) > 0.5
        controller_correct = float(controller_top.get("label", 0.0)) > 0.5
        ranker_correct = float(ranker_top.get("label", 0.0)) > 0.5
        oracle_rows.append((group_type, oracle_correct))
        controller_rows.append((group_type, controller_correct))
        ranker_rows.append((group_type, ranker_correct))
        if ranker_correct and not controller_correct:
            wins_vs_controller += 1
        elif controller_correct and not ranker_correct:
            regressions_vs_controller += 1
        if ranker_correct and not oracle_correct:
            wins_vs_oracle += 1
        elif oracle_correct and not ranker_correct:
            regressions_vs_oracle += 1

    oracle_flags = [is_correct for _, is_correct in oracle_rows]
    controller_flags = [is_correct for _, is_correct in controller_rows]
    ranker_flags = [is_correct for _, is_correct in ranker_rows]
    return {
        "groups_evaluated": int(len(ranker_rows)),
        "oracle_group_accuracy": _mean([1.0 if value else 0.0 for value in oracle_flags]),
        "controller_group_accuracy": _mean([1.0 if value else 0.0 for value in controller_flags]),
        "branch_ranker_group_accuracy": _mean([1.0 if value else 0.0 for value in ranker_flags]),
        "branch_ranker_wins_vs_controller": int(wins_vs_controller),
        "branch_ranker_regressions_vs_controller": int(regressions_vs_controller),
        "branch_ranker_wins_vs_oracle": int(wins_vs_oracle),
        "branch_ranker_regressions_vs_oracle": int(regressions_vs_oracle),
        "by_group_type": {
            "oracle": _group_type_breakdown_from_flags(oracle_rows),
            "controller": _group_type_breakdown_from_flags(controller_rows),
            "branch_ranker": _group_type_breakdown_from_flags(ranker_rows),
        },
    }


def _group_type_breakdown_from_flags(rows: list[tuple[str, bool]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = {}
    for group_type, is_correct in rows:
        grouped.setdefault(str(group_type), []).append(1.0 if bool(is_correct) else 0.0)
    return {
        group_type: {
            "groups": int(len(values)),
            "group_accuracy": _mean(values),
        }
        for group_type, values in sorted(grouped.items())
    }


def _surface_text_chooser_metrics(benchmark_rows: list[dict[str, object]]) -> dict[str, object]:
    correctness_rows: list[tuple[str, bool]] = []
    sample_rows: list[dict[str, object]] = []
    for row in benchmark_rows:
        candidates = list(row.get("candidates", []))
        if not candidates:
            continue
        scored = []
        for candidate in candidates:
            scored.append((float(_surface_text_score(str(candidate.get("text", "")))), dict(candidate)))
        scored.sort(
            key=lambda pair: (
                float(pair[0]),
                float(pair[1].get("ambiguity_score", 0.0)),
                -int(pair[1].get("candidate_index", 0)),
            ),
            reverse=True,
        )
        chosen_score, chosen = scored[0]
        is_correct = float(chosen.get("label", 0.0)) > 0.5
        group_type = str(row.get("group_type", "other"))
        correctness_rows.append((group_type, bool(is_correct)))
        if len(sample_rows) < 8:
            sample_rows.append(
                {
                    "problem_id": str(row.get("problem_id", "")),
                    "group_type": group_type,
                    "chosen_candidate_index": int(chosen.get("candidate_index", -1)),
                    "chosen_label": float(chosen.get("label", 0.0)),
                    "chosen_surface_score": float(chosen_score),
                }
            )
    return {
        "groups_evaluated": int(len(correctness_rows)),
        "group_accuracy": _mean([1.0 if is_correct else 0.0 for _, is_correct in correctness_rows]),
        "by_group_type": _group_type_breakdown_from_flags(correctness_rows),
        "sample_choices": sample_rows,
    }


def _challenger_only_rows(benchmark_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    challenger_rows: list[dict[str, object]] = []
    for row in benchmark_rows:
        allowed_indexes = set(int(index) for index in row.get("ambiguous_candidate_indexes", []))
        allowed_indexes.add(int(row.get("gold_candidate_index", -1)))
        filtered_candidates = [
            {
                "text": str(candidate.get("text", "")),
                "label": float(candidate.get("label", 0.0)),
            }
            for candidate in row.get("candidates", [])
            if int(candidate.get("candidate_index", -1)) in allowed_indexes
        ]
        challenger_rows.append(
            {
                "problem_id": str(row.get("problem_id", "")),
                "prompt": str(row.get("prompt", "")),
                "candidates": filtered_candidates,
            }
        )
    return challenger_rows


def _scores_group_type_breakdown(
    scores,
    labels,
    group_ids,
    eval_examples,
    benchmark_group_types: dict[str, str],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = {}
    unique_groups = sorted(set(int(value) for value in group_ids.tolist()))
    for group_index in unique_groups:
        idx = [position for position, value in enumerate(group_ids.tolist()) if int(value) == int(group_index)]
        if not idx:
            continue
        local_scores = [(float(scores[position]), int(position)) for position in idx]
        local_scores.sort(key=lambda pair: pair[0], reverse=True)
        pred_position = int(local_scores[0][1])
        pred_correct = float(labels[pred_position]) > 0.5
        problem_id = str(eval_examples[int(group_index)].problem_id)
        group_type = benchmark_group_types.get(problem_id, "other")
        grouped.setdefault(group_type, []).append(1.0 if pred_correct else 0.0)
    return {
        group_type: {
            "groups": int(len(values)),
            "group_accuracy": _mean(values),
        }
        for group_type, values in sorted(grouped.items())
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run text/oracle/fused reranker baselines plus controller-path metrics on the structural ambiguity benchmark.")
    ap.add_argument("--train_dataset", required=True)
    ap.add_argument("--benchmark_jsonl", required=True)
    ap.add_argument("--controller_json", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_dir", default="out/ai")
    ap.add_argument("--feature_modes", nargs="+", default=["text", "oracle", "text+oracle"])
    ap.add_argument("--text_encoder", choices=["hashed", "hf"], default="hashed")
    ap.add_argument("--text_dim", type=int, default=256)
    ap.add_argument("--hf_model", default="")
    ap.add_argument("--hf_max_length", type=int, default=256)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    client = AnchoredOracleClient()
    translator = HeuristicAnchoredTranslator()
    out_dir = _local_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark_rows = _read_jsonl(_local_path(args.benchmark_jsonl))
    controller_report = json.loads(_local_path(args.controller_json).read_text(encoding="utf-8"))

    tag_counter: Counter[str] = Counter()
    type_counter: Counter[str] = Counter()
    ambiguous_candidates_per_group: list[float] = []
    benchmark_group_types = {str(row.get("problem_id", "")): str(row.get("group_type", "other")) for row in benchmark_rows}
    for row in benchmark_rows:
        type_counter.update([str(row.get("group_type", "other"))])
        tag_counter.update(list(row.get("ambiguous_tags", [])))
        ambiguous_candidates_per_group.append(float(len(row.get("ambiguous_candidate_indexes", []))))

    challenger_rows = _challenger_only_rows(benchmark_rows)
    out_path = _local_path(args.out_json)
    challenger_jsonl = out_path.with_name(f"{out_path.stem}_challenger_only.jsonl")
    _write_jsonl(challenger_jsonl, challenger_rows)

    summary: dict[str, object] = {
        "train_dataset": str(args.train_dataset),
        "benchmark_jsonl": str(args.benchmark_jsonl),
        "challenger_only_jsonl": str(challenger_jsonl),
        "controller_json": str(args.controller_json),
        "benchmark_composition": {
            "groups_total": int(len(benchmark_rows)),
            "group_types": dict(sorted(type_counter.items())),
            "ambiguity_tag_counts": dict(sorted(tag_counter.items())),
            "mean_ambiguous_candidates_per_group": _mean(ambiguous_candidates_per_group),
        },
        "controller_path": _controller_subset_metrics(benchmark_rows, controller_report),
        "controller_path_challenger_only": _controller_subset_metrics_challenger_only(benchmark_rows, controller_report),
        "surface_text_chooser": _surface_text_chooser_metrics(benchmark_rows),
        "surface_text_chooser_challenger_only": _surface_text_chooser_metrics(benchmark_rows=[
            {
                **row,
                "candidates": [
                    candidate
                    for candidate in row.get("candidates", [])
                    if int(candidate.get("candidate_index", -1)) in set(int(index) for index in row.get("ambiguous_candidate_indexes", [])) | {int(row.get("gold_candidate_index", -1))}
                ],
            }
            for row in benchmark_rows
        ]),
        "reranker_runs": [],
    }

    controller_groups_by_problem = {
        str(group.get("problem_id", "")): dict(group)
        for group in controller_report.get("group_reports", [])
    }
    controller_rows: list[tuple[str, bool]] = []
    oracle_rows: list[tuple[str, bool]] = []
    ranker_rows: list[tuple[str, bool]] = []
    for row in benchmark_rows:
        problem_id = str(row.get("problem_id", ""))
        group_type = benchmark_group_types.get(problem_id, "other")
        group = controller_groups_by_problem.get(problem_id)
        if group is None:
            continue
        oracle_rows.append((group_type, bool(group.get("baseline_top_correct", False))))
        controller_rows.append((group_type, bool(group.get("controller_top_correct", False))))
        ranker_rows.append((group_type, bool(group.get("ranker_top_correct", False))))
    summary["controller_path"]["by_group_type"] = {
        "oracle": _group_type_breakdown_from_flags(oracle_rows),
        "controller": _group_type_breakdown_from_flags(controller_rows),
        "branch_ranker": _group_type_breakdown_from_flags(ranker_rows),
    }

    X_eval_challenger_cache = None
    y_eval_challenger_cache = None
    groups_eval_challenger_cache = None
    eval_examples_challenger_cache = None

    for mode in args.feature_modes:
        sanitized_mode = str(mode).replace("+", "_plus_")
        X_train, y_train, groups_train, train_examples = _dataset_matrices(
            str(args.train_dataset),
            client=client,
            translator=translator,
            feature_mode=str(mode),
            text_encoder=str(args.text_encoder),
            text_dim=int(args.text_dim),
            hf_model=str(args.hf_model),
            hf_max_length=int(args.hf_max_length),
        )
        X_eval, y_eval, groups_eval, eval_examples = _dataset_matrices(
            str(args.benchmark_jsonl),
            client=client,
            translator=translator,
            feature_mode=str(mode),
            text_encoder=str(args.text_encoder),
            text_dim=int(args.text_dim),
            hf_model=str(args.hf_model),
            hf_max_length=int(args.hf_max_length),
        )
        X_eval_challenger, y_eval_challenger, groups_eval_challenger, eval_examples_challenger = _dataset_matrices(
            str(challenger_jsonl),
            client=client,
            translator=translator,
            feature_mode=str(mode),
            text_encoder=str(args.text_encoder),
            text_dim=int(args.text_dim),
            hf_model=str(args.hf_model),
            hf_max_length=int(args.hf_max_length),
        )
        model = PairwiseMLPReranker(input_dim=int(X_train.shape[1]), hidden_dim=int(args.hidden_dim), seed=int(args.seed))
        history = model.fit(
            X_train,
            y_train,
            groups_train,
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )
        model_path = out_dir / f"reranker_{sanitized_mode}.npz"
        model.save(str(model_path))
        train_scores = model.score(X_train)
        eval_scores = model.score(X_eval)
        eval_scores_challenger = model.score(X_eval_challenger)
        train_metrics = ranking_metrics(train_scores, y_train, groups_train)
        eval_metrics = ranking_metrics(eval_scores, y_eval, groups_eval)
        eval_metrics_challenger = ranking_metrics(eval_scores_challenger, y_eval_challenger, groups_eval_challenger)
        summary["reranker_runs"].append(
            {
                "feature_mode": str(mode),
                "model_path": str(model_path),
                "train_examples": int(len(train_examples)),
                "eval_examples": int(len(eval_examples)),
                "eval_examples_challenger_only": int(len(eval_examples_challenger)),
                "train_loss_last": float(history[-1] if history else 0.0),
                "train_metrics": train_metrics,
                "eval_metrics": eval_metrics,
                "eval_metrics_challenger_only": eval_metrics_challenger,
                "eval_by_group_type": _scores_group_type_breakdown(eval_scores, y_eval, groups_eval, eval_examples, benchmark_group_types),
                "eval_by_group_type_challenger_only": _scores_group_type_breakdown(
                    eval_scores_challenger,
                    y_eval_challenger,
                    groups_eval_challenger,
                    eval_examples_challenger,
                    benchmark_group_types,
                ),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(summary, indent=2)
    out_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())