from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml_oracle.datasets import load_reasoning_examples
from src.ml_oracle.translator import HeuristicAnchoredTranslator


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mean_x = _mean(xs)
    mean_y = _mean(ys)
    centered_x = [float(value - mean_x) for value in xs]
    centered_y = [float(value - mean_y) for value in ys]
    denom_x = math.sqrt(sum(value * value for value in centered_x))
    denom_y = math.sqrt(sum(value * value for value in centered_y))
    if math.isclose(denom_x, 0.0, abs_tol=1e-12) or math.isclose(denom_y, 0.0, abs_tol=1e-12):
        return 0.0
    numerator = sum(value_x * value_y for value_x, value_y in zip(centered_x, centered_y))
    return float(numerator / (denom_x * denom_y))


def _pairwise_preference(gold_value: float, wrong_values: list[float]) -> tuple[float, float]:
    if not wrong_values:
        return 0.0, 0.0
    wins = 0.0
    ties = 0.0
    for value in wrong_values:
        if gold_value > value:
            wins += 1.0
        elif math.isclose(gold_value, value, rel_tol=1e-9, abs_tol=1e-9):
            ties += 1.0
    total = float(len(wrong_values))
    return float(wins / total), float(ties / total)


def _state_edge_features(translator: HeuristicAnchoredTranslator, text: str, *, prompt: str) -> dict[str, float]:
    states = translator.reasoning_states_for_trace(text, prompt=prompt)
    total_states = max(1, len(states))
    dependency_edges = sum(len(state.dependency_step_indexes) for state in states)
    dependency_states = sum(1 for state in states if state.dependency_step_indexes)
    carried_states = sum(1 for state in states if state.carried_result)
    answer_support_states = sum(1 for state in states if state.answer_support)
    answer_support_dependency_states = sum(1 for state in states if state.answer_support and state.dependency_step_indexes)
    introduced_values = sum(len(state.introduced_values) for state in states)
    dependency_values = sum(len(state.dependency_values) for state in states)
    result_states = sum(1 for state in states if state.result is not None)

    dependency_spans: list[float] = []
    chain_depths: list[int] = []
    step_depth: dict[int, int] = {}
    for state in states:
        if state.dependency_step_indexes:
            dependency_spans.extend(float(state.step_index - dep_index) for dep_index in state.dependency_step_indexes)
            depth = 1 + max(step_depth.get(int(dep_index), 1) for dep_index in state.dependency_step_indexes)
        else:
            depth = 1
        step_depth[int(state.step_index)] = int(depth)
        chain_depths.append(int(depth))

    terminal_dependency_depth = 0.0
    if states:
        last_state = states[-1]
        if last_state.dependency_step_indexes:
            terminal_dependency_depth = float(max(last_state.step_index - dep_index for dep_index in last_state.dependency_step_indexes))

    return {
        "dependency_edge_count": float(dependency_edges),
        "dependency_state_fraction": float(dependency_states / total_states),
        "carried_result_fraction": float(carried_states / total_states),
        "answer_support_dependency_fraction": float(answer_support_dependency_states / total_states),
        "answer_support_given_dependency": float(answer_support_dependency_states / max(1, dependency_states)),
        "introduced_value_fraction": float(introduced_values / total_states),
        "dependency_value_fraction": float(dependency_values / total_states),
        "result_state_fraction": float(result_states / total_states),
        "mean_dependency_span": _mean(dependency_spans),
        "max_chain_depth": float(max(chain_depths) if chain_depths else 0.0),
        "normalized_chain_depth": float((max(chain_depths) if chain_depths else 0.0) / total_states),
        "terminal_dependency_depth": float(terminal_dependency_depth),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze explicit state-edge features on low-margin branch candidates.")
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--controller_json", required=True)
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    translator = HeuristicAnchoredTranslator()
    examples = load_reasoning_examples(str(args.input_jsonl))
    example_map = {str(example.problem_id): example for example in examples}
    report = json.loads(Path(str(args.controller_json)).read_text(encoding="utf-8"))
    threshold = float(report.get("controller_config", {}).get("controller_margin_threshold", 0.0))

    feature_names: set[str] = set()
    gold_feature_values: dict[str, list[float]] = {}
    wrong_feature_values: dict[str, list[float]] = {}
    argmax_success: dict[str, int] = {}
    argmax_total: dict[str, int] = {}
    pairwise_win_rates: dict[str, list[float]] = {}
    pairwise_tie_rates: dict[str, list[float]] = {}
    row_values: dict[str, list[float]] = {}
    row_labels: list[float] = []

    groups_total = 0
    eligible_groups = 0
    low_margin_groups = 0
    for group in report.get("group_reports", []):
        groups_total += 1
        if float(group.get("oracle_margin", 0.0)) > threshold:
            continue
        low_margin_groups += 1
        example = example_map.get(str(group.get("problem_id", "")))
        if example is None:
            continue
        branch_candidates = [candidate for candidate in group.get("candidates", []) if bool(candidate.get("in_branch", False))]
        if len(branch_candidates) < 2:
            continue
        gold_candidates = [candidate for candidate in branch_candidates if float(candidate.get("label", 0.0)) > 0.5]
        if not gold_candidates:
            continue
        eligible_groups += 1

        feature_maps: list[dict[str, float]] = []
        for candidate in branch_candidates:
            candidate_index = int(candidate.get("candidate_index", 0))
            trace = example.candidates[candidate_index]
            feature_map = _state_edge_features(translator, trace.text, prompt=example.prompt)
            feature_maps.append(feature_map)
            for key, value in feature_map.items():
                feature_names.add(str(key))
                row_values.setdefault(str(key), []).append(float(value))
            row_labels.append(float(candidate.get("label", 0.0)))

        gold_index = next(index for index, candidate in enumerate(branch_candidates) if float(candidate.get("label", 0.0)) > 0.5)
        gold_map = feature_maps[gold_index]
        wrong_maps = [feature_maps[index] for index, candidate in enumerate(branch_candidates) if float(candidate.get("label", 0.0)) <= 0.5]

        for feature_name in feature_names:
            gold_value = float(gold_map.get(feature_name, 0.0))
            wrong_values = [float(feature_map.get(feature_name, 0.0)) for feature_map in wrong_maps]
            gold_feature_values.setdefault(feature_name, []).append(gold_value)
            wrong_feature_values.setdefault(feature_name, []).extend(wrong_values)
            if wrong_values:
                win_rate, tie_rate = _pairwise_preference(gold_value, wrong_values)
                pairwise_win_rates.setdefault(feature_name, []).append(win_rate)
                pairwise_tie_rates.setdefault(feature_name, []).append(tie_rate)

            argmax_total[feature_name] = argmax_total.get(feature_name, 0) + 1
            ranked = sorted(
                range(len(branch_candidates)),
                key=lambda index: (
                    float(feature_maps[index].get(feature_name, 0.0)),
                    float(branch_candidates[index].get("full_score", 0.0)),
                ),
                reverse=True,
            )
            if float(branch_candidates[int(ranked[0])].get("label", 0.0)) > 0.5:
                argmax_success[feature_name] = argmax_success.get(feature_name, 0) + 1

    feature_rankings = []
    for feature_name in sorted(feature_names):
        feature_rankings.append(
            {
                "feature": feature_name,
                "gold_mean": _mean(gold_feature_values.get(feature_name, [])),
                "wrong_mean": _mean(wrong_feature_values.get(feature_name, [])),
                "gold_minus_wrong_mean": _mean(gold_feature_values.get(feature_name, [])) - _mean(wrong_feature_values.get(feature_name, [])),
                "argmax_group_accuracy": float(argmax_success.get(feature_name, 0) / max(1, argmax_total.get(feature_name, 0))),
                "pairwise_win_rate": _mean(pairwise_win_rates.get(feature_name, [])),
                "pairwise_tie_rate": _mean(pairwise_tie_rates.get(feature_name, [])),
                "candidate_correctness_correlation": _pearson(row_values.get(feature_name, []), row_labels),
                "groups_evaluated": int(argmax_total.get(feature_name, 0)),
            }
        )

    feature_rankings.sort(
        key=lambda row: (
            float(row["argmax_group_accuracy"]),
            float(row["pairwise_win_rate"]),
            float(row["gold_minus_wrong_mean"]),
        ),
        reverse=True,
    )

    summary = {
        "input_jsonl": str(args.input_jsonl),
        "controller_json": str(args.controller_json),
        "groups_total": int(groups_total),
        "low_margin_groups": int(low_margin_groups),
        "eligible_branch_groups": int(eligible_groups),
        "feature_rankings": feature_rankings,
    }

    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if str(args.out_json).strip():
        out_path = Path(str(args.out_json))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())