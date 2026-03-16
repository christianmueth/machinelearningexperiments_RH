from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml_oracle.datasets import load_reasoning_examples, materialize_dataset
from src.ml_oracle.frozen_oracle_client import AnchoredOracleClient
from src.ml_oracle.reranker import PairwiseMLPReranker, ranking_metrics
from src.ml_oracle.translator import HeuristicAnchoredTranslator


DEFAULT_BRANCH_RANKER_FEATURE_NAMES = (
    "oracle_score",
    "result_reuse_fraction",
    "result_reuse_count",
    "answer_support_fraction",
    "answer_support_count",
)

STATE_EDGE_BRANCH_RANKER_FEATURE_NAMES = (
    "dependency_edge_count",
    "dependency_state_fraction",
    "dependency_value_fraction",
    "answer_support_dependency_fraction",
    "answer_support_given_dependency",
    "max_chain_depth",
    "normalized_chain_depth",
    "terminal_dependency_depth",
)

ALLOWED_BRANCH_RANKER_FEATURE_NAMES = set(DEFAULT_BRANCH_RANKER_FEATURE_NAMES) | set(STATE_EDGE_BRANCH_RANKER_FEATURE_NAMES)


def _parse_branch_ranker_feature_names(raw: str) -> tuple[str, ...]:
    if not str(raw).strip():
        return DEFAULT_BRANCH_RANKER_FEATURE_NAMES
    feature_names = tuple(item.strip() for item in str(raw).split(",") if item.strip())
    invalid = [name for name in feature_names if name not in ALLOWED_BRANCH_RANKER_FEATURE_NAMES]
    if invalid:
        raise ValueError(f"Unsupported branch ranker feature names: {', '.join(invalid)}")
    return feature_names if feature_names else DEFAULT_BRANCH_RANKER_FEATURE_NAMES


def _jsonable_signature(signature: tuple[tuple[object, ...], ...]) -> list[list[object]]:
    return [list(row) for row in signature]


def _prefix_texts(translator: HeuristicAnchoredTranslator, candidate_text: str) -> tuple[str, ...]:
    steps = translator.split_reasoning_steps(candidate_text)
    if not steps:
        cleaned = str(candidate_text).strip()
        return (cleaned,) if cleaned else ()
    prefixes: list[str] = []
    for index in range(len(steps)):
        prefixes.append("Reasoning:\n" + "\n".join(steps[: index + 1]))
    return tuple(prefixes)


def _oracle_vector_for_text(
    text: str,
    *,
    prompt: str,
    client: AnchoredOracleClient,
    translator: HeuristicAnchoredTranslator,
) -> np.ndarray:
    query = translator.query_for_trace(text, prompt=prompt)
    return client.oracle_vector(query)


def _group_margin(scores: np.ndarray) -> float:
    local = np.sort(np.asarray(scores, dtype=np.float64))[::-1]
    if local.size <= 1:
        return 0.0
    return float(local[0] - local[1])


def _structured_tiebreak_features(
    translator: HeuristicAnchoredTranslator,
    candidate_text: str,
    *,
    prompt: str,
) -> dict[str, float]:
    steps = translator.split_reasoning_steps(candidate_text)
    states = translator.reasoning_states_for_trace(candidate_text, prompt=prompt)
    answer_value = translator._answer_value(candidate_text)
    reasoning_text = translator._reasoning_text(candidate_text)
    reasoning_lines = [line.strip() for line in reasoning_text.splitlines() if line.strip()]

    valid_states = 0
    answer_support_count = 0
    correct_equations = 0
    incorrect_equations = 0
    reused_results = 0
    approx_states = 0
    final_resolution = 0.0
    dependency_edges = 0
    dependency_states = 0
    answer_support_dependency_states = 0
    introduced_values = 0
    dependency_values = 0
    result_states = 0
    carried_states = 0
    dependency_spans: list[float] = []
    chain_depths: list[int] = []
    step_depth: dict[int, int] = {}

    for index, state in enumerate(states):
        has_arithmetic_shape = bool(
            state.equation_correct is not False and (state.equation_text or state.result is not None or len(state.operands) >= 2)
        )
        if has_arithmetic_shape:
            valid_states += 1
        if state.answer_support:
            answer_support_count += 1
        if state.equation_correct is True:
            correct_equations += 1
        elif state.equation_correct is False:
            incorrect_equations += 1
        if state.contains_approx_language:
            approx_states += 1
        if state.dependency_step_indexes:
            dependency_states += 1
            dependency_edges += len(state.dependency_step_indexes)
            dependency_spans.extend(float(state.step_index - dep_index) for dep_index in state.dependency_step_indexes)
        if state.answer_support and state.dependency_step_indexes:
            answer_support_dependency_states += 1
        if state.carried_result:
            carried_states += 1
        introduced_values += len(state.introduced_values)
        dependency_values += len(state.dependency_values)
        if state.result is not None:
            result_states += 1
        if state.dependency_step_indexes:
            depth = 1 + max(step_depth.get(int(dep_index), 1) for dep_index in state.dependency_step_indexes)
        else:
            depth = 1
        step_depth[int(state.step_index)] = int(depth)
        chain_depths.append(int(depth))

        if state.result is not None:
            later_states = states[index + 1 :]
            if any(
                any(math.isclose(float(state.result), float(quantity), rel_tol=1e-6, abs_tol=1e-6) for quantity in later.quantities)
                for later in later_states
            ):
                reused_results += 1

    if states:
        last_state = states[-1]
        if last_state.answer_support:
            final_resolution += 1.0
        if last_state.equation_correct is True:
            final_resolution += 0.5
        elif last_state.equation_correct is False:
            final_resolution -= 0.5
        if answer_value is not None and any(
            math.isclose(float(answer_value), float(quantity), rel_tol=1e-6, abs_tol=1e-6) for quantity in last_state.quantities
        ):
            final_resolution += 0.5

    terminal_dependency_depth = 0.0
    if states:
        last_state = states[-1]
        if last_state.dependency_step_indexes:
            terminal_dependency_depth = float(max(last_state.step_index - dep_index for dep_index in last_state.dependency_step_indexes))

    total_states = max(1, len(states))
    total_steps = max(1, len(steps))
    total_equations = max(1, correct_equations + incorrect_equations)
    total_lines = max(1, len(reasoning_lines))
    return {
        "step_count": float(len(steps)),
        "state_count": float(len(states)),
        "reasoning_line_count": float(len(reasoning_lines)),
        "structured_step_fraction": float(len(steps) / total_lines),
        "valid_state_count": float(valid_states),
        "valid_state_fraction": float(valid_states / total_states),
        "answer_support_count": float(answer_support_count),
        "answer_support_fraction": float(answer_support_count / total_states),
        "correct_equation_count": float(correct_equations),
        "incorrect_equation_count": float(incorrect_equations),
        "equation_consistency": float((correct_equations - incorrect_equations) / total_equations),
        "result_reuse_count": float(reused_results),
        "result_reuse_fraction": float(reused_results / total_states),
        "step_density": float(len(states) / total_steps),
        "final_resolution": float(final_resolution),
        "approx_fraction": float(approx_states / total_states),
        "dependency_edge_count": float(dependency_edges),
        "dependency_state_fraction": float(dependency_states / total_states),
        "carried_result_fraction": float(carried_states / total_states),
        "answer_support_dependency_fraction": float(answer_support_dependency_states / total_states),
        "answer_support_given_dependency": float(answer_support_dependency_states / max(1, dependency_states)),
        "introduced_value_fraction": float(introduced_values / total_states),
        "dependency_value_fraction": float(dependency_values / total_states),
        "result_state_fraction": float(result_states / total_states),
        "mean_dependency_span": float(sum(dependency_spans) / len(dependency_spans)) if dependency_spans else 0.0,
        "max_chain_depth": float(max(chain_depths) if chain_depths else 0.0),
        "normalized_chain_depth": float((max(chain_depths) if chain_depths else 0.0) / total_states),
        "terminal_dependency_depth": float(terminal_dependency_depth),
    }


def _branch_ranker_feature_vector(candidate_report: dict[str, object], feature_names: tuple[str, ...]) -> np.ndarray:
    tiebreak_features = candidate_report.get("tiebreak_features", {})
    if not isinstance(tiebreak_features, dict):
        tiebreak_features = {}
    values: list[float] = []
    for feature_name in feature_names:
        if feature_name == "oracle_score":
            values.append(float(candidate_report.get("full_score", 0.0)))
        else:
            values.append(float(tiebreak_features.get(feature_name, 0.0)))
    return np.asarray(values, dtype=np.float64)


def _standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    if X.size == 0:
        return np.zeros((0,), dtype=np.float64), np.ones((0,), dtype=np.float64)
    mean = np.mean(X, axis=0)
    scale = np.std(X, axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    return mean.astype(np.float64), scale.astype(np.float64)


def _standardize_apply(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.size == 0:
        return X.astype(np.float64)
    return ((X - mean) / scale).astype(np.float64)


def _choose_signature_representative(candidate_reports: list[dict[str, object]], local_indexes: list[int]) -> int:
    ranked = sorted(
        [int(local_index) for local_index in local_indexes],
        key=lambda local_index: (
            float(candidate_reports[local_index].get("adjusted_score", 0.0)),
            float(candidate_reports[local_index].get("full_score", 0.0)),
            -int(candidate_reports[local_index].get("candidate_index", local_index)),
        ),
        reverse=True,
    )
    return int(ranked[0])


def _dedup_branch_locals(
    candidate_reports: list[dict[str, object]],
    branch_locals: list[int],
) -> tuple[list[int], dict[int, list[int]], int]:
    grouped: dict[tuple[tuple[object, ...], ...], list[int]] = {}
    for local_index in branch_locals:
        signature = tuple(
            tuple(row) if isinstance(row, list) else tuple(row)
            for row in candidate_reports[int(local_index)].get("state_signature", [])
        )
        grouped.setdefault(signature, []).append(int(local_index))

    representative_map: dict[int, list[int]] = {}
    duplicate_count = 0
    for local_indexes in grouped.values():
        representative = _choose_signature_representative(candidate_reports, local_indexes)
        representative_map[int(representative)] = sorted(int(local_index) for local_index in local_indexes)
        duplicate_count += max(0, len(local_indexes) - 1)

    representative_locals = sorted(
        representative_map.keys(),
        key=lambda local_index: (
            float(candidate_reports[local_index].get("adjusted_score", 0.0)),
            float(candidate_reports[local_index].get("full_score", 0.0)),
            -int(candidate_reports[local_index].get("candidate_index", local_index)),
        ),
        reverse=True,
    )
    return representative_locals, representative_map, int(duplicate_count)


def _rerank_branch_band(
    adjusted_scores: np.ndarray,
    branch_locals: list[int],
    branch_ranker_scores: np.ndarray,
) -> np.ndarray:
    rescored = np.asarray(adjusted_scores, dtype=np.float64).copy()
    if len(branch_locals) <= 1:
        return rescored
    branch_values = np.asarray([float(adjusted_scores[local_index]) for local_index in branch_locals], dtype=np.float64)
    upper = float(np.max(branch_values) + 1e-6)
    lower = float(np.min(branch_values) - 1e-6)
    assigned_scores = np.linspace(upper, lower, num=len(branch_locals), dtype=np.float64)
    ranked_positions = sorted(
        range(len(branch_locals)),
        key=lambda index: (
            float(branch_ranker_scores[index]),
            float(adjusted_scores[int(branch_locals[index])]),
        ),
        reverse=True,
    )
    for rank_index, position in enumerate(ranked_positions):
        rescored[int(branch_locals[position])] = float(assigned_scores[rank_index])
    return rescored


def _apply_representative_scores_to_duplicates(
    candidate_reports: list[dict[str, object]],
    reranked_scores: np.ndarray,
    representative_map: dict[int, list[int]],
) -> np.ndarray:
    rescored = np.asarray(reranked_scores, dtype=np.float64).copy()
    for representative, local_indexes in representative_map.items():
        representative_score = float(rescored[int(representative)])
        for local_index in local_indexes:
            rescored[int(local_index)] = representative_score
            candidate_reports[int(local_index)]["ranker_adjusted_score"] = representative_score
            candidate_reports[int(local_index)]["dedup_representative_index"] = int(representative)
            candidate_reports[int(local_index)]["dedup_is_representative"] = bool(int(local_index) == int(representative))
    return rescored


def _apply_signature_consensus_scores(
    candidate_reports: list[dict[str, object]],
    adjusted_scores: np.ndarray,
    reranked_scores: np.ndarray,
    representative_map: dict[int, list[int]],
) -> np.ndarray:
    rescored = np.asarray(reranked_scores, dtype=np.float64).copy()
    for representative, local_indexes in representative_map.items():
        representative_score = float(rescored[int(representative)])
        ordered_locals = sorted(
            [int(local_index) for local_index in local_indexes],
            key=lambda local_index: (
                float(adjusted_scores[int(local_index)]),
                float(candidate_reports[int(local_index)].get("full_score", 0.0)),
                -int(candidate_reports[int(local_index)].get("candidate_index", local_index)),
            ),
            reverse=True,
        )
        for offset, local_index in enumerate(ordered_locals):
            projected_score = float(representative_score - (offset * 1e-9))
            rescored[int(local_index)] = projected_score
            candidate_reports[int(local_index)]["ranker_adjusted_score"] = projected_score
            candidate_reports[int(local_index)]["dedup_representative_index"] = int(representative)
            candidate_reports[int(local_index)]["dedup_is_representative"] = bool(int(local_index) == int(representative))
    return rescored


def _controller_diagnostics_for_candidate(
    example_prompt: str,
    candidate_text: str,
    *,
    model: PairwiseMLPReranker,
    client: AnchoredOracleClient,
    translator: HeuristicAnchoredTranslator,
) -> dict[str, object]:
    prefix_scores: list[float] = []
    for prefix_text in _prefix_texts(translator, candidate_text):
        vector = _oracle_vector_for_text(prefix_text, prompt=example_prompt, client=client, translator=translator)
        prefix_score = float(model.score(np.asarray([vector], dtype=np.float64))[0])
        prefix_scores.append(prefix_score)
    progress = float(prefix_scores[-1] - prefix_scores[0]) if len(prefix_scores) >= 2 else 0.0
    answer_support = float(max(0.0, translator._answer_support_score(candidate_text)))
    step_query_count = int(len(translator.queries_for_trace(candidate_text, prompt=example_prompt)))
    return {
        "prefix_scores": prefix_scores,
        "progress": progress,
        "answer_support": answer_support,
        "step_query_count": step_query_count,
        "tiebreak_features": _structured_tiebreak_features(
            translator,
            candidate_text,
            prompt=example_prompt,
        ),
    }


def _score_dataset_with_controller(
    examples,
    base_scores: np.ndarray,
    labels: np.ndarray,
    *,
    model: PairwiseMLPReranker,
    client: AnchoredOracleClient,
    translator: HeuristicAnchoredTranslator,
    controller_margin_threshold: float,
    controller_top_k: int,
    controller_score_window: float,
    progress_weight: float,
    answer_support_weight: float,
    step_query_weight: float,
    controller_step_query_delta: int,
    controller_step_query_window: float,
    valid_state_weight: float,
    answer_support_fraction_weight: float,
    equation_consistency_weight: float,
    result_reuse_weight: float,
    final_resolution_weight: float,
    approx_penalty_weight: float,
    branch_ranker_feature_names: tuple[str, ...],
    branch_ranker_model: PairwiseMLPReranker | None = None,
    branch_ranker_mean: np.ndarray | None = None,
    branch_ranker_scale: np.ndarray | None = None,
    collect_branch_training_data: bool = False,
    enable_branch_structural_dedup: bool = False,
    enable_branch_signature_consensus: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], dict[str, int], np.ndarray, np.ndarray, np.ndarray]:
    controller_scores = np.asarray(base_scores, dtype=np.float64).copy()
    ranker_scores = controller_scores.copy()
    group_reports: list[dict[str, object]] = []
    branch_train_X: list[np.ndarray] = []
    branch_train_y: list[float] = []
    branch_train_groups: list[int] = []
    low_margin_refined = 0
    low_margin_changed = 0
    low_margin_reranked = 0
    low_margin_ranker_changed = 0
    low_margin_branch_duplicates = 0
    low_margin_groups_with_branch_duplicates = 0

    cursor = 0
    for group_index, example in enumerate(examples):
        count = len(example.candidates)
        idx = np.arange(cursor, cursor + count, dtype=np.int64)
        group_full_scores = np.asarray(base_scores[idx], dtype=np.float64)
        baseline_order = np.argsort(-group_full_scores, kind="mergesort")
        baseline_top_local = int(baseline_order[0])
        baseline_top_global = int(idx[baseline_top_local])
        group_margin = _group_margin(group_full_scores)
        branch_ranker_locals: list[int] = []
        candidate_reports: list[dict[str, object]] = []

        if float(group_margin) <= float(controller_margin_threshold):
            low_margin_refined += 1
            diagnostics = [
                _controller_diagnostics_for_candidate(
                    example.prompt,
                    candidate.text,
                    model=model,
                    client=client,
                    translator=translator,
                )
                for candidate in example.candidates
            ]
            step_query_counts = [int(diagnostic["step_query_count"]) for diagnostic in diagnostics]
            branch_locals = _branch_locals_for_group(
                group_full_scores,
                top_k=int(controller_top_k),
                score_window=float(controller_score_window),
                step_query_counts=step_query_counts,
                step_query_delta=int(controller_step_query_delta),
                step_query_window=float(controller_step_query_window),
            )
            adjusted_scores = group_full_scores.copy()
            for local_index, candidate in enumerate(example.candidates):
                diagnostic = diagnostics[local_index]
                prefix_scores = list(diagnostic["prefix_scores"])
                progress = float(diagnostic["progress"])
                answer_support = float(diagnostic["answer_support"])
                step_query_count = int(diagnostic["step_query_count"])
                tiebreak_features = dict(diagnostic["tiebreak_features"])
                diagnostic_bonus = (
                    float(progress_weight) * progress
                    + float(answer_support_weight) * answer_support
                    + float(step_query_weight) * float(step_query_count)
                    + float(valid_state_weight) * float(tiebreak_features["valid_state_fraction"])
                    + float(answer_support_fraction_weight) * float(tiebreak_features["answer_support_fraction"])
                    + float(equation_consistency_weight) * float(tiebreak_features["equation_consistency"])
                    + float(result_reuse_weight) * float(tiebreak_features["result_reuse_fraction"])
                    + float(final_resolution_weight) * float(tiebreak_features["final_resolution"])
                    - float(approx_penalty_weight) * float(tiebreak_features["approx_fraction"])
                )
                if int(local_index) in branch_locals:
                    adjusted_scores[local_index] = float(group_full_scores[local_index] + diagnostic_bonus)
                candidate_reports.append(
                    {
                        "candidate_index": int(local_index),
                        "full_score": float(group_full_scores[local_index]),
                        "adjusted_score": float(adjusted_scores[local_index]),
                        "ranker_adjusted_score": float(adjusted_scores[local_index]),
                        "branch_ranker_score": 0.0,
                        "label": float(candidate.label),
                        "answer_line": candidate.text.splitlines()[0] if candidate.text.splitlines() else candidate.text[:160],
                        "progress": progress,
                        "answer_support": answer_support,
                        "step_query_count": step_query_count,
                        "tiebreak_features": tiebreak_features,
                        "state_signature": _jsonable_signature(
                            translator.structural_state_signature_for_trace(candidate.text, prompt=example.prompt)
                        ),
                        "prefix_scores": prefix_scores,
                        "in_branch": bool(int(local_index) in branch_locals),
                        "dedup_representative_index": int(local_index),
                        "dedup_is_representative": bool(int(local_index) in branch_locals),
                    }
                )
            controller_scores[idx] = adjusted_scores
            controller_order = np.argsort(-adjusted_scores, kind="mergesort")
            controller_top_global = int(idx[int(controller_order[0])])
            if controller_top_global != baseline_top_global:
                low_margin_changed += 1

            if bool(enable_branch_structural_dedup) or bool(enable_branch_signature_consensus):
                dedup_branch_locals, representative_map, duplicate_count = _dedup_branch_locals(candidate_reports, branch_locals)
            else:
                dedup_branch_locals = list(branch_locals)
                representative_map = {int(local_index): [int(local_index)] for local_index in branch_locals}
                duplicate_count = 0
            low_margin_branch_duplicates += int(duplicate_count)
            if int(duplicate_count) > 0:
                low_margin_groups_with_branch_duplicates += 1

            branch_candidates = [candidate_reports[local_index] for local_index in dedup_branch_locals]
            branch_labels = [float(candidate_report["label"]) for candidate_report in branch_candidates]
            if collect_branch_training_data and len(dedup_branch_locals) >= 2 and any(label > 0.5 for label in branch_labels) and any(label <= 0.5 for label in branch_labels):
                for candidate_report in branch_candidates:
                    branch_train_X.append(_branch_ranker_feature_vector(candidate_report, branch_ranker_feature_names))
                    branch_train_y.append(float(candidate_report["label"]))
                    branch_train_groups.append(int(group_index))

            if (
                branch_ranker_model is not None
                and branch_ranker_mean is not None
                and branch_ranker_scale is not None
                and len(dedup_branch_locals) >= 2
            ):
                branch_matrix = np.asarray(
                    [_branch_ranker_feature_vector(candidate_reports[local_index], branch_ranker_feature_names) for local_index in dedup_branch_locals],
                    dtype=np.float64,
                )
                branch_ranker_values = branch_ranker_model.score(
                    _standardize_apply(branch_matrix, branch_ranker_mean, branch_ranker_scale)
                )
                reranked_scores = _rerank_branch_band(adjusted_scores, dedup_branch_locals, branch_ranker_values)
                if bool(enable_branch_structural_dedup):
                    reranked_scores = _apply_representative_scores_to_duplicates(candidate_reports, reranked_scores, representative_map)
                elif bool(enable_branch_signature_consensus):
                    reranked_scores = _apply_signature_consensus_scores(candidate_reports, adjusted_scores, reranked_scores, representative_map)
                ranker_scores[idx] = reranked_scores
                branch_ranker_locals = list(dedup_branch_locals)
                low_margin_reranked += 1
                if int(np.argmax(reranked_scores)) != int(np.argmax(adjusted_scores)):
                    low_margin_ranker_changed += 1
                for branch_position, local_index in enumerate(dedup_branch_locals):
                    candidate_reports[local_index]["branch_ranker_score"] = float(branch_ranker_values[branch_position])
                    candidate_reports[local_index]["ranker_adjusted_score"] = float(reranked_scores[local_index])
                for representative, local_indexes in representative_map.items():
                    representative_score = float(candidate_reports[representative]["branch_ranker_score"])
                    for local_index in local_indexes:
                        candidate_reports[local_index]["branch_ranker_score"] = representative_score
            else:
                ranker_scores[idx] = adjusted_scores
                for representative, local_indexes in representative_map.items():
                    for local_index in local_indexes:
                        candidate_reports[local_index]["dedup_representative_index"] = int(representative)
                        candidate_reports[local_index]["dedup_is_representative"] = bool(int(local_index) == int(representative))
        else:
            branch_locals = _branch_locals_for_group(
                group_full_scores,
                top_k=int(controller_top_k),
                score_window=float(controller_score_window),
            )
            for local_index, candidate in enumerate(example.candidates):
                tiebreak_features = _structured_tiebreak_features(
                    translator,
                    candidate.text,
                    prompt=example.prompt,
                )
                candidate_reports.append(
                    {
                        "candidate_index": int(local_index),
                        "full_score": float(group_full_scores[local_index]),
                        "adjusted_score": float(group_full_scores[local_index]),
                        "ranker_adjusted_score": float(group_full_scores[local_index]),
                        "branch_ranker_score": 0.0,
                        "label": float(candidate.label),
                        "answer_line": candidate.text.splitlines()[0] if candidate.text.splitlines() else candidate.text[:160],
                        "progress": 0.0,
                        "answer_support": float(max(0.0, translator._answer_support_score(candidate.text))),
                        "step_query_count": int(len(translator.queries_for_trace(candidate.text, prompt=example.prompt))),
                        "tiebreak_features": tiebreak_features,
                        "state_signature": _jsonable_signature(
                            translator.structural_state_signature_for_trace(candidate.text, prompt=example.prompt)
                        ),
                        "prefix_scores": [],
                        "in_branch": bool(int(local_index) in branch_locals),
                        "dedup_representative_index": int(local_index),
                        "dedup_is_representative": bool(int(local_index) in branch_locals),
                    }
                )
            ranker_scores[idx] = group_full_scores

        group_reports.append(
            {
                "problem_id": str(example.problem_id),
                "group_index": int(group_index),
                "oracle_margin": float(group_margin),
                "baseline_top_correct": bool(float(labels[baseline_top_global]) > 0.5),
                "controller_top_correct": bool(float(labels[idx[np.argmax(controller_scores[idx])]]) > 0.5),
                "ranker_top_correct": bool(float(labels[idx[np.argmax(ranker_scores[idx])]]) > 0.5),
                "branch_local_indexes": branch_locals,
                "ranker_branch_local_indexes": branch_ranker_locals,
                "candidates": candidate_reports,
            }
        )
        cursor += count

    stats = {
        "low_margin_groups_refined": int(low_margin_refined),
        "low_margin_groups_changed": int(low_margin_changed),
        "low_margin_groups_reranked": int(low_margin_reranked),
        "low_margin_groups_changed_by_ranker": int(low_margin_ranker_changed),
        "low_margin_branch_duplicates": int(low_margin_branch_duplicates),
        "low_margin_groups_with_branch_duplicates": int(low_margin_groups_with_branch_duplicates),
    }
    train_X = np.asarray(branch_train_X, dtype=np.float64) if branch_train_X else np.zeros((0, len(branch_ranker_feature_names)), dtype=np.float64)
    train_y = np.asarray(branch_train_y, dtype=np.float64) if branch_train_y else np.zeros((0,), dtype=np.float64)
    train_groups = np.asarray(branch_train_groups, dtype=np.int64) if branch_train_groups else np.zeros((0,), dtype=np.int64)
    return controller_scores, ranker_scores, group_reports, stats, train_X, train_y, train_groups


def _branch_locals_for_group(
    scores: np.ndarray,
    *,
    top_k: int,
    score_window: float,
    step_query_counts: list[int] | None = None,
    step_query_delta: int = 1,
    step_query_window: float = 0.0,
) -> list[int]:
    baseline_order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    branch_order = baseline_order[: max(1, int(top_k))]
    top_local = int(branch_order[0])
    top_score = float(scores[top_local])
    branch_locals = [
        int(local_index)
        for local_index in branch_order.tolist()
        if float(top_score - scores[int(local_index)]) <= float(score_window)
    ]
    if not branch_locals:
        branch_locals = [top_local]
    if not step_query_counts:
        return branch_locals

    branch_best_step_queries = max(int(step_query_counts[local_index]) for local_index in branch_locals)
    for local_index, candidate_score in enumerate(np.asarray(scores, dtype=np.float64).tolist()):
        if int(local_index) in branch_locals:
            continue
        if int(step_query_counts[local_index]) < int(branch_best_step_queries + step_query_delta):
            continue
        if float(top_score - float(candidate_score)) > float(step_query_window):
            continue
        branch_locals.append(int(local_index))
    return sorted(set(int(local_index) for local_index in branch_locals))


def main() -> int:
    ap = argparse.ArgumentParser(description="Run an oracle-guided controller prototype using partial reasoning prefixes and low-margin refinement.")
    ap.add_argument("--train_dataset", required=True)
    ap.add_argument("--eval_dataset", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--controller_margin_threshold", type=float, default=0.0)
    ap.add_argument("--controller_score_window", type=float, default=0.001)
    ap.add_argument("--controller_top_k", type=int, default=3)
    ap.add_argument("--progress_weight", type=float, default=0.0)
    ap.add_argument("--answer_support_weight", type=float, default=0.2)
    ap.add_argument("--step_query_weight", type=float, default=0.0001)
    ap.add_argument("--controller_step_query_delta", type=int, default=1)
    ap.add_argument("--controller_step_query_window", type=float, default=0.006)
    ap.add_argument("--valid_state_weight", type=float, default=0.0)
    ap.add_argument("--answer_support_fraction_weight", type=float, default=0.0)
    ap.add_argument("--equation_consistency_weight", type=float, default=0.0)
    ap.add_argument("--result_reuse_weight", type=float, default=0.0)
    ap.add_argument("--final_resolution_weight", type=float, default=0.0)
    ap.add_argument("--approx_penalty_weight", type=float, default=0.0)
    ap.add_argument("--enable_branch_ranker", action="store_true")
    ap.add_argument("--branch_ranker_hidden_dim", type=int, default=8)
    ap.add_argument("--branch_ranker_epochs", type=int, default=75)
    ap.add_argument("--branch_ranker_lr", type=float, default=5e-3)
    ap.add_argument("--branch_ranker_weight_decay", type=float, default=1e-4)
    ap.add_argument("--branch_ranker_feature_names", default="")
    ap.add_argument("--enable_branch_structural_dedup", action="store_true")
    ap.add_argument("--enable_branch_signature_consensus", action="store_true")
    args = ap.parse_args()
    branch_ranker_feature_names = _parse_branch_ranker_feature_names(str(args.branch_ranker_feature_names))

    translator = HeuristicAnchoredTranslator()
    client = AnchoredOracleClient()

    train_examples = load_reasoning_examples(str(args.train_dataset))
    eval_examples = load_reasoning_examples(str(args.eval_dataset))
    X_train, y_train, groups_train = materialize_dataset(
        train_examples,
        client=client,
        translator=translator,
        feature_mode="oracle",
    )
    X_eval, y_eval, groups_eval = materialize_dataset(
        eval_examples,
        client=client,
        translator=translator,
        feature_mode="oracle",
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
    train_scores = model.score(X_train)
    eval_scores = model.score(X_eval)

    train_controller_scores, _, _, train_controller_stats, branch_train_X, branch_train_y, branch_train_groups = _score_dataset_with_controller(
        train_examples,
        train_scores,
        y_train,
        model=model,
        client=client,
        translator=translator,
        controller_margin_threshold=float(args.controller_margin_threshold),
        controller_top_k=int(args.controller_top_k),
        controller_score_window=float(args.controller_score_window),
        progress_weight=float(args.progress_weight),
        answer_support_weight=float(args.answer_support_weight),
        step_query_weight=float(args.step_query_weight),
        controller_step_query_delta=int(args.controller_step_query_delta),
        controller_step_query_window=float(args.controller_step_query_window),
        valid_state_weight=float(args.valid_state_weight),
        answer_support_fraction_weight=float(args.answer_support_fraction_weight),
        equation_consistency_weight=float(args.equation_consistency_weight),
        result_reuse_weight=float(args.result_reuse_weight),
        final_resolution_weight=float(args.final_resolution_weight),
        approx_penalty_weight=float(args.approx_penalty_weight),
        branch_ranker_feature_names=branch_ranker_feature_names,
        collect_branch_training_data=bool(args.enable_branch_ranker),
        enable_branch_structural_dedup=bool(args.enable_branch_structural_dedup),
        enable_branch_signature_consensus=bool(args.enable_branch_signature_consensus),
    )

    branch_ranker_model: PairwiseMLPReranker | None = None
    branch_ranker_mean: np.ndarray | None = None
    branch_ranker_scale: np.ndarray | None = None
    branch_ranker_history: list[float] = []
    branch_ranker_status = "disabled"
    if bool(args.enable_branch_ranker):
        eligible_train_groups = int(len(np.unique(branch_train_groups))) if branch_train_groups.size else 0
        if branch_train_X.shape[0] >= 2 and eligible_train_groups >= 2:
            branch_ranker_mean, branch_ranker_scale = _standardize_fit(branch_train_X)
            branch_ranker_model = PairwiseMLPReranker(
                input_dim=int(branch_train_X.shape[1]),
                hidden_dim=int(args.branch_ranker_hidden_dim),
                seed=int(args.seed),
            )
            branch_ranker_history = branch_ranker_model.fit(
                _standardize_apply(branch_train_X, branch_ranker_mean, branch_ranker_scale),
                branch_train_y,
                branch_train_groups,
                epochs=int(args.branch_ranker_epochs),
                lr=float(args.branch_ranker_lr),
                weight_decay=float(args.branch_ranker_weight_decay),
            )
            branch_ranker_status = "trained"
        else:
            branch_ranker_status = "insufficient_low_margin_branch_training_data"

    controller_scores, ranker_scores, group_reports, eval_controller_stats, _, _, _ = _score_dataset_with_controller(
        eval_examples,
        eval_scores,
        y_eval,
        model=model,
        client=client,
        translator=translator,
        controller_margin_threshold=float(args.controller_margin_threshold),
        controller_top_k=int(args.controller_top_k),
        controller_score_window=float(args.controller_score_window),
        progress_weight=float(args.progress_weight),
        answer_support_weight=float(args.answer_support_weight),
        step_query_weight=float(args.step_query_weight),
        controller_step_query_delta=int(args.controller_step_query_delta),
        controller_step_query_window=float(args.controller_step_query_window),
        valid_state_weight=float(args.valid_state_weight),
        answer_support_fraction_weight=float(args.answer_support_fraction_weight),
        equation_consistency_weight=float(args.equation_consistency_weight),
        result_reuse_weight=float(args.result_reuse_weight),
        final_resolution_weight=float(args.final_resolution_weight),
        approx_penalty_weight=float(args.approx_penalty_weight),
        branch_ranker_feature_names=branch_ranker_feature_names,
        branch_ranker_model=branch_ranker_model,
        branch_ranker_mean=branch_ranker_mean,
        branch_ranker_scale=branch_ranker_scale,
        enable_branch_structural_dedup=bool(args.enable_branch_structural_dedup),
        enable_branch_signature_consensus=bool(args.enable_branch_signature_consensus),
    )

    summary = {
        "train_dataset": str(args.train_dataset),
        "eval_dataset": str(args.eval_dataset),
        "controller_config": {
            "hidden_dim": int(args.hidden_dim),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "controller_margin_threshold": float(args.controller_margin_threshold),
            "controller_score_window": float(args.controller_score_window),
            "controller_top_k": int(args.controller_top_k),
            "progress_weight": float(args.progress_weight),
            "answer_support_weight": float(args.answer_support_weight),
            "step_query_weight": float(args.step_query_weight),
            "controller_step_query_delta": int(args.controller_step_query_delta),
            "controller_step_query_window": float(args.controller_step_query_window),
            "valid_state_weight": float(args.valid_state_weight),
            "answer_support_fraction_weight": float(args.answer_support_fraction_weight),
            "equation_consistency_weight": float(args.equation_consistency_weight),
            "result_reuse_weight": float(args.result_reuse_weight),
            "final_resolution_weight": float(args.final_resolution_weight),
            "approx_penalty_weight": float(args.approx_penalty_weight),
            "enable_branch_ranker": bool(args.enable_branch_ranker),
            "branch_ranker_hidden_dim": int(args.branch_ranker_hidden_dim),
            "branch_ranker_epochs": int(args.branch_ranker_epochs),
            "branch_ranker_lr": float(args.branch_ranker_lr),
            "branch_ranker_weight_decay": float(args.branch_ranker_weight_decay),
            "branch_ranker_feature_names": list(branch_ranker_feature_names),
            "enable_branch_structural_dedup": bool(args.enable_branch_structural_dedup),
            "enable_branch_signature_consensus": bool(args.enable_branch_signature_consensus),
        },
        "oracle_branch": {
            "train_loss_last": float(history[-1] if history else 0.0),
            "train_metrics": ranking_metrics(train_scores, y_train, groups_train),
            "eval_metrics": ranking_metrics(eval_scores, y_eval, groups_eval),
        },
        "controller_branch": {
            "eval_metrics": ranking_metrics(controller_scores, y_eval, groups_eval),
            "train_metrics": ranking_metrics(train_controller_scores, y_train, groups_train),
            "low_margin_groups_refined": int(eval_controller_stats["low_margin_groups_refined"]),
            "low_margin_groups_changed": int(eval_controller_stats["low_margin_groups_changed"]),
            "train_low_margin_groups_refined": int(train_controller_stats["low_margin_groups_refined"]),
            "train_low_margin_groups_changed": int(train_controller_stats["low_margin_groups_changed"]),
        },
        "branch_ranker_branch": {
            "enabled": bool(args.enable_branch_ranker),
            "status": str(branch_ranker_status),
            "feature_names": list(branch_ranker_feature_names),
            "train_branch_rows": int(branch_train_X.shape[0]),
            "train_branch_groups": int(len(np.unique(branch_train_groups))) if branch_train_groups.size else 0,
            "train_loss_last": float(branch_ranker_history[-1] if branch_ranker_history else 0.0),
            "eval_metrics": ranking_metrics(ranker_scores, y_eval, groups_eval),
            "low_margin_groups_reranked": int(eval_controller_stats["low_margin_groups_reranked"]),
            "low_margin_groups_changed_vs_controller": int(eval_controller_stats["low_margin_groups_changed_by_ranker"]),
            "low_margin_branch_duplicates": int(eval_controller_stats["low_margin_branch_duplicates"]),
            "low_margin_groups_with_branch_duplicates": int(eval_controller_stats["low_margin_groups_with_branch_duplicates"]),
        },
        "group_reports": group_reports,
    }

    out_path = Path(str(args.out_json))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())