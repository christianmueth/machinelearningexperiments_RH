from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _candidate_feature_map(candidate: dict[str, object]) -> dict[str, float]:
    features = {
        "full_score": float(candidate.get("full_score", 0.0)),
        "adjusted_score": float(candidate.get("adjusted_score", 0.0)),
        "progress": float(candidate.get("progress", 0.0)),
        "answer_support": float(candidate.get("answer_support", 0.0)),
        "step_query_count": float(candidate.get("step_query_count", 0.0)),
    }
    tiebreak = candidate.get("tiebreak_features", {})
    if isinstance(tiebreak, dict):
        for key, value in tiebreak.items():
            features[str(key)] = float(value)
    return features


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = max(0.0, min(1.0, float(q))) * float(len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = float(position - lower)
    return float((1.0 - weight) * ordered[lower] + weight * ordered[upper])


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


def _normalized_feature_scores(candidates: list[dict[str, object]], feature_name: str) -> list[float]:
    values = [float(_candidate_feature_map(candidate).get(feature_name, 0.0)) for candidate in candidates]
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if math.isclose(maximum, minimum, rel_tol=1e-9, abs_tol=1e-9):
        return [0.0 for _ in values]
    scale = float(maximum - minimum)
    return [float((value - minimum) / scale) for value in values]


def _top_candidate_by_feature(candidates: list[dict[str, object]], feature_name: str) -> dict[str, object]:
    return sorted(
        candidates,
        key=lambda candidate: (
            float(_candidate_feature_map(candidate).get(feature_name, 0.0)),
            float(candidate.get("full_score", 0.0)),
        ),
        reverse=True,
    )[0]


def _top_candidate_by_combined_features(candidates: list[dict[str, object]], feature_names: tuple[str, str]) -> dict[str, object]:
    combined_scores = [0.0 for _ in candidates]
    for feature_name in feature_names:
        normalized = _normalized_feature_scores(candidates, feature_name)
        combined_scores = [float(score + addition) for score, addition in zip(combined_scores, normalized)]
    ranked_indexes = sorted(
        range(len(candidates)),
        key=lambda index: (
            float(combined_scores[index]),
            float(candidates[index].get("full_score", 0.0)),
        ),
        reverse=True,
    )
    return candidates[int(ranked_indexes[0])]


def _interaction_summary(
    groups: list[dict[str, object]],
    feature_names: tuple[str, str],
) -> dict[str, object]:
    feature_a, feature_b = feature_names
    row_feature_a: list[float] = []
    row_feature_b: list[float] = []
    row_labels: list[float] = []
    group_total = 0
    feature_a_success = 0
    feature_b_success = 0
    combined_success = 0
    argmax_agreement = 0
    combined_unique_wins = 0
    combined_regressions = 0

    for group in groups:
        candidates = [candidate for candidate in group.get("candidates", []) if bool(candidate.get("in_branch", False))]
        if len(candidates) < 2:
            continue
        gold_candidates = [candidate for candidate in candidates if float(candidate.get("label", 0.0)) > 0.5]
        if not gold_candidates:
            continue

        for candidate in candidates:
            feature_map = _candidate_feature_map(candidate)
            row_feature_a.append(float(feature_map.get(feature_a, 0.0)))
            row_feature_b.append(float(feature_map.get(feature_b, 0.0)))
            row_labels.append(float(candidate.get("label", 0.0)))

        group_total += 1
        top_a = _top_candidate_by_feature(candidates, feature_a)
        top_b = _top_candidate_by_feature(candidates, feature_b)
        top_combined = _top_candidate_by_combined_features(candidates, feature_names)

        a_correct = float(top_a.get("label", 0.0)) > 0.5
        b_correct = float(top_b.get("label", 0.0)) > 0.5
        combined_correct = float(top_combined.get("label", 0.0)) > 0.5
        if a_correct:
            feature_a_success += 1
        if b_correct:
            feature_b_success += 1
        if combined_correct:
            combined_success += 1
        if int(top_a.get("candidate_index", -1)) == int(top_b.get("candidate_index", -2)):
            argmax_agreement += 1
        if combined_correct and not a_correct and not b_correct:
            combined_unique_wins += 1
        if (a_correct or b_correct) and not combined_correct:
            combined_regressions += 1

    high_a = _quantile(row_feature_a, 0.75)
    high_b = _quantile(row_feature_b, 0.75)
    quadrant_rows = {
        "both_high": [],
        "feature_a_high_only": [],
        "feature_b_high_only": [],
        "neither_high": [],
    }
    for value_a, value_b, label in zip(row_feature_a, row_feature_b, row_labels):
        if value_a >= high_a and value_b >= high_b:
            quadrant_rows["both_high"].append(label)
        elif value_a >= high_a:
            quadrant_rows["feature_a_high_only"].append(label)
        elif value_b >= high_b:
            quadrant_rows["feature_b_high_only"].append(label)
        else:
            quadrant_rows["neither_high"].append(label)

    return {
        "features": [feature_a, feature_b],
        "rows": int(len(row_labels)),
        "groups_evaluated": int(group_total),
        "feature_correlation": _pearson(row_feature_a, row_feature_b),
        "feature_a_correctness_correlation": _pearson(row_feature_a, row_labels),
        "feature_b_correctness_correlation": _pearson(row_feature_b, row_labels),
        "argmax_agreement_rate": float(argmax_agreement / group_total) if group_total else 0.0,
        "feature_a_argmax_group_accuracy": float(feature_a_success / group_total) if group_total else 0.0,
        "feature_b_argmax_group_accuracy": float(feature_b_success / group_total) if group_total else 0.0,
        "combined_argmax_group_accuracy": float(combined_success / group_total) if group_total else 0.0,
        "combined_gain_vs_best_single": float(combined_success / group_total) - max(
            float(feature_a_success / group_total) if group_total else 0.0,
            float(feature_b_success / group_total) if group_total else 0.0,
        ) if group_total else 0.0,
        "combined_unique_wins": int(combined_unique_wins),
        "combined_regressions": int(combined_regressions),
        "high_thresholds": {
            feature_a: float(high_a),
            feature_b: float(high_b),
        },
        "high_quadrant_accuracy": {
            name: {
                "count": int(len(labels)),
                "accuracy": _mean([float(label) for label in labels]),
            }
            for name, labels in quadrant_rows.items()
        },
    }


def _parse_interaction_pairs(raw_pairs: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for raw_pair in raw_pairs:
        pieces = [piece.strip() for piece in str(raw_pair).split(",") if piece.strip()]
        if len(pieces) != 2:
            continue
        parsed.append((pieces[0], pieces[1]))
    return parsed


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


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze low-margin controller-branch features offline before enabling any tie-break layer.")
    ap.add_argument("--controller_json", required=True)
    ap.add_argument("--out_json", default="")
    ap.add_argument("--max_margin", type=float, default=None)
    ap.add_argument(
        "--interaction_pair",
        action="append",
        default=[
            "result_reuse_fraction,answer_support_fraction",
            "result_reuse_fraction,answer_support_count",
        ],
        help="Comma-separated feature pair to analyze for complementarity. May be passed multiple times.",
    )
    args = ap.parse_args()

    report = json.loads(Path(str(args.controller_json)).read_text(encoding="utf-8"))
    threshold = args.max_margin
    if threshold is None:
        threshold = float(report.get("controller_config", {}).get("controller_margin_threshold", 0.0))

    group_reports = list(report.get("group_reports", []))
    low_margin_groups = [
        group
        for group in group_reports
        if float(group.get("oracle_margin", 0.0)) <= float(threshold)
    ]

    feature_names: set[str] = set()
    eligible_groups = 0
    groups_gold_in_branch = 0
    groups_branch_changed = 0
    gold_feature_values: dict[str, list[float]] = {}
    wrong_feature_values: dict[str, list[float]] = {}
    argmax_success: dict[str, int] = {}
    argmax_total: dict[str, int] = {}
    argmin_success: dict[str, int] = {}
    argmin_total: dict[str, int] = {}
    pairwise_win_rates: dict[str, list[float]] = {}
    pairwise_tie_rates: dict[str, list[float]] = {}

    for group in low_margin_groups:
        candidates = [candidate for candidate in group.get("candidates", []) if bool(candidate.get("in_branch", False))]
        if len(candidates) < 2:
            continue
        eligible_groups += 1
        if bool(group.get("baseline_top_correct", False)) != bool(group.get("controller_top_correct", False)):
            groups_branch_changed += 1

        gold_candidates = [candidate for candidate in candidates if float(candidate.get("label", 0.0)) > 0.5]
        if not gold_candidates:
            continue
        gold_candidate = gold_candidates[0]
        groups_gold_in_branch += 1
        wrong_candidates = [candidate for candidate in candidates if float(candidate.get("label", 0.0)) <= 0.5]

        feature_maps = [_candidate_feature_map(candidate) for candidate in candidates]
        gold_map = _candidate_feature_map(gold_candidate)
        for feature_map in feature_maps:
            feature_names.update(feature_map.keys())

        for feature_name in feature_names:
            gold_value = float(gold_map.get(feature_name, 0.0))
            wrong_values = [float(_candidate_feature_map(candidate).get(feature_name, 0.0)) for candidate in wrong_candidates]
            gold_feature_values.setdefault(feature_name, []).append(gold_value)
            wrong_feature_values.setdefault(feature_name, []).extend(wrong_values)

            if wrong_values:
                win_rate, tie_rate = _pairwise_preference(gold_value, wrong_values)
                pairwise_win_rates.setdefault(feature_name, []).append(win_rate)
                pairwise_tie_rates.setdefault(feature_name, []).append(tie_rate)

        for feature_name in feature_names:
            argmax_total[feature_name] = argmax_total.get(feature_name, 0) + 1
            argmin_total[feature_name] = argmin_total.get(feature_name, 0) + 1
            ranked = sorted(
                candidates,
                key=lambda candidate: (
                    float(_candidate_feature_map(candidate).get(feature_name, 0.0)),
                    float(candidate.get("full_score", 0.0)),
                ),
                reverse=True,
            )
            top_candidate = ranked[0]
            if float(top_candidate.get("label", 0.0)) > 0.5:
                argmax_success[feature_name] = argmax_success.get(feature_name, 0) + 1
            bottom_ranked = sorted(
                candidates,
                key=lambda candidate: (
                    float(_candidate_feature_map(candidate).get(feature_name, 0.0)),
                    float(candidate.get("full_score", 0.0)),
                ),
            )
            bottom_candidate = bottom_ranked[0]
            if float(bottom_candidate.get("label", 0.0)) > 0.5:
                argmin_success[feature_name] = argmin_success.get(feature_name, 0) + 1

    feature_summaries = []
    for feature_name in sorted(feature_names):
        gold_values = gold_feature_values.get(feature_name, [])
        wrong_values = wrong_feature_values.get(feature_name, [])
        feature_summaries.append(
            {
                "feature": feature_name,
                "gold_mean": _mean(gold_values),
                "wrong_mean": _mean(wrong_values),
                "gold_minus_wrong_mean": _mean(gold_values) - _mean(wrong_values),
                "argmax_group_accuracy": float(argmax_success.get(feature_name, 0) / argmax_total.get(feature_name, 1)),
                "argmin_group_accuracy": float(argmin_success.get(feature_name, 0) / argmin_total.get(feature_name, 1)),
                "best_direction": "max"
                if float(argmax_success.get(feature_name, 0) / argmax_total.get(feature_name, 1))
                >= float(argmin_success.get(feature_name, 0) / argmin_total.get(feature_name, 1))
                else "min",
                "best_direction_group_accuracy": max(
                    float(argmax_success.get(feature_name, 0) / argmax_total.get(feature_name, 1)),
                    float(argmin_success.get(feature_name, 0) / argmin_total.get(feature_name, 1)),
                ),
                "pairwise_win_rate": _mean(pairwise_win_rates.get(feature_name, [])),
                "pairwise_tie_rate": _mean(pairwise_tie_rates.get(feature_name, [])),
                "groups_evaluated": int(argmax_total.get(feature_name, 0)),
            }
        )

    feature_summaries.sort(
        key=lambda row: (
            float(row["best_direction_group_accuracy"]),
            float(row["pairwise_win_rate"]),
            float(row["gold_minus_wrong_mean"]),
        ),
        reverse=True,
    )

    summary = {
        "controller_json": str(args.controller_json),
        "max_margin": float(threshold),
        "groups_total": int(len(group_reports)),
        "low_margin_groups": int(len(low_margin_groups)),
        "eligible_branch_groups": int(eligible_groups),
        "gold_in_branch_groups": int(groups_gold_in_branch),
        "groups_with_controller_decision_change": int(groups_branch_changed),
        "feature_rankings": feature_summaries,
        "interaction_analyses": [
            _interaction_summary(low_margin_groups, feature_pair)
            for feature_pair in _parse_interaction_pairs(list(args.interaction_pair))
        ],
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