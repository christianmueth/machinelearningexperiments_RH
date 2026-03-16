from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


FEATURE_NAMES = (
    "full_score",
    "result_reuse_fraction",
    "result_reuse_count",
    "answer_support_fraction",
    "answer_support_count",
    "valid_state_fraction",
    "equation_consistency",
)


def _feature(candidate: dict[str, object], feature_name: str) -> float:
    if feature_name == "full_score":
        return float(candidate.get("full_score", 0.0))
    tiebreak = candidate.get("tiebreak_features", {})
    if isinstance(tiebreak, dict):
        return float(tiebreak.get(feature_name, 0.0))
    return 0.0


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _top_candidate(candidates: list[dict[str, object]], score_name: str) -> dict[str, object]:
    return sorted(
        candidates,
        key=lambda candidate: (
            float(candidate.get(score_name, 0.0)),
            float(candidate.get("full_score", 0.0)),
            -int(candidate.get("candidate_index", 0)),
        ),
        reverse=True,
    )[0]


def _structural_signature(candidate: dict[str, object]) -> tuple[float, ...]:
    return tuple(round(_feature(candidate, feature_name), 6) for feature_name in FEATURE_NAMES[1:])


def _row_summary(candidate: dict[str, object]) -> dict[str, object]:
    return {
        "candidate_index": int(candidate.get("candidate_index", -1)),
        "label": float(candidate.get("label", 0.0)),
        "full_score": float(candidate.get("full_score", 0.0)),
        "adjusted_score": float(candidate.get("adjusted_score", 0.0)),
        "ranker_adjusted_score": float(candidate.get("ranker_adjusted_score", 0.0)),
        "branch_ranker_score": float(candidate.get("branch_ranker_score", 0.0)),
        "answer_line": str(candidate.get("answer_line", "")),
        "features": {feature_name: _feature(candidate, feature_name) for feature_name in FEATURE_NAMES},
    }


def _append_feature_deltas(store: dict[str, list[float]], chosen: dict[str, object], replaced: dict[str, object]) -> None:
    for feature_name in FEATURE_NAMES:
        store.setdefault(feature_name, []).append(_feature(chosen, feature_name) - _feature(replaced, feature_name))
    store.setdefault("branch_ranker_score", []).append(
        float(chosen.get("branch_ranker_score", 0.0)) - float(replaced.get("branch_ranker_score", 0.0))
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit controller-vs-branch-ranker outcome deltas on a controller report.")
    ap.add_argument("--controller_json", required=True)
    ap.add_argument("--out_json", default="")
    ap.add_argument("--max_examples", type=int, default=8)
    args = ap.parse_args()

    report = json.loads(Path(str(args.controller_json)).read_text(encoding="utf-8"))
    threshold = float(report.get("controller_config", {}).get("controller_margin_threshold", 0.0))

    wins: list[dict[str, object]] = []
    regressions: list[dict[str, object]] = []
    unchanged = 0
    low_margin_total = 0
    high_margin_total = 0
    low_margin_controller_correct = 0
    low_margin_ranker_correct = 0
    high_margin_controller_correct = 0
    high_margin_ranker_correct = 0
    duplicate_signature_groups = 0
    win_feature_deltas: dict[str, list[float]] = {}
    regression_feature_deltas: dict[str, list[float]] = {}

    for group in report.get("group_reports", []):
        candidates = list(group.get("candidates", []))
        if not candidates:
            continue
        controller_top = _top_candidate(candidates, "adjusted_score")
        ranker_top = _top_candidate(candidates, "ranker_adjusted_score")
        low_margin = float(group.get("oracle_margin", 0.0)) <= threshold
        controller_correct = float(controller_top.get("label", 0.0)) > 0.5
        ranker_correct = float(ranker_top.get("label", 0.0)) > 0.5

        if low_margin:
            low_margin_total += 1
            low_margin_controller_correct += int(controller_correct)
            low_margin_ranker_correct += int(ranker_correct)
        else:
            high_margin_total += 1
            high_margin_controller_correct += int(controller_correct)
            high_margin_ranker_correct += int(ranker_correct)

        signatures = [_structural_signature(candidate) for candidate in candidates if bool(candidate.get("in_branch", False))]
        if len(signatures) != len(set(signatures)):
            duplicate_signature_groups += 1

        if controller_correct == ranker_correct:
            unchanged += 1
            continue

        row = {
            "problem_id": str(group.get("problem_id", "")),
            "group_index": int(group.get("group_index", -1)),
            "oracle_margin": float(group.get("oracle_margin", 0.0)),
            "low_margin": bool(low_margin),
            "controller_candidate": _row_summary(controller_top),
            "ranker_candidate": _row_summary(ranker_top),
            "branch_local_indexes": list(group.get("branch_local_indexes", [])),
            "ranker_branch_local_indexes": list(group.get("ranker_branch_local_indexes", [])),
            "branch_size": int(sum(1 for candidate in candidates if bool(candidate.get("in_branch", False)))),
            "duplicate_structural_signature_in_branch": bool(len(signatures) != len(set(signatures))),
        }
        if ranker_correct and not controller_correct:
            wins.append(row)
            _append_feature_deltas(win_feature_deltas, ranker_top, controller_top)
        elif controller_correct and not ranker_correct:
            regressions.append(row)
            _append_feature_deltas(regression_feature_deltas, ranker_top, controller_top)

    summary = {
        "controller_json": str(args.controller_json),
        "groups_total": int(len(report.get("group_reports", []))),
        "low_margin_groups": int(low_margin_total),
        "high_margin_groups": int(high_margin_total),
        "duplicate_structural_signature_groups": int(duplicate_signature_groups),
        "wins_vs_controller": int(len(wins)),
        "regressions_vs_controller": int(len(regressions)),
        "unchanged_groups": int(unchanged),
        "low_margin_controller_accuracy": float(low_margin_controller_correct / low_margin_total) if low_margin_total else 0.0,
        "low_margin_ranker_accuracy": float(low_margin_ranker_correct / low_margin_total) if low_margin_total else 0.0,
        "high_margin_controller_accuracy": float(high_margin_controller_correct / high_margin_total) if high_margin_total else 0.0,
        "high_margin_ranker_accuracy": float(high_margin_ranker_correct / high_margin_total) if high_margin_total else 0.0,
        "win_feature_deltas_mean": {feature_name: _mean(values) for feature_name, values in win_feature_deltas.items()},
        "regression_feature_deltas_mean": {feature_name: _mean(values) for feature_name, values in regression_feature_deltas.items()},
        "wins_sample": wins[: max(0, int(args.max_examples))],
        "regressions_sample": regressions[: max(0, int(args.max_examples))],
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