from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml_oracle.datasets import load_reasoning_examples
from src.ml_oracle.translator import HeuristicAnchoredTranslator


def _legacy_signature(candidate: dict[str, object]) -> tuple[object, ...]:
    tiebreak = candidate.get("tiebreak_features", {})
    if not isinstance(tiebreak, dict):
        tiebreak = {}
    return (
        round(float(tiebreak.get("result_reuse_fraction", 0.0)), 6),
        round(float(tiebreak.get("result_reuse_count", 0.0)), 6),
        round(float(tiebreak.get("answer_support_fraction", 0.0)), 6),
        round(float(tiebreak.get("answer_support_count", 0.0)), 6),
        round(float(tiebreak.get("valid_state_fraction", 0.0)), 6),
        round(float(tiebreak.get("equation_consistency", 0.0)), 6),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit translator state-resolution strength using explicit structural state signatures.")
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--controller_json", default="")
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    translator = HeuristicAnchoredTranslator()
    examples = load_reasoning_examples(str(args.input_jsonl))
    example_map = {str(example.problem_id): example for example in examples}

    controller_report = {}
    if str(args.controller_json).strip():
        controller_report = json.loads(Path(str(args.controller_json)).read_text(encoding="utf-8"))

    total_groups = 0
    legacy_duplicate_groups = 0
    state_duplicate_groups = 0
    legacy_duplicate_branch_groups = 0
    state_duplicate_branch_groups = 0
    improved_branch_groups = 0
    legacy_duplicate_candidates = 0
    state_duplicate_candidates = 0
    branch_legacy_duplicate_candidates = 0
    branch_state_duplicate_candidates = 0
    sample_rows: list[dict[str, object]] = []

    if controller_report:
        groups_iter = list(controller_report.get("group_reports", []))
    else:
        groups_iter = [
            {
                "problem_id": str(example.problem_id),
                "candidates": [
                    {"candidate_index": index, "in_branch": True, "tiebreak_features": {}}
                    for index, _ in enumerate(example.candidates)
                ],
            }
            for example in examples
        ]

    for group in groups_iter:
        problem_id = str(group.get("problem_id", ""))
        example = example_map.get(problem_id)
        if example is None:
            continue
        candidates = list(group.get("candidates", []))
        if not candidates:
            continue
        total_groups += 1
        legacy_signatures = [_legacy_signature(candidate) for candidate in candidates]
        state_signatures = [
            translator.structural_state_signature_for_trace(example.candidates[int(candidate.get("candidate_index", 0))].text, prompt=example.prompt)
            for candidate in candidates
        ]
        legacy_duplicates = len(legacy_signatures) - len(set(legacy_signatures))
        state_duplicates = len(state_signatures) - len(set(state_signatures))
        legacy_duplicate_candidates += max(0, int(legacy_duplicates))
        state_duplicate_candidates += max(0, int(state_duplicates))
        if legacy_duplicates > 0:
            legacy_duplicate_groups += 1
        if state_duplicates > 0:
            state_duplicate_groups += 1

        branch_candidates = [candidate for candidate in candidates if bool(candidate.get("in_branch", False))]
        branch_legacy_signatures = [_legacy_signature(candidate) for candidate in branch_candidates]
        branch_state_signatures = [
            translator.structural_state_signature_for_trace(example.candidates[int(candidate.get("candidate_index", 0))].text, prompt=example.prompt)
            for candidate in branch_candidates
        ]
        branch_legacy_duplicates = len(branch_legacy_signatures) - len(set(branch_legacy_signatures))
        branch_state_duplicates = len(branch_state_signatures) - len(set(branch_state_signatures))
        branch_legacy_duplicate_candidates += max(0, int(branch_legacy_duplicates))
        branch_state_duplicate_candidates += max(0, int(branch_state_duplicates))
        if branch_legacy_duplicates > 0:
            legacy_duplicate_branch_groups += 1
        if branch_state_duplicates > 0:
            state_duplicate_branch_groups += 1
        if branch_legacy_duplicates > branch_state_duplicates:
            improved_branch_groups += 1

        if len(sample_rows) < 8 and branch_legacy_duplicates > 0:
            sample_rows.append(
                {
                    "problem_id": problem_id,
                    "legacy_branch_duplicates": int(branch_legacy_duplicates),
                    "state_branch_duplicates": int(branch_state_duplicates),
                    "candidate_indexes": [int(candidate.get("candidate_index", 0)) for candidate in branch_candidates],
                }
            )

    summary = {
        "input_jsonl": str(args.input_jsonl),
        "controller_json": str(args.controller_json),
        "groups_total": int(total_groups),
        "legacy_duplicate_groups": int(legacy_duplicate_groups),
        "state_duplicate_groups": int(state_duplicate_groups),
        "legacy_duplicate_branch_groups": int(legacy_duplicate_branch_groups),
        "state_duplicate_branch_groups": int(state_duplicate_branch_groups),
        "branch_groups_improved_by_state_signature": int(improved_branch_groups),
        "legacy_duplicate_candidates": int(legacy_duplicate_candidates),
        "state_duplicate_candidates": int(state_duplicate_candidates),
        "legacy_duplicate_branch_candidates": int(branch_legacy_duplicate_candidates),
        "state_duplicate_branch_candidates": int(branch_state_duplicate_candidates),
        "sample_rows": sample_rows,
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