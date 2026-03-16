from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.scripts.generate_candidate_traces import (
    _append_unique_candidate_row,
    _clean_generated_text,
    _extract_candidate_answer,
    _make_near_miss_numeric,
    _structural_challenger_rows,
)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a fresh grouped challenger batch from an existing grouped dataset using deliberate structural challenger constructors.")
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--out_summary_json", default="")
    ap.add_argument("--target_candidates", type=int, default=6)
    ap.add_argument("--near_miss_count", type=int, default=2)
    ap.add_argument("--dependency_conflict_count", type=int, default=3)
    ap.add_argument("--state_reuse_count", type=int, default=2)
    ap.add_argument("--wrong_answer_from_state_count", type=int, default=0)
    ap.add_argument("--reuse_original_wrongs", action="store_true")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    random.seed(int(args.seed))
    input_rows = _read_jsonl(Path(str(args.input_jsonl)))
    output_rows: list[dict[str, object]] = []
    generated_counts: dict[str, int] = {
        "near_miss_numeric": 0,
        "structural_dependency_conflict": 0,
        "structural_state_reuse": 0,
        "structural_wrong_answer_from_state": 0,
        "reused_original_wrong": 0,
    }
    groups_with_anchor = 0
    groups_with_minimum = 0
    candidate_counts: list[float] = []

    for row in input_rows:
        candidates = list(row.get("candidates", []))
        gold_candidates = [candidate for candidate in candidates if float(candidate.get("label", 0.0)) > 0.5]
        if len(gold_candidates) != 1:
            continue
        gold = dict(gold_candidates[0])
        anchor_text = _clean_generated_text(str(gold.get("text", "")))
        gold_answer = _extract_candidate_answer(anchor_text)
        if not anchor_text or not gold_answer:
            continue
        groups_with_anchor += 1

        final_rows: list[dict[str, object]] = []
        final_seen: set[tuple[str, str]] = set()
        _append_unique_candidate_row(
            final_rows,
            {
                "text": anchor_text,
                "role_tag": "anchor",
                "provenance_tag": "natural_correct",
            },
            seen_keys=final_seen,
            deduplicate_candidates=True,
        )

        for index in range(max(0, int(args.near_miss_count))):
            candidate_text = _make_near_miss_numeric(anchor_text, gold_answer=str(gold_answer))
            if not candidate_text:
                continue
            if _append_unique_candidate_row(
                final_rows,
                {
                    "text": str(candidate_text),
                    "role_tag": f"near_miss_{index + 1}",
                    "provenance_tag": "near_miss_numeric",
                },
                seen_keys=final_seen,
                deduplicate_candidates=True,
            ):
                generated_counts["near_miss_numeric"] += 1

        for challenger_row in _structural_challenger_rows(
            anchor_text,
            gold_answer=str(gold_answer),
            dependency_conflict_count=int(args.dependency_conflict_count),
            state_reuse_count=int(args.state_reuse_count),
            wrong_answer_from_state_count=int(args.wrong_answer_from_state_count),
        ):
            if _append_unique_candidate_row(
                final_rows,
                challenger_row,
                seen_keys=final_seen,
                deduplicate_candidates=True,
            ):
                generated_counts[str(challenger_row.get("provenance_tag", ""))] = generated_counts.get(str(challenger_row.get("provenance_tag", "")), 0) + 1

        if bool(args.reuse_original_wrongs):
            for original in candidates:
                if float(original.get("label", 0.0)) > 0.5:
                    continue
                if len(final_rows) >= int(args.target_candidates):
                    break
                if _append_unique_candidate_row(
                    final_rows,
                    {
                        "text": _clean_generated_text(str(original.get("text", ""))),
                        "role_tag": str(original.get("role_tag", "natural_sample") or "natural_sample"),
                        "provenance_tag": str(original.get("provenance_tag", "natural_wrong") or "natural_wrong"),
                    },
                    seen_keys=final_seen,
                    deduplicate_candidates=True,
                ):
                    generated_counts["reused_original_wrong"] += 1

        final_candidates: list[dict[str, object]] = []
        for candidate_row in final_rows[: max(2, int(args.target_candidates))]:
            label = 1.0 if _extract_candidate_answer(str(candidate_row.get("text", ""))) == str(gold_answer) else 0.0
            final_candidates.append(
                {
                    "text": str(candidate_row.get("text", "")),
                    "label": float(label),
                    "role_tag": str(candidate_row.get("role_tag", "")),
                    "provenance_tag": str(candidate_row.get("provenance_tag", "")),
                }
            )
        random.shuffle(final_candidates)
        negative_count = sum(1 for candidate in final_candidates if float(candidate.get("label", 0.0)) <= 0.5)
        groups_with_minimum += int(negative_count >= 3)
        candidate_counts.append(float(len(final_candidates)))
        output_rows.append(
            {
                "problem_id": str(row.get("problem_id", "")),
                "prompt": str(row.get("prompt", "")),
                "candidates": final_candidates,
            }
        )

    out_jsonl = Path(str(args.out_jsonl))
    _write_jsonl(out_jsonl, output_rows)
    summary = {
        "input_jsonl": str(args.input_jsonl),
        "out_jsonl": str(out_jsonl),
        "groups_total": int(len(input_rows)),
        "groups_written": int(len(output_rows)),
        "groups_with_anchor": int(groups_with_anchor),
        "groups_with_at_least_three_negatives": int(groups_with_minimum),
        "target_candidates": int(args.target_candidates),
        "near_miss_count": int(args.near_miss_count),
        "dependency_conflict_count": int(args.dependency_conflict_count),
        "state_reuse_count": int(args.state_reuse_count),
        "wrong_answer_from_state_count": int(args.wrong_answer_from_state_count),
        "reuse_original_wrongs": bool(args.reuse_original_wrongs),
        "mean_candidates_per_group": _mean(candidate_counts),
        "generated_counts": dict(sorted(generated_counts.items())),
        "sample_problem_ids": [str(row.get("problem_id", "")) for row in output_rows[:12]],
    }
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if str(args.out_summary_json).strip():
        out_summary = Path(str(args.out_summary_json))
        out_summary.parent.mkdir(parents=True, exist_ok=True)
        out_summary.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote {out_summary}")
    print(f"wrote {out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())