from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _read_grouped_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _extract_last_numeric(text: str) -> str | None:
    matches = re.findall(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))
    return matches[-1] if matches else None


def _extract_candidate_answer(text: str) -> str:
    raw = str(text).strip()
    answer_match = re.search(r"^answer\s*:\s*(.+)$", raw, flags=re.IGNORECASE | re.MULTILINE)
    if answer_match:
        tail = answer_match.group(1).strip()
        numeric = _extract_last_numeric(tail)
        return numeric if numeric is not None else tail.lower()
    match = re.search(r"final answer\s*:\s*(.+)$", raw, flags=re.IGNORECASE | re.MULTILINE)
    if match:
        tail = match.group(1).strip()
        numeric = _extract_last_numeric(tail)
        return numeric if numeric is not None else tail.lower()
    numeric = _extract_last_numeric(raw)
    return numeric if numeric is not None else raw.lower()


def _fraction(count: int, total: int) -> float:
    return float(count) / float(total) if total else 0.0


def summarize_groups(rows: list[dict[str, object]], *, model: str, dataset: str) -> dict[str, object]:
    groups_total = len(rows)
    candidates_per_group: list[int] = []
    positives_per_group: list[int] = []
    unique_answers_per_group: list[int] = []
    unique_texts_per_group: list[int] = []
    role_counts: dict[str, int] = {}
    role_positive_counts: dict[str, int] = {}

    for row in rows:
        candidates = list(row.get("candidates", []))
        candidate_count = len(candidates)
        positive_count = sum(1 for cand in candidates if float(cand.get("label", 0.0)) > 0.5)
        answers = {_extract_candidate_answer(str(cand.get("text", ""))) for cand in candidates}
        texts = {str(cand.get("text", "")).strip() for cand in candidates if str(cand.get("text", "")).strip()}
        candidates_per_group.append(candidate_count)
        positives_per_group.append(positive_count)
        unique_answers_per_group.append(len(answers))
        unique_texts_per_group.append(len(texts))
        for cand in candidates:
            role_tag = str(cand.get("role_tag", "")).strip()
            if not role_tag:
                continue
            role_counts[role_tag] = role_counts.get(role_tag, 0) + 1
            if float(cand.get("label", 0.0)) > 0.5:
                role_positive_counts[role_tag] = role_positive_counts.get(role_tag, 0) + 1

    groups_with_positive = sum(1 for count in positives_per_group if count >= 1)
    groups_with_1p_2n = sum(
        1
        for candidate_count, positive_count in zip(candidates_per_group, positives_per_group)
        if positive_count >= 1 and (candidate_count - positive_count) >= 2
    )
    groups_with_multiple_positives = sum(1 for count in positives_per_group if count >= 2)
    groups_with_zero_positive = sum(1 for count in positives_per_group if count == 0)

    summary = {
        "model": model,
        "dataset": dataset,
        "n_examples": groups_total,
        "groups_total": groups_total,
        "groups_with_positive": groups_with_positive,
        "groups_with_zero_positive": groups_with_zero_positive,
        "groups_with_1p_2n": groups_with_1p_2n,
        "groups_with_multiple_positives": groups_with_multiple_positives,
        "mean_candidates_per_group": _fraction(sum(candidates_per_group), groups_total),
        "mean_positives_per_group": _fraction(sum(positives_per_group), groups_total),
        "mean_unique_final_answers": _fraction(sum(unique_answers_per_group), groups_total),
        "mean_unique_candidate_texts": _fraction(sum(unique_texts_per_group), groups_total),
        "fraction_groups_with_positive": _fraction(groups_with_positive, groups_total),
        "fraction_groups_with_1p_2n": _fraction(groups_with_1p_2n, groups_total),
        "fraction_groups_with_multiple_positives": _fraction(groups_with_multiple_positives, groups_total),
    }
    if role_counts:
        summary["role_stats"] = {
            role_tag: {
                "candidate_count": int(role_counts[role_tag]),
                "positive_count": int(role_positive_counts.get(role_tag, 0)),
                "positive_rate": _fraction(role_positive_counts.get(role_tag, 0), role_counts[role_tag]),
            }
            for role_tag in sorted(role_counts)
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize grouped candidate-trace JSONL for benchmark viability.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--model", default="unknown")
    parser.add_argument("--dataset", default="unknown")
    args = parser.parse_args()

    input_path = Path(str(args.input_jsonl))
    rows = _read_grouped_jsonl(input_path)
    summary = summarize_groups(rows, model=str(args.model), dataset=str(args.dataset))
    summary["input_jsonl"] = str(input_path)

    rendered = json.dumps(summary, indent=2)
    print(rendered)

    if str(args.output_json).strip():
        output_path = Path(str(args.output_json))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())