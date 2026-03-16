from __future__ import annotations

import argparse
import json
import random
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


def _truncate(text: str, limit: int) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)] + "..."


def _is_viable_group(row: dict[str, object], *, min_negatives: int) -> bool:
    candidates = list(row.get("candidates", []))
    positives = sum(1 for cand in candidates if float(cand.get("label", 0.0)) > 0.5)
    negatives = len(candidates) - positives
    return positives >= 1 and negatives >= min_negatives


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample grouped candidate traces for manual hardness inspection.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_negatives", type=int, default=2)
    parser.add_argument("--require_viable", action="store_true")
    parser.add_argument("--snippet_chars", type=int, default=220)
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    rows = _read_grouped_jsonl(Path(str(args.input_jsonl)))
    if bool(args.require_viable):
        rows = [row for row in rows if _is_viable_group(row, min_negatives=int(args.min_negatives))]

    rng = random.Random(int(args.seed))
    sample = list(rows)
    rng.shuffle(sample)
    sample = sample[: max(0, int(args.limit))]

    inspection_rows: list[dict[str, object]] = []
    for row in sample:
        candidates = list(row.get("candidates", []))
        inspection_rows.append(
            {
                "problem_id": str(row.get("problem_id", "")),
                "prompt": str(row.get("prompt", "")),
                "n_candidates": len(candidates),
                "n_positives": sum(1 for cand in candidates if float(cand.get("label", 0.0)) > 0.5),
                "unique_final_answers": len({_extract_candidate_answer(str(cand.get("text", ""))) for cand in candidates}),
                "candidates": [
                    {
                        "label": float(cand.get("label", 0.0)),
                        "role_tag": str(cand.get("role_tag", "")),
                        "provenance_tag": str(cand.get("provenance_tag", "")),
                        "final_answer": _extract_candidate_answer(str(cand.get("text", ""))),
                        "text_snippet": _truncate(str(cand.get("text", "")), int(args.snippet_chars)),
                    }
                    for cand in candidates
                ],
            }
        )

    rendered = json.dumps(inspection_rows, indent=2)
    print(rendered)

    if str(args.output_json).strip():
        output_path = Path(str(args.output_json))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())