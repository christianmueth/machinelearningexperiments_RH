from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


def _extract_numeric_answer(text: str) -> str | None:
    matches = re.findall(r"-?\d+(?:\.\d+)?", str(text))
    return matches[-1] if matches else None


def _perturb_numeric_string(value: str, *, rng: random.Random) -> str:
    if "." in value:
        numeric = float(value)
        offset = rng.choice([-2.0, -1.0, 1.0, 2.0])
        candidate = numeric + offset
        return str(int(candidate)) if candidate.is_integer() else f"{candidate:.2f}"
    numeric = int(value)
    return str(numeric + rng.choice([-3, -2, -1, 1, 2, 3]))


def _build_candidate_group(prompt: str, answer: str, *, rng: random.Random, limit: int) -> list[dict[str, object]]:
    answer_text = str(answer).strip()
    numeric_answer = _extract_numeric_answer(answer_text)
    gold_final = numeric_answer if numeric_answer is not None else answer_text

    candidates: list[dict[str, object]] = [
        {
            "text": f"Reasoning: follow the prompt constraints carefully and compute the result step by step. Final answer: {gold_final}",
            "label": 1.0,
        }
    ]

    distractors: list[str] = []
    if numeric_answer is not None:
        for _ in range(max(2, limit - 1)):
            wrong = _perturb_numeric_string(numeric_answer, rng=rng)
            distractors.append(
                f"Reasoning: use a plausible but incorrect arithmetic shortcut. Final answer: {wrong}"
            )
    else:
        distractors.extend(
            [
                "Reasoning: infer a conclusion that is too weak to settle the problem. Final answer: cannot determine.",
                "Reasoning: apply the wrong rule and reverse the intended implication. Final answer: false.",
                "Reasoning: ignore part of the prompt and jump to a generic conclusion. Final answer: unknown.",
            ]
        )

    for text in distractors[: max(1, limit - 1)]:
        candidates.append({"text": text, "label": 0.0})
    candidates = candidates[:limit]
    rng.shuffle(candidates)
    return candidates


def _read_jsonl_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _load_hf_records(dataset_name: str, split: str, subset: str | None) -> list[dict[str, object]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Hugging Face datasets is not installed. Install ai/requirements-training.txt first or use --input_jsonl."
        ) from exc

    if subset:
        dataset = load_dataset(dataset_name, subset, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    return [dict(row) for row in dataset]


def _select_fields(record: dict[str, object], prompt_field: str, answer_field: str) -> tuple[str, str]:
    prompt = str(record[prompt_field]).strip()
    answer = str(record[answer_field]).strip()
    return prompt, answer


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare grouped candidate-trace JSONL for the anchored-oracle reranker.")
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--input_jsonl", default="")
    ap.add_argument("--hf_dataset", default="")
    ap.add_argument("--hf_subset", default="")
    ap.add_argument("--hf_split", default="train")
    ap.add_argument("--prompt_field", default="question")
    ap.add_argument("--answer_field", default="answer")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--candidates_per_problem", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not str(args.input_jsonl).strip() and not str(args.hf_dataset).strip():
        raise SystemExit("Provide either --input_jsonl or --hf_dataset.")

    rng = random.Random(int(args.seed))
    if str(args.input_jsonl).strip():
        records = _read_jsonl_records(Path(str(args.input_jsonl)))
    else:
        records = _load_hf_records(str(args.hf_dataset), str(args.hf_split), str(args.hf_subset).strip() or None)

    rows: list[dict[str, object]] = []
    for index, record in enumerate(records[: max(0, int(args.limit))]):
        prompt, answer = _select_fields(record, str(args.prompt_field), str(args.answer_field))
        candidates = _build_candidate_group(prompt, answer, rng=rng, limit=max(2, int(args.candidates_per_problem)))
        rows.append(
            {
                "problem_id": str(record.get("problem_id", f"sample_{index:06d}")),
                "prompt": prompt,
                "candidates": candidates,
            }
        )

    out_path = Path(str(args.out_jsonl))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"wrote {out_path}")
    print(f"n_examples={len(rows)}")
    print(f"candidates_per_problem={max(2, int(args.candidates_per_problem))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())