from __future__ import annotations

import argparse
import json
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


def _positive_count(row: dict[str, object]) -> int:
    candidates = list(row.get("candidates", []))
    return sum(1 for cand in candidates if float(cand.get("label", 0.0)) > 0.5)


def _negative_count(row: dict[str, object]) -> int:
    candidates = list(row.get("candidates", []))
    positives = _positive_count(row)
    return len(candidates) - positives


def _keep_row(
    row: dict[str, object],
    *,
    min_positives: int,
    max_positives: int,
    exact_positives: int,
    min_negatives: int,
) -> bool:
    positives = _positive_count(row)
    negatives = _negative_count(row)
    if int(exact_positives) >= 0 and positives != int(exact_positives):
        return False
    if positives < int(min_positives):
        return False
    if int(max_positives) >= 0 and positives > int(max_positives):
        return False
    if negatives < int(min_negatives):
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter grouped candidate JSONL by positive/negative label structure.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_summary_json", default="")
    parser.add_argument("--min_positives", type=int, default=0)
    parser.add_argument("--max_positives", type=int, default=-1)
    parser.add_argument("--exact_positives", type=int, default=-1)
    parser.add_argument("--min_negatives", type=int, default=0)
    args = parser.parse_args()

    input_path = Path(str(args.input_jsonl))
    rows = _read_grouped_jsonl(input_path)
    kept_rows = [
        row
        for row in rows
        if _keep_row(
            row,
            min_positives=int(args.min_positives),
            max_positives=int(args.max_positives),
            exact_positives=int(args.exact_positives),
            min_negatives=int(args.min_negatives),
        )
    ]

    output_path = Path(str(args.output_jsonl))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in kept_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "groups_total": int(len(rows)),
        "groups_kept": int(len(kept_rows)),
        "fraction_kept": (float(len(kept_rows)) / float(len(rows))) if rows else 0.0,
        "min_positives": int(args.min_positives),
        "max_positives": int(args.max_positives),
        "exact_positives": int(args.exact_positives),
        "min_negatives": int(args.min_negatives),
    }
    rendered = json.dumps(summary, indent=2)
    print(rendered)

    if str(args.output_summary_json).strip():
        summary_path = Path(str(args.output_summary_json))
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())