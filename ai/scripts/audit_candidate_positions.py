from __future__ import annotations

import argparse
import json
from collections import Counter
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


def _fraction(count: float, total: int) -> float:
    return float(count) / float(total) if total else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit positive-label position statistics for grouped candidate JSONL.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    rows = _read_grouped_jsonl(Path(str(args.input_jsonl)))
    positive_count_histogram: Counter[int] = Counter()
    positive_position_histogram: Counter[int] = Counter()
    groups_with_positive = 0
    groups_with_first_positive = 0
    expected_first_positive_total = 0.0

    for row in rows:
        candidates = list(row.get("candidates", []))
        positive_positions = [
            index for index, cand in enumerate(candidates) if float(cand.get("label", 0.0)) > 0.5
        ]
        positive_count = len(positive_positions)
        positive_count_histogram[positive_count] += 1
        if positive_count == 0:
            continue
        groups_with_positive += 1
        positive_position_histogram.update(positive_positions)
        expected_first_positive_total += float(positive_count) / float(len(candidates)) if candidates else 0.0
        if 0 in positive_positions:
            groups_with_first_positive += 1

    summary = {
        "input_jsonl": str(args.input_jsonl),
        "groups_total": int(len(rows)),
        "groups_with_positive": int(groups_with_positive),
        "positive_count_histogram": {
            str(key): int(value) for key, value in sorted(positive_count_histogram.items())
        },
        "positive_position_histogram": {
            str(key): int(value) for key, value in sorted(positive_position_histogram.items())
        },
        "observed_first_position_positive_rate": _fraction(groups_with_first_positive, groups_with_positive),
        "expected_first_position_positive_rate_from_label_mix": _fraction(
            expected_first_positive_total,
            groups_with_positive,
        ),
    }
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