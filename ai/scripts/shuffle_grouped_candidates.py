from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Shuffle candidate order within each grouped JSONL example.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = _read_jsonl(Path(str(args.input_jsonl)))
    rng = random.Random(int(args.seed))
    for row in rows:
        candidates = list(row.get("candidates", []))
        rng.shuffle(candidates)
        row["candidates"] = candidates

    output_path = Path(str(args.output_jsonl))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(json.dumps({
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(output_path),
        "groups_total": len(rows),
        "seed": int(args.seed),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())