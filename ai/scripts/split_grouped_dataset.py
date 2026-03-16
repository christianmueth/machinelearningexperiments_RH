from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _read_jsonl(path: Path) -> list[str]:
    rows: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(line)
    return rows


def _write_jsonl(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row)
            handle.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Split a grouped JSONL reasoning dataset into train/eval files without breaking groups.")
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--eval_jsonl", required=True)
    ap.add_argument("--train_fraction", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_summary_json", default="")
    args = ap.parse_args()

    input_path = Path(str(args.input_jsonl))
    rows = _read_jsonl(input_path)
    rng = random.Random(int(args.seed))
    indexed_rows = list(enumerate(rows))
    rng.shuffle(indexed_rows)

    group_count = len(indexed_rows)
    if group_count == 0:
        train_rows: list[str] = []
        eval_rows: list[str] = []
    else:
        train_count = max(1, min(group_count - 1, int(round(float(args.train_fraction) * group_count)))) if group_count > 1 else 1
        train_pairs = sorted(indexed_rows[:train_count], key=lambda pair: pair[0])
        eval_pairs = sorted(indexed_rows[train_count:], key=lambda pair: pair[0])
        train_rows = [row for _, row in train_pairs]
        eval_rows = [row for _, row in eval_pairs]

    _write_jsonl(Path(str(args.train_jsonl)), train_rows)
    _write_jsonl(Path(str(args.eval_jsonl)), eval_rows)

    summary = {
        "input_jsonl": str(args.input_jsonl),
        "train_jsonl": str(args.train_jsonl),
        "eval_jsonl": str(args.eval_jsonl),
        "seed": int(args.seed),
        "train_fraction": float(args.train_fraction),
        "groups_total": int(group_count),
        "groups_train": int(len(train_rows)),
        "groups_eval": int(len(eval_rows)),
    }
    print(json.dumps(summary, indent=2))
    if str(args.out_summary_json).strip():
        out_path = Path(str(args.out_summary_json))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())