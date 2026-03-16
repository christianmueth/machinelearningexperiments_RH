from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml_oracle.datasets import load_reasoning_examples
from src.ml_oracle.translator import HeuristicAnchoredTranslator


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit reasoning-trace structure for controller and step-aware translator readiness.")
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    translator = HeuristicAnchoredTranslator()
    examples = load_reasoning_examples(str(args.input_jsonl))

    total_candidates = 0
    total_positive = 0
    step_counts: list[int] = []
    state_counts: list[int] = []
    step_query_counts: list[int] = []
    answer_support_positive = 0
    step_marker_count = 0
    multiline_reasoning_count = 0

    for example in examples:
        for candidate in example.candidates:
            total_candidates += 1
            total_positive += int(candidate.label > 0.5)
            text = str(candidate.text)
            steps = translator.split_reasoning_steps(text)
            states = translator.reasoning_states_for_trace(text, prompt=example.prompt)
            queries = translator.queries_for_trace(text, prompt=example.prompt)
            step_counts.append(int(len(steps)))
            state_counts.append(int(len(states)))
            step_query_counts.append(int(len(queries)))
            if float(translator._answer_support_score(text)) > 0.0:
                answer_support_positive += 1
            if re.search(r"(^|\n)\s*step\s*\d+\s*:", text, flags=re.IGNORECASE):
                step_marker_count += 1
            reasoning = translator._reasoning_text(text)
            if "\n" in reasoning.strip():
                multiline_reasoning_count += 1

    def _frac(count: int) -> float:
        return float(count) / float(total_candidates) if total_candidates else 0.0

    summary = {
        "input_jsonl": str(args.input_jsonl),
        "groups": int(len(examples)),
        "candidates": int(total_candidates),
        "positives": int(total_positive),
        "mean_steps": float(statistics.fmean(step_counts)) if step_counts else 0.0,
        "mean_states": float(statistics.fmean(state_counts)) if state_counts else 0.0,
        "mean_step_queries": float(statistics.fmean(step_query_counts)) if step_query_counts else 0.0,
        "fraction_ge_2_steps": float(sum(1 for value in step_counts if value >= 2) / len(step_counts)) if step_counts else 0.0,
        "fraction_ge_3_steps": float(sum(1 for value in step_counts if value >= 3) / len(step_counts)) if step_counts else 0.0,
        "fraction_ge_4_steps": float(sum(1 for value in step_counts if value >= 4) / len(step_counts)) if step_counts else 0.0,
        "fraction_with_step_markers": _frac(step_marker_count),
        "fraction_multiline_reasoning": _frac(multiline_reasoning_count),
        "fraction_positive_answer_support": _frac(answer_support_positive),
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