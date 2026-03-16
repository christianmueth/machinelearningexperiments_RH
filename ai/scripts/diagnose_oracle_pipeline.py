from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml_oracle.datasets import load_reasoning_examples
from src.ml_oracle.frozen_oracle_client import AnchoredOracleClient
from src.ml_oracle.translator import HeuristicAnchoredTranslator


def _query_key(query) -> tuple[object, ...]:
    return (
        float(query.u),
        tuple(str(part) for part in query.feature_families),
        str(query.sigma_mode),
        str(query.cluster_window),
        bool(query.include_perturbation_features),
        str(query.pipeline_tag),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose where oracle candidate-level variation collapses in grouped datasets.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--limit_groups", type=int, default=10)
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    examples = load_reasoning_examples(str(args.input_jsonl))
    translator = HeuristicAnchoredTranslator()
    client = AnchoredOracleClient()

    query_histogram: Counter[tuple[object, ...]] = Counter()
    vector_histogram: Counter[tuple[float, ...]] = Counter()
    group_rows: list[dict[str, object]] = []

    for example in examples[: max(0, int(args.limit_groups))]:
        group_query_keys: list[tuple[object, ...]] = []
        group_vectors: list[tuple[float, ...]] = []
        candidates: list[dict[str, object]] = []
        for candidate_index, candidate in enumerate(example.candidates):
            query = candidate.oracle_query or translator.query_for_trace(candidate.text, prompt=example.prompt)
            query_key = _query_key(query)
            oracle_vector = tuple(np.round(client.oracle_vector(query), 8).tolist())
            query_histogram[query_key] += 1
            vector_histogram[oracle_vector] += 1
            group_query_keys.append(query_key)
            group_vectors.append(oracle_vector)
            candidates.append(
                {
                    "candidate_index": int(candidate_index),
                    "label": float(candidate.label),
                    "answer_line": candidate.text.splitlines()[0] if candidate.text.splitlines() else candidate.text[:120],
                    "query_key": {
                        "u": float(query.u),
                        "feature_families": list(query.feature_families),
                        "sigma_mode": str(query.sigma_mode),
                        "cluster_window": str(query.cluster_window),
                        "include_perturbation_features": bool(query.include_perturbation_features),
                        "pipeline_tag": str(query.pipeline_tag),
                    },
                }
            )
        group_rows.append(
            {
                "problem_id": str(example.problem_id),
                "prompt": str(example.prompt),
                "n_candidates": int(len(example.candidates)),
                "unique_query_keys": int(len(set(group_query_keys))),
                "unique_oracle_vectors": int(len(set(group_vectors))),
                "candidates": candidates,
            }
        )

    summary = {
        "input_jsonl": str(args.input_jsonl),
        "groups_analyzed": int(len(group_rows)),
        "unique_query_keys_total": int(len(query_histogram)),
        "unique_oracle_vectors_total": int(len(vector_histogram)),
        "query_key_histogram": [
            {
                "count": int(count),
                "query_key": {
                    "u": float(key[0]),
                    "feature_families": list(key[1]),
                    "sigma_mode": str(key[2]),
                    "cluster_window": str(key[3]),
                    "include_perturbation_features": bool(key[4]),
                    "pipeline_tag": str(key[5]),
                },
            }
            for key, count in query_histogram.most_common()
        ],
        "groups": group_rows,
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