from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml_oracle.datasets import load_reasoning_examples, materialize_dataset
from src.ml_oracle.frozen_oracle_client import AnchoredOracleClient
from src.ml_oracle.reranker import PairwiseMLPReranker, ranking_metrics
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


def _candidate_trace_stats(translator: HeuristicAnchoredTranslator, text: str, *, prompt: str) -> dict[str, object]:
    numeric_tokens = translator._extract_numeric_tokens(text)
    correct_equations, incorrect_equations = translator._equation_stats(text)
    answer_support = translator._answer_support_score(text)
    answer_match = len(translator._extract_numeric_tokens(text.split("Reasoning:", 1)[0])) > 0
    reasoning_steps = translator.split_reasoning_steps(text)
    reasoning_states = translator.reasoning_states_for_trace(text, prompt=prompt)
    step_queries = translator.queries_for_trace(text, prompt=prompt)
    return {
        "n_numeric_tokens": int(len(numeric_tokens)),
        "correct_equations": int(correct_equations),
        "incorrect_equations": int(incorrect_equations),
        "answer_support_score": float(answer_support),
        "has_answer_numeric": bool(answer_match),
        "reasoning_step_lines": int(len(reasoning_steps)),
        "reasoning_states": int(len(reasoning_states)),
        "step_query_count": int(len(step_queries)),
        "step_query_us": [float(query.u) for query in step_queries],
        "step_operations": [str(state.operation) for state in reasoning_states],
        "contains_approx_language": bool(any(token in str(text).lower() for token in ("approx", "approximately", "maybe", "guess", "unknown", "cannot"))),
    }


def _group_margin(scores: np.ndarray) -> float:
    local = np.sort(np.asarray(scores, dtype=np.float64))[::-1]
    if local.size <= 1:
        return 0.0
    return float(local[0] - local[1])


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "max": 0.0, "mean": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _bucket_summary(groups: list[dict[str, object]]) -> dict[str, object]:
    if not groups:
        return {
            "group_count": 0,
            "oracle_accuracy": 0.0,
            "avg_margin": 0.0,
            "avg_unique_queries": 0.0,
            "avg_unique_u_values": 0.0,
            "avg_top_score": 0.0,
            "avg_second_score": 0.0,
            "avg_top_incorrect_equations": 0.0,
            "avg_gold_incorrect_equations": 0.0,
        }
    return {
        "group_count": int(len(groups)),
        "oracle_accuracy": float(sum(1 for row in groups if row["oracle_top_correct"]) / len(groups)),
        "avg_margin": float(statistics.fmean(float(row["oracle_margin"]) for row in groups)),
        "avg_unique_queries": float(statistics.fmean(float(row["unique_query_keys"]) for row in groups)),
        "avg_unique_u_values": float(statistics.fmean(float(row["unique_u_values"]) for row in groups)),
        "avg_top_score": float(statistics.fmean(float(row["top_score"]) for row in groups)),
        "avg_second_score": float(statistics.fmean(float(row["second_score"]) for row in groups)),
        "avg_top_incorrect_equations": float(statistics.fmean(float(row["top_candidate"]["trace_stats"]["incorrect_equations"]) for row in groups)),
        "avg_gold_incorrect_equations": float(statistics.fmean(float(row["gold_candidate"]["trace_stats"]["incorrect_equations"]) for row in groups)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose oracle-only uncertainty, query spread, and low-margin failure modes.")
    parser.add_argument("--train_dataset", required=True)
    parser.add_argument("--eval_dataset", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--low_margin_percentile", type=float, default=25.0)
    parser.add_argument("--max_failure_groups", type=int, default=15)
    parser.add_argument("--step_aware_queries", action="store_true")
    args = parser.parse_args()

    translator = HeuristicAnchoredTranslator(use_step_aware_queries=bool(args.step_aware_queries))
    client = AnchoredOracleClient()

    train_examples = load_reasoning_examples(str(args.train_dataset))
    eval_examples = load_reasoning_examples(str(args.eval_dataset))
    X_train, y_train, groups_train = materialize_dataset(
        train_examples,
        client=client,
        translator=translator,
        feature_mode="oracle",
    )
    X_eval, y_eval, groups_eval = materialize_dataset(
        eval_examples,
        client=client,
        translator=translator,
        feature_mode="oracle",
    )

    model = PairwiseMLPReranker(input_dim=int(X_train.shape[1]), hidden_dim=int(args.hidden_dim), seed=int(args.seed))
    history = model.fit(
        X_train,
        y_train,
        groups_train,
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    train_scores = model.score(X_train)
    eval_scores = model.score(X_eval)

    eval_rows: list[dict[str, object]] = []
    query_histogram: Counter[tuple[object, ...]] = Counter()
    u_histogram: Counter[float] = Counter()
    cursor = 0
    for example_index, example in enumerate(eval_examples):
        count = len(example.candidates)
        idx = np.arange(cursor, cursor + count, dtype=np.int64)
        local_scores = eval_scores[idx]
        local_labels = y_eval[idx]
        order = idx[np.argsort(-local_scores, kind="mergesort")]
        top_idx = int(order[0])
        second_idx = int(order[1]) if order.size > 1 else int(order[0])
        gold_local_pos = int(np.argmax(local_labels))
        gold_idx = int(idx[gold_local_pos])

        candidates: list[dict[str, object]] = []
        query_keys: list[tuple[object, ...]] = []
        u_values: list[float] = []
        for candidate_index, candidate in enumerate(example.candidates):
            global_idx = int(idx[candidate_index])
            query = candidate.oracle_query or translator.query_for_trace(candidate.text, prompt=example.prompt)
            trace_stats = _candidate_trace_stats(translator, candidate.text, prompt=example.prompt)
            key = _query_key(query)
            query_histogram[key] += 1
            u_histogram[float(query.u)] += 1
            query_keys.append(key)
            u_values.append(float(query.u))
            candidates.append(
                {
                    "candidate_index": int(candidate_index),
                    "score": float(eval_scores[global_idx]),
                    "label": float(candidate.label),
                    "is_top_ranked": bool(global_idx == top_idx),
                    "is_gold": bool(global_idx == gold_idx),
                    "answer_line": candidate.text.splitlines()[0] if candidate.text.splitlines() else candidate.text[:200],
                    "query": {
                        "u": float(query.u),
                        "include_perturbation_features": bool(query.include_perturbation_features),
                        "feature_families": list(query.feature_families),
                    },
                    "trace_stats": trace_stats,
                }
            )

        top_candidate = next(item for item in candidates if item["is_top_ranked"])
        gold_candidate = next(item for item in candidates if item["is_gold"])
        eval_rows.append(
            {
                "problem_id": str(example.problem_id),
                "example_index": int(example_index),
                "prompt": str(example.prompt),
                "oracle_margin": float(_group_margin(local_scores)),
                "oracle_top_correct": bool(top_idx == gold_idx),
                "top_score": float(eval_scores[top_idx]),
                "second_score": float(eval_scores[second_idx]),
                "unique_query_keys": int(len(set(query_keys))),
                "unique_u_values": int(len(set(u_values))),
                "top_candidate": top_candidate,
                "gold_candidate": gold_candidate,
                "candidates": candidates,
            }
        )
        cursor += count

    margins = [float(row["oracle_margin"]) for row in eval_rows]
    threshold = float(np.percentile(np.asarray(margins, dtype=np.float64), float(args.low_margin_percentile))) if margins else 0.0
    low_margin_groups = [row for row in eval_rows if float(row["oracle_margin"]) <= threshold]
    high_margin_groups = [row for row in eval_rows if float(row["oracle_margin"]) > threshold]
    wrong_groups = [row for row in eval_rows if not bool(row["oracle_top_correct"])]
    wrong_groups.sort(key=lambda row: (float(row["oracle_margin"]), -float(row["unique_query_keys"])))

    summary = {
        "train_dataset": str(args.train_dataset),
        "eval_dataset": str(args.eval_dataset),
        "oracle_model": {
            "step_aware_queries": bool(args.step_aware_queries),
            "seed": int(args.seed),
            "hidden_dim": int(args.hidden_dim),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "train_loss_last": float(history[-1] if history else 0.0),
            "train_metrics": ranking_metrics(train_scores, y_train, groups_train),
            "eval_metrics": ranking_metrics(eval_scores, y_eval, groups_eval),
        },
        "eval_margin_stats": _quantiles(margins),
        "low_margin_threshold": float(threshold),
        "low_margin_summary": _bucket_summary(low_margin_groups),
        "high_margin_summary": _bucket_summary(high_margin_groups),
        "query_u_histogram": [{"u": float(u), "count": int(count)} for u, count in sorted(u_histogram.items())],
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
        "failure_examples": wrong_groups[: max(0, int(args.max_failure_groups))],
    }

    out_path = Path(str(args.out_json))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())