from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.scripts.run_residual_reranker_experiment import _candidate_text_rows, _fit_tfidf, _materialize_branch, _standardize_from_train, _transform_tfidf
from src.ml_oracle.datasets import load_reasoning_examples
from src.ml_oracle.frozen_oracle_client import AnchoredOracleClient
from src.ml_oracle.reranker import PairwiseMLPReranker, ranking_metrics
from src.ml_oracle.translator import HeuristicAnchoredTranslator


def _prepare_text_features(
    dataset_path: str,
    *,
    client: AnchoredOracleClient,
    translator: HeuristicAnchoredTranslator,
    text_encoder: str,
    text_dim: int,
    tfidf_max_features: int,
    hf_model: str,
    hf_max_length: int,
    tfidf_state: tuple[dict[str, int], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[dict[str, int], np.ndarray] | None]:
    examples = load_reasoning_examples(str(dataset_path))
    if str(text_encoder) == "tfidf":
        texts, labels, groups = _candidate_text_rows(examples)
        if tfidf_state is None:
            tfidf_state = _fit_tfidf(texts, max_features=int(tfidf_max_features))
        vocab, idf = tfidf_state
        X = _transform_tfidf(texts, vocab=vocab, idf=idf)
        return X, labels, groups, tfidf_state
    X, labels, groups, _ = _materialize_branch(
        str(dataset_path),
        feature_mode="text",
        client=client,
        translator=translator,
        text_encoder=str(text_encoder),
        text_dim=int(text_dim),
        hf_model=str(hf_model),
        hf_max_length=int(hf_max_length),
    )
    return X, labels, groups, tfidf_state


def _group_margin_map(scores: np.ndarray, group_ids: np.ndarray) -> dict[int, float]:
    margin_map: dict[int, float] = {}
    for group in np.unique(group_ids):
        idx = np.flatnonzero(group_ids == int(group))
        if idx.size < 2:
            margin_map[int(group)] = 0.0
            continue
        local = np.sort(np.asarray(scores[idx], dtype=np.float64))[::-1]
        margin_map[int(group)] = float(local[0] - local[1])
    return margin_map


def _group_top_correct_map(scores: np.ndarray, labels: np.ndarray, group_ids: np.ndarray) -> dict[int, bool]:
    result: dict[int, bool] = {}
    for group in np.unique(group_ids):
        idx = np.flatnonzero(group_ids == int(group))
        if idx.size == 0:
            result[int(group)] = False
            continue
        local_scores = np.asarray(scores[idx], dtype=np.float64)
        order = np.argsort(-local_scores, kind="mergesort")
        top_index = int(idx[order[0]])
        result[int(group)] = bool(float(labels[top_index]) > 0.5)
    return result


def _subset_metrics(scores: np.ndarray, labels: np.ndarray, group_ids: np.ndarray, selected_groups: set[int]) -> dict[str, float]:
    if not selected_groups:
        return {"group_accuracy": 0.0, "mrr": 0.0, "ndcg": 0.0}
    mask = np.asarray([int(group) in selected_groups for group in group_ids], dtype=bool)
    if not mask.any():
        return {"group_accuracy": 0.0, "mrr": 0.0, "ndcg": 0.0}
    return ranking_metrics(scores[mask], labels[mask], group_ids[mask])


def _margin_regime_summary(
    oracle_scores: np.ndarray,
    text_scores: np.ndarray,
    gated_scores: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray,
    *,
    margin_map: dict[int, float],
    margin_threshold: float,
) -> dict[str, object]:
    low_margin_groups = {int(group) for group, margin in margin_map.items() if float(margin) <= float(margin_threshold)}
    high_margin_groups = {int(group) for group, margin in margin_map.items() if float(margin) > float(margin_threshold)}

    oracle_correct = _group_top_correct_map(oracle_scores, labels, group_ids)
    text_correct = _group_top_correct_map(text_scores, labels, group_ids)
    gated_correct = _group_top_correct_map(gated_scores, labels, group_ids)

    def _group_overlap(groups: set[int]) -> dict[str, int]:
        return {
            "group_count": int(len(groups)),
            "oracle_correct": int(sum(1 for group in groups if oracle_correct.get(group, False))),
            "text_correct": int(sum(1 for group in groups if text_correct.get(group, False))),
            "gated_correct": int(sum(1 for group in groups if gated_correct.get(group, False))),
            "text_helps_when_oracle_wrong": int(
                sum(1 for group in groups if (not oracle_correct.get(group, False)) and text_correct.get(group, False))
            ),
            "gated_helps_when_oracle_wrong": int(
                sum(1 for group in groups if (not oracle_correct.get(group, False)) and gated_correct.get(group, False))
            ),
            "gated_hurts_when_oracle_right": int(
                sum(1 for group in groups if oracle_correct.get(group, False) and (not gated_correct.get(group, False)))
            ),
        }

    return {
        "margin_threshold": float(margin_threshold),
        "low_margin": {
            **_group_overlap(low_margin_groups),
            "metrics": {
                "oracle": _subset_metrics(oracle_scores, labels, group_ids, low_margin_groups),
                "text": _subset_metrics(text_scores, labels, group_ids, low_margin_groups),
                "gated": _subset_metrics(gated_scores, labels, group_ids, low_margin_groups),
            },
        },
        "high_margin": {
            **_group_overlap(high_margin_groups),
            "metrics": {
                "oracle": _subset_metrics(oracle_scores, labels, group_ids, high_margin_groups),
                "text": _subset_metrics(text_scores, labels, group_ids, high_margin_groups),
                "gated": _subset_metrics(gated_scores, labels, group_ids, high_margin_groups),
            },
        },
    }


def _apply_gate(
    oracle_scores: np.ndarray,
    text_scores: np.ndarray,
    group_ids: np.ndarray,
    *,
    margin_map: dict[int, float],
    margin_threshold: float,
    text_lambda: float,
) -> np.ndarray:
    final_scores = np.asarray(oracle_scores, dtype=np.float64).copy()
    for group in np.unique(group_ids):
        group_margin = float(margin_map.get(int(group), 0.0))
        if group_margin > float(margin_threshold):
            continue
        idx = np.flatnonzero(group_ids == int(group))
        final_scores[idx] += float(text_lambda) * np.asarray(text_scores[idx], dtype=np.float64)
    return final_scores


def main() -> int:
    ap = argparse.ArgumentParser(description="Run confidence-gated oracle-first fusion experiments.")
    ap.add_argument("--train_dataset", required=True)
    ap.add_argument("--eval_dataset", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--text_encoder", choices=["hashed", "hf", "tfidf"], default="hf")
    ap.add_argument("--text_dim", type=int, default=256)
    ap.add_argument("--tfidf_max_features", type=int, default=2048)
    ap.add_argument("--hf_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--hf_max_length", type=int, default=256)
    ap.add_argument("--oracle_hidden_dim", type=int, default=64)
    ap.add_argument("--text_hidden_dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--text_lambda", type=float, default=0.1)
    ap.add_argument("--margin_percentile", type=float, default=25.0)
    ap.add_argument("--margin_threshold_override", type=float, default=float("nan"))
    ap.add_argument("--standardize_branch_scores", action="store_true")
    args = ap.parse_args()

    client = AnchoredOracleClient()
    translator = HeuristicAnchoredTranslator()

    X_train_text, y_train, groups_train, tfidf_state = _prepare_text_features(
        str(args.train_dataset),
        client=client,
        translator=translator,
        text_encoder=str(args.text_encoder),
        text_dim=int(args.text_dim),
        tfidf_max_features=int(args.tfidf_max_features),
        hf_model=str(args.hf_model),
        hf_max_length=int(args.hf_max_length),
    )
    X_eval_text, y_eval, groups_eval, _ = _prepare_text_features(
        str(args.eval_dataset),
        client=client,
        translator=translator,
        text_encoder=str(args.text_encoder),
        text_dim=int(args.text_dim),
        tfidf_max_features=int(args.tfidf_max_features),
        hf_model=str(args.hf_model),
        hf_max_length=int(args.hf_max_length),
        tfidf_state=tfidf_state,
    )

    oracle_text_encoder = "hashed" if str(args.text_encoder) == "tfidf" else str(args.text_encoder)
    X_train_oracle, _, _, n_train = _materialize_branch(
        str(args.train_dataset),
        feature_mode="oracle",
        client=client,
        translator=translator,
        text_encoder=oracle_text_encoder,
        text_dim=int(args.text_dim),
        hf_model=str(args.hf_model),
        hf_max_length=int(args.hf_max_length),
    )
    X_eval_oracle, _, _, n_eval = _materialize_branch(
        str(args.eval_dataset),
        feature_mode="oracle",
        client=client,
        translator=translator,
        text_encoder=oracle_text_encoder,
        text_dim=int(args.text_dim),
        hf_model=str(args.hf_model),
        hf_max_length=int(args.hf_max_length),
    )

    oracle_model = PairwiseMLPReranker(input_dim=int(X_train_oracle.shape[1]), hidden_dim=int(args.oracle_hidden_dim), seed=int(args.seed))
    text_model = PairwiseMLPReranker(input_dim=int(X_train_text.shape[1]), hidden_dim=int(args.text_hidden_dim), seed=int(args.seed) + 1000)
    oracle_history = oracle_model.fit(X_train_oracle, y_train, groups_train, epochs=int(args.epochs), lr=float(args.lr), weight_decay=float(args.weight_decay))
    text_history = text_model.fit(X_train_text, y_train, groups_train, epochs=int(args.epochs), lr=float(args.lr), weight_decay=float(args.weight_decay))

    train_oracle_scores = oracle_model.score(X_train_oracle)
    eval_oracle_scores = oracle_model.score(X_eval_oracle)
    train_text_scores = text_model.score(X_train_text)
    eval_text_scores = text_model.score(X_eval_text)

    score_stats: dict[str, float] = {}
    if bool(args.standardize_branch_scores):
        train_oracle_scores, eval_oracle_scores, oracle_mean, oracle_std = _standardize_from_train(train_oracle_scores, eval_oracle_scores)
        train_text_scores, eval_text_scores, text_mean, text_std = _standardize_from_train(train_text_scores, eval_text_scores)
        score_stats = {
            "oracle_train_score_mean": oracle_mean,
            "oracle_train_score_std": oracle_std,
            "text_train_score_mean": text_mean,
            "text_train_score_std": text_std,
        }

    train_margin_map = _group_margin_map(train_oracle_scores, groups_train)
    margin_values = np.asarray(list(train_margin_map.values()), dtype=np.float64)
    if np.isfinite(float(args.margin_threshold_override)):
        margin_threshold = float(args.margin_threshold_override)
    else:
        margin_threshold = float(np.percentile(margin_values, float(args.margin_percentile))) if margin_values.size else 0.0
    eval_margin_map = _group_margin_map(eval_oracle_scores, groups_eval)

    train_gated_scores = _apply_gate(
        train_oracle_scores,
        train_text_scores,
        groups_train,
        margin_map=train_margin_map,
        margin_threshold=margin_threshold,
        text_lambda=float(args.text_lambda),
    )
    eval_gated_scores = _apply_gate(
        eval_oracle_scores,
        eval_text_scores,
        groups_eval,
        margin_map=eval_margin_map,
        margin_threshold=margin_threshold,
        text_lambda=float(args.text_lambda),
    )

    train_gate_active_groups = sum(1 for margin in train_margin_map.values() if margin <= margin_threshold)
    eval_gate_active_groups = sum(1 for margin in eval_margin_map.values() if margin <= margin_threshold)

    summary = {
        "train_dataset": str(args.train_dataset),
        "eval_dataset": str(args.eval_dataset),
        "text_encoder": str(args.text_encoder),
        "hf_model": str(args.hf_model),
        "text_lambda": float(args.text_lambda),
        "margin_percentile": float(args.margin_percentile),
        "margin_threshold": margin_threshold,
        "margin_threshold_override": None if not np.isfinite(float(args.margin_threshold_override)) else float(args.margin_threshold_override),
        "standardize_branch_scores": bool(args.standardize_branch_scores),
        "seed": int(args.seed),
        "train_examples": int(n_train),
        "eval_examples": int(n_eval),
        "oracle_branch": {
            "train_loss_last": float(oracle_history[-1] if oracle_history else 0.0),
            "train_metrics": ranking_metrics(train_oracle_scores, y_train, groups_train),
            "eval_metrics": ranking_metrics(eval_oracle_scores, y_eval, groups_eval),
        },
        "text_branch": {
            "train_loss_last": float(text_history[-1] if text_history else 0.0),
            "train_metrics": ranking_metrics(train_text_scores, y_train, groups_train),
            "eval_metrics": ranking_metrics(eval_text_scores, y_eval, groups_eval),
        },
        "gated": {
            "train_metrics": ranking_metrics(train_gated_scores, y_train, groups_train),
            "eval_metrics": ranking_metrics(eval_gated_scores, y_eval, groups_eval),
            "train_gate_active_groups": int(train_gate_active_groups),
            "eval_gate_active_groups": int(eval_gate_active_groups),
        },
        "train_margin_diagnostics": _margin_regime_summary(
            train_oracle_scores,
            train_text_scores,
            train_gated_scores,
            y_train,
            groups_train,
            margin_map=train_margin_map,
            margin_threshold=margin_threshold,
        ),
        "eval_margin_diagnostics": _margin_regime_summary(
            eval_oracle_scores,
            eval_text_scores,
            eval_gated_scores,
            y_eval,
            groups_eval,
            margin_map=eval_margin_map,
            margin_threshold=margin_threshold,
        ),
        "score_stats": score_stats,
    }

    out_path = Path(str(args.out_json))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())