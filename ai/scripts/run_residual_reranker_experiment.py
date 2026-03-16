from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml_oracle.datasets import load_reasoning_examples, materialize_dataset
from src.ml_oracle.frozen_oracle_client import AnchoredOracleClient
from src.ml_oracle.oracle_schema import ReasoningExample
from src.ml_oracle.reranker import PairwiseMLPReranker, ranking_metrics
from src.ml_oracle.translator import HeuristicAnchoredTranslator


def _materialize_branch(
    dataset_path: str,
    *,
    feature_mode: str,
    client: AnchoredOracleClient,
    translator: HeuristicAnchoredTranslator,
    text_encoder: str,
    text_dim: int,
    hf_model: str,
    hf_max_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    examples = load_reasoning_examples(dataset_path)
    X, y, groups = materialize_dataset(
        examples,
        client=client,
        translator=translator,
        text_dim=int(text_dim),
        feature_mode=str(feature_mode),
        text_encoder_name=str(text_encoder),
        hf_model=str(hf_model),
        hf_max_length=int(hf_max_length),
    )
    return X, y, groups, len(examples)


def _candidate_text_rows(examples: list[ReasoningExample]) -> tuple[list[str], np.ndarray, np.ndarray]:
    texts: list[str] = []
    labels: list[float] = []
    groups: list[int] = []
    for group_index, example in enumerate(examples):
        for candidate in example.candidates:
            texts.append(f"{example.prompt} {candidate.text}")
            labels.append(float(candidate.label))
            groups.append(int(group_index))
    return texts, np.asarray(labels, dtype=np.float64), np.asarray(groups, dtype=np.int64)


def _tokenize_text(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", str(text).lower())


def _fit_tfidf(train_texts: list[str], *, max_features: int) -> tuple[dict[str, int], np.ndarray]:
    doc_freq: dict[str, int] = {}
    term_totals: dict[str, int] = {}
    for text in train_texts:
        tokens = _tokenize_text(text)
        unique_tokens = set(tokens)
        for token in unique_tokens:
            doc_freq[token] = doc_freq.get(token, 0) + 1
        for token in tokens:
            term_totals[token] = term_totals.get(token, 0) + 1
    ranked_tokens = sorted(term_totals, key=lambda token: (-term_totals[token], token))[: max(1, int(max_features))]
    vocab = {token: index for index, token in enumerate(ranked_tokens)}
    n_docs = max(1, len(train_texts))
    idf = np.zeros((len(vocab),), dtype=np.float64)
    for token, index in vocab.items():
        idf[index] = math.log((1.0 + n_docs) / (1.0 + doc_freq.get(token, 0))) + 1.0
    return vocab, idf


def _transform_tfidf(texts: list[str], *, vocab: dict[str, int], idf: np.ndarray) -> np.ndarray:
    rows: list[np.ndarray] = []
    for text in texts:
        tokens = _tokenize_text(text)
        counts: dict[int, int] = {}
        for token in tokens:
            if token not in vocab:
                continue
            index = vocab[token]
            counts[index] = counts.get(index, 0) + 1
        row = np.zeros((len(vocab),), dtype=np.float64)
        token_total = max(1, sum(counts.values()))
        for index, count in counts.items():
            row[index] = (float(count) / float(token_total)) * idf[index]
        norm = np.linalg.norm(row)
        if norm > 0.0:
            row /= norm
        rows.append(row)
    if not rows:
        return np.zeros((0, len(vocab)), dtype=np.float64)
    return np.vstack(rows).astype(np.float64)


def _standardize_from_train(train_scores: np.ndarray, eval_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    train_scores = np.asarray(train_scores, dtype=np.float64)
    eval_scores = np.asarray(eval_scores, dtype=np.float64)
    mean = float(np.mean(train_scores)) if train_scores.size else 0.0
    std = float(np.std(train_scores)) if train_scores.size else 1.0
    if std <= 1e-12:
        std = 1.0
    return (train_scores - mean) / std, (eval_scores - mean) / std, mean, std


def main() -> int:
    ap = argparse.ArgumentParser(description="Run residual reranker experiments: oracle_score + lambda * text_score.")
    ap.add_argument("--train_dataset", required=True)
    ap.add_argument("--eval_dataset", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--text_encoder", choices=["hashed", "hf", "tfidf"], default="hashed")
    ap.add_argument("--text_dim", type=int, default=256)
    ap.add_argument("--tfidf_max_features", type=int, default=2048)
    ap.add_argument("--hf_model", default="")
    ap.add_argument("--hf_max_length", type=int, default=256)
    ap.add_argument("--oracle_hidden_dim", type=int, default=64)
    ap.add_argument("--text_hidden_dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--text_lambda", type=float, default=0.1)
    ap.add_argument("--standardize_branch_scores", action="store_true")
    args = ap.parse_args()

    client = AnchoredOracleClient()
    translator = HeuristicAnchoredTranslator()

    train_examples = load_reasoning_examples(str(args.train_dataset))
    eval_examples = load_reasoning_examples(str(args.eval_dataset))
    n_train = len(train_examples)
    n_eval = len(eval_examples)
    if str(args.text_encoder) == "tfidf":
        train_texts, y_train, groups_train = _candidate_text_rows(train_examples)
        eval_texts, y_eval, groups_eval = _candidate_text_rows(eval_examples)
        vocab, idf = _fit_tfidf(train_texts, max_features=int(args.tfidf_max_features))
        X_train_text = _transform_tfidf(train_texts, vocab=vocab, idf=idf)
        X_eval_text = _transform_tfidf(eval_texts, vocab=vocab, idf=idf)
    else:
        X_train_text, y_train, groups_train, _ = _materialize_branch(
            str(args.train_dataset),
            feature_mode="text",
            client=client,
            translator=translator,
            text_encoder=str(args.text_encoder),
            text_dim=int(args.text_dim),
            hf_model=str(args.hf_model),
            hf_max_length=int(args.hf_max_length),
        )
        X_eval_text, y_eval, groups_eval, _ = _materialize_branch(
            str(args.eval_dataset),
            feature_mode="text",
            client=client,
            translator=translator,
            text_encoder=str(args.text_encoder),
            text_dim=int(args.text_dim),
            hf_model=str(args.hf_model),
            hf_max_length=int(args.hf_max_length),
        )
    oracle_branch_text_encoder = "hashed" if str(args.text_encoder) == "tfidf" else str(args.text_encoder)
    X_train_oracle, _, _, _ = _materialize_branch(
        str(args.train_dataset),
        feature_mode="oracle",
        client=client,
        translator=translator,
        text_encoder=oracle_branch_text_encoder,
        text_dim=int(args.text_dim),
        hf_model=str(args.hf_model),
        hf_max_length=int(args.hf_max_length),
    )
    X_eval_oracle, _, _, _ = _materialize_branch(
        str(args.eval_dataset),
        feature_mode="oracle",
        client=client,
        translator=translator,
        text_encoder=oracle_branch_text_encoder,
        text_dim=int(args.text_dim),
        hf_model=str(args.hf_model),
        hf_max_length=int(args.hf_max_length),
    )

    oracle_model = PairwiseMLPReranker(
        input_dim=int(X_train_oracle.shape[1]),
        hidden_dim=int(args.oracle_hidden_dim),
        seed=int(args.seed),
    )
    text_model = PairwiseMLPReranker(
        input_dim=int(X_train_text.shape[1]),
        hidden_dim=int(args.text_hidden_dim),
        seed=int(args.seed) + 1000,
    )

    oracle_history = oracle_model.fit(
        X_train_oracle,
        y_train,
        groups_train,
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    text_history = text_model.fit(
        X_train_text,
        y_train,
        groups_train,
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

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

    train_residual_scores = train_oracle_scores + float(args.text_lambda) * train_text_scores
    eval_residual_scores = eval_oracle_scores + float(args.text_lambda) * eval_text_scores

    summary = {
        "train_dataset": str(args.train_dataset),
        "eval_dataset": str(args.eval_dataset),
        "text_encoder": str(args.text_encoder),
        "hf_model": str(args.hf_model),
        "text_lambda": float(args.text_lambda),
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
        "residual": {
            "train_metrics": ranking_metrics(train_residual_scores, y_train, groups_train),
            "eval_metrics": ranking_metrics(eval_residual_scores, y_eval, groups_eval),
        },
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