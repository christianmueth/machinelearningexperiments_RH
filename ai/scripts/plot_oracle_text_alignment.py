from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.scripts.run_residual_reranker_experiment import _candidate_text_rows, _fit_tfidf, _materialize_branch, _standardize_from_train, _transform_tfidf
from src.ml_oracle.datasets import load_reasoning_examples
from src.ml_oracle.frozen_oracle_client import AnchoredOracleClient
from src.ml_oracle.reranker import PairwiseMLPReranker
from src.ml_oracle.translator import HeuristicAnchoredTranslator


def _branch_matrices(
    dataset_path: str,
    *,
    client: AnchoredOracleClient,
    translator: HeuristicAnchoredTranslator,
    text_encoder: str,
    text_dim: int,
    tfidf_max_features: int,
    hf_model: str,
    hf_max_length: int,
) -> tuple[list, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[dict[str, int], np.ndarray] | None]:
    examples = load_reasoning_examples(str(dataset_path))
    if str(text_encoder) == "tfidf":
        texts, labels, groups = _candidate_text_rows(examples)
        vocab, idf = _fit_tfidf(texts, max_features=int(tfidf_max_features))
        X_text = _transform_tfidf(texts, vocab=vocab, idf=idf)
        tfidf_state: tuple[dict[str, int], np.ndarray] | None = (vocab, idf)
    else:
        X_text, labels, groups, _ = _materialize_branch(
            str(dataset_path),
            feature_mode="text",
            client=client,
            translator=translator,
            text_encoder=str(text_encoder),
            text_dim=int(text_dim),
            hf_model=str(hf_model),
            hf_max_length=int(hf_max_length),
        )
        tfidf_state = None
    oracle_text_encoder = "hashed" if str(text_encoder) == "tfidf" else str(text_encoder)
    X_oracle, _, _, _ = _materialize_branch(
        str(dataset_path),
        feature_mode="oracle",
        client=client,
        translator=translator,
        text_encoder=oracle_text_encoder,
        text_dim=int(text_dim),
        hf_model=str(hf_model),
        hf_max_length=int(hf_max_length),
    )
    return examples, X_text, labels, groups, X_oracle, tfidf_state


def _eval_branch_matrices(
    dataset_path: str,
    *,
    client: AnchoredOracleClient,
    translator: HeuristicAnchoredTranslator,
    text_encoder: str,
    text_dim: int,
    hf_model: str,
    hf_max_length: int,
    tfidf_state: tuple[dict[str, int], np.ndarray] | None,
) -> tuple[list, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    examples = load_reasoning_examples(str(dataset_path))
    if str(text_encoder) == "tfidf":
        texts, labels, groups = _candidate_text_rows(examples)
        if tfidf_state is None:
            raise ValueError("TF-IDF state must be provided for eval materialization")
        vocab, idf = tfidf_state
        X_text = _transform_tfidf(texts, vocab=vocab, idf=idf)
    else:
        X_text, labels, groups, _ = _materialize_branch(
            str(dataset_path),
            feature_mode="text",
            client=client,
            translator=translator,
            text_encoder=str(text_encoder),
            text_dim=int(text_dim),
            hf_model=str(hf_model),
            hf_max_length=int(hf_max_length),
        )
    oracle_text_encoder = "hashed" if str(text_encoder) == "tfidf" else str(text_encoder)
    X_oracle, _, _, _ = _materialize_branch(
        str(dataset_path),
        feature_mode="oracle",
        client=client,
        translator=translator,
        text_encoder=oracle_text_encoder,
        text_dim=int(text_dim),
        hf_model=str(hf_model),
        hf_max_length=int(hf_max_length),
    )
    return examples, X_text, labels, groups, X_oracle


def _stable_desc_order(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    local = np.asarray(values[idx], dtype=np.float64)
    order = np.argsort(-local, kind="mergesort")
    return idx[order]


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot oracle-vs-text alignment on eval groups.")
    ap.add_argument("--train_dataset", required=True)
    ap.add_argument("--eval_dataset", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", default="")
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
    args = ap.parse_args()

    client = AnchoredOracleClient()
    translator = HeuristicAnchoredTranslator()

    train_examples, X_train_text, y_train, groups_train, X_train_oracle, tfidf_state = _branch_matrices(
        str(args.train_dataset),
        client=client,
        translator=translator,
        text_encoder=str(args.text_encoder),
        text_dim=int(args.text_dim),
        tfidf_max_features=int(args.tfidf_max_features),
        hf_model=str(args.hf_model),
        hf_max_length=int(args.hf_max_length),
    )
    eval_examples, X_eval_text, y_eval, groups_eval, X_eval_oracle = _eval_branch_matrices(
        str(args.eval_dataset),
        client=client,
        translator=translator,
        text_encoder=str(args.text_encoder),
        text_dim=int(args.text_dim),
        hf_model=str(args.hf_model),
        hf_max_length=int(args.hf_max_length),
        tfidf_state=tfidf_state,
    )

    oracle_model = PairwiseMLPReranker(input_dim=int(X_train_oracle.shape[1]), hidden_dim=int(args.oracle_hidden_dim), seed=int(args.seed))
    text_model = PairwiseMLPReranker(input_dim=int(X_train_text.shape[1]), hidden_dim=int(args.text_hidden_dim), seed=int(args.seed) + 1000)
    oracle_model.fit(X_train_oracle, y_train, groups_train, epochs=int(args.epochs), lr=float(args.lr), weight_decay=float(args.weight_decay))
    text_model.fit(X_train_text, y_train, groups_train, epochs=int(args.epochs), lr=float(args.lr), weight_decay=float(args.weight_decay))

    train_oracle = oracle_model.score(X_train_oracle)
    eval_oracle = oracle_model.score(X_eval_oracle)
    train_text = text_model.score(X_train_text)
    eval_text = text_model.score(X_eval_text)
    train_oracle, eval_oracle, _, _ = _standardize_from_train(train_oracle, eval_oracle)
    train_text, eval_text, _, _ = _standardize_from_train(train_text, eval_text)

    rows: list[dict[str, object]] = []
    for group in np.unique(groups_eval):
        idx = np.flatnonzero(groups_eval == int(group))
        oracle_order = _stable_desc_order(eval_oracle, idx)
        text_order = _stable_desc_order(eval_text, idx)
        oracle_top = int(oracle_order[0]) if oracle_order.size else -1
        oracle_margin = float(eval_oracle[oracle_order[0]] - eval_oracle[oracle_order[1]]) if oracle_order.size >= 2 else 0.0
        prompt = eval_examples[int(group)].prompt
        for candidate_position, global_index in enumerate(idx.tolist()):
            rows.append(
                {
                    "group_id": int(group),
                    "candidate_index": int(candidate_position),
                    "label": float(y_eval[global_index]),
                    "oracle_score": float(eval_oracle[global_index]),
                    "text_score": float(eval_text[global_index]),
                    "oracle_rank": int(np.where(oracle_order == global_index)[0][0]) + 1,
                    "text_rank": int(np.where(text_order == global_index)[0][0]) + 1,
                    "oracle_top": bool(global_index == oracle_top),
                    "oracle_margin": oracle_margin,
                    "prompt": prompt,
                }
            )

    out_csv = Path(str(args.out_csv))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["group_id"])
        writer.writeheader()
        writer.writerows(rows)

    x = np.asarray([row["oracle_score"] for row in rows], dtype=np.float64)
    y = np.asarray([row["text_score"] for row in rows], dtype=np.float64)
    labels = np.asarray([row["label"] for row in rows], dtype=np.float64)
    oracle_top_mask = np.asarray([bool(row["oracle_top"]) for row in rows], dtype=bool)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    negatives = labels <= 0.5
    positives = labels > 0.5
    ax.scatter(x[negatives], y[negatives], s=26, c="#8a8f98", alpha=0.7, label="negative")
    ax.scatter(x[positives], y[positives], s=38, c="#d95f02", alpha=0.9, label="positive")
    ax.scatter(x[oracle_top_mask], y[oracle_top_mask], s=90, facecolors="none", edgecolors="#1b9e77", linewidths=1.2, label="oracle top")
    ax.set_xlabel("Oracle Score (standardized)")
    ax.set_ylabel("Text Score (standardized)")
    ax.set_title("Oracle vs Text Alignment on Eval Candidates")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    out_png = Path(str(args.out_png))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    summary = {
        "text_encoder": str(args.text_encoder),
        "hf_model": str(args.hf_model),
        "seed": int(args.seed),
        "n_eval_candidates": int(len(rows)),
        "n_eval_groups": int(len(np.unique(groups_eval))),
        "oracle_text_correlation": float(np.corrcoef(x, y)[0, 1]) if len(rows) >= 2 else 0.0,
        "positive_mean_oracle_score": float(np.mean(x[positives])) if positives.any() else 0.0,
        "negative_mean_oracle_score": float(np.mean(x[negatives])) if negatives.any() else 0.0,
        "positive_mean_text_score": float(np.mean(y[positives])) if positives.any() else 0.0,
        "negative_mean_text_score": float(np.mean(y[negatives])) if negatives.any() else 0.0,
        "out_csv": str(out_csv),
        "out_png": str(out_png),
    }
    print(json.dumps(summary, indent=2))
    if str(args.out_json).strip():
        out_json = Path(str(args.out_json))
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out_json}")
    print(f"wrote {out_csv}")
    print(f"wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())