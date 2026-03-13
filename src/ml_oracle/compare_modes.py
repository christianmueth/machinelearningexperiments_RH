from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml_oracle.datasets import load_reasoning_examples, materialize_dataset
from src.ml_oracle.frozen_oracle_client import AnchoredOracleClient
from src.ml_oracle.reranker import PairwiseMLPReranker, group_accuracy
from src.ml_oracle.translator import HeuristicAnchoredTranslator


def _fit_and_eval(*, train_path: str, eval_path: str, feature_mode: str, text_dim: int, hidden_dim: int, epochs: int, lr: float, weight_decay: float, seed: int) -> dict[str, float | str]:
    client = AnchoredOracleClient()
    translator = HeuristicAnchoredTranslator()
    train_examples = load_reasoning_examples(train_path)
    eval_examples = load_reasoning_examples(eval_path)
    X_train, y_train, g_train = materialize_dataset(train_examples, client=client, translator=translator, text_dim=int(text_dim), feature_mode=str(feature_mode))
    X_eval, y_eval, g_eval = materialize_dataset(eval_examples, client=client, translator=translator, text_dim=int(text_dim), feature_mode=str(feature_mode))

    model = PairwiseMLPReranker(input_dim=int(X_train.shape[1]), hidden_dim=int(hidden_dim), seed=int(seed))
    history = model.fit(X_train, y_train, g_train, epochs=int(epochs), lr=float(lr), weight_decay=float(weight_decay))
    train_acc = group_accuracy(model.score(X_train), y_train, g_train)
    eval_acc = group_accuracy(model.score(X_eval), y_eval, g_eval)
    return {
        "feature_mode": str(feature_mode),
        "train_groups": float(len(set(g_train.tolist()))),
        "eval_groups": float(len(set(g_eval.tolist()))),
        "input_dim": float(X_train.shape[1]),
        "train_loss_last": float(history[-1] if history else 0.0),
        "train_group_accuracy": float(train_acc),
        "eval_group_accuracy": float(eval_acc),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare text-only, oracle-only, and text+oracle rerankers on a fixed anchored-oracle train/eval split.")
    ap.add_argument("--train_dataset", required=True)
    ap.add_argument("--eval_dataset", required=True)
    ap.add_argument("--text_dim", type=int, default=256)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_csv", default="out/ml_oracle_mode_comparison.csv")
    args = ap.parse_args()

    rows = []
    for feature_mode in ["text", "oracle", "text+oracle"]:
        rows.append(
            _fit_and_eval(
                train_path=str(args.train_dataset),
                eval_path=str(args.eval_dataset),
                feature_mode=str(feature_mode),
                text_dim=int(args.text_dim),
                hidden_dim=int(args.hidden_dim),
                epochs=int(args.epochs),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                seed=int(args.seed),
            )
        )
    out_path = Path(str(args.out_csv))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    for row in rows:
        print(f"mode={row['feature_mode']} train_acc={row['train_group_accuracy']:.6g} eval_acc={row['eval_group_accuracy']:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())