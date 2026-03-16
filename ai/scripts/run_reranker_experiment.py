from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml_oracle.datasets import load_reasoning_examples, materialize_dataset
from src.ml_oracle.frozen_oracle_client import AnchoredOracleClient
from src.ml_oracle.reranker import PairwiseMLPReranker, ranking_metrics
from src.ml_oracle.translator import HeuristicAnchoredTranslator


def _dataset_matrices(
    dataset_path: str,
    *,
    client: AnchoredOracleClient,
    translator: HeuristicAnchoredTranslator,
    feature_mode: str,
    oracle_feature_groups: tuple[str, ...] | None,
    text_encoder: str,
    text_dim: int,
    hf_model: str,
    hf_max_length: int,
):
    examples = load_reasoning_examples(dataset_path)
    X, y, groups = materialize_dataset(
        examples,
        client=client,
        translator=translator,
        text_dim=int(text_dim),
        feature_mode=str(feature_mode),
        oracle_feature_groups=oracle_feature_groups,
        text_encoder_name=str(text_encoder),
        hf_model=str(hf_model),
        hf_max_length=int(hf_max_length),
    )
    return X, y, groups, len(examples)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run repeatable train/eval reranker experiments across feature modes.")
    ap.add_argument("--train_dataset", required=True)
    ap.add_argument("--eval_dataset", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_dir", default="out/ai")
    ap.add_argument("--feature_modes", nargs="+", default=["text", "oracle", "text+oracle"])
    ap.add_argument("--oracle_feature_groups", default="")
    ap.add_argument("--text_encoder", choices=["hashed", "hf"], default="hashed")
    ap.add_argument("--text_dim", type=int, default=256)
    ap.add_argument("--hf_model", default="")
    ap.add_argument("--hf_max_length", type=int, default=256)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    client = AnchoredOracleClient()
    translator = HeuristicAnchoredTranslator()
    out_dir = Path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_groups = tuple(part.strip() for part in str(args.oracle_feature_groups).split(",") if part.strip()) or None

    summary: dict[str, object] = {
        "train_dataset": str(args.train_dataset),
        "eval_dataset": str(args.eval_dataset),
        "text_encoder": str(args.text_encoder),
        "hf_model": str(args.hf_model),
        "runs": [],
    }

    for mode in args.feature_modes:
        sanitized_mode = str(mode).replace("+", "_plus_")
        X_train, y_train, groups_train, n_train = _dataset_matrices(
            str(args.train_dataset),
            client=client,
            translator=translator,
            feature_mode=str(mode),
            oracle_feature_groups=feature_groups,
            text_encoder=str(args.text_encoder),
            text_dim=int(args.text_dim),
            hf_model=str(args.hf_model),
            hf_max_length=int(args.hf_max_length),
        )
        X_eval, y_eval, groups_eval, n_eval = _dataset_matrices(
            str(args.eval_dataset),
            client=client,
            translator=translator,
            feature_mode=str(mode),
            oracle_feature_groups=feature_groups,
            text_encoder=str(args.text_encoder),
            text_dim=int(args.text_dim),
            hf_model=str(args.hf_model),
            hf_max_length=int(args.hf_max_length),
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
        model_path = out_dir / f"{Path(str(args.out_json)).stem}_{sanitized_mode}.npz"
        model.save(str(model_path))
        train_metrics = ranking_metrics(model.score(X_train), y_train, groups_train)
        eval_metrics = ranking_metrics(model.score(X_eval), y_eval, groups_eval)
        run_summary = {
            "feature_mode": str(mode),
            "train_examples": int(n_train),
            "eval_examples": int(n_eval),
            "model_path": str(model_path),
            "train_loss_last": float(history[-1] if history else 0.0),
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
        }
        summary["runs"].append(run_summary)
        print(json.dumps(run_summary, indent=2))

    out_path = Path(str(args.out_json))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())