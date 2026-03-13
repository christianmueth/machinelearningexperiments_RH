from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml_oracle.datasets import load_reasoning_examples, materialize_dataset
from src.ml_oracle.frozen_oracle_client import AnchoredOracleClient
from src.ml_oracle.reranker import PairwiseMLPReranker, group_accuracy
from src.ml_oracle.translator import HeuristicAnchoredTranslator


def main() -> int:
    ap = argparse.ArgumentParser(description="Train a candidate-trace reranker using anchored oracle features plus hashed text embeddings.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out_model", default="out/ml_oracle_reranker.npz")
    ap.add_argument("--feature_mode", choices=["text", "oracle", "text+oracle"], default="text+oracle")
    ap.add_argument("--oracle_feature_groups", default="")
    ap.add_argument("--text_dim", type=int, default=256)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    examples = load_reasoning_examples(args.dataset)
    client = AnchoredOracleClient()
    translator = HeuristicAnchoredTranslator()
    X, y, groups = materialize_dataset(
        examples,
        client=client,
        translator=translator,
        text_dim=int(args.text_dim),
        feature_mode=str(args.feature_mode),
        oracle_feature_groups=tuple(part.strip() for part in str(args.oracle_feature_groups).split(",") if part.strip()) or None,
    )

    model = PairwiseMLPReranker(input_dim=int(X.shape[1]), hidden_dim=int(args.hidden_dim), seed=int(args.seed))
    history = model.fit(X, y, groups, epochs=int(args.epochs), lr=float(args.lr), weight_decay=float(args.weight_decay))

    out_model = Path(str(args.out_model))
    out_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_model))
    scores = model.score(X)
    print(f"feature_mode={args.feature_mode}")
    if str(args.oracle_feature_groups).strip():
        print(f"oracle_feature_groups={args.oracle_feature_groups}")
    print(f"train_loss_last={history[-1] if history else 0.0:.6g}")
    print(f"train_group_accuracy={group_accuracy(scores, y, groups):.6g}")
    print(f"wrote {out_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())