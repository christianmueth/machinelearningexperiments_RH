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
    ap = argparse.ArgumentParser(description="Evaluate a saved anchored-oracle reranker on grouped candidate-trace data.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--feature_mode", choices=["text", "oracle", "text+oracle"], default="text+oracle")
    ap.add_argument("--oracle_feature_groups", default="")
    ap.add_argument("--text_dim", type=int, default=256)
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
    model = PairwiseMLPReranker.load(str(args.model))
    scores = model.score(X)
    print(f"feature_mode={args.feature_mode}")
    if str(args.oracle_feature_groups).strip():
        print(f"oracle_feature_groups={args.oracle_feature_groups}")
    print(f"group_accuracy={group_accuracy(scores, y, groups):.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())