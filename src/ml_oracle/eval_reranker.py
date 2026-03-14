from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml_oracle.datasets import load_reasoning_examples, materialize_dataset
from src.ml_oracle.frozen_oracle_client import AnchoredOracleClient
from src.ml_oracle.reranker import PairwiseMLPReranker, ranking_metrics
from src.ml_oracle.translator import HeuristicAnchoredTranslator


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate a saved anchored-oracle reranker on grouped candidate-trace data.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--feature_mode", choices=["text", "oracle", "text+oracle"], default="text+oracle")
    ap.add_argument("--oracle_feature_groups", default="")
    ap.add_argument("--text_encoder", choices=["hashed", "hf"], default="hashed")
    ap.add_argument("--text_dim", type=int, default=256)
    ap.add_argument("--hf_model", default="")
    ap.add_argument("--hf_max_length", type=int, default=256)
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
        text_encoder_name=str(args.text_encoder),
        hf_model=str(args.hf_model),
        hf_max_length=int(args.hf_max_length),
    )
    model = PairwiseMLPReranker.load(str(args.model))
    scores = model.score(X)
    print(f"feature_mode={args.feature_mode}")
    print(f"text_encoder={args.text_encoder}")
    if str(args.hf_model).strip():
        print(f"hf_model={args.hf_model}")
    if str(args.oracle_feature_groups).strip():
        print(f"oracle_feature_groups={args.oracle_feature_groups}")
    metrics = ranking_metrics(scores, y, groups)
    print(f"group_accuracy={metrics['group_accuracy']:.6g}")
    print(f"mrr={metrics['mrr']:.6g}")
    print(f"ndcg={metrics['ndcg']:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())