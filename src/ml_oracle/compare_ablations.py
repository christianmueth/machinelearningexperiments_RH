from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml_oracle.datasets import load_reasoning_examples, materialize_dataset
from src.ml_oracle.feature_registry import FEATURE_GROUPS
from src.ml_oracle.frozen_oracle_client import AnchoredOracleClient
from src.ml_oracle.reranker import PairwiseMLPReranker, group_accuracy
from src.ml_oracle.translator import HeuristicAnchoredTranslator


def _fit_and_eval(*, train_path: str, eval_path: str, feature_mode: str, oracle_feature_groups: tuple[str, ...] | None, text_dim: int, hidden_dim: int, epochs: int, lr: float, weight_decay: float, seed: int) -> dict[str, float | str]:
    client = AnchoredOracleClient()
    translator = HeuristicAnchoredTranslator()
    train_examples = load_reasoning_examples(train_path)
    eval_examples = load_reasoning_examples(eval_path)
    X_train, y_train, g_train = materialize_dataset(
        train_examples,
        client=client,
        translator=translator,
        text_dim=int(text_dim),
        feature_mode=str(feature_mode),
        oracle_feature_groups=oracle_feature_groups,
    )
    X_eval, y_eval, g_eval = materialize_dataset(
        eval_examples,
        client=client,
        translator=translator,
        text_dim=int(text_dim),
        feature_mode=str(feature_mode),
        oracle_feature_groups=oracle_feature_groups,
    )
    model = PairwiseMLPReranker(input_dim=int(X_train.shape[1]), hidden_dim=int(hidden_dim), seed=int(seed))
    history = model.fit(X_train, y_train, g_train, epochs=int(epochs), lr=float(lr), weight_decay=float(weight_decay))
    return {
        "feature_mode": str(feature_mode),
        "oracle_feature_groups": ",".join(oracle_feature_groups or ()),
        "train_groups": float(len(set(g_train.tolist()))),
        "eval_groups": float(len(set(g_eval.tolist()))),
        "input_dim": float(X_train.shape[1]),
        "train_loss_last": float(history[-1] if history else 0.0),
        "train_group_accuracy": float(group_accuracy(model.score(X_train), y_train, g_train)),
        "eval_group_accuracy": float(group_accuracy(model.score(X_eval), y_eval, g_eval)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run anchored feature-family ablations on the fixed train/eval split.")
    ap.add_argument("--train_dataset", required=True)
    ap.add_argument("--eval_dataset", required=True)
    ap.add_argument("--text_dim", type=int, default=256)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_csv", default="out/ml_oracle_ablation_comparison.csv")
    args = ap.parse_args()

    oracle_runs: list[tuple[str, tuple[str, ...]]] = [
        ("oracle_full", tuple(FEATURE_GROUPS.keys())),
        ("oracle_closure", ("closure",)),
        ("oracle_packet", ("packet",)),
        ("oracle_zero", ("zero",)),
        ("oracle_fe_stability", ("fe_stability",)),
        ("oracle_closure_plus_packet", ("closure", "packet")),
        ("oracle_zero_plus_fe_stability", ("zero", "fe_stability")),
        ("oracle_all_cumulative", ("closure", "packet", "zero", "fe_stability")),
        ("text_plus_oracle_full", tuple(FEATURE_GROUPS.keys())),
        ("text_plus_oracle_zero_plus_fe", ("zero", "fe_stability")),
        ("text_plus_oracle_packet", ("packet",)),
    ]

    rows: list[dict[str, float | str]] = []
    rows.append(
        _fit_and_eval(
            train_path=str(args.train_dataset),
            eval_path=str(args.eval_dataset),
            feature_mode="text",
            oracle_feature_groups=None,
            text_dim=int(args.text_dim),
            hidden_dim=int(args.hidden_dim),
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            seed=int(args.seed),
        )
    )
    for run_name, groups in oracle_runs:
        feature_mode = "oracle" if run_name.startswith("oracle_") else "text+oracle"
        row = _fit_and_eval(
            train_path=str(args.train_dataset),
            eval_path=str(args.eval_dataset),
            feature_mode=feature_mode,
            oracle_feature_groups=groups,
            text_dim=int(args.text_dim),
            hidden_dim=int(args.hidden_dim),
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            seed=int(args.seed),
        )
        row["run_name"] = run_name
        rows.append(row)
    if "run_name" not in rows[0]:
        rows[0]["run_name"] = "text_only"

    out_path = Path(str(args.out_csv))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    for row in rows:
        print(
            f"run={row['run_name']} mode={row['feature_mode']} groups={row['oracle_feature_groups']} train_acc={row['train_group_accuracy']:.6g} eval_acc={row['eval_group_accuracy']:.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())