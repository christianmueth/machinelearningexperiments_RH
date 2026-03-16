from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent


DEFAULT_FEATURE_SETS: tuple[tuple[str, str], ...] = (
    ("baseline", "oracle_score,result_reuse_fraction,result_reuse_count,answer_support_fraction,answer_support_count"),
    ("depth_pair", "oracle_score,result_reuse_fraction,result_reuse_count,answer_support_fraction,answer_support_count,max_chain_depth,dependency_edge_count"),
    ("dependency_support_pair", "oracle_score,result_reuse_fraction,result_reuse_count,answer_support_fraction,answer_support_count,answer_support_dependency_fraction,answer_support_given_dependency"),
    ("hybrid_four", "oracle_score,result_reuse_fraction,result_reuse_count,answer_support_fraction,answer_support_count,max_chain_depth,dependency_edge_count,answer_support_dependency_fraction,answer_support_given_dependency"),
)


def _parse_feature_set(raw: str) -> tuple[str, str]:
    if "=" not in str(raw):
        raise ValueError(f"Feature set must use name=feature1,feature2 form: {raw}")
    name, feature_csv = str(raw).split("=", 1)
    return str(name).strip(), str(feature_csv).strip()


def _bool_flag_args(flag: str, enabled: bool) -> list[str]:
    return [flag] if bool(enabled) else []


def _run_variant(
    *,
    train_dataset: str,
    eval_dataset: str,
    out_json: Path,
    feature_csv: str,
    controller_config: dict[str, object],
) -> dict[str, object]:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "run_oracle_controller_experiment.py"),
        "--train_dataset",
        str(train_dataset),
        "--eval_dataset",
        str(eval_dataset),
        "--out_json",
        str(out_json),
        "--hidden_dim",
        str(int(controller_config.get("hidden_dim", 64))),
        "--epochs",
        str(int(controller_config.get("epochs", 50))),
        "--lr",
        str(float(controller_config.get("lr", 1e-2))),
        "--weight_decay",
        str(float(controller_config.get("weight_decay", 1e-4))),
        "--seed",
        str(int(controller_config.get("seed", 0))),
        "--controller_margin_threshold",
        str(float(controller_config.get("controller_margin_threshold", 0.0))),
        "--controller_score_window",
        str(float(controller_config.get("controller_score_window", 0.001))),
        "--controller_top_k",
        str(int(controller_config.get("controller_top_k", 3))),
        "--progress_weight",
        str(float(controller_config.get("progress_weight", 0.0))),
        "--answer_support_weight",
        str(float(controller_config.get("answer_support_weight", 0.2))),
        "--step_query_weight",
        str(float(controller_config.get("step_query_weight", 0.0001))),
        "--controller_step_query_delta",
        str(int(controller_config.get("controller_step_query_delta", 1))),
        "--controller_step_query_window",
        str(float(controller_config.get("controller_step_query_window", 0.006))),
        "--valid_state_weight",
        str(float(controller_config.get("valid_state_weight", 0.0))),
        "--answer_support_fraction_weight",
        str(float(controller_config.get("answer_support_fraction_weight", 0.0))),
        "--equation_consistency_weight",
        str(float(controller_config.get("equation_consistency_weight", 0.0))),
        "--result_reuse_weight",
        str(float(controller_config.get("result_reuse_weight", 0.0))),
        "--final_resolution_weight",
        str(float(controller_config.get("final_resolution_weight", 0.0))),
        "--approx_penalty_weight",
        str(float(controller_config.get("approx_penalty_weight", 0.0))),
        "--enable_branch_ranker",
        "--branch_ranker_hidden_dim",
        str(int(controller_config.get("branch_ranker_hidden_dim", 8))),
        "--branch_ranker_epochs",
        str(int(controller_config.get("branch_ranker_epochs", 75))),
        "--branch_ranker_lr",
        str(float(controller_config.get("branch_ranker_lr", 5e-3))),
        "--branch_ranker_weight_decay",
        str(float(controller_config.get("branch_ranker_weight_decay", 1e-4))),
        "--branch_ranker_feature_names",
        str(feature_csv),
        *_bool_flag_args("--enable_branch_structural_dedup", bool(controller_config.get("enable_branch_structural_dedup", False))),
        *_bool_flag_args("--enable_branch_signature_consensus", bool(controller_config.get("enable_branch_signature_consensus", False))),
    ]
    print(f"running: {' '.join(command)}")
    subprocess.run(command, check=True, cwd=str(REPO_ROOT))
    return json.loads(out_json.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Run offline branch-ranker feature-set ablations on the main structured benchmark without changing the default live path.")
    ap.add_argument("--train_dataset", required=True)
    ap.add_argument("--eval_dataset", required=True)
    ap.add_argument("--reference_report", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--feature_set", action="append", default=[])
    args = ap.parse_args()

    reference_report = json.loads(Path(str(args.reference_report)).read_text(encoding="utf-8"))
    controller_config = dict(reference_report.get("controller_config", {}))
    feature_sets = [_parse_feature_set(raw) for raw in args.feature_set] if args.feature_set else list(DEFAULT_FEATURE_SETS)

    out_dir = Path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    variant_summaries: list[dict[str, object]] = []
    baseline_eval_accuracy = None
    for name, feature_csv in feature_sets:
        variant_json = out_dir / f"branch_ranker_ablation_{name}.json"
        report = _run_variant(
            train_dataset=str(args.train_dataset),
            eval_dataset=str(args.eval_dataset),
            out_json=variant_json,
            feature_csv=str(feature_csv),
            controller_config=controller_config,
        )
        branch_metrics = dict(report.get("branch_ranker_branch", {}).get("eval_metrics", {}))
        controller_metrics = dict(report.get("controller_branch", {}).get("eval_metrics", {}))
        summary = {
            "name": str(name),
            "feature_names": list(report.get("branch_ranker_branch", {}).get("feature_names", [])),
            "out_json": str(variant_json),
            "branch_ranker_eval_metrics": branch_metrics,
            "controller_eval_metrics": controller_metrics,
            "gain_vs_controller_group_accuracy": float(branch_metrics.get("group_accuracy", 0.0) - controller_metrics.get("group_accuracy", 0.0)),
            "low_margin_groups_changed_vs_controller": int(report.get("branch_ranker_branch", {}).get("low_margin_groups_changed_vs_controller", 0)),
            "train_branch_rows": int(report.get("branch_ranker_branch", {}).get("train_branch_rows", 0)),
            "train_branch_groups": int(report.get("branch_ranker_branch", {}).get("train_branch_groups", 0)),
            "status": str(report.get("branch_ranker_branch", {}).get("status", "")),
        }
        if baseline_eval_accuracy is None:
            baseline_eval_accuracy = float(branch_metrics.get("group_accuracy", 0.0))
        summary["gain_vs_ablation_baseline_group_accuracy"] = float(branch_metrics.get("group_accuracy", 0.0) - float(baseline_eval_accuracy))
        variant_summaries.append(summary)

    variant_summaries.sort(
        key=lambda row: (
            float(row.get("branch_ranker_eval_metrics", {}).get("group_accuracy", 0.0)),
            float(row.get("branch_ranker_eval_metrics", {}).get("mrr", 0.0)),
            float(row.get("branch_ranker_eval_metrics", {}).get("ndcg", 0.0)),
        ),
        reverse=True,
    )

    rendered = {
        "train_dataset": str(args.train_dataset),
        "eval_dataset": str(args.eval_dataset),
        "reference_report": str(args.reference_report),
        "feature_sets": [
            {"name": name, "feature_csv": feature_csv}
            for name, feature_csv in feature_sets
        ],
        "variants": variant_summaries,
        "best_variant": variant_summaries[0] if variant_summaries else {},
    }
    out_json = Path(str(args.out_json))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rendered, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rendered, indent=2))
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())