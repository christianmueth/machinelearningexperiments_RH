from __future__ import annotations

import argparse
import shutil
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent


def _run_python(script_name: str, args: list[str]) -> None:
    command = [sys.executable, str(SCRIPT_DIR / script_name), *args]
    print(f"running: {' '.join(command)}")
    subprocess.run(command, check=True, cwd=str(REPO_ROOT))


def _copy_if_present(source: Path, destination: Path) -> str:
    if not source.exists():
        return ""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(destination))
    return str(destination)


def _copy_dir_if_present(source: Path, destination: Path) -> str:
    if not source.exists() or not source.is_dir():
        return ""
    if destination.exists():
        shutil.rmtree(str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(str(source), str(destination))
    return str(destination)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run the fixed structured-dataset evaluation stack: audits, grouped split, oracle baseline, safe controller, uncertainty, and offline branch-feature analysis."
    )
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--artifact_prefix", default="")
    ap.add_argument("--train_fraction", type=float, default=0.67)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out_dir", default="out/ai")
    ap.add_argument("--bundle_dir", default="")
    ap.add_argument("--skip_structure_audit", action="store_true")
    ap.add_argument("--skip_position_audit", action="store_true")
    ap.add_argument("--enable_branch_ranker", action="store_true")
    ap.add_argument("--enable_branch_structural_dedup", action="store_true")
    ap.add_argument("--enable_branch_signature_consensus", action="store_true")
    ap.add_argument("--run_branch_ranker_feature_ablation", action="store_true")
    args = ap.parse_args()

    input_path = Path(str(args.input_jsonl))
    prefix = str(args.artifact_prefix).strip() or input_path.stem
    out_dir = REPO_ROOT / str(args.out_dir)
    audits_dir = out_dir / "audits"
    diagnostics_dir = out_dir / "diagnostics"
    summaries_dir = out_dir / "dataset_summaries"

    train_jsonl = input_path.with_name(f"{prefix}_train.jsonl")
    eval_jsonl = input_path.with_name(f"{prefix}_eval.jsonl")

    structure_json = audits_dir / f"{prefix}_structure.json"
    positions_json = audits_dir / f"{prefix}_positions.json"
    split_summary_json = summaries_dir / f"{prefix}_split_summary.json"
    oracle_baseline_json = out_dir / f"{prefix}_oracle_baseline.json"
    oracle_uncertainty_json = diagnostics_dir / f"{prefix}_oracle_uncertainty.json"
    controller_safe_json = out_dir / f"{prefix}_oracle_controller_safe.json"
    branch_feature_json = diagnostics_dir / f"{prefix}_branch_feature_analysis.json"
    branch_ranker_audit_json = diagnostics_dir / f"{prefix}_branch_ranker_deltas.json"
    branch_ranker_ablation_json = diagnostics_dir / f"{prefix}_branch_ranker_feature_ablation.json"
    branch_ranker_ablation_dir = diagnostics_dir / f"{prefix}_branch_ranker_feature_ablation"
    state_resolution_json = diagnostics_dir / f"{prefix}_state_resolution.json"
    state_edge_feature_json = diagnostics_dir / f"{prefix}_state_edge_feature_analysis.json"
    ambiguity_benchmark_jsonl = diagnostics_dir / f"{prefix}_structural_ambiguity_benchmark.jsonl"
    ambiguity_summary_json = diagnostics_dir / f"{prefix}_structural_ambiguity_summary.json"
    ambiguity_suite_json = diagnostics_dir / f"{prefix}_structural_ambiguity_suite.json"
    pipeline_summary_json = summaries_dir / f"{prefix}_structured_pipeline_summary.json"

    bundle_dir = Path(str(args.bundle_dir)) if str(args.bundle_dir).strip() else (out_dir / "checkpoints" / prefix)

    if not bool(args.skip_structure_audit):
        _run_python(
            "audit_reasoning_structure.py",
            ["--input_jsonl", str(input_path), "--out_json", str(structure_json)],
        )

    if not bool(args.skip_position_audit):
        _run_python(
            "audit_candidate_positions.py",
            ["--input_jsonl", str(input_path), "--output_json", str(positions_json)],
        )

    _run_python(
        "split_grouped_dataset.py",
        [
            "--input_jsonl",
            str(input_path),
            "--train_jsonl",
            str(train_jsonl),
            "--eval_jsonl",
            str(eval_jsonl),
            "--train_fraction",
            str(float(args.train_fraction)),
            "--seed",
            str(int(args.seed)),
            "--out_summary_json",
            str(split_summary_json),
        ],
    )

    _run_python(
        "run_reranker_experiment.py",
        [
            "--train_dataset",
            str(train_jsonl),
            "--eval_dataset",
            str(eval_jsonl),
            "--out_json",
            str(oracle_baseline_json),
            "--out_dir",
            str(out_dir),
            "--feature_modes",
            "oracle",
        ],
    )

    _run_python(
        "diagnose_oracle_uncertainty.py",
        [
            "--train_dataset",
            str(train_jsonl),
            "--eval_dataset",
            str(eval_jsonl),
            "--out_json",
            str(oracle_uncertainty_json),
        ],
    )

    controller_args = [
        "--train_dataset",
        str(train_jsonl),
        "--eval_dataset",
        str(eval_jsonl),
        "--out_json",
        str(controller_safe_json),
    ]
    if bool(args.enable_branch_ranker):
        controller_args.append("--enable_branch_ranker")
    if bool(args.enable_branch_structural_dedup):
        controller_args.append("--enable_branch_structural_dedup")
    if bool(args.enable_branch_signature_consensus):
        controller_args.append("--enable_branch_signature_consensus")
    _run_python("run_oracle_controller_experiment.py", controller_args)

    _run_python(
        "analyze_controller_branch_features.py",
        [
            "--controller_json",
            str(controller_safe_json),
            "--out_json",
            str(branch_feature_json),
        ],
    )

    _run_python(
        "audit_reasoning_state_resolution.py",
        [
            "--input_jsonl",
            str(eval_jsonl),
            "--controller_json",
            str(controller_safe_json),
            "--out_json",
            str(state_resolution_json),
        ],
    )

    _run_python(
        "analyze_state_edge_features.py",
        [
            "--input_jsonl",
            str(eval_jsonl),
            "--controller_json",
            str(controller_safe_json),
            "--out_json",
            str(state_edge_feature_json),
        ],
    )

    _run_python(
        "build_structural_ambiguity_benchmark.py",
        [
            "--input_jsonl",
            str(eval_jsonl),
            "--controller_json",
            str(controller_safe_json),
            "--out_jsonl",
            str(ambiguity_benchmark_jsonl),
            "--out_summary_json",
            str(ambiguity_summary_json),
        ],
    )

    _run_python(
        "run_structural_ambiguity_suite.py",
        [
            "--train_dataset",
            str(train_jsonl),
            "--benchmark_jsonl",
            str(ambiguity_benchmark_jsonl),
            "--controller_json",
            str(controller_safe_json),
            "--out_json",
            str(ambiguity_suite_json),
            "--out_dir",
            str(out_dir),
        ],
    )

    if bool(args.enable_branch_ranker):
        _run_python(
            "audit_branch_ranker_deltas.py",
            [
                "--controller_json",
                str(controller_safe_json),
                "--out_json",
                str(branch_ranker_audit_json),
            ],
        )
        if bool(args.run_branch_ranker_feature_ablation):
            _run_python(
                "run_branch_ranker_feature_ablation.py",
                [
                    "--train_dataset",
                    str(train_jsonl),
                    "--eval_dataset",
                    str(eval_jsonl),
                    "--reference_report",
                    str(controller_safe_json),
                    "--out_json",
                    str(branch_ranker_ablation_json),
                    "--out_dir",
                    str(branch_ranker_ablation_dir),
                ],
            )

    summary = {
        "input_jsonl": str(input_path),
        "artifact_prefix": str(prefix),
        "train_jsonl": str(train_jsonl),
        "eval_jsonl": str(eval_jsonl),
        "outputs": {
            "structure_audit": str(structure_json) if not bool(args.skip_structure_audit) else "",
            "position_audit": str(positions_json) if not bool(args.skip_position_audit) else "",
            "split_summary": str(split_summary_json),
            "oracle_baseline": str(oracle_baseline_json),
            "oracle_uncertainty": str(oracle_uncertainty_json),
            "controller_report": str(controller_safe_json),
            "controller_safe": str(controller_safe_json),
            "branch_feature_analysis": str(branch_feature_json),
            "state_resolution": str(state_resolution_json),
            "state_edge_feature_analysis": str(state_edge_feature_json),
            "structural_ambiguity_benchmark": str(ambiguity_benchmark_jsonl),
            "structural_ambiguity_summary": str(ambiguity_summary_json),
            "structural_ambiguity_suite": str(ambiguity_suite_json),
            "branch_ranker_deltas": str(branch_ranker_audit_json) if bool(args.enable_branch_ranker) else "",
            "branch_ranker_feature_ablation": str(branch_ranker_ablation_json) if bool(args.enable_branch_ranker and args.run_branch_ranker_feature_ablation) else "",
        },
        "config": {
            "train_fraction": float(args.train_fraction),
            "seed": int(args.seed),
            "out_dir": str(out_dir),
            "bundle_dir": str(bundle_dir),
            "enable_branch_ranker": bool(args.enable_branch_ranker),
            "enable_branch_structural_dedup": bool(args.enable_branch_structural_dedup),
            "enable_branch_signature_consensus": bool(args.enable_branch_signature_consensus),
            "run_branch_ranker_feature_ablation": bool(args.run_branch_ranker_feature_ablation),
        },
    }
    pipeline_summary_json.parent.mkdir(parents=True, exist_ok=True)
    pipeline_summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    bundle_manifest = {
        "artifact_prefix": str(prefix),
        "input_jsonl": str(input_path),
        "bundle_dir": str(bundle_dir),
        "files": {
            "dataset_jsonl": _copy_if_present(input_path, bundle_dir / f"{prefix}.jsonl"),
            "train_jsonl": _copy_if_present(train_jsonl, bundle_dir / f"{prefix}_train.jsonl"),
            "eval_jsonl": _copy_if_present(eval_jsonl, bundle_dir / f"{prefix}_eval.jsonl"),
            "structure_audit": _copy_if_present(structure_json, bundle_dir / f"{prefix}_structure_audit.json") if not bool(args.skip_structure_audit) else "",
            "position_audit": _copy_if_present(positions_json, bundle_dir / f"{prefix}_position_audit.json") if not bool(args.skip_position_audit) else "",
            "split_summary": _copy_if_present(split_summary_json, bundle_dir / f"{prefix}_split_summary.json"),
            "oracle_baseline": _copy_if_present(oracle_baseline_json, bundle_dir / f"{prefix}_oracle_baseline.json"),
            "oracle_uncertainty": _copy_if_present(oracle_uncertainty_json, bundle_dir / f"{prefix}_oracle_uncertainty.json"),
            "controller_report": _copy_if_present(controller_safe_json, bundle_dir / f"{prefix}_controller_report.json"),
            "controller_safe_report": _copy_if_present(controller_safe_json, bundle_dir / f"{prefix}_controller_safe_report.json"),
            "branch_feature_analysis": _copy_if_present(branch_feature_json, bundle_dir / f"{prefix}_branch_feature_analysis.json"),
            "state_resolution": _copy_if_present(state_resolution_json, bundle_dir / f"{prefix}_state_resolution.json"),
            "state_edge_feature_analysis": _copy_if_present(state_edge_feature_json, bundle_dir / f"{prefix}_state_edge_feature_analysis.json"),
            "structural_ambiguity_benchmark": _copy_if_present(ambiguity_benchmark_jsonl, bundle_dir / f"{prefix}_structural_ambiguity_benchmark.jsonl"),
            "structural_ambiguity_summary": _copy_if_present(ambiguity_summary_json, bundle_dir / f"{prefix}_structural_ambiguity_summary.json"),
            "structural_ambiguity_suite": _copy_if_present(ambiguity_suite_json, bundle_dir / f"{prefix}_structural_ambiguity_suite.json"),
            "branch_ranker_deltas": _copy_if_present(branch_ranker_audit_json, bundle_dir / f"{prefix}_branch_ranker_deltas.json") if bool(args.enable_branch_ranker) else "",
            "branch_ranker_feature_ablation": _copy_if_present(branch_ranker_ablation_json, bundle_dir / f"{prefix}_branch_ranker_feature_ablation.json") if bool(args.enable_branch_ranker and args.run_branch_ranker_feature_ablation) else "",
            "branch_ranker_feature_ablation_dir": _copy_dir_if_present(branch_ranker_ablation_dir, bundle_dir / f"{prefix}_branch_ranker_feature_ablation") if bool(args.enable_branch_ranker and args.run_branch_ranker_feature_ablation) else "",
            "pipeline_summary": _copy_if_present(pipeline_summary_json, bundle_dir / f"{prefix}_pipeline_summary.json"),
        },
    }
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_manifest_path = bundle_dir / "bundle_manifest.json"
    bundle_manifest_path.write_text(json.dumps(bundle_manifest, indent=2) + "\n", encoding="utf-8")
    summary["bundle_manifest"] = str(bundle_manifest_path)
    pipeline_summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"wrote {pipeline_summary_json}")
    print(f"wrote {bundle_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())