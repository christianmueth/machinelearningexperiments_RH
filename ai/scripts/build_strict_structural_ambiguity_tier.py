from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml_oracle.datasets import load_reasoning_examples, materialize_dataset
from src.ml_oracle.frozen_oracle_client import AnchoredOracleClient
from src.ml_oracle.reranker import PairwiseMLPReranker
from src.ml_oracle.translator import HeuristicAnchoredTranslator


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _parse_csv_arg(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _extract_answer_line(text: str) -> str:
    match = re.search(r"(?:answer|final answer)\s*:\s*([^\n]+)", str(text), flags=re.IGNORECASE)
    return str(match.group(1)).strip() if match is not None else ""


def _extract_answer_value(text: str) -> float | None:
    answer_line = _extract_answer_line(text)
    if not answer_line:
        return None
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", answer_line)
    if not numbers:
        return None
    try:
        return float(numbers[-1].replace(",", ""))
    except ValueError:
        return None


def _reasoning_prefix(text: str) -> str:
    match = re.search(r"(?:answer|final answer)\s*:", str(text), flags=re.IGNORECASE)
    return str(text)[: match.start()] if match is not None else str(text)


def _looks_incomplete(text: str) -> bool:
    stripped = str(text).rstrip()
    if not stripped:
        return True
    if stripped.endswith((":", "$", "=", "*", "/", "+", "-", "(", ",")):
        return True
    has_answer = bool(re.search(r"(?:^|\n)\s*(?:answer|final answer)\s*:", stripped, flags=re.IGNORECASE))
    return bool(re.search(r"(?:Step\s+\d+:[^\n]*)$", stripped) and not has_answer and stripped[-1] not in ".!?")


def _equation_count(text: str) -> int:
    return len(re.findall(r"-?\d[\d,]*(?:\.\d+)?\s*[+\-*/x]\s*-?\d[\d,]*(?:\.\d+)?\s*=\s*-?\d[\d,]*(?:\.\d+)?", str(text), flags=re.IGNORECASE))


def _step_count(text: str) -> int:
    explicit = re.findall(r"(?:^|\n)\s*Step\s+\d+\s*:", str(text), flags=re.IGNORECASE)
    if explicit:
        return len(explicit)
    return len([line.strip() for line in str(text).splitlines() if line.strip()])


def _surface_text_score(text: str) -> float:
    raw = str(text)
    reasoning = _reasoning_prefix(raw)
    answer_line = _extract_answer_line(raw)
    answer_value = _extract_answer_value(raw)
    has_answer = bool(answer_line)
    incomplete = _looks_incomplete(raw)
    eq_count = _equation_count(raw)
    steps = _step_count(raw)
    reasoning_numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", reasoning)
    answer_mentioned = False
    if answer_value is not None:
        normalized_answer = str(int(answer_value)) if float(answer_value).is_integer() else str(answer_value)
        answer_mentioned = any(token.replace(",", "") == normalized_answer for token in reasoning_numbers)
    distinct_reasoning_numbers = len({token.replace(",", "") for token in reasoning_numbers})
    token_count = len(re.findall(r"\S+", raw))

    score = 0.0
    score += 1.4 if has_answer else -1.0
    score += 1.2 if not incomplete else -1.5
    score += min(float(steps), 6.0) * 0.12
    score += min(float(eq_count), 4.0) * 0.22
    score += min(float(distinct_reasoning_numbers), 8.0) * 0.05
    score += 0.45 if answer_mentioned else -0.15
    if has_answer and answer_line.endswith((".", "!", "?")):
        score += 0.05
    if token_count < 8:
        score -= 0.4
    if token_count > 120:
        score -= 0.25
    if re.search(r"\b(?:maybe|guess|probably|approximately|approx)\b", raw, flags=re.IGNORECASE):
        score -= 0.35
    return float(score)


def _surface_choice_result(candidates: list[dict[str, object]]) -> dict[str, object]:
    scored = []
    for candidate in candidates:
        scored.append(
            (
                float(_surface_text_score(str(candidate.get("text", "")))),
                dict(candidate),
            )
        )
    scored.sort(
        key=lambda pair: (
            float(pair[0]),
            float(pair[1].get("ambiguity_score", 0.0)),
            -int(pair[1].get("candidate_index", 0)),
        ),
        reverse=True,
    )
    if not scored:
        return {
            "correct": False,
            "chosen_candidate_index": -1,
            "gold_surface_score": 0.0,
            "best_wrong_surface_score": 0.0,
            "gold_minus_best_wrong": 0.0,
        }

    gold_scores = [float(score) for score, candidate in scored if float(candidate.get("label", 0.0)) > 0.5]
    wrong_scores = [float(score) for score, candidate in scored if float(candidate.get("label", 0.0)) <= 0.5]
    gold_surface_score = max(gold_scores) if gold_scores else 0.0
    best_wrong_surface_score = max(wrong_scores) if wrong_scores else 0.0
    chosen_score, chosen_candidate = scored[0]
    return {
        "correct": bool(float(chosen_candidate.get("label", 0.0)) > 0.5),
        "chosen_candidate_index": int(chosen_candidate.get("candidate_index", -1)),
        "chosen_surface_score": float(chosen_score),
        "gold_surface_score": float(gold_surface_score),
        "best_wrong_surface_score": float(best_wrong_surface_score),
        "gold_minus_best_wrong": float(gold_surface_score - best_wrong_surface_score),
    }


def _dataset_matrices(dataset_path: str, *, feature_mode: str):
    client = AnchoredOracleClient()
    translator = HeuristicAnchoredTranslator()
    examples = load_reasoning_examples(dataset_path)
    X, y, groups = materialize_dataset(
        examples,
        client=client,
        translator=translator,
        feature_mode=str(feature_mode),
        text_dim=256,
        oracle_feature_groups=None,
        text_encoder_name="hashed",
        hf_model="",
        hf_max_length=256,
    )
    return X, y, groups, examples


def _challenger_model_correctness(challenger_jsonl: Path, model_path: Path) -> dict[str, bool]:
    X, y, groups, examples = _dataset_matrices(str(challenger_jsonl), feature_mode="text+oracle")
    model = PairwiseMLPReranker.load(str(model_path))
    scores = model.score(X)
    correctness: dict[str, bool] = {}
    for group_index in sorted(set(int(value) for value in groups.tolist())):
        idx = [position for position, value in enumerate(groups.tolist()) if int(value) == int(group_index)]
        ranked = sorted(idx, key=lambda position: float(scores[position]), reverse=True)
        correctness[str(examples[int(group_index)].problem_id)] = bool(ranked and float(y[int(ranked[0])]) > 0.5)
    return correctness


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a stricter structural ambiguity tier where both a surface chooser and fused text+oracle reranker fail.")
    ap.add_argument("--benchmark_jsonl", required=True)
    ap.add_argument("--challenger_jsonl", required=True)
    ap.add_argument("--suite_json", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--out_summary_json", default="")
    ap.add_argument("--max_surface_margin", type=float, default=0.0)
    ap.add_argument("--group_type_allowlist", default="")
    ap.add_argument("--emit_group_type_slices", action="store_true")
    args = ap.parse_args()

    benchmark_rows = _read_jsonl(Path(str(args.benchmark_jsonl)))
    challenger_rows = _read_jsonl(Path(str(args.challenger_jsonl)))
    suite = json.loads(Path(str(args.suite_json)).read_text(encoding="utf-8"))
    fused_model_path = None
    for run in suite.get("reranker_runs", []):
        if str(run.get("feature_mode", "")) == "text+oracle":
            fused_model_path = Path(str(run.get("model_path", "")))
            break
    if fused_model_path is None or not fused_model_path.exists():
        raise FileNotFoundError("Could not locate text+oracle model from suite_json")

    challenger_by_problem = {str(row.get("problem_id", "")): row for row in challenger_rows}
    fused_correctness = _challenger_model_correctness(Path(str(args.challenger_jsonl)), fused_model_path)
    allowed_group_types = set(_parse_csv_arg(str(args.group_type_allowlist)))

    strict_rows: list[dict[str, object]] = []
    type_counter: Counter[str] = Counter()
    surface_margin_values: list[float] = []
    for row in benchmark_rows:
        group_type = str(row.get("group_type", "other"))
        if allowed_group_types and group_type not in allowed_group_types:
            continue
        problem_id = str(row.get("problem_id", ""))
        challenger_row = challenger_by_problem.get(problem_id)
        if challenger_row is None:
            continue
        allowed_indexes = set(int(index) for index in row.get("ambiguous_candidate_indexes", []))
        allowed_indexes.add(int(row.get("gold_candidate_index", -1)))
        filtered_candidates = [
            dict(candidate)
            for candidate in row.get("candidates", [])
            if int(candidate.get("candidate_index", -1)) in allowed_indexes
        ]
        surface_result = _surface_choice_result(filtered_candidates)
        surface_correct = bool(surface_result.get("correct", False))
        fused_correct = bool(fused_correctness.get(problem_id, False))
        surface_margin = float(surface_result.get("gold_minus_best_wrong", 0.0))
        if fused_correct:
            continue
        if surface_correct and surface_margin > float(args.max_surface_margin):
            continue
        enriched = dict(row)
        enriched["strict_tier_reasons"] = {
            "surface_text_chooser_correct": bool(surface_correct),
            "fused_text_oracle_correct": bool(fused_correct),
            "surface_choice_candidate_index": int(surface_result.get("chosen_candidate_index", -1)),
            "gold_minus_best_wrong_surface_score": float(surface_margin),
        }
        strict_rows.append(enriched)
        type_counter.update([group_type])
        surface_margin_values.append(float(surface_margin))

    out_jsonl = Path(str(args.out_jsonl))
    _write_jsonl(out_jsonl, strict_rows)
    summary = {
        "benchmark_jsonl": str(args.benchmark_jsonl),
        "challenger_jsonl": str(args.challenger_jsonl),
        "suite_json": str(args.suite_json),
        "out_jsonl": str(out_jsonl),
        "max_surface_margin": float(args.max_surface_margin),
        "group_type_allowlist": sorted(allowed_group_types),
        "groups_total": int(len(benchmark_rows)),
        "groups_strict": int(len(strict_rows)),
        "strict_fraction": float(len(strict_rows) / len(benchmark_rows)) if benchmark_rows else 0.0,
        "group_types": dict(sorted(type_counter.items())),
        "mean_gold_minus_best_wrong_surface_score": _mean(surface_margin_values),
        "sample_problem_ids": [str(row.get("problem_id", "")) for row in strict_rows[:12]],
    }
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if str(args.out_summary_json).strip():
        out_summary = Path(str(args.out_summary_json))
        out_summary.parent.mkdir(parents=True, exist_ok=True)
        out_summary.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote {out_summary}")
    if bool(args.emit_group_type_slices):
        stem = out_jsonl.stem
        suffix = out_jsonl.suffix
        summary_stem = Path(str(args.out_summary_json)).stem if str(args.out_summary_json).strip() else ""
        summary_suffix = Path(str(args.out_summary_json)).suffix if str(args.out_summary_json).strip() else ""
        for group_type in sorted(type_counter):
            typed_rows = [row for row in strict_rows if str(row.get("group_type", "other")) == group_type]
            typed_path = out_jsonl.with_name(f"{stem}_{group_type}{suffix}")
            _write_jsonl(typed_path, typed_rows)
            print(f"wrote {typed_path}")
            if summary_stem and summary_suffix:
                typed_summary = out_summary.with_name(f"{summary_stem}_{group_type}{summary_suffix}")
                typed_summary_payload = {
                    **summary,
                    "out_jsonl": str(typed_path),
                    "groups_strict": int(len(typed_rows)),
                    "strict_fraction": float(len(typed_rows) / len(benchmark_rows)) if benchmark_rows else 0.0,
                    "group_types": {group_type: int(len(typed_rows))},
                    "mean_gold_minus_best_wrong_surface_score": _mean([
                        float(row.get("strict_tier_reasons", {}).get("gold_minus_best_wrong_surface_score", 0.0))
                        for row in typed_rows
                    ]),
                    "sample_problem_ids": [str(row.get("problem_id", "")) for row in typed_rows[:12]],
                }
                typed_summary.write_text(json.dumps(typed_summary_payload, indent=2) + "\n", encoding="utf-8")
                print(f"wrote {typed_summary}")
    print(f"wrote {out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())