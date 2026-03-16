from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml_oracle.translator import HeuristicAnchoredTranslator


def _local_path(raw_path: str | Path) -> Path:
    path = Path(str(raw_path).replace("\\", "/"))
    resolved = path if path.is_absolute() else REPO_ROOT / path
    if sys.platform == "win32":
        raw = str(resolved)
        if not raw.startswith("\\\\?\\") and len(raw) >= 240:
            return Path("\\\\?\\" + raw)
    return resolved


def _read_jsonl_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    if math.isclose(float(denominator), 0.0, abs_tol=1e-12):
        return 0.0
    return float(numerator / denominator)


def _jaccard(left: set[object], right: set[object]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return float(len(left & right) / len(left | right))


def _sequence_overlap(left: list[str], right: list[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return float(SequenceMatcher(a=left, b=right).ratio())


def _extract_answer_value(text: str) -> float | None:
    match = re.search(r"(?:answer|final answer)\s*:\s*([^\n]+)", str(text), flags=re.IGNORECASE)
    if match is None:
        return None
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", match.group(1))
    if not numbers:
        return None
    try:
        return float(numbers[-1].replace(",", ""))
    except ValueError:
        return None


def _extract_answer_line(text: str) -> str:
    match = re.search(r"(?:answer|final answer)\s*:\s*([^\n]+)", str(text), flags=re.IGNORECASE)
    return str(match.group(1)).strip() if match is not None else ""


def _reasoning_prefix(text: str) -> str:
    match = re.search(r"(?:answer|final answer)\s*:", str(text), flags=re.IGNORECASE)
    return str(text)[: match.start()] if match is not None else str(text)


def _has_answer_line(text: str) -> bool:
    return bool(re.search(r"(?:^|\n)\s*(?:answer|final answer)\s*:", str(text), flags=re.IGNORECASE))


def _looks_incomplete(text: str) -> bool:
    stripped = str(text).rstrip()
    if not stripped:
        return True
    if stripped.endswith((":", "$", "=", "*", "/", "+", "-", "(", ",")):
        return True
    return bool(re.search(r"(?:Step\s+\d+:[^\n]*)$", stripped) and not _has_answer_line(stripped) and stripped[-1] not in ".!?")


def _equation_count(text: str) -> int:
    return len(re.findall(r"-?\d[\d,]*(?:\.\d+)?\s*[+\-*/x]\s*-?\d[\d,]*(?:\.\d+)?\s*=\s*-?\d[\d,]*(?:\.\d+)?", str(text), flags=re.IGNORECASE))


def _surface_text_score(text: str) -> float:
    raw = str(text)
    reasoning = _reasoning_prefix(raw)
    answer_line = _extract_answer_line(raw)
    answer_value = _extract_answer_value(raw)
    has_answer = bool(answer_line)
    incomplete = _looks_incomplete(raw)
    eq_count = _equation_count(raw)
    steps = len(re.findall(r"(?:^|\n)\s*Step\s+\d+\s*:", raw, flags=re.IGNORECASE)) or len([line.strip() for line in raw.splitlines() if line.strip()])
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


def _parse_csv_arg(raw: str) -> set[str]:
    return {item.strip() for item in str(raw).split(",") if item.strip()}


def _rounded_values(values: tuple[float, ...]) -> tuple[float, ...]:
    rounded: list[float] = []
    for value in values:
        normalized = round(float(value), 6)
        if not any(math.isclose(float(normalized), float(existing), rel_tol=1e-9, abs_tol=1e-9) for existing in rounded):
            rounded.append(float(normalized))
    return tuple(rounded)


def _analyze_candidate(
    translator: HeuristicAnchoredTranslator,
    *,
    prompt: str,
    text: str,
    candidate_index: int,
    label: float,
    role_tag: str,
    provenance_tag: str,
) -> dict[str, object]:
    states = translator.reasoning_states_for_trace(text, prompt=prompt)
    operations = [str(state.operation) for state in states]
    dependency_edges = {
        (int(dep_index), int(state.step_index))
        for state in states
        for dep_index in state.dependency_step_indexes
    }
    introduced_values = {
        round(float(value), 6)
        for state in states
        for value in state.introduced_values
    }
    dependency_values = {
        round(float(value), 6)
        for state in states
        for value in state.dependency_values
    }
    signature_rows = set(translator.structural_state_signature_for_trace(text, prompt=prompt))
    total_states = max(1, len(states))
    valid_state_count = sum(1 for state in states if state.result is not None or len(state.operands) >= 2)
    answer_support_count = sum(1 for state in states if state.answer_support)
    dependency_state_count = sum(1 for state in states if state.dependency_step_indexes)
    answer_support_dependency_count = sum(1 for state in states if state.answer_support and state.dependency_step_indexes)
    carried_result_count = sum(1 for state in states if state.carried_result)
    correct_equation_count = sum(1 for state in states if state.equation_correct is True)
    incorrect_equation_count = sum(1 for state in states if state.equation_correct is False)
    equation_total = correct_equation_count + incorrect_equation_count
    depth_map: dict[int, int] = {}
    chain_depths: list[int] = []
    for state in states:
        if state.dependency_step_indexes:
            depth = 1 + max(depth_map.get(int(dep_index), 1) for dep_index in state.dependency_step_indexes)
        else:
            depth = 1
        depth_map[int(state.step_index)] = int(depth)
        chain_depths.append(int(depth))

    return {
        "candidate_index": int(candidate_index),
        "label": float(label),
        "role_tag": str(role_tag),
        "provenance_tag": str(provenance_tag),
        "text": str(text),
        "surface_text_score": float(_surface_text_score(text)),
        "answer_value": _extract_answer_value(text),
        "has_answer_line": bool(_has_answer_line(text)),
        "incomplete_trace": bool(_looks_incomplete(text)),
        "step_count": int(len(translator.split_reasoning_steps(text))),
        "state_count": int(len(states)),
        "valid_state_count": int(valid_state_count),
        "valid_state_fraction": float(valid_state_count / total_states),
        "answer_support_count": int(answer_support_count),
        "answer_support_fraction": float(answer_support_count / total_states),
        "dependency_edge_count": int(len(dependency_edges)),
        "dependency_state_fraction": float(dependency_state_count / total_states),
        "answer_support_dependency_fraction": float(answer_support_dependency_count / total_states),
        "answer_support_given_dependency": float(answer_support_dependency_count / max(1, dependency_state_count)),
        "carried_result_count": int(carried_result_count),
        "carried_result_fraction": float(carried_result_count / total_states),
        "correct_equation_count": int(correct_equation_count),
        "incorrect_equation_count": int(incorrect_equation_count),
        "equation_consistency": float(correct_equation_count / equation_total) if equation_total else 0.0,
        "max_chain_depth": float(max(chain_depths) if chain_depths else 0.0),
        "normalized_chain_depth": float((max(chain_depths) if chain_depths else 0.0) / total_states),
        "operations": operations,
        "dependency_edges": sorted([list(edge) for edge in dependency_edges]),
        "introduced_values": list(_rounded_values(tuple(float(value) for value in introduced_values))),
        "dependency_values": list(_rounded_values(tuple(float(value) for value in dependency_values))),
        "signature_rows": len(signature_rows),
        "_dependency_edge_set": dependency_edges,
        "_introduced_value_set": introduced_values,
        "_signature_set": signature_rows,
    }


def _comparison_to_gold(candidate: dict[str, object], gold: dict[str, object]) -> dict[str, float | None]:
    edge_overlap = _jaccard(set(candidate["_dependency_edge_set"]), set(gold["_dependency_edge_set"]))
    introduced_overlap = _jaccard(set(candidate["_introduced_value_set"]), set(gold["_introduced_value_set"]))
    signature_overlap = _jaccard(set(candidate["_signature_set"]), set(gold["_signature_set"]))
    operation_overlap = _sequence_overlap(list(candidate["operations"]), list(gold["operations"]))
    depth_ratio = min(1.0, _safe_ratio(float(candidate["max_chain_depth"]), max(1.0, float(gold["max_chain_depth"]))))
    answer_distance = None
    if candidate.get("answer_value") is not None and gold.get("answer_value") is not None:
        answer_distance = abs(float(candidate["answer_value"]) - float(gold["answer_value"]))
    return {
        "operation_overlap": float(operation_overlap),
        "dependency_edge_overlap": float(edge_overlap),
        "introduced_value_overlap": float(introduced_overlap),
        "signature_overlap": float(signature_overlap),
        "depth_ratio": float(depth_ratio),
        "answer_distance": None if answer_distance is None else float(answer_distance),
    }


def _ambiguity_tags(candidate: dict[str, object], gold: dict[str, object], comparison: dict[str, float | None]) -> list[str]:
    tags: list[str] = []
    if bool(candidate.get("incomplete_trace", False)):
        tags.append("incomplete_trace")
    if float(candidate.get("label", 0.0)) > 0.5:
        return tags

    operation_overlap = float(comparison["operation_overlap"])
    edge_overlap = float(comparison["dependency_edge_overlap"])
    signature_overlap = float(comparison["signature_overlap"])
    introduced_overlap = float(comparison["introduced_value_overlap"])
    answer_distance = comparison.get("answer_distance")
    equation_consistency = float(candidate.get("equation_consistency", 0.0))
    valid_state_fraction = float(candidate.get("valid_state_fraction", 0.0))
    answer_support_fraction = float(candidate.get("answer_support_fraction", 0.0))
    has_answer_line = bool(candidate.get("has_answer_line", False))
    dependency_edge_count = int(candidate.get("dependency_edge_count", 0))
    gold_dependency_edge_count = int(gold.get("dependency_edge_count", 0))

    if (
        operation_overlap >= 0.75
        and max(edge_overlap, signature_overlap, introduced_overlap) >= 0.55
        and answer_distance is not None
        and answer_distance > 0.0
    ):
        tags.append("arithmetic_slip")
    if (
        operation_overlap >= 0.6
        and dependency_edge_count > 0
        and gold_dependency_edge_count > 0
        and edge_overlap < 0.5
    ):
        tags.append("dependency_conflict")
    if (
        gold_dependency_edge_count >= 2
        and dependency_edge_count <= max(0, gold_dependency_edge_count - 1)
        and float(candidate.get("max_chain_depth", 0.0)) + 0.5 < float(gold.get("max_chain_depth", 0.0))
    ):
        tags.append("dependency_omission")
    if has_answer_line and answer_support_fraction <= 0.0:
        tags.append("unsupported_answer")
    if signature_overlap >= 0.8:
        tags.append("structural_near_duplicate")
    if valid_state_fraction >= 0.55 and equation_consistency >= 0.5 and operation_overlap >= 0.55:
        tags.append("structurally_plausible_wrong")
    return tags


def _ambiguity_score(candidate: dict[str, object], gold: dict[str, object], comparison: dict[str, float | None]) -> float:
    equation_consistency = float(candidate.get("equation_consistency", 0.0))
    if int(candidate.get("correct_equation_count", 0)) + int(candidate.get("incorrect_equation_count", 0)) == 0:
        equation_consistency = 0.5 if float(candidate.get("valid_state_fraction", 0.0)) >= 0.5 else 0.0
    plausibility = _mean(
        [
            float(candidate.get("valid_state_fraction", 0.0)),
            equation_consistency,
            max(float(candidate.get("answer_support_fraction", 0.0)), 0.25 if bool(candidate.get("has_answer_line", False)) else 0.0),
            max(float(candidate.get("answer_support_dependency_fraction", 0.0)), float(candidate.get("carried_result_fraction", 0.0))),
        ]
    )
    alignment = _mean(
        [
            float(comparison["operation_overlap"]),
            float(comparison["signature_overlap"]),
            max(float(comparison["dependency_edge_overlap"]), float(comparison["introduced_value_overlap"])),
            float(comparison["depth_ratio"]),
        ]
    )
    score = 0.6 * alignment + 0.4 * plausibility
    surface_margin = float(candidate.get("surface_text_score", 0.0)) - float(gold.get("surface_text_score", 0.0))
    if surface_margin >= 0.0:
        score += 0.08
    elif surface_margin >= -0.15:
        score += 0.04
    if 0.25 <= float(comparison["dependency_edge_overlap"]) < 0.9 and float(comparison["operation_overlap"]) >= 0.6:
        score += 0.05
    if float(candidate.get("valid_state_fraction", 0.0)) >= 0.55 and max(float(candidate.get("carried_result_fraction", 0.0)), float(candidate.get("answer_support_fraction", 0.0))) >= 0.2:
        score += 0.04
    if bool(candidate.get("incomplete_trace", False)):
        score -= 0.2
    if float(candidate.get("label", 0.0)) > 0.5:
        score = 1.0
    return float(max(0.0, min(1.0, score)))


def _group_type(tags: set[str]) -> str:
    arithmetic = "arithmetic_slip" in tags
    dependency = bool({"dependency_conflict", "dependency_omission"} & tags)
    structural = bool({"structurally_plausible_wrong", "structural_near_duplicate"} & tags)
    if arithmetic and dependency:
        return "mixed"
    if dependency:
        return "dependency"
    if arithmetic:
        return "arithmetic"
    if structural:
        return "structural"
    return "other"


def _strip_private_fields(candidate: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in candidate.items() if not str(key).startswith("_")}


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a structural ambiguity stress-test benchmark from grouped reasoning candidates.")
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--controller_json", default="")
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--out_summary_json", default="")
    ap.add_argument("--min_ambiguity_score", type=float, default=0.6)
    ap.add_argument("--max_gold_surface_margin", type=float, default=None)
    ap.add_argument("--min_wrong_surface_margin", type=float, default=None)
    ap.add_argument("--min_surface_hard_candidates", type=int, default=0)
    ap.add_argument("--group_type_allowlist", default="")
    args = ap.parse_args()

    translator = HeuristicAnchoredTranslator()
    input_path = _local_path(args.input_jsonl)
    records = _read_jsonl_records(input_path)
    controller_groups: dict[str, dict[str, object]] = {}
    if str(args.controller_json).strip():
        controller_report = json.loads(_local_path(args.controller_json).read_text(encoding="utf-8"))
        controller_groups = {
            str(group.get("problem_id", "")): dict(group)
            for group in controller_report.get("group_reports", [])
        }

    min_ambiguity_score = float(args.min_ambiguity_score)
    max_gold_surface_margin = None if args.max_gold_surface_margin is None else float(args.max_gold_surface_margin)
    min_wrong_surface_margin = None if args.min_wrong_surface_margin is None else float(args.min_wrong_surface_margin)
    group_type_allowlist = _parse_csv_arg(str(args.group_type_allowlist))
    included_records: list[dict[str, object]] = []
    tag_counter: Counter[str] = Counter()
    group_type_counter: Counter[str] = Counter()
    low_margin_included = 0
    branch_included = 0
    surface_hard_included = 0

    for row in records:
        prompt = str(row.get("prompt", ""))
        raw_candidates = list(row.get("candidates", []))
        analyzed_candidates: list[dict[str, object]] = []
        for candidate_index, candidate in enumerate(raw_candidates):
            candidate_row = dict(candidate)
            analyzed_candidates.append(
                _analyze_candidate(
                    translator,
                    prompt=prompt,
                    text=str(candidate_row.get("text", "")),
                    candidate_index=int(candidate_index),
                    label=float(candidate_row.get("label", 0.0)),
                    role_tag=str(candidate_row.get("role_tag", "")),
                    provenance_tag=str(candidate_row.get("provenance_tag", "")),
                )
            )

        gold_candidates = [candidate for candidate in analyzed_candidates if float(candidate.get("label", 0.0)) > 0.5]
        if len(gold_candidates) != 1:
            continue
        gold = gold_candidates[0]

        ambiguous_candidates: list[dict[str, object]] = []
        all_tags: set[str] = set()
        for candidate in analyzed_candidates:
            comparison = _comparison_to_gold(candidate, gold)
            tags = _ambiguity_tags(candidate, gold, comparison)
            score = _ambiguity_score(candidate, gold, comparison)
            candidate["comparison_to_gold"] = comparison
            candidate["ambiguity_tags"] = tags
            candidate["ambiguity_score"] = float(score)
            candidate["surface_score_margin_vs_gold"] = float(candidate.get("surface_text_score", 0.0)) - float(gold.get("surface_text_score", 0.0))
            candidate["benchmark_worthy"] = bool(
                float(candidate.get("label", 0.0)) <= 0.5
                and score >= min_ambiguity_score
                and bool({"arithmetic_slip", "dependency_conflict", "dependency_omission", "structurally_plausible_wrong", "structural_near_duplicate"} & set(tags))
                and not bool(candidate.get("incomplete_trace", False))
            )
            all_tags.update(tags)
            if bool(candidate["benchmark_worthy"]):
                ambiguous_candidates.append(candidate)

        if not ambiguous_candidates:
            continue

        wrong_surface_margins = [
            float(candidate.get("surface_score_margin_vs_gold", 0.0))
            for candidate in ambiguous_candidates
        ]
        best_wrong_surface_margin = max(wrong_surface_margins) if wrong_surface_margins else float("-inf")
        surface_hard_candidate_count = sum(1 for margin in wrong_surface_margins if margin >= 0.0)

        ambiguous_candidates.sort(
            key=lambda candidate: (
                float(candidate.get("ambiguity_score", 0.0)),
                float(candidate.get("surface_score_margin_vs_gold", 0.0)),
                float(candidate.get("comparison_to_gold", {}).get("signature_overlap", 0.0)),
                float(candidate.get("comparison_to_gold", {}).get("operation_overlap", 0.0)),
            ),
            reverse=True,
        )
        problem_id = str(row.get("problem_id", ""))
        controller_group = controller_groups.get(problem_id, {})
        if controller_group:
            if float(controller_group.get("oracle_margin", 0.0)) <= 0.0:
                low_margin_included += 1
            if any(bool(candidate.get("in_branch", False)) for candidate in controller_group.get("candidates", [])):
                branch_included += 1

        group_type = _group_type(set(tag for candidate in ambiguous_candidates for tag in candidate.get("ambiguity_tags", [])))
        if group_type_allowlist and group_type not in group_type_allowlist:
            continue
        if max_gold_surface_margin is not None and (-best_wrong_surface_margin) > max_gold_surface_margin:
            continue
        if min_wrong_surface_margin is not None and best_wrong_surface_margin < min_wrong_surface_margin:
            continue
        if surface_hard_candidate_count < int(args.min_surface_hard_candidates):
            continue
        group_type_counter.update([group_type])
        for candidate in ambiguous_candidates:
            tag_counter.update(list(candidate.get("ambiguity_tags", [])))
        if best_wrong_surface_margin >= 0.0:
            surface_hard_included += 1

        included_records.append(
            {
                "problem_id": problem_id,
                "prompt": prompt,
                "gold_candidate_index": int(gold.get("candidate_index", -1)),
                "group_type": group_type,
                "max_ambiguity_score": float(max(candidate.get("ambiguity_score", 0.0) for candidate in ambiguous_candidates)),
                "gold_surface_text_score": float(gold.get("surface_text_score", 0.0)),
                "best_wrong_surface_text_score": float(float(gold.get("surface_text_score", 0.0)) + best_wrong_surface_margin),
                "gold_minus_best_wrong_surface_score": float(-best_wrong_surface_margin),
                "surface_hard_candidate_count": int(surface_hard_candidate_count),
                "ambiguous_candidate_indexes": [int(candidate.get("candidate_index", -1)) for candidate in ambiguous_candidates],
                "ambiguous_tags": sorted(set(tag for candidate in ambiguous_candidates for tag in candidate.get("ambiguity_tags", []))),
                "controller_context": {
                    "oracle_margin": float(controller_group.get("oracle_margin", 0.0)),
                    "baseline_top_correct": bool(controller_group.get("baseline_top_correct", False)),
                    "controller_top_correct": bool(controller_group.get("controller_top_correct", False)),
                    "ranker_top_correct": bool(controller_group.get("ranker_top_correct", False)),
                    "branch_local_indexes": list(controller_group.get("branch_local_indexes", [])),
                    "ranker_branch_local_indexes": list(controller_group.get("ranker_branch_local_indexes", [])),
                }
                if controller_group
                else {},
                "candidates": [_strip_private_fields(candidate) for candidate in analyzed_candidates],
            }
        )

    out_jsonl = _local_path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for record in included_records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    summary = {
        "input_jsonl": str(args.input_jsonl),
        "controller_json": str(args.controller_json),
        "out_jsonl": str(out_jsonl),
        "min_ambiguity_score": float(min_ambiguity_score),
        "max_gold_surface_margin": max_gold_surface_margin,
        "min_wrong_surface_margin": min_wrong_surface_margin,
        "min_surface_hard_candidates": int(args.min_surface_hard_candidates),
        "group_type_allowlist": sorted(group_type_allowlist),
        "groups_total": int(len(records)),
        "groups_included": int(len(included_records)),
        "included_fraction": _safe_ratio(float(len(included_records)), float(len(records))),
        "group_types": dict(sorted(group_type_counter.items())),
        "ambiguity_tag_counts": dict(sorted(tag_counter.items())),
        "surface_hard_included_groups": int(surface_hard_included),
        "low_margin_included_groups": int(low_margin_included),
        "controller_context_groups": int(sum(1 for record in included_records if record.get("controller_context"))),
        "branch_context_groups": int(branch_included),
        "sample_problem_ids": [str(record.get("problem_id", "")) for record in included_records[:12]],
    }

    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if str(args.out_summary_json).strip():
        out_summary = _local_path(args.out_summary_json)
        try:
            out_summary.parent.mkdir(parents=True, exist_ok=True)
            out_summary.write_text(rendered + "\n", encoding="utf-8")
            print(f"wrote {out_summary}")
        except OSError as exc:
            print(f"warning: failed to write summary json {out_summary}: {exc}")
    print(f"wrote {out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())