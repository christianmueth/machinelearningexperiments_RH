from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

import numpy as np


ROLE_SPECS: tuple[dict[str, str], ...] = (
    {
        "role_tag": "best",
        "provenance_tag": "natural_role_best",
        "instruction": "Solve carefully. Be concise and accurate.",
    },
    {
        "role_tag": "fast",
        "provenance_tag": "natural_role_fast",
        "instruction": "Solve quickly using mental math. Keep it short and intuitive.",
    },
    {
        "role_tag": "steps",
        "provenance_tag": "natural_role_steps",
        "instruction": "Solve with short step-by-step reasoning. Keep the decomposition brief.",
    },
    {
        "role_tag": "check",
        "provenance_tag": "natural_role_check",
        "instruction": "Solve the problem, then double-check the arithmetic before giving the answer.",
    },
    {
        "role_tag": "alt",
        "provenance_tag": "natural_role_alt",
        "instruction": "Solve using a different method than the most obvious one.",
    },
)


def _read_jsonl_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _append_jsonl_record(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _load_hf_records(dataset_name: str, split: str, subset: str | None, *, local_files_only: bool) -> list[dict[str, object]]:
    from datasets import DownloadConfig, load_dataset

    download_config = DownloadConfig(local_files_only=bool(local_files_only))

    if subset:
        dataset = load_dataset(dataset_name, subset, split=split, download_config=download_config)
    else:
        dataset = load_dataset(dataset_name, split=split, download_config=download_config)
    return [dict(row) for row in dataset]


def _extract_gold_answer(text: str) -> str:
    raw = str(text).strip()
    if "####" in raw:
        raw = raw.split("####")[-1].strip()
    numeric = _extract_last_numeric(raw)
    return numeric if numeric is not None else raw.strip().lower()


def _extract_last_numeric(text: str) -> str | None:
    matches = re.findall(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))
    return matches[-1] if matches else None


def _numeric_matches(text: str) -> list[re.Match[str]]:
    return list(re.finditer(r"-?\d[\d,]*(?:\.\d+)?", str(text)))


def _extract_candidate_answer(text: str) -> str:
    raw = str(text).strip()
    answer_match = re.search(r"^answer\s*:\s*(.+)$", raw, flags=re.IGNORECASE | re.MULTILINE)
    if answer_match:
        candidate = answer_match.group(1).strip()
        numeric = _extract_last_numeric(candidate)
        return numeric if numeric is not None else candidate.lower()
    match = re.search(r"final answer\s*:\s*(.+)$", raw, flags=re.IGNORECASE | re.MULTILINE)
    if match:
        candidate = match.group(1).strip()
        numeric = _extract_last_numeric(candidate)
        return numeric if numeric is not None else candidate.lower()
    numeric = _extract_last_numeric(raw)
    return numeric if numeric is not None else raw.lower()


def _clean_generated_text(text: str) -> str:
    cleaned = str(text).strip()
    restart_markers = (
        "\nHuman:",
        "\nUser:",
        "\nQ:",
        "\nQuestion:",
        "\nProblem:",
        ".Human:",
        ".User:",
        ".Q:",
        ".Question:",
        ".Problem:",
        ".Write ",
    )
    for marker in restart_markers:
        pos = cleaned.find(marker)
        if pos >= 0:
            cleaned = cleaned[:pos].rstrip()
    final_match = re.search(r"(?is)(final answer\s*:\s*[^\n]+)", cleaned)
    if final_match:
        cleaned = cleaned[: final_match.end()].rstrip()
    return cleaned


def _candidate_dedup_key(text: str) -> tuple[str, str]:
    cleaned = _clean_generated_text(text)
    normalized = " ".join(cleaned.split())
    return (_extract_candidate_answer(cleaned), normalized)


def _perturb_numeric_string(value: str, *, avoid: set[str] | None = None) -> str:
    avoid_values = set(avoid or set())
    if "." in str(value):
        numeric = float(value)
        decimal_places = max(1, len(str(value).split(".", 1)[1]))
        offsets = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)
        for offset in random.sample(list(offsets), k=len(offsets)):
            candidate = numeric + offset
            rendered = f"{candidate:.{decimal_places}f}"
            if rendered not in avoid_values:
                return rendered
        fallback = numeric + 1.5
        return f"{fallback:.{decimal_places}f}"

    numeric = int(value)
    offsets = (-5, -3, -2, -1, 1, 2, 3, 5)
    for offset in random.sample(list(offsets), k=len(offsets)):
        shifted = numeric + offset
        if numeric > 0 and shifted <= 0:
            continue
        candidate = str(shifted)
        if candidate not in avoid_values:
            return candidate
    fallback = numeric + 7 if numeric >= 0 else numeric - 7
    if numeric > 0 and fallback <= 0:
        fallback = numeric + 9
    return str(fallback)


def _replace_last_numeric(text: str, replacement: str) -> str:
    raw = str(text)
    matches = _numeric_matches(raw)
    if not matches:
        return raw
    match = matches[-1]
    return raw[: match.start()] + str(replacement) + raw[match.end() :]


def _replace_answer_numeric(text: str, replacement: str) -> str:
    patterns = (
        r"(^answer\s*:\s*)(-?\d+(?:\.\d+)?)",
        r"(^final answer\s*:\s*)(-?\d+(?:\.\d+)?)",
    )
    for pattern in patterns:
        match = re.search(pattern, str(text), flags=re.IGNORECASE | re.MULTILINE)
        if match:
            start, end = match.span(2)
            return str(text)[:start] + str(replacement) + str(text)[end:]
    return _replace_last_numeric(str(text), str(replacement))


def _perturb_nonfinal_numeric(text: str, *, gold_answer: str) -> str:
    raw = str(text)
    matches = _numeric_matches(raw)
    if len(matches) <= 1:
        return raw

    answer_value = _extract_candidate_answer(raw)
    candidate_indexes: list[int] = []
    fallback_indexes: list[int] = []
    for index, match in enumerate(matches[:-1]):
        original_value = match.group(0).replace(",", "")
        numeric_value = _extract_last_numeric(original_value)
        if numeric_value is None:
            continue
        absolute_value = abs(float(numeric_value))
        if absolute_value >= 4.0:
            candidate_indexes.append(index)
        else:
            fallback_indexes.append(index)
    candidate_indexes = list(reversed(candidate_indexes or fallback_indexes))
    for index in candidate_indexes:
        match = matches[index]
        original_value = match.group(0).replace(",", "")
        replacement = _perturb_numeric_string(original_value, avoid={original_value, str(gold_answer), str(answer_value)})
        if replacement == original_value:
            continue
        return raw[: match.start()] + replacement + raw[match.end() :]
    return raw


def _make_near_miss_numeric(text: str, *, gold_answer: str) -> str | None:
    cleaned = _clean_generated_text(text)
    answer_value = _extract_candidate_answer(cleaned)
    numeric_answer = _extract_last_numeric(answer_value)
    if numeric_answer is None:
        return None

    wrong_answer = _perturb_numeric_string(numeric_answer, avoid={str(gold_answer), numeric_answer})
    candidate = _perturb_nonfinal_numeric(cleaned, gold_answer=str(gold_answer))
    candidate = _replace_answer_numeric(candidate, wrong_answer)
    candidate = _clean_generated_text(candidate)
    if _extract_candidate_answer(candidate) == str(gold_answer):
        return None
    return candidate


def _render_number_like(reference: str, value: float) -> str:
    raw = str(reference).replace(",", "")
    if "." in raw:
        decimals = len(raw.split(".", 1)[1])
        return f"{float(value):.{decimals}f}"
    rounded = int(round(float(value)))
    return str(rounded)


def _compute_binary(op: str, left: float, right: float) -> float | None:
    if op == "+":
        return float(left + right)
    if op == "-":
        return float(left - right)
    if op in {"*", "x", "X"}:
        return float(left * right)
    if op == "/":
        if abs(float(right)) <= 1e-12:
            return None
        return float(left / right)
    return None


def _equation_records(text: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_index, line in enumerate(str(text).splitlines()):
        match = re.search(
            r"(?P<a>-?\d[\d,]*(?:\.\d+)?)\s*(?P<op>[+\-*/xX])\s*(?P<b>-?\d[\d,]*(?:\.\d+)?)\s*=\s*(?P<c>-?\d[\d,]*(?:\.\d+)?)",
            line,
        )
        if match is None:
            continue
        records.append(
            {
                "line_index": int(line_index),
                "line": str(line),
                "match": match,
                "a_raw": str(match.group("a")),
                "b_raw": str(match.group("b")),
                "c_raw": str(match.group("c")),
                "a": float(str(match.group("a")).replace(",", "")),
                "b": float(str(match.group("b")).replace(",", "")),
                "c": float(str(match.group("c")).replace(",", "")),
                "op": str(match.group("op")),
            }
        )
    return records


def _replace_equation_values(line: str, match: re.Match[str], *, a_raw: str, b_raw: str, c_raw: str) -> str:
    replacement = f"{a_raw} {match.group('op')} {b_raw} = {c_raw}"
    return str(line)[: match.start()] + replacement + str(line)[match.end() :]


def _replace_answer_value(text: str, replacement: str) -> str:
    return _clean_generated_text(_replace_answer_numeric(str(text), str(replacement)))


def _split_reasoning_and_answer(text: str) -> tuple[str, str] | None:
    cleaned = _clean_generated_text(text)
    match = re.search(r"(?im)^\s*(answer|final answer)\s*:\s*.*$", cleaned)
    if match is None:
        return None
    reasoning = cleaned[: match.start()].rstrip()
    answer_line = cleaned[match.start() :].strip()
    if not reasoning or not answer_line:
        return None
    return reasoning, answer_line


def _next_step_number(text: str) -> int:
    matches = re.findall(r"(?:^|\n)\s*Step\s+(\d+)\s*:", str(text), flags=re.IGNORECASE)
    if matches:
        return max(int(value) for value in matches) + 1
    nonempty_lines = [line for line in str(text).splitlines() if line.strip()]
    return max(1, len(nonempty_lines) + 1)


def _surface_polish_candidate(text: str, *, answer_value: str) -> str:
    parts = _split_reasoning_and_answer(text)
    if parts is None:
        return _clean_generated_text(text)
    reasoning, answer_line = parts
    step_number = _next_step_number(reasoning)
    reinforcement = f"Step {step_number}: Use that carried result as the final answer, so the value is {answer_value}."
    return _clean_generated_text(f"{reasoning}\n{reinforcement}\n{answer_line}")


def _candidate_variants(base_text: str, *, answer_value: str) -> list[str]:
    variants: list[str] = []
    for candidate in (
        _surface_polish_candidate(base_text, answer_value=str(answer_value)),
        _clean_generated_text(base_text),
    ):
        if candidate and candidate not in variants:
            variants.append(candidate)
    return variants


def _make_wrong_answer_from_state(text: str, *, gold_answer: str, variant_index: int = 0) -> str | None:
    cleaned = _clean_generated_text(text)
    equations = _equation_records(cleaned)
    if not equations:
        return None
    candidates: list[str] = []
    for equation in reversed(equations[:-1] or equations):
        candidate_answer = _render_number_like(str(equation["c_raw"]), float(equation["c"]))
        if candidate_answer == str(gold_answer):
            continue
        candidate = _replace_answer_value(cleaned, candidate_answer)
        if _extract_candidate_answer(candidate) == str(gold_answer):
            continue
        for variant in _candidate_variants(candidate, answer_value=str(candidate_answer)):
            if variant not in candidates:
                candidates.append(variant)
    return candidates[int(variant_index)] if int(variant_index) < len(candidates) else None


def _make_dependency_conflict(text: str, *, gold_answer: str, variant_index: int = 0) -> str | None:
    cleaned = _clean_generated_text(text)
    lines = cleaned.splitlines()
    equations = _equation_records(cleaned)
    if len(equations) < 2:
        return None
    candidates: list[str] = []
    for equation_index in range(len(equations) - 1, 0, -1):
        equation = equations[equation_index]
        prior_results = [float(previous["c"]) for previous in equations[:equation_index]]
        prior_result_raws = [str(previous["c_raw"]) for previous in equations[:equation_index]]
        choices = list(zip(prior_results, prior_result_raws))
        choices.reverse()
        for prior_value, prior_raw in choices:
            if abs(float(prior_value) - float(equation["a"])) <= 1e-9 or abs(float(prior_value) - float(equation["b"])) <= 1e-9:
                continue
            new_result = _compute_binary(str(equation["op"]), float(prior_value), float(equation["b"]))
            if new_result is None:
                continue
            rendered_result = _render_number_like(str(equation["c_raw"]), float(new_result))
            if rendered_result == str(gold_answer):
                continue
            updated_lines = list(lines)
            updated_lines[int(equation["line_index"])] = _replace_equation_values(
                str(equation["line"]),
                equation["match"],
                a_raw=str(prior_raw),
                b_raw=str(equation["b_raw"]),
                c_raw=rendered_result,
            )
            candidate = _replace_answer_value("\n".join(updated_lines), rendered_result)
            if _extract_candidate_answer(candidate) == str(gold_answer):
                continue
            for variant in _candidate_variants(candidate, answer_value=str(rendered_result)):
                if variant not in candidates:
                    candidates.append(variant)
    return candidates[int(variant_index)] if int(variant_index) < len(candidates) else None


def _make_bad_state_reuse(text: str, *, gold_answer: str, variant_index: int = 0) -> str | None:
    cleaned = _clean_generated_text(text)
    lines = cleaned.splitlines()
    equations = _equation_records(cleaned)
    if not equations:
        return None
    candidates: list[str] = []
    for equation in reversed(equations):
        replacement_b = None
        replacement_b_raw = None
        if equations.index(equation) > 0:
            previous = equations[equations.index(equation) - 1]
            if abs(float(previous["c"]) - float(equation["b"])) > 1e-9:
                replacement_b = float(previous["c"])
                replacement_b_raw = str(previous["c_raw"])
        if replacement_b is None:
            if abs(float(equation["a"]) - float(equation["b"])) <= 1e-9:
                continue
            replacement_b = float(equation["a"])
            replacement_b_raw = str(equation["a_raw"])
        new_result = _compute_binary(str(equation["op"]), float(equation["a"]), float(replacement_b))
        if new_result is None:
            continue
        rendered_result = _render_number_like(str(equation["c_raw"]), float(new_result))
        if rendered_result == str(gold_answer):
            continue
        updated_lines = list(lines)
        updated_lines[int(equation["line_index"])] = _replace_equation_values(
            str(equation["line"]),
            equation["match"],
            a_raw=str(equation["a_raw"]),
            b_raw=str(replacement_b_raw),
            c_raw=rendered_result,
        )
        candidate = _replace_answer_value("\n".join(updated_lines), rendered_result)
        if _extract_candidate_answer(candidate) == str(gold_answer):
            continue
        for variant in _candidate_variants(candidate, answer_value=str(rendered_result)):
            if variant not in candidates:
                candidates.append(variant)
    return candidates[int(variant_index)] if int(variant_index) < len(candidates) else None


def _structural_challenger_rows(
    text: str,
    *,
    gold_answer: str,
    dependency_conflict_count: int,
    state_reuse_count: int,
    wrong_answer_from_state_count: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    constructors: list[tuple[int, str, str, callable]] = [
        (int(dependency_conflict_count), "dependency_conflict", "structural_dependency_conflict", _make_dependency_conflict),
        (int(state_reuse_count), "state_reuse", "structural_state_reuse", _make_bad_state_reuse),
        (int(wrong_answer_from_state_count), "wrong_state_answer", "structural_wrong_answer_from_state", _make_wrong_answer_from_state),
    ]
    for count, role_prefix, provenance_tag, constructor in constructors:
        for index in range(max(0, int(count))):
            candidate_text = constructor(str(text), gold_answer=str(gold_answer), variant_index=index)
            if not candidate_text:
                continue
            rows.append(
                {
                    "text": str(candidate_text),
                    "role_tag": f"{role_prefix}_{index + 1}",
                    "provenance_tag": str(provenance_tag),
                }
            )
    return rows


def _append_unique_candidate_row(
    rows: list[dict[str, str]],
    row: dict[str, str],
    *,
    seen_keys: set[tuple[str, str]],
    deduplicate_candidates: bool,
) -> bool:
    text = str(row.get("text", "")).strip()
    if not text:
        return False
    dedup_key = _candidate_dedup_key(text)
    if bool(deduplicate_candidates) and dedup_key in seen_keys:
        return False
    seen_keys.add(dedup_key)
    rows.append(dict(row))
    return True


def _build_prompt(question: str, *, prompt_style: str, role_instruction: str = "") -> str:
    style = str(prompt_style).strip().lower()
    prefix = f"{role_instruction.strip()}\n\n" if str(role_instruction).strip() else ""
    if style == "structured_steps":
        return (
            f"{prefix}"
            "Solve the following arithmetic word problem.\n\n"
            "Write the reasoning as explicit numbered steps, one arithmetic step per line.\n"
            "Keep each step short and concrete. Finish with a final answer line.\n\n"
            f"Problem:\n{question.strip()}\n\n"
            "Format exactly:\n"
            "Step 1: <first arithmetic step>\n"
            "Step 2: <next arithmetic step>\n"
            "Step 3: <next arithmetic step if needed>\n"
            "Answer: <number>"
        )
    if style == "answer_first":
        return (
            f"{prefix}"
            "Solve the following math problem.\n\n"
            "Write the final numeric answer first on the first line.\n"
            "Then explain your reasoning briefly in 1-3 sentences.\n\n"
            f"Problem:\n{question.strip()}\n\n"
            "Format exactly:\n"
            "Answer: <number>\n"
            "Reasoning: <short explanation>"
        )
    if style == "strict":
        return (
            f"{prefix}"
            "Solve the math problem. Output exactly two lines.\n"
            "Line 1: Reasoning: <one short sentence>\n"
            "Line 2: Final answer: <number only>\n"
            f"Problem: {question.strip()}"
        )
    if style == "answer_only":
        return (
            f"{prefix}"
            "Solve the math problem and output exactly one line in this format: Final answer: <number only>.\n"
            f"Problem: {question.strip()}"
        )
    if style == "concise":
        return (
            f"{prefix}"
            "Solve the problem with at most two short reasoning steps. End exactly with 'Final answer: <number>'.\n"
            f"Problem: {question.strip()}\n"
            "Answer:"
        )
    return (
        f"{prefix}"
        "Solve the following problem. Give a short reasoning trace and end with 'Final answer: ...'.\n"
        f"Problem: {question.strip()}\n"
        "Trace:"
    )


def _select_role_specs(num_candidates: int) -> list[dict[str, str]]:
    if num_candidates <= len(ROLE_SPECS):
        return list(ROLE_SPECS[: max(1, num_candidates)])
    selected = list(ROLE_SPECS)
    while len(selected) < max(1, num_candidates):
        source = ROLE_SPECS[(len(selected) - len(ROLE_SPECS)) % len(ROLE_SPECS)]
        repeat_index = 1 + (len(selected) // len(ROLE_SPECS))
        selected.append(
            {
                "role_tag": f"{source['role_tag']}_{repeat_index}",
                "provenance_tag": source["provenance_tag"],
                "instruction": source["instruction"],
            }
        )
    return selected


def _configure_hf_environment(*, local_files_only: bool, disable_implicit_token: bool) -> None:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if disable_implicit_token:
        os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    if local_files_only:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def _load_generator(model_name: str, *, local_files_only: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=bool(local_files_only))
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=bool(local_files_only))
        model_kind = "seq2seq"
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=bool(local_files_only))
        model_kind = "causal"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, model_kind, device


def _tokenize_prompt(
    *,
    prompt: str,
    tokenizer,
    model_kind: str,
    max_input_length: int,
    use_chat_template: bool,
):
    if bool(use_chat_template) and model_kind == "causal" and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are solving short arithmetic word problems. Follow the requested output format exactly."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            truncation=True,
            max_length=int(max_input_length),
        )

    return tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=int(max_input_length),
        padding=False,
    )


def _set_generation_seed(seed: int) -> None:
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _generate_candidates(
    *,
    prompt: str,
    tokenizer,
    model,
    model_kind: str,
    device: str,
    num_candidates: int,
    temperature: float,
    top_p: float,
    max_input_length: int,
    max_new_tokens: int,
    use_chat_template: bool,
) -> list[str]:
    import torch

    batch = _tokenize_prompt(
        prompt=prompt,
        tokenizer=tokenizer,
        model_kind=model_kind,
        max_input_length=int(max_input_length),
        use_chat_template=bool(use_chat_template),
    )
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.no_grad():
        outputs = model.generate(
            **batch,
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            num_return_sequences=int(num_candidates),
            max_new_tokens=int(max_new_tokens),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    texts: list[str] = []
    input_ids = batch["input_ids"]
    input_len = int(input_ids.shape[1])
    for seq in outputs:
        if model_kind == "causal":
            decoded = tokenizer.decode(seq[input_len:], skip_special_tokens=True)
        else:
            decoded = tokenizer.decode(seq, skip_special_tokens=True)
        cleaned = _clean_generated_text(decoded)
        if cleaned:
            texts.append(cleaned)
    return texts


def _generate_role_conditioned_candidates(
    *,
    question: str,
    prompt_style: str,
    tokenizer,
    model,
    model_kind: str,
    device: str,
    num_candidates: int,
    temperature: float,
    top_p: float,
    max_input_length: int,
    max_new_tokens: int,
    use_chat_template: bool,
) -> list[dict[str, str]]:
    generated_rows: list[dict[str, str]] = []
    for role_spec in _select_role_specs(int(num_candidates)):
        prompt = _build_prompt(
            question,
            prompt_style=str(prompt_style),
            role_instruction=str(role_spec["instruction"]),
        )
        outputs = _generate_candidates(
            prompt=prompt,
            tokenizer=tokenizer,
            model=model,
            model_kind=model_kind,
            device=device,
            num_candidates=1,
            temperature=float(temperature),
            top_p=float(top_p),
            max_input_length=int(max_input_length),
            max_new_tokens=int(max_new_tokens),
            use_chat_template=bool(use_chat_template),
        )
        text = outputs[0] if outputs else ""
        generated_rows.append(
            {
                "text": text,
                "role_tag": str(role_spec["role_tag"]),
                "provenance_tag": str(role_spec["provenance_tag"]),
            }
        )
    return generated_rows


def _generate_sampling_candidates(
    *,
    question: str,
    prompt_style: str,
    tokenizer,
    model,
    model_kind: str,
    device: str,
    num_candidates: int,
    temperature: float,
    top_p: float,
    max_input_length: int,
    max_new_tokens: int,
    use_chat_template: bool,
    deduplicate_candidates: bool,
    max_regen_rounds: int,
) -> list[dict[str, str]]:
    prompt = _build_prompt(question, prompt_style=prompt_style)
    target = max(1, int(num_candidates))
    seen_keys: set[tuple[str, str]] = set()
    unique_rows: list[dict[str, str]] = []
    rounds_remaining = max(0, int(max_regen_rounds))

    while len(unique_rows) < target:
        needed = target - len(unique_rows)
        generated = _generate_candidates(
            prompt=prompt,
            tokenizer=tokenizer,
            model=model,
            model_kind=model_kind,
            device=device,
            num_candidates=needed,
            temperature=float(temperature),
            top_p=float(top_p),
            max_input_length=int(max_input_length),
            max_new_tokens=int(max_new_tokens),
            use_chat_template=bool(use_chat_template),
        )
        if not generated:
            break
        for text in generated:
            row = {"text": text}
            if not bool(deduplicate_candidates):
                unique_rows.append(row)
                if len(unique_rows) >= target:
                    break
                continue
            dedup_key = _candidate_dedup_key(text)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            unique_rows.append(row)
            if len(unique_rows) >= target:
                break
        if not bool(deduplicate_candidates):
            break
        if len(unique_rows) >= target:
            break
        if rounds_remaining <= 0:
            break
        rounds_remaining -= 1
    return unique_rows[:target]


def _generate_two_stage_candidates(
    *,
    question: str,
    gold_answer: str,
    prompt_style: str,
    tokenizer,
    model,
    model_kind: str,
    device: str,
    num_candidates: int,
    temperature: float,
    top_p: float,
    max_input_length: int,
    max_new_tokens: int,
    use_chat_template: bool,
    deduplicate_candidates: bool,
    max_regen_rounds: int,
    positive_search_rounds: int,
    near_miss_count: int,
    dependency_conflict_count: int,
    state_reuse_count: int,
    wrong_answer_from_state_count: int,
) -> list[dict[str, str]]:
    prompt = _build_prompt(question, prompt_style=prompt_style)
    target = max(2, int(num_candidates))
    pool_seen: set[tuple[str, str]] = set()
    positives: list[dict[str, str]] = []
    negatives: list[dict[str, str]] = []

    search_rounds_remaining = max(1, int(positive_search_rounds))
    while search_rounds_remaining > 0 and not positives:
        generated = _generate_candidates(
            prompt=prompt,
            tokenizer=tokenizer,
            model=model,
            model_kind=model_kind,
            device=device,
            num_candidates=max(target, 2),
            temperature=float(temperature),
            top_p=float(top_p),
            max_input_length=int(max_input_length),
            max_new_tokens=int(max_new_tokens),
            use_chat_template=bool(use_chat_template),
        )
        for text in generated:
            row = {"text": text}
            if not _append_unique_candidate_row(
                positives if _extract_candidate_answer(text) == str(gold_answer) else negatives,
                row,
                seen_keys=pool_seen,
                deduplicate_candidates=bool(deduplicate_candidates),
            ):
                continue
        search_rounds_remaining -= 1

    if not positives:
        return _generate_sampling_candidates(
            question=question,
            prompt_style=prompt_style,
            tokenizer=tokenizer,
            model=model,
            model_kind=model_kind,
            device=device,
            num_candidates=target,
            temperature=float(temperature),
            top_p=float(top_p),
            max_input_length=int(max_input_length),
            max_new_tokens=int(max_new_tokens),
            use_chat_template=bool(use_chat_template),
            deduplicate_candidates=bool(deduplicate_candidates),
            max_regen_rounds=int(max_regen_rounds),
        )

    final_rows: list[dict[str, str]] = []
    final_seen: set[tuple[str, str]] = set()
    anchor_row = {
        "text": str(positives[0]["text"]),
        "role_tag": "anchor",
        "provenance_tag": "natural_correct",
    }
    _append_unique_candidate_row(
        final_rows,
        anchor_row,
        seen_keys=final_seen,
        deduplicate_candidates=bool(deduplicate_candidates),
    )

    desired_near_misses = min(max(0, int(near_miss_count)), max(0, target - 1))
    near_miss_attempts = 0
    near_miss_index = 0
    while near_miss_index < desired_near_misses and near_miss_attempts < (desired_near_misses * 6 + 6):
        near_miss_attempts += 1
        candidate_text = _make_near_miss_numeric(str(anchor_row["text"]), gold_answer=str(gold_answer))
        if not candidate_text:
            continue
        if _extract_candidate_answer(candidate_text) == str(gold_answer):
            continue
        if _append_unique_candidate_row(
            final_rows,
            {
                "text": candidate_text,
                "role_tag": f"near_miss_{near_miss_index + 1}",
                "provenance_tag": "near_miss_numeric",
            },
            seen_keys=final_seen,
            deduplicate_candidates=bool(deduplicate_candidates),
        ):
            near_miss_index += 1

    for challenger_row in _structural_challenger_rows(
        str(anchor_row["text"]),
        gold_answer=str(gold_answer),
        dependency_conflict_count=int(dependency_conflict_count),
        state_reuse_count=int(state_reuse_count),
        wrong_answer_from_state_count=int(wrong_answer_from_state_count),
    ):
        if len(final_rows) >= target:
            break
        _append_unique_candidate_row(
            final_rows,
            challenger_row,
            seen_keys=final_seen,
            deduplicate_candidates=bool(deduplicate_candidates),
        )

    negative_index = 0
    while len(final_rows) < target and negative_index < len(negatives):
        negative_text = str(negatives[negative_index]["text"])
        negative_index += 1
        _append_unique_candidate_row(
            final_rows,
            {
                "text": negative_text,
                "role_tag": "natural_sample",
                "provenance_tag": "natural_wrong",
            },
            seen_keys=final_seen,
            deduplicate_candidates=bool(deduplicate_candidates),
        )

    regen_rounds_remaining = max(0, int(max_regen_rounds))
    while len(final_rows) < target and regen_rounds_remaining >= 0:
        needed = target - len(final_rows)
        generated = _generate_candidates(
            prompt=prompt,
            tokenizer=tokenizer,
            model=model,
            model_kind=model_kind,
            device=device,
            num_candidates=max(1, needed),
            temperature=float(temperature),
            top_p=float(top_p),
            max_input_length=int(max_input_length),
            max_new_tokens=int(max_new_tokens),
            use_chat_template=bool(use_chat_template),
        )
        for text in generated:
            if _extract_candidate_answer(text) == str(gold_answer):
                continue
            _append_unique_candidate_row(
                final_rows,
                {
                    "text": text,
                    "role_tag": "natural_sample",
                    "provenance_tag": "natural_wrong",
                },
                seen_keys=final_seen,
                deduplicate_candidates=bool(deduplicate_candidates),
            )
            if len(final_rows) >= target:
                break
        if len(final_rows) >= target or regen_rounds_remaining == 0:
            break
        regen_rounds_remaining -= 1

    while len(final_rows) < target and near_miss_attempts < (target * 10):
        near_miss_attempts += 1
        candidate_text = _make_near_miss_numeric(str(anchor_row["text"]), gold_answer=str(gold_answer))
        if not candidate_text:
            continue
        _append_unique_candidate_row(
            final_rows,
            {
                "text": candidate_text,
                "role_tag": f"near_miss_{near_miss_index + 1}",
                "provenance_tag": "near_miss_numeric",
            },
            seen_keys=final_seen,
            deduplicate_candidates=bool(deduplicate_candidates),
        )
        near_miss_index += 1

    return final_rows[:target]


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate grouped candidate-trace JSONL using a frozen Hugging Face model.")
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--out_config_json", default="")
    ap.add_argument("--out_summary_json", default="")
    ap.add_argument("--model", default="google/flan-t5-small")
    ap.add_argument("--input_jsonl", default="")
    ap.add_argument("--hf_dataset", default="")
    ap.add_argument("--hf_subset", default="")
    ap.add_argument("--hf_split", default="train")
    ap.add_argument("--prompt_field", default="question")
    ap.add_argument("--answer_field", default="answer")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--start_index", type=int, default=0)
    ap.add_argument("--num_candidates", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_input_length", type=int, default=256)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flush_every", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--concise_prompt", action="store_true")
    ap.add_argument("--prompt_style", choices=["trace", "concise", "strict", "answer_only", "answer_first", "structured_steps"], default="trace")
    ap.add_argument("--candidate_strategy", choices=["sampling", "roles", "two_stage"], default="sampling")
    ap.add_argument("--deduplicate_candidates", action="store_true")
    ap.add_argument("--max_regen_rounds", type=int, default=3)
    ap.add_argument("--positive_search_rounds", type=int, default=4)
    ap.add_argument("--two_stage_near_miss_count", type=int, default=2)
    ap.add_argument("--two_stage_dependency_conflict_count", type=int, default=0)
    ap.add_argument("--two_stage_state_reuse_count", type=int, default=0)
    ap.add_argument("--two_stage_wrong_answer_from_state_count", type=int, default=0)
    ap.add_argument("--require_positive", action="store_true")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--disable_implicit_hf_token", action="store_true")
    ap.add_argument("--use_chat_template", action="store_true")
    args = ap.parse_args()

    if not str(args.input_jsonl).strip() and not str(args.hf_dataset).strip():
        raise SystemExit("Provide either --input_jsonl or --hf_dataset.")

    _configure_hf_environment(
        local_files_only=bool(args.local_files_only),
        disable_implicit_token=bool(args.disable_implicit_hf_token),
    )

    if str(args.input_jsonl).strip():
        records = _read_jsonl_records(Path(str(args.input_jsonl)))
    else:
        records = _load_hf_records(
            str(args.hf_dataset),
            str(args.hf_split),
            str(args.hf_subset).strip() or None,
            local_files_only=bool(args.local_files_only),
        )

    _set_generation_seed(int(args.seed))
    tokenizer, model, model_kind, device = _load_generator(
        str(args.model),
        local_files_only=bool(args.local_files_only),
    )

    out_path = Path(str(args.out_jsonl))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing_problem_ids: set[str] = set()
    if bool(args.resume) and out_path.exists():
        for row in _read_jsonl_records(out_path):
            existing_problem_ids.add(str(row.get("problem_id", "")))
    elif out_path.exists():
        out_path.write_text("", encoding="utf-8")

    kept = 0
    skipped = 0
    skipped_existing = 0
    total_candidates = 0
    total_positives = 0
    groups_with_positive = 0
    groups_with_1p_2n = 0
    start_index = max(0, int(args.start_index))
    stop_index = start_index + max(0, int(args.limit))
    selected_records = records[start_index:stop_index]
    prompt_style = str(args.prompt_style)
    candidate_strategy = str(args.candidate_strategy)
    if bool(args.concise_prompt) and prompt_style == "trace":
        prompt_style = "concise"
    for index, record in enumerate(selected_records, start=start_index):
        prompt = str(record[str(args.prompt_field)]).strip()
        problem_id = str(record.get("problem_id", f"sample_{index:06d}"))
        if problem_id in existing_problem_ids:
            skipped_existing += 1
            continue
        gold_answer = _extract_gold_answer(str(record[str(args.answer_field)]))
        if candidate_strategy == "roles":
            generated_rows = _generate_role_conditioned_candidates(
                question=prompt,
                prompt_style=prompt_style,
                tokenizer=tokenizer,
                model=model,
                model_kind=model_kind,
                device=device,
                num_candidates=max(1, int(args.num_candidates)),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                max_input_length=int(args.max_input_length),
                max_new_tokens=int(args.max_new_tokens),
                use_chat_template=bool(args.use_chat_template),
            )
        elif candidate_strategy == "two_stage":
            generated_rows = _generate_two_stage_candidates(
                question=prompt,
                gold_answer=gold_answer,
                prompt_style=prompt_style,
                tokenizer=tokenizer,
                model=model,
                model_kind=model_kind,
                device=device,
                num_candidates=max(2, int(args.num_candidates)),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                max_input_length=int(args.max_input_length),
                max_new_tokens=int(args.max_new_tokens),
                use_chat_template=bool(args.use_chat_template),
                deduplicate_candidates=bool(args.deduplicate_candidates),
                max_regen_rounds=int(args.max_regen_rounds),
                positive_search_rounds=int(args.positive_search_rounds),
                near_miss_count=int(args.two_stage_near_miss_count),
                dependency_conflict_count=int(args.two_stage_dependency_conflict_count),
                state_reuse_count=int(args.two_stage_state_reuse_count),
                wrong_answer_from_state_count=int(args.two_stage_wrong_answer_from_state_count),
            )
        else:
            generated_rows = _generate_sampling_candidates(
                question=prompt,
                prompt_style=prompt_style,
                tokenizer=tokenizer,
                model=model,
                model_kind=model_kind,
                device=device,
                num_candidates=max(2, int(args.num_candidates)),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                max_input_length=int(args.max_input_length),
                max_new_tokens=int(args.max_new_tokens),
                use_chat_template=bool(args.use_chat_template),
                deduplicate_candidates=bool(args.deduplicate_candidates),
                max_regen_rounds=int(args.max_regen_rounds),
            )
        candidates: list[dict[str, object]] = []
        positive_count = 0
        for generated_row in generated_rows:
            text = str(generated_row.get("text", ""))
            label = 1.0 if _extract_candidate_answer(text) == gold_answer else 0.0
            positive_count += int(label > 0.5)
            candidate_record: dict[str, object] = {"text": text, "label": label}
            if str(generated_row.get("role_tag", "")).strip():
                candidate_record["role_tag"] = str(generated_row["role_tag"])
            if str(generated_row.get("provenance_tag", "")).strip():
                candidate_record["provenance_tag"] = str(generated_row["provenance_tag"])
            candidates.append(candidate_record)
        random.shuffle(candidates)
        if bool(args.require_positive) and positive_count == 0:
            skipped += 1
            continue
        total_candidates += len(candidates)
        total_positives += positive_count
        groups_with_positive += int(positive_count >= 1)
        groups_with_1p_2n += int(positive_count >= 1 and (len(candidates) - positive_count) >= 2)
        row = {
            "problem_id": problem_id,
            "prompt": prompt,
            "candidates": candidates,
        }
        _append_jsonl_record(out_path, row)
        existing_problem_ids.add(problem_id)
        kept += 1
        if kept % max(1, int(args.flush_every)) == 0:
            print(f"progress_written={kept} progress_skipped_no_positive={skipped} progress_skipped_existing={skipped_existing}", flush=True)

    config = {
        "model": str(args.model),
        "out_jsonl": str(out_path),
        "input_jsonl": str(args.input_jsonl),
        "hf_dataset": str(args.hf_dataset),
        "hf_subset": str(args.hf_subset),
        "hf_split": str(args.hf_split),
        "prompt_field": str(args.prompt_field),
        "answer_field": str(args.answer_field),
        "start_index": int(args.start_index),
        "limit": int(args.limit),
        "num_candidates": int(args.num_candidates),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "max_input_length": int(args.max_input_length),
        "max_new_tokens": int(args.max_new_tokens),
        "seed": int(args.seed),
        "flush_every": int(args.flush_every),
        "resume": bool(args.resume),
        "concise_prompt": bool(args.concise_prompt),
        "prompt_style": prompt_style,
        "candidate_strategy": candidate_strategy,
        "deduplicate_candidates": bool(args.deduplicate_candidates),
        "max_regen_rounds": int(args.max_regen_rounds),
        "positive_search_rounds": int(args.positive_search_rounds),
        "two_stage_near_miss_count": int(args.two_stage_near_miss_count),
        "two_stage_dependency_conflict_count": int(args.two_stage_dependency_conflict_count),
        "two_stage_state_reuse_count": int(args.two_stage_state_reuse_count),
        "two_stage_wrong_answer_from_state_count": int(args.two_stage_wrong_answer_from_state_count),
        "require_positive": bool(args.require_positive),
        "local_files_only": bool(args.local_files_only),
        "disable_implicit_hf_token": bool(args.disable_implicit_hf_token),
        "use_chat_template": bool(args.use_chat_template),
    }
    summary = {
        "model": str(args.model),
        "dataset": str(args.hf_dataset or args.input_jsonl or "unknown"),
        "seed": int(args.seed),
        "out_jsonl": str(out_path),
        "start_index": int(args.start_index),
        "limit": int(args.limit),
        "groups_total": int(kept),
        "groups_skipped_no_positive": int(skipped),
        "groups_skipped_existing": int(skipped_existing),
        "groups_with_positive": int(groups_with_positive),
        "groups_with_1p_2n": int(groups_with_1p_2n),
        "mean_candidates_per_group": _fraction(total_candidates, kept),
        "mean_positives_per_group": _fraction(total_positives, kept),
        "fraction_groups_with_positive": _fraction(groups_with_positive, kept),
        "fraction_groups_with_1p_2n": _fraction(groups_with_1p_2n, kept),
        "device": str(device),
    }

    if str(args.out_config_json).strip():
        config_path = Path(str(args.out_config_json))
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {config_path}")

    if str(args.out_summary_json).strip():
        summary_path = Path(str(args.out_summary_json))
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {summary_path}")

    print(f"generator_model={args.model}")
    print(f"device={device}")
    print(f"wrote {out_path}")
    print(f"n_examples={kept}")
    print(f"skipped_no_positive={skipped}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())