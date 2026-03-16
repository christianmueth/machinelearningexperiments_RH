from __future__ import annotations

import math
import re
import statistics

from .oracle_schema import AnchoredOracleQuery, ReasoningState


class HeuristicAnchoredTranslator:
    def __init__(self, *, default_u: float = 0.24, zero_u: float = 0.16, use_step_aware_queries: bool = False) -> None:
        self.default_u = float(default_u)
        self.zero_u = float(zero_u)
        self.use_step_aware_queries = bool(use_step_aware_queries)
        self.available_us: tuple[float, ...] = (0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24)
        self._equation_pattern = re.compile(
            r"(-?\d[\d,]*(?:\.\d+)?)\s*([+\-*/x])\s*(-?\d[\d,]*(?:\.\d+)?)\s*=\s*(-?\d[\d,]*(?:\.\d+)?)",
            flags=re.IGNORECASE,
        )

    @staticmethod
    def _extract_numeric_tokens(text: str) -> list[str]:
        return re.findall(r"-?\d[\d,]*(?:\.\d+)?", str(text))

    @staticmethod
    def _normalize_numeric(token: str) -> float | None:
        try:
            return float(str(token).replace(",", ""))
        except ValueError:
            return None

    def _snap_u(self, value: float) -> float:
        return min(self.available_us, key=lambda candidate: abs(float(candidate) - float(value)))

    @staticmethod
    def _round_structural_value(value: float | None) -> float | None:
        if value is None:
            return None
        rounded = round(float(value), 6)
        if math.isclose(float(rounded), round(float(rounded)), rel_tol=1e-9, abs_tol=1e-9):
            return float(int(round(float(rounded))))
        return float(rounded)

    @staticmethod
    def _unique_values(values: tuple[float, ...] | list[float]) -> tuple[float, ...]:
        ordered: list[float] = []
        for value in values:
            rounded = HeuristicAnchoredTranslator._round_structural_value(float(value))
            if rounded is None:
                continue
            if not any(math.isclose(float(rounded), float(existing), rel_tol=1e-9, abs_tol=1e-9) for existing in ordered):
                ordered.append(float(rounded))
        return tuple(ordered)

    def _reasoning_text(self, text: str) -> str:
        raw = str(text)
        if "Reasoning:" in raw:
            return raw.split("Reasoning:", 1)[1].strip()
        return raw.strip()

    def _answer_value(self, text: str) -> float | None:
        answer_match = re.search(r"(?:answer|final answer)\s*:\s*([^\n]+)", str(text), flags=re.IGNORECASE)
        if answer_match is None:
            return None
        answer_numbers = self._extract_numeric_tokens(answer_match.group(1))
        if not answer_numbers:
            return None
        return self._normalize_numeric(answer_numbers[-1])

    def _equation_stats(self, text: str) -> tuple[int, int]:
        correct = 0
        incorrect = 0
        for left_raw, operator, right_raw, result_raw in self._equation_pattern.findall(str(text)):
            left = self._normalize_numeric(left_raw)
            right = self._normalize_numeric(right_raw)
            result = self._normalize_numeric(result_raw)
            if left is None or right is None or result is None:
                continue
            if operator.lower() == "x":
                predicted = left * right
            elif operator == "+":
                predicted = left + right
            elif operator == "-":
                predicted = left - right
            elif operator == "*":
                predicted = left * right
            else:
                if abs(right) <= 1e-12:
                    continue
                predicted = left / right
            if math.isclose(float(predicted), float(result), rel_tol=1e-6, abs_tol=1e-6):
                correct += 1
            else:
                incorrect += 1
        return correct, incorrect

    def _answer_support_score(self, text: str) -> float:
        numbers = self._extract_numeric_tokens(text)
        if not numbers:
            return -0.02
        answer_value = self._answer_value(text)
        if answer_value is None:
            return -0.01
        reasoning = self._reasoning_text(text)
        reasoning_numbers = [self._normalize_numeric(token) for token in self._extract_numeric_tokens(reasoning)]
        reasoning_numbers = [value for value in reasoning_numbers if value is not None]
        if any(math.isclose(float(answer_value), float(value), rel_tol=1e-6, abs_tol=1e-6) for value in reasoning_numbers):
            return 0.02
        return -0.01

    def split_reasoning_steps(self, text: str) -> tuple[str, ...]:
        reasoning = self._reasoning_text(text)
        if not reasoning:
            return ()
        normalized = reasoning.replace(";", "\n")
        normalized = re.sub(r"\bthen\b", "\nthen", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\btherefore\b", "\ntherefore", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bso\b", "\nso", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bhence\b", "\nhence", normalized, flags=re.IGNORECASE)
        parts: list[str] = []
        for line in normalized.splitlines():
            line = line.strip()
            if not line:
                continue
            parts.extend(part.strip() for part in re.split(r"(?<=[.!?])\s+", line) if part.strip())
        filtered = [part for part in parts if self._extract_numeric_tokens(part) or re.search(r"[=+\-*/x]", part)]
        return tuple(filtered)

    def _infer_operation(self, step_text: str) -> str:
        lowered = str(step_text).lower()
        match = self._equation_pattern.search(step_text)
        if match is not None:
            operator = match.group(2).lower()
            return {
                "+": "add",
                "-": "subtract",
                "*": "multiply",
                "x": "multiply",
                "/": "divide",
            }.get(operator, "equation")
        if any(token in lowered for token in ("twice", "double", "triple", "times", "product", "multiply")):
            return "multiply"
        if any(token in lowered for token in ("divide", "ratio", "quotient", "per", "each")):
            return "divide"
        if any(token in lowered for token in ("left", "remain", "remaining", "difference", "minus", "subtract")):
            return "subtract"
        if any(token in lowered for token in ("total", "sum", "add", "plus", "together", "combined")):
            return "add"
        return "observe"

    def reasoning_states_for_trace(self, text: str, *, prompt: str | None = None) -> tuple[ReasoningState, ...]:
        answer_value = self._answer_value(text)
        states: list[ReasoningState] = []
        seen_values: list[float] = []
        prior_results: list[tuple[int, float]] = []
        for step_index, step_text in enumerate(self.split_reasoning_steps(text)):
            quantity_tokens = self._extract_numeric_tokens(step_text)
            quantities = tuple(value for value in (self._normalize_numeric(token) for token in quantity_tokens) if value is not None)
            match = self._equation_pattern.search(step_text)
            equation_text = match.group(0) if match is not None else ""
            equation_correct: bool | None = None
            operands: tuple[float, ...] = ()
            result: float | None = None
            if match is not None:
                left = self._normalize_numeric(match.group(1))
                right = self._normalize_numeric(match.group(3))
                result = self._normalize_numeric(match.group(4))
                operands = tuple(value for value in (left, right) if value is not None)
                eq_correct, eq_incorrect = self._equation_stats(equation_text)
                if eq_correct > 0:
                    equation_correct = True
                elif eq_incorrect > 0:
                    equation_correct = False
            elif len(quantities) >= 2:
                operands = tuple(quantities[:2])
                result = quantities[-1]

            answer_support = bool(
                answer_value is not None
                and any(math.isclose(float(answer_value), float(value), rel_tol=1e-6, abs_tol=1e-6) for value in quantities)
            )
            dependency_step_indexes: list[int] = []
            dependency_values: list[float] = []
            introduced_values: list[float] = []
            for value in quantities:
                if any(math.isclose(float(value), float(previous), rel_tol=1e-6, abs_tol=1e-6) for previous in seen_values):
                    dependency_values.append(float(value))
                else:
                    introduced_values.append(float(value))
            for previous_index, previous_result in prior_results:
                if any(math.isclose(float(previous_result), float(value), rel_tol=1e-6, abs_tol=1e-6) for value in quantities):
                    dependency_step_indexes.append(int(previous_index))
                    dependency_values.append(float(previous_result))
            carried_result = bool(
                result is not None
                and any(math.isclose(float(result), float(value), rel_tol=1e-6, abs_tol=1e-6) for value in dependency_values)
            )
            states.append(
                ReasoningState(
                    step_index=int(step_index),
                    source_text=str(step_text),
                    quantities=tuple(float(value) for value in quantities),
                    operation=self._infer_operation(step_text),
                    operands=tuple(float(value) for value in operands),
                    result=None if result is None else float(result),
                    equation_text=str(equation_text),
                    equation_correct=equation_correct,
                    answer_support=bool(answer_support),
                    contains_approx_language=bool(any(token in step_text.lower() for token in ("approx", "approximately", "maybe", "guess", "unknown", "cannot"))),
                    dependency_step_indexes=tuple(sorted(set(int(index) for index in dependency_step_indexes))),
                    dependency_values=self._unique_values(dependency_values),
                    introduced_values=self._unique_values(introduced_values),
                    carried_result=bool(carried_result),
                )
            )
            for value in quantities:
                if not any(math.isclose(float(value), float(previous), rel_tol=1e-6, abs_tol=1e-6) for previous in seen_values):
                    seen_values.append(float(value))
            if result is not None:
                prior_results.append((int(step_index), float(result)))
        if not states and prompt is not None:
            prompt_numbers = tuple(value for value in (self._normalize_numeric(token) for token in self._extract_numeric_tokens(prompt)) if value is not None)
            if prompt_numbers:
                states.append(
                    ReasoningState(
                        step_index=0,
                        source_text=str(prompt),
                        quantities=tuple(float(value) for value in prompt_numbers),
                        operation="observe",
                        introduced_values=self._unique_values(prompt_numbers),
                    )
                )
        return tuple(states)

    def reasoning_state_graph_for_trace(self, text: str, *, prompt: str | None = None) -> tuple[dict[str, object], ...]:
        graph_rows: list[dict[str, object]] = []
        for state in self.reasoning_states_for_trace(text, prompt=prompt):
            graph_rows.append(
                {
                    "step_index": int(state.step_index),
                    "operation": str(state.operation),
                    "operands": [self._round_structural_value(value) for value in state.operands],
                    "result": self._round_structural_value(state.result),
                    "quantities": [self._round_structural_value(value) for value in state.quantities],
                    "dependency_step_indexes": [int(index) for index in state.dependency_step_indexes],
                    "dependency_values": [self._round_structural_value(value) for value in state.dependency_values],
                    "introduced_values": [self._round_structural_value(value) for value in state.introduced_values],
                    "answer_support": bool(state.answer_support),
                    "equation_correct": state.equation_correct,
                    "carried_result": bool(state.carried_result),
                }
            )
        return tuple(graph_rows)

    def structural_state_signature_for_trace(self, text: str, *, prompt: str | None = None) -> tuple[tuple[object, ...], ...]:
        signature_rows: list[tuple[object, ...]] = []
        for state in self.reasoning_states_for_trace(text, prompt=prompt):
            signature_rows.append(
                (
                    str(state.operation),
                    tuple(self._round_structural_value(value) for value in state.operands),
                    self._round_structural_value(state.result),
                    tuple(int(index) for index in state.dependency_step_indexes),
                    tuple(self._round_structural_value(value) for value in state.dependency_values),
                    tuple(self._round_structural_value(value) for value in state.introduced_values),
                    bool(state.answer_support),
                    state.equation_correct,
                    bool(state.carried_result),
                )
            )
        return tuple(signature_rows)

    def _query_for_reasoning_state(self, state: ReasoningState, *, prompt: str | None = None, is_final: bool = False) -> AnchoredOracleQuery:
        combined = f"{prompt or ''} {state.source_text}".lower()
        tokens = set(re.findall(r"[a-z0-9_]+", combined))
        u_value = 0.20
        if state.operation in {"add", "subtract"}:
            u_value += 0.01
        elif state.operation in {"multiply", "divide"}:
            u_value += 0.02
        if state.equation_correct is True:
            u_value += 0.02
        elif state.equation_correct is False:
            u_value -= 0.03
        if state.answer_support:
            u_value += 0.02 if is_final else 0.01
        if state.contains_approx_language:
            u_value -= 0.02
        if len(state.quantities) >= 4:
            u_value += 0.01
        if any(float(value).is_integer() is False for value in state.quantities):
            u_value -= 0.01
        if any(token in tokens for token in {"therefore", "so", "hence", "total", "remaining", "left", "cost", "earned"}):
            u_value += 0.01
        snapped = self._snap_u(min(0.24, max(0.16, u_value)))
        include_perturbation_features = bool(snapped >= 0.22 and state.equation_correct is not False and (state.answer_support or len(state.quantities) >= 3))
        return AnchoredOracleQuery(
            u=float(snapped),
            feature_families=("closure", "spectral", "global"),
            sigma_mode="anchored_default",
            cluster_window="canonical_t28",
            include_perturbation_features=bool(include_perturbation_features),
            pipeline_tag="anchored_a3_v1",
        )

    def queries_for_trace(self, text: str, *, prompt: str | None = None) -> tuple[AnchoredOracleQuery, ...]:
        combined = f"{prompt or ''} {text}".lower()
        tokens = set(re.findall(r"[a-z0-9_]+", combined))
        if tokens & {"zero", "zeros", "root", "roots", "equation", "solve", "candidate"}:
            return (
                AnchoredOracleQuery(
                    u=float(self.zero_u),
                    feature_families=("closure", "spectral", "global"),
                    sigma_mode="anchored_default",
                    cluster_window="canonical_t28",
                    include_perturbation_features=False,
                    pipeline_tag="anchored_a3_v1",
                ),
            )

        states = self.reasoning_states_for_trace(text, prompt=prompt)
        if not states:
            u_value, include_perturbation_features = self._u_for_arithmetic_trace(text, prompt=prompt)
            return (
                AnchoredOracleQuery(
                    u=float(u_value),
                    feature_families=("closure", "spectral", "global"),
                    sigma_mode="anchored_default",
                    cluster_window="canonical_t28",
                    include_perturbation_features=bool(include_perturbation_features),
                    pipeline_tag="anchored_a3_v1",
                ),
            )

        queries: list[AnchoredOracleQuery] = []
        seen: set[tuple[float, bool]] = set()
        for index, state in enumerate(states):
            query = self._query_for_reasoning_state(state, prompt=prompt, is_final=index == len(states) - 1)
            key = (float(query.u), bool(query.include_perturbation_features))
            if key not in seen:
                seen.add(key)
                queries.append(query)
        return tuple(queries)

    def _u_for_arithmetic_trace(self, text: str, *, prompt: str | None) -> tuple[float, bool]:
        combined = f"{prompt or ''} {text}".lower()
        tokens = set(re.findall(r"[a-z0-9_]+", combined))
        correct_equations, incorrect_equations = self._equation_stats(text)
        numbers = self._extract_numeric_tokens(text)
        answer_support = self._answer_support_score(text)
        u_value = 0.21
        u_value += 0.01 * min(correct_equations, 2)
        u_value -= 0.02 * min(incorrect_equations, 2)
        u_value += answer_support
        if any(token in tokens for token in {"approx", "approximately", "maybe", "guess", "unknown", "cannot"}):
            u_value -= 0.02
        if any(token in tokens for token in {"therefore", "so", "hence", "total", "remaining", "left", "cost", "earned"}):
            u_value += 0.01
        if any("." in token for token in numbers):
            u_value -= 0.01
        if len(numbers) >= 6:
            u_value += 0.01
        snapped = self._snap_u(min(0.24, max(0.16, u_value)))
        include_perturbation_features = bool(snapped >= 0.22 and incorrect_equations == 0)
        return snapped, include_perturbation_features

    def _legacy_query_for_trace(self, text: str, *, prompt: str | None = None) -> AnchoredOracleQuery:
        combined = f"{prompt or ''} {text}".lower()
        tokens = set(re.findall(r"[a-z0-9_]+", combined))
        u_value = self.default_u
        if tokens & {"zero", "zeros", "root", "roots", "equation", "solve", "candidate"}:
            u_value = self.zero_u
        if tokens & {"consistency", "coherence", "coherent", "symmetry", "functional", "verify", "verifier"}:
            u_value = self.default_u
        if not (tokens & {"zero", "zeros", "root", "roots", "equation", "solve", "candidate"}):
            u_value, include_perturbation_features = self._u_for_arithmetic_trace(text, prompt=prompt)
        else:
            include_perturbation_features = False
        return AnchoredOracleQuery(
            u=float(u_value),
            feature_families=("closure", "spectral", "global"),
            sigma_mode="anchored_default",
            cluster_window="canonical_t28",
            include_perturbation_features=bool(include_perturbation_features),
            pipeline_tag="anchored_a3_v1",
        )

    def query_for_trace(self, text: str, *, prompt: str | None = None) -> AnchoredOracleQuery:
        if not self.use_step_aware_queries:
            return self._legacy_query_for_trace(text, prompt=prompt)
        queries = self.queries_for_trace(text, prompt=prompt)
        if not queries:
            return AnchoredOracleQuery(
                u=float(self.default_u),
                feature_families=("closure", "spectral", "global"),
                sigma_mode="anchored_default",
                cluster_window="canonical_t28",
                include_perturbation_features=True,
                pipeline_tag="anchored_a3_v1",
            )
        states = self.reasoning_states_for_trace(text, prompt=prompt)
        mean_u = statistics.fmean(float(query.u) for query in queries)
        if states and states[-1].answer_support:
            mean_u += 0.01
        if any(state.equation_correct is False for state in states):
            mean_u -= 0.01
        if len(queries) >= 3:
            mean_u += 0.01
        include_perturbation_features = bool(any(query.include_perturbation_features for query in queries))
        families = tuple(dict.fromkeys(name for query in queries for name in query.feature_families)) or ("closure", "spectral", "global")
        return AnchoredOracleQuery(
            u=float(self._snap_u(min(0.24, max(0.16, mean_u)))),
            feature_families=families,
            sigma_mode="anchored_default",
            cluster_window="canonical_t28",
            include_perturbation_features=bool(include_perturbation_features),
            pipeline_tag="anchored_a3_v1",
        )