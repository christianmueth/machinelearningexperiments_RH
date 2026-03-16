from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .frozen_oracle_client import AnchoredOracleClient
from .oracle_schema import AnchoredOracleQuery, CandidateTrace, ReasoningExample
from .text_encoders import TextEncoder, build_text_encoder
from .translator import HeuristicAnchoredTranslator


def _local_path(path: str | Path) -> Path:
    resolved = Path(path)
    if sys.platform == "win32":
        raw = str(resolved)
        if not raw.startswith("\\?\\") and len(raw) >= 240:
            return Path("\\?\\" + raw)
    return resolved


def _query_from_dict(data: dict[str, object]) -> AnchoredOracleQuery:
    return AnchoredOracleQuery(
        u=float(data.get("u", 0.24)),
        feature_families=tuple(data.get("feature_families", ("closure", "spectral", "global", "stability"))),
        sigma_mode=str(data.get("sigma_mode", "anchored_default")),
        cluster_window=str(data.get("cluster_window", "canonical_t28")),
        include_perturbation_features=bool(data.get("include_perturbation_features", True)),
        pipeline_tag=str(data.get("pipeline_tag", "anchored_a3_v1")),
    )


def load_reasoning_examples(path: str | Path) -> list[ReasoningExample]:
    examples: list[ReasoningExample] = []
    with _local_path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            candidates: list[CandidateTrace] = []
            for cand in raw["candidates"]:
                query = _query_from_dict(cand["oracle_query"]) if "oracle_query" in cand else None
                oracle_features = tuple(float(x) for x in cand["oracle_features"]) if "oracle_features" in cand else None
                candidates.append(
                    CandidateTrace(
                        text=str(cand["text"]),
                        label=float(cand["label"]),
                        oracle_query=query,
                        oracle_features=oracle_features,
                    )
                )
            examples.append(
                ReasoningExample(
                    problem_id=str(raw["problem_id"]),
                    prompt=str(raw["prompt"]),
                    candidates=tuple(candidates),
                )
            )
    return examples


def materialize_dataset(
    examples: list[ReasoningExample],
    *,
    client: AnchoredOracleClient,
    translator: HeuristicAnchoredTranslator,
    text_dim: int = 256,
    feature_mode: str = "text+oracle",
    oracle_feature_groups: tuple[str, ...] | list[str] | None = None,
    text_encoder_name: str = "hashed",
    hf_model: str = "",
    hf_max_length: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_mode = str(feature_mode).strip().lower()
    if feature_mode not in {"text", "oracle", "text+oracle"}:
        raise ValueError("feature_mode must be one of: text, oracle, text+oracle")
    text_encoder: TextEncoder = build_text_encoder(
        text_encoder=str(text_encoder_name),
        text_dim=int(text_dim),
        hf_model=str(hf_model),
        max_length=int(hf_max_length),
    )
    rows: list[np.ndarray] = []
    labels: list[float] = []
    groups: list[int] = []
    for group_index, example in enumerate(examples):
        for candidate in example.candidates:
            query = candidate.oracle_query or translator.query_for_trace(candidate.text, prompt=example.prompt)
            if candidate.oracle_features is None:
                oracle_vec = client.oracle_vector(query, feature_groups=oracle_feature_groups)
            else:
                oracle_vec = np.asarray(candidate.oracle_features, dtype=np.float64)
            text_vec = text_encoder.encode(f"{example.prompt} {candidate.text}")
            if feature_mode == "text":
                row = text_vec
            elif feature_mode == "oracle":
                row = oracle_vec
            else:
                row = np.concatenate([text_vec, oracle_vec], axis=0)
            rows.append(np.asarray(row, dtype=np.float64))
            labels.append(float(candidate.label))
            groups.append(int(group_index))
    output_dim = int(text_encoder.output_dim)
    return (
        np.vstack(rows).astype(np.float64) if rows else np.zeros((0, output_dim), dtype=np.float64),
        np.asarray(labels, dtype=np.float64),
        np.asarray(groups, dtype=np.int64),
    )