from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .frozen_oracle_client import AnchoredOracleClient
from .oracle_schema import AnchoredOracleQuery, CandidateTrace, ReasoningExample
from .translator import HeuristicAnchoredTranslator


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
    with Path(path).open("r", encoding="utf-8") as handle:
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


def hashed_text_embedding(text: str, *, dim: int) -> np.ndarray:
    vec = np.zeros((int(dim),), dtype=np.float64)
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    if not tokens:
        return vec
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], byteorder="little", signed=False) % int(dim)
        sign = 1.0 if (digest[4] % 2 == 0) else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 0.0:
        vec /= norm
    return vec


def materialize_dataset(
    examples: list[ReasoningExample],
    *,
    client: AnchoredOracleClient,
    translator: HeuristicAnchoredTranslator,
    text_dim: int = 256,
    feature_mode: str = "text+oracle",
    oracle_feature_groups: tuple[str, ...] | list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_mode = str(feature_mode).strip().lower()
    if feature_mode not in {"text", "oracle", "text+oracle"}:
        raise ValueError("feature_mode must be one of: text, oracle, text+oracle")
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
            text_vec = hashed_text_embedding(f"{example.prompt} {candidate.text}", dim=int(text_dim))
            if feature_mode == "text":
                row = text_vec
            elif feature_mode == "oracle":
                row = oracle_vec
            else:
                row = np.concatenate([text_vec, oracle_vec], axis=0)
            rows.append(np.asarray(row, dtype=np.float64))
            labels.append(float(candidate.label))
            groups.append(int(group_index))
    return (
        np.vstack(rows).astype(np.float64) if rows else np.zeros((0, int(text_dim)), dtype=np.float64),
        np.asarray(labels, dtype=np.float64),
        np.asarray(groups, dtype=np.int64),
    )