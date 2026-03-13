from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class AnchoredOracleQuery:
    u: float = 0.24
    feature_families: tuple[str, ...] = ("closure", "spectral", "global", "stability")
    sigma_mode: str = "anchored_default"
    cluster_window: str = "canonical_t28"
    include_perturbation_features: bool = True
    pipeline_tag: str = "anchored_a3_v1"


@dataclass(frozen=True)
class AnchoredOracleResponse:
    query: AnchoredOracleQuery
    feature_names: tuple[str, ...]
    feature_vector: np.ndarray
    feature_map: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateTrace:
    text: str
    label: float
    oracle_query: AnchoredOracleQuery | None = None
    oracle_features: tuple[float, ...] | None = None


@dataclass(frozen=True)
class ReasoningExample:
    problem_id: str
    prompt: str
    candidates: tuple[CandidateTrace, ...]