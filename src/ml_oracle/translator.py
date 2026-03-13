from __future__ import annotations

import re

from .oracle_schema import AnchoredOracleQuery


class HeuristicAnchoredTranslator:
    def __init__(self, *, default_u: float = 0.24, zero_u: float = 0.16) -> None:
        self.default_u = float(default_u)
        self.zero_u = float(zero_u)

    def query_for_trace(self, text: str, *, prompt: str | None = None) -> AnchoredOracleQuery:
        combined = f"{prompt or ''} {text}".lower()
        tokens = set(re.findall(r"[a-z0-9_]+", combined))

        u_value = self.default_u
        if tokens & {"zero", "zeros", "root", "roots", "equation", "solve", "candidate"}:
            u_value = self.zero_u
        if tokens & {"consistency", "coherence", "coherent", "symmetry", "functional", "verify", "verifier"}:
            u_value = self.default_u

        families = ["closure", "spectral", "global"]
        if tokens & {"stable", "stability", "robust", "perturbation", "sensitivity"}:
            families.append("stability")
        if not families:
            families = ["closure", "spectral", "global", "stability"]

        return AnchoredOracleQuery(
            u=float(u_value),
            feature_families=tuple(dict.fromkeys(families)),
            sigma_mode="anchored_default",
            cluster_window="canonical_t28",
            include_perturbation_features=True,
            pipeline_tag="anchored_a3_v1",
        )