from __future__ import annotations

import math

import numpy as np


BASE_FEATURE_NAMES: tuple[str, ...] = (
    "query.u",
    "closure.beta2",
    "closure.beta3",
    "closure.c",
    "coeff.A1_star",
    "coeff.A2_star",
    "coeff.A3_star",
    "spectral.err1",
    "spectral.err2",
    "spectral.err3",
    "spectral.coeff_err_pass",
    "spectral.radius",
    "global.rel_l2_logdet",
    "global.completion_rich_rel_l2_logdet",
    "global.rel_l2_logabs",
    "global.completion_rich_rel_l2_logabs",
    "global.min_abs_sigma_zero",
    "global.t_at_min_abs_sigma_zero",
    "global.best_zero_abs",
    "global.best_zero_t",
    "global.n_zero_candidates",
    "stability.best_u_by_fe_completion",
    "stability.best_u_by_zero_candidate",
    "stability.spread_completion_rich_rel_l2_logdet",
    "stability.spread_min_abs_det_sigma_zero",
    "stability.n_coeff_err_pass",
    "analytic.fe_zero_abs",
    "analytic.fe_median_rel_det_defect_all",
    "analytic.fe_median_rel_det_defect_zero_critical",
    "analytic.sigma_at_min_abs",
    "analytic.critical_line_preference_gap",
    "tracking.abs_det_candidate",
    "tracking.abs_det_transverse_min",
    "tracking.sigma_at_transverse_min",
    "tracking.critical_line_preferred",
    "rigidity.base_err2",
    "rigidity.base_radius",
    "rigidity.phase_twist_penalty",
    "rigidity.radius_rms_shift",
    "analytic_stability.base_median_rel_det_defect_all",
    "analytic_stability.wide_minus_base",
    "analytic_stability.primecut_minus_base",
)

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "closure": (
        "query.u",
        "closure.beta2",
        "closure.beta3",
        "closure.c",
    ),
    "packet": (
        "coeff.A1_star",
        "coeff.A2_star",
        "coeff.A3_star",
        "spectral.err1",
        "spectral.err2",
        "spectral.err3",
        "spectral.coeff_err_pass",
        "spectral.radius",
        "rigidity.base_err2",
        "rigidity.base_radius",
        "rigidity.phase_twist_penalty",
        "rigidity.radius_rms_shift",
    ),
    "zero": (
        "global.min_abs_sigma_zero",
        "global.t_at_min_abs_sigma_zero",
        "global.best_zero_abs",
        "global.best_zero_t",
        "global.n_zero_candidates",
        "tracking.abs_det_candidate",
        "tracking.abs_det_transverse_min",
        "tracking.sigma_at_transverse_min",
        "tracking.critical_line_preferred",
    ),
    "fe_stability": (
        "global.rel_l2_logdet",
        "global.completion_rich_rel_l2_logdet",
        "global.rel_l2_logabs",
        "global.completion_rich_rel_l2_logabs",
        "stability.best_u_by_fe_completion",
        "stability.best_u_by_zero_candidate",
        "stability.spread_completion_rich_rel_l2_logdet",
        "stability.spread_min_abs_det_sigma_zero",
        "stability.n_coeff_err_pass",
        "analytic.fe_zero_abs",
        "analytic.fe_median_rel_det_defect_all",
        "analytic.fe_median_rel_det_defect_zero_critical",
        "analytic.sigma_at_min_abs",
        "analytic.critical_line_preference_gap",
        "analytic_stability.base_median_rel_det_defect_all",
        "analytic_stability.wide_minus_base",
        "analytic_stability.primecut_minus_base",
    ),
}


class AnchoredFeatureRegistry:
    def __init__(self) -> None:
        self.base_feature_names = BASE_FEATURE_NAMES
        self.feature_groups = FEATURE_GROUPS
        self.feature_names = self.base_feature_names + tuple(f"{name}__missing" for name in self.base_feature_names)

    def empty_feature_map(self) -> dict[str, float]:
        return {name: math.nan for name in self.base_feature_names}

    def vectorize(self, feature_map: dict[str, float]) -> np.ndarray:
        values = np.asarray([float(feature_map.get(name, math.nan)) for name in self.base_feature_names], dtype=float)
        missing = ~np.isfinite(values)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return np.concatenate([values, missing.astype(float)], axis=0).astype(np.float64)

    def feature_group_names(self, groups: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
        if not groups:
            return self.base_feature_names
        selected: list[str] = []
        for group in groups:
            key = str(group).strip()
            if key not in self.feature_groups:
                raise KeyError(f"unknown anchored feature group: {key}")
            for name in self.feature_groups[key]:
                if name not in selected:
                    selected.append(name)
        return tuple(selected)

    def select_vector(self, feature_map: dict[str, float], groups: tuple[str, ...] | list[str] | None) -> np.ndarray:
        names = self.feature_group_names(groups)
        values = np.asarray([float(feature_map.get(name, math.nan)) for name in names], dtype=float)
        missing = ~np.isfinite(values)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return np.concatenate([values, missing.astype(float)], axis=0).astype(np.float64)