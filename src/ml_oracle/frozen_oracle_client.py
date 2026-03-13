from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .feature_registry import AnchoredFeatureRegistry
from .oracle_schema import AnchoredOracleQuery, AnchoredOracleResponse


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "out"

ANCHORED_ARTIFACTS = {
    "completed_summary": OUT_DIR / "completed_global_object_anchored_default_summary.csv",
    "completed_stability": OUT_DIR / "completed_global_object_anchored_default_stability.csv",
    "fe_summary": OUT_DIR / "analytic_characterization_a3default_quartic_a4m02_anchored_fe_summary.csv",
    "sigma_scan_summary": OUT_DIR / "analytic_characterization_a3default_quartic_a4m02_anchored_sigma_scan_summary.csv",
    "rigidity": OUT_DIR / "analytic_characterization_a3default_quartic_a4m02_anchored_rigidity.csv",
    "analytic_stability": OUT_DIR / "analytic_characterization_a3default_quartic_a4m02_anchored_stability.csv",
    "zero_tracking": OUT_DIR / "analytic_characterization_a3default_quartic_a4m02_anchored_zero_tracking.csv",
}


class AnchoredOracleClient:
    def __init__(self, repo_root: Path | None = None) -> None:
        self.repo_root = Path(repo_root) if repo_root is not None else REPO_ROOT
        self.registry = AnchoredFeatureRegistry()
        self._load_artifacts()

    def available_us(self) -> tuple[float, ...]:
        return tuple(sorted(self.completed_summary["u"].astype(float).tolist()))

    def query(self, query: AnchoredOracleQuery) -> AnchoredOracleResponse:
        feature_map = self.registry.empty_feature_map()
        feature_map["query.u"] = float(query.u)

        summary_row = self._row_for_u(self.completed_summary, float(query.u))
        self._fill_completed_summary(feature_map, summary_row)
        self._fill_completed_stability(feature_map)

        track_row = self._row_for_u(self.zero_tracking, float(query.u), required=False)
        if track_row is not None:
            self._fill_tracking(feature_map, track_row)

        if math.isclose(float(query.u), 0.24, rel_tol=0.0, abs_tol=1e-12):
            self._fill_analytic_summary(feature_map)
            self._fill_sigma_scan(feature_map)
            if bool(query.include_perturbation_features):
                self._fill_rigidity(feature_map)
                self._fill_analytic_stability(feature_map)

        vector = self.registry.vectorize(feature_map)
        return AnchoredOracleResponse(
            query=query,
            feature_names=self.registry.feature_names,
            feature_vector=vector,
            feature_map=feature_map,
        )

    def export_feature_table(self, *, us: list[float] | None = None) -> pd.DataFrame:
        values = list(us) if us is not None else list(self.available_us())
        rows: list[dict[str, float | str]] = []
        for u in values:
            response = self.query(AnchoredOracleQuery(u=float(u)))
            row: dict[str, float | str] = {"pipeline_tag": response.query.pipeline_tag, "cluster_window": response.query.cluster_window}
            row.update({name: float(value) for name, value in response.feature_map.items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def oracle_vector(self, query: AnchoredOracleQuery, *, feature_groups: tuple[str, ...] | list[str] | None = None) -> np.ndarray:
        response = self.query(query)
        return self.registry.select_vector(response.feature_map, feature_groups)

    def _artifact_path(self, key: str) -> Path:
        path = ANCHORED_ARTIFACTS[key]
        if not path.exists():
            raise FileNotFoundError(f"missing anchored artifact: {path}")
        return path

    def _load_artifacts(self) -> None:
        self.completed_summary = pd.read_csv(self._artifact_path("completed_summary"))
        self.completed_stability = pd.read_csv(self._artifact_path("completed_stability"))
        self.fe_summary = pd.read_csv(self._artifact_path("fe_summary"))
        self.sigma_scan_summary = pd.read_csv(self._artifact_path("sigma_scan_summary"))
        self.rigidity = pd.read_csv(self._artifact_path("rigidity"))
        self.analytic_stability = pd.read_csv(self._artifact_path("analytic_stability"))
        self.zero_tracking = pd.read_csv(self._artifact_path("zero_tracking"))

    @staticmethod
    def _row_for_u(df: pd.DataFrame, u: float, *, required: bool = True) -> pd.Series | None:
        mask = np.isclose(df["u"].astype(float), float(u), atol=1e-12)
        sub = df.loc[mask].copy()
        if sub.empty:
            if required:
                raise KeyError(f"no anchored row for u={u}")
            return None
        return sub.iloc[0]

    def _fill_completed_summary(self, feature_map: dict[str, float], row: pd.Series) -> None:
        feature_map["closure.beta2"] = float(row["beta2"])
        feature_map["closure.beta3"] = float(row["beta3"])
        feature_map["closure.c"] = float(row["c"])
        feature_map["coeff.A1_star"] = float(row["A1_star"])
        feature_map["coeff.A2_star"] = float(row["A2_star"])
        feature_map["coeff.A3_star"] = float(row["A3_star"])
        feature_map["spectral.err1"] = float(row["err1"])
        feature_map["spectral.err2"] = float(row["err2"])
        feature_map["spectral.err3"] = float(row["err3"])
        feature_map["spectral.coeff_err_pass"] = float(bool(row["coeff_err_pass"]))
        feature_map["spectral.radius"] = float(row["spectral_radius"])
        feature_map["global.rel_l2_logdet"] = float(row["rel_l2_logdet_conj1ms"])
        feature_map["global.completion_rich_rel_l2_logdet"] = float(row["completion_rich_rel_l2_logdet_conj1ms"])
        feature_map["global.rel_l2_logabs"] = float(row["rel_l2_logabs_det_conj1ms"])
        feature_map["global.completion_rich_rel_l2_logabs"] = float(row["completion_rich_rel_l2_logabs_det_conj1ms"])
        feature_map["global.min_abs_sigma_zero"] = float(row["min_abs_det_sigma_zero"])
        feature_map["global.t_at_min_abs_sigma_zero"] = float(row["t_at_min_abs_det_sigma_zero"])
        feature_map["global.best_zero_abs"] = float(row["best_zero_candidate_abs_det"])
        feature_map["global.best_zero_t"] = float(row["best_zero_candidate_t"])
        feature_map["global.n_zero_candidates"] = float(row["n_zero_candidates"])

    def _fill_completed_stability(self, feature_map: dict[str, float]) -> None:
        row = self.completed_stability.iloc[0]
        feature_map["stability.best_u_by_fe_completion"] = float(row["best_u_by_fe_completion"])
        feature_map["stability.best_u_by_zero_candidate"] = float(row["best_u_by_zero_candidate"])
        feature_map["stability.spread_completion_rich_rel_l2_logdet"] = float(row["spread_completion_rich_rel_l2_logdet_conj1ms"])
        feature_map["stability.spread_min_abs_det_sigma_zero"] = float(row["spread_min_abs_det_sigma_zero"])
        feature_map["stability.n_coeff_err_pass"] = float(row["n_coeff_err_pass"])

    def _fill_analytic_summary(self, feature_map: dict[str, float]) -> None:
        row = self.fe_summary.iloc[0]
        feature_map["analytic.fe_zero_abs"] = float(row["zero_candidate_abs_det"])
        feature_map["analytic.fe_median_rel_det_defect_all"] = float(row["median_rel_det_defect_all"])
        feature_map["analytic.fe_median_rel_det_defect_zero_critical"] = float(row["median_rel_det_defect_zero_on_critical"])

    def _fill_sigma_scan(self, feature_map: dict[str, float]) -> None:
        row = self.sigma_scan_summary.iloc[0]
        feature_map["analytic.sigma_at_min_abs"] = float(row["sigma_at_min_abs_det"])
        feature_map["analytic.critical_line_preference_gap"] = float(row["critical_line_preference_gap"])

    def _fill_tracking(self, feature_map: dict[str, float], row: pd.Series) -> None:
        feature_map["tracking.abs_det_candidate"] = float(row["abs_det_candidate"])
        feature_map["tracking.abs_det_transverse_min"] = float(row["abs_det_transverse_min"])
        feature_map["tracking.sigma_at_transverse_min"] = float(row["sigma_at_transverse_min"])
        feature_map["tracking.critical_line_preferred"] = float(bool(row["critical_line_preferred"]))

    def _fill_rigidity(self, feature_map: dict[str, float]) -> None:
        df = self.rigidity.set_index("label")
        base = df.loc["base"]
        pos = df.loc["phase_twist_pos"]
        neg = df.loc["phase_twist_neg"]
        up = df.loc["r_all_up"]
        down = df.loc["r_all_down"]
        feature_map["rigidity.base_err2"] = float(base["err2"])
        feature_map["rigidity.base_radius"] = float(base["spectral_radius"])
        feature_map["rigidity.phase_twist_penalty"] = float(
            0.5 * (float(pos["median_rel_det_defect_zero_on_critical"]) + float(neg["median_rel_det_defect_zero_on_critical"]))
            - float(base["median_rel_det_defect_zero_on_critical"])
        )
        feature_map["rigidity.radius_rms_shift"] = float(
            0.5 * (float(up["det_logabs_rms_diff_sigma_half"]) + float(down["det_logabs_rms_diff_sigma_half"]))
        )

    def _fill_analytic_stability(self, feature_map: dict[str, float]) -> None:
        df = self.analytic_stability.set_index("label")
        base = df.loc["base"]
        wide = df.loc["wide"]
        primecut = df.loc["primecut_alt"]
        feature_map["analytic_stability.base_median_rel_det_defect_all"] = float(base["median_rel_det_defect_all"])
        feature_map["analytic_stability.wide_minus_base"] = float(wide["median_rel_det_defect_all"] - base["median_rel_det_defect_all"])
        feature_map["analytic_stability.primecut_minus_base"] = float(primecut["median_rel_det_defect_all"] - base["median_rel_det_defect_all"])