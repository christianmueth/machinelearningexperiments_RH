from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


# ---------- Core thresholds ----------


SELECTOR_VERSION = "selector_v1"


@dataclass(frozen=True)
class GateThresholds:
    gate_tin_inv_max: float = 500.0
    gate_sep2_min: float = 0.01
    gate_lam_cost_max: float = 0.8


@dataclass(frozen=True)
class AcceptThresholds:
    tau_logabs: float
    tau_arg: float


@dataclass(frozen=True)
class ThresholdBundle:
    gate: GateThresholds
    accept: AcceptThresholds


STRICT = AcceptThresholds(tau_logabs=0.2, tau_arg=0.5)
DIAG = AcceptThresholds(tau_logabs=0.2, tau_arg=0.7)
FRONTIER_Q2TR = AcceptThresholds(tau_logabs=0.05, tau_arg=0.6)
FRONTIER_Q2 = AcceptThresholds(tau_logabs=0.1, tau_arg=1.0)
FRONTIER_Q2PHI2 = AcceptThresholds(tau_logabs=0.1, tau_arg=1.0)


# ---------- Helpers ----------


WANT_COLS = [
    "side",
    "idx",
    "sigma",
    "t",
    "Tin_inv_norm1_est_N2",
    "sep2_N2",
    "lam_match_cost",
    "dlog_abs",
    "darg",
]


def _coerce_num(s: pd.Series | None) -> pd.Series:
    if s is None:
        return pd.Series([], dtype=float)
    return pd.to_numeric(s, errors="coerce")


def _min_finite(s: pd.Series | None) -> float:
    if s is None:
        return float("nan")
    a = pd.to_numeric(s, errors="coerce")
    a = a[np.isfinite(a.to_numpy(dtype=float, na_value=np.nan))]
    if a.empty:
        return float("nan")
    return float(a.min())


def _estimate_dt(tvals: Iterable[float]) -> float:
    s = sorted(set(float(t) for t in tvals if math.isfinite(float(t))))
    if len(s) < 2:
        return float("nan")
    diffs = [b - a for a, b in zip(s, s[1:]) if b - a > 0]
    if not diffs:
        return float("nan")
    return float(min(diffs))


def _segments_width_max(t_sorted: np.ndarray, *, dt: float) -> tuple[int, float]:
    if t_sorted.size == 0:
        return (0, 0.0)
    if t_sorted.size == 1:
        return (1, 0.0)

    if not math.isfinite(dt) or dt <= 0:
        dt = float(np.nanmin(np.diff(t_sorted))) if t_sorted.size >= 2 else float("nan")
        if not math.isfinite(dt) or dt <= 0:
            return (1, 0.0)

    gaps = np.diff(t_sorted)
    breaks = np.where(gaps > (1.5 * dt))[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [t_sorted.size - 1]))
    widths = t_sorted[ends] - t_sorted[starts]
    return (int(len(starts)), float(np.nanmax(widths) if widths.size else 0.0))


def _infer_obs_from_name(p: Path) -> str:
    raw = p.name.lower()
    tok = "_" + re.sub(r"[^a-z0-9]+", "_", raw) + "_"
    if "_q2tr_" in tok or "_q2_tr_" in tok:
        return "Q2_tr"
    if "_q2phi2_" in tok or "_q2_phi2_" in tok:
        return "Q2_phi2"
    if "_q2_" in tok:
        return "Q2"
    return p.stem


def _group_key(p: Path) -> str:
    # Replace observable token by {OBS} to group triads.
    name = p.name
    name = re.sub(r"_Q2tr_", "_{OBS}_", name, flags=re.IGNORECASE)
    name = re.sub(r"_Q2phi2_", "_{OBS}_", name, flags=re.IGNORECASE)
    name = re.sub(r"_Q2_", "_{OBS}_", name, flags=re.IGNORECASE)
    return name


def _load_aug_df(path: Path, *, side: str) -> pd.DataFrame:
    usecols = [c for c in WANT_COLS if c]
    df = pd.read_csv(path, usecols=lambda c: (c in usecols))
    if side and ("side" in df.columns):
        df = df[df["side"].astype(str) == side].copy()
    return df


def _tier01_mask(df: pd.DataFrame, gate: GateThresholds) -> pd.Series:
    ok = pd.Series(True, index=df.index)

    if "Tin_inv_norm1_est_N2" in df.columns and math.isfinite(gate.gate_tin_inv_max):
        ok &= _coerce_num(df["Tin_inv_norm1_est_N2"]) <= gate.gate_tin_inv_max

    if "sep2_N2" in df.columns and math.isfinite(gate.gate_sep2_min):
        ok &= _coerce_num(df["sep2_N2"]) >= gate.gate_sep2_min

    if "lam_match_cost" in df.columns and math.isfinite(gate.gate_lam_cost_max):
        ok &= _coerce_num(df["lam_match_cost"]) <= gate.gate_lam_cost_max

    return ok.fillna(False).astype(bool)


def _accept_mask(df: pd.DataFrame, th: ThresholdBundle) -> pd.Series:
    tier01 = _tier01_mask(df, th.gate)
    ok = tier01.copy()
    dlog = _coerce_num(df.get("dlog_abs"))
    darg = _coerce_num(df.get("darg"))
    ok &= dlog <= th.accept.tau_logabs
    ok &= darg <= th.accept.tau_arg
    return ok.fillna(False).astype(bool)


def _guess_delta_summary_path(aug_csv: Path) -> Path | None:
    name = aug_csv.name
    if name.endswith("_delta__aug_lam.csv"):
        cand = aug_csv.with_name(name.replace("_delta__aug_lam.csv", "_delta_summary.json"))
        return cand if cand.exists() else None
    if name.endswith("__aug_lam.csv"):
        cand = aug_csv.with_name(name.replace("__aug_lam.csv", "_summary.json"))
        return cand if cand.exists() else None
    return None


def _read_summary(summary_path: Path | None) -> dict[str, Any]:
    if summary_path is None or (not summary_path.exists()):
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _fixed_gauge_flag(summary: dict[str, Any]) -> bool | None:
    norm = summary.get("normalization")
    if not isinstance(norm, dict):
        return None
    lsfit = norm.get("lsfit")
    if not isinstance(lsfit, dict):
        return None
    mode = lsfit.get("mode")
    if not isinstance(mode, str):
        return None
    return mode.lower() == "external_json"


def _mirror_sigma_stats(
    df: pd.DataFrame,
    *,
    tier01: pd.Series,
    sigma_center: float,
    round_decimals: int = 6,
) -> dict[str, Any]:
    if df.empty or ("sigma" not in df.columns) or ("t" not in df.columns):
        return {
            "mirror_pairs": 0,
            "mirror_pair_frac_tier01": float("nan"),
            "mirror_pair_frac_all": float("nan"),
            "mirror_sigma_tol_used": float("nan"),
            "mirror_dlog_mean": float("nan"),
            "mirror_dlog_p50": float("nan"),
            "mirror_dlog_p90": float("nan"),
            "mirror_dlog_iqr": float("nan"),
            "mirror_darg_mean": float("nan"),
            "mirror_darg_p50": float("nan"),
            "mirror_darg_p90": float("nan"),
            "mirror_darg_iqr": float("nan"),
            "mirror_dlog_min": float("nan"),
            "mirror_darg_min": float("nan"),
        }

    sigma = _coerce_num(df.get("sigma"))
    t = _coerce_num(df.get("t"))
    dlog = _coerce_num(df.get("dlog_abs"))
    darg = _coerce_num(df.get("darg"))

    # Build per-t buckets so mirror pairing can be tolerant (nearest sigma within a
    # data-driven tolerance) rather than requiring exact rounded (t,sigma) matches.
    buckets: dict[float, list[tuple[float, float, float]]] = {}
    for i in df.index.tolist():
        sig_i = float(sigma.loc[i]) if i in sigma.index else float("nan")
        t_i = float(t.loc[i]) if i in t.index else float("nan")
        if not (math.isfinite(sig_i) and math.isfinite(t_i)):
            continue
        t_key = float(np.round(t_i, round_decimals))
        dlog_i = float(dlog.loc[i]) if (i in dlog.index and math.isfinite(float(dlog.loc[i]))) else float("nan")
        darg_i = float(darg.loc[i]) if (i in darg.index and math.isfinite(float(darg.loc[i]))) else float("nan")
        buckets.setdefault(t_key, []).append((sig_i, dlog_i, darg_i))

    # Sort buckets by sigma and compute a global sigma tolerance fallback.
    tol_global = float("nan")
    try:
        all_sig = np.array([float(x) for x in sigma.tolist() if math.isfinite(float(x))], dtype=float)
        all_sig = np.unique(np.sort(all_sig))
        if all_sig.size >= 2:
            diffs = np.diff(all_sig)
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            if diffs.size:
                tol_global = 0.51 * float(np.nanmedian(diffs))
    except Exception:
        tol_global = float("nan")

    bucket_arrays: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, float, int]] = {}
    for t_key, rows in buckets.items():
        rows_s = sorted(rows, key=lambda r: float(r[0]))
        sigs = np.array([r[0] for r in rows_s], dtype=float)
        dlogs = np.array([r[1] for r in rows_s], dtype=float)
        dargs = np.array([r[2] for r in rows_s], dtype=float)
        uniq_n = 0
        tol = float("nan")
        try:
            u = np.unique(sigs[np.isfinite(sigs)])
            uniq_n = int(u.size)
            if u.size >= 2:
                diffs = np.diff(np.sort(u))
                diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
                if diffs.size:
                    tol = 0.51 * float(np.nanmedian(diffs))
        except Exception:
            tol = float("nan")
            uniq_n = 0
        if not math.isfinite(tol):
            tol = float(tol_global) if math.isfinite(float(tol_global)) else 0.0
        bucket_arrays[t_key] = (sigs, dlogs, dargs, float(tol), int(uniq_n))

    diffs_dlog: list[float] = []
    diffs_darg: list[float] = []
    pairs = 0
    tier01_n = int(tier01.sum())
    n_total = int(len(df))
    tol_used: list[float] = []

    for i in df.index.tolist():
        if not bool(tier01.loc[i]):
            continue
        if not (math.isfinite(float(sigma.loc[i])) and math.isfinite(float(t.loc[i]))):
            continue
        sig = float(sigma.loc[i])
        tt = float(t.loc[i])
        sig_m = (2.0 * float(sigma_center)) - sig

        t_key = float(np.round(tt, round_decimals))
        cur = bucket_arrays.get(t_key)
        if cur is None:
            continue
        sigs, dlogs, dargs, tol, uniq_n = cur
        if sigs.size == 0:
            continue

        # Mirror symmetry is not meaningful if there's no sigma variation at this t.
        if int(uniq_n) < 2:
            continue

        # Avoid self-matching at the symmetry center (sigma_m == sigma).
        if abs(float(sig_m) - float(sig)) <= 1e-12:
            continue

        # Nearest sigma match.
        j = int(np.searchsorted(sigs, sig_m, side="left"))
        cand = []
        if 0 <= j < sigs.size:
            cand.append(j)
        if 0 <= (j - 1) < sigs.size:
            cand.append(j - 1)
        if not cand:
            continue
        best_j = min(cand, key=lambda k: abs(float(sigs[k]) - float(sig_m)))
        if float(abs(float(sigs[best_j]) - float(sig_m))) > float(tol):
            continue

        pairs += 1
        tol_used.append(float(tol))

        if math.isfinite(float(dlog.loc[i])) and math.isfinite(float(dlogs[best_j])):
            diffs_dlog.append(abs(float(dlog.loc[i]) - float(dlogs[best_j])))
        if math.isfinite(float(darg.loc[i])) and math.isfinite(float(dargs[best_j])):
            diffs_darg.append(abs(float(darg.loc[i]) - float(dargs[best_j])))

    def _stats(a: list[float]) -> dict[str, float]:
        if not a:
            return {
                "mean": float("nan"),
                "p50": float("nan"),
                "p90": float("nan"),
                "iqr": float("nan"),
                "min": float("nan"),
            }
        x = np.array(a, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return {
                "mean": float("nan"),
                "p50": float("nan"),
                "p90": float("nan"),
                "iqr": float("nan"),
                "min": float("nan"),
            }
        p25 = float(np.nanpercentile(x, 25))
        p50 = float(np.nanpercentile(x, 50))
        p75 = float(np.nanpercentile(x, 75))
        return {
            "mean": float(np.nanmean(x)),
            "p50": p50,
            "p90": float(np.nanpercentile(x, 90)),
            "iqr": float(p75 - p25),
            "min": float(np.nanmin(x)),
        }

    sdlog = _stats(diffs_dlog)
    sdarg = _stats(diffs_darg)

    return {
        "mirror_pairs": int(pairs),
        "mirror_pair_frac_tier01": (float(pairs) / float(tier01_n)) if tier01_n else float("nan"),
        "mirror_pair_frac_all": (float(pairs) / float(n_total)) if n_total else float("nan"),
        "mirror_sigma_tol_used": float(np.nanmedian(np.array(tol_used, dtype=float))) if tol_used else float("nan"),
        "mirror_dlog_mean": float(sdlog["mean"]),
        "mirror_dlog_p50": float(sdlog["p50"]),
        "mirror_dlog_p90": float(sdlog["p90"]),
        "mirror_dlog_iqr": float(sdlog["iqr"]),
        "mirror_darg_mean": float(sdarg["mean"]),
        "mirror_darg_p50": float(sdarg["p50"]),
        "mirror_darg_p90": float(sdarg["p90"]),
        "mirror_darg_iqr": float(sdarg["iqr"]),
        "mirror_dlog_min": float(sdlog["min"]),
        "mirror_darg_min": float(sdarg["min"]),
    }


def _obs_metrics(
    aug_csv: Path,
    *,
    side: str,
    gate: GateThresholds,
    strict: AcceptThresholds,
    diag: AcceptThresholds,
    frontier: AcceptThresholds,
    sigma_center: float,
) -> dict[str, Any]:
    df = _load_aug_df(aug_csv, side=side)

    t = _coerce_num(df.get("t"))
    tvals = [float(x) for x in t.tolist() if math.isfinite(float(x))]
    dt = _estimate_dt(tvals)

    tier01 = _tier01_mask(df, gate)
    n_total = int(len(df))
    n_tier01 = int(tier01.sum())

    dlog = _coerce_num(df.get("dlog_abs"))
    darg = _coerce_num(df.get("darg"))

    # Aggregate per-artifact summaries used for sigma-family mirror proxies.
    sigma_all_med = float("nan")
    sigma_tier01_med = float("nan")
    dlog_tier01_med = float("nan")
    darg_tier01_med = float("nan")
    try:
        if "sigma" in df.columns:
            sig_all = pd.to_numeric(df.loc[:, "sigma"], errors="coerce")
            sig_all = sig_all[np.isfinite(sig_all.to_numpy(dtype=float, na_value=np.nan))]
            if not sig_all.empty:
                sigma_all_med = float(np.nanmedian(sig_all.to_numpy(dtype=float)))

            sig_t01 = pd.to_numeric(df.loc[tier01, "sigma"], errors="coerce")
            sig_t01 = sig_t01[np.isfinite(sig_t01.to_numpy(dtype=float, na_value=np.nan))]
            if not sig_t01.empty:
                sigma_tier01_med = float(np.nanmedian(sig_t01.to_numpy(dtype=float)))

        dlog_t01 = pd.to_numeric(df.loc[tier01, "dlog_abs"], errors="coerce") if "dlog_abs" in df.columns else pd.Series([], dtype=float)
        dlog_t01 = dlog_t01[np.isfinite(dlog_t01.to_numpy(dtype=float, na_value=np.nan))]
        if not dlog_t01.empty:
            dlog_tier01_med = float(np.nanmedian(dlog_t01.to_numpy(dtype=float)))

        darg_t01 = pd.to_numeric(df.loc[tier01, "darg"], errors="coerce") if "darg" in df.columns else pd.Series([], dtype=float)
        darg_t01 = darg_t01[np.isfinite(darg_t01.to_numpy(dtype=float, na_value=np.nan))]
        if not darg_t01.empty:
            darg_tier01_med = float(np.nanmedian(darg_t01.to_numpy(dtype=float)))
    except Exception:
        pass

    def _acc(th_acc: AcceptThresholds) -> pd.Series:
        th = ThresholdBundle(gate=gate, accept=th_acc)
        return _accept_mask(df, th)

    acc_strict = _acc(strict)
    acc_diag = _acc(diag)
    acc_frontier = _acc(frontier)

    # Best-t proxies (used later for cross-rung stability/shift diagnostics).
    best_t_tier01 = float("nan")
    best_t_strict = float("nan")
    best_sigma_tier01 = float("nan")
    best_sigma_strict = float("nan")
    try:
        if ("t" in df.columns) and ("dlog_abs" in df.columns):
            tt = pd.to_numeric(df.loc[tier01, "t"], errors="coerce")
            dd = pd.to_numeric(df.loc[tier01, "dlog_abs"], errors="coerce")
            ss = pd.to_numeric(df.loc[tier01, "sigma"], errors="coerce") if "sigma" in df.columns else pd.Series([], dtype=float)
            ok = (
                np.isfinite(tt.to_numpy(dtype=float, na_value=np.nan))
                & np.isfinite(dd.to_numpy(dtype=float, na_value=np.nan))
                & np.isfinite(ss.to_numpy(dtype=float, na_value=np.nan))
            )
            if ok.any():
                j = int(np.nanargmin(dd.to_numpy(dtype=float)[ok]))
                best_t_tier01 = float(tt.to_numpy(dtype=float)[ok][j])
                best_sigma_tier01 = float(ss.to_numpy(dtype=float)[ok][j])

            tt2 = pd.to_numeric(df.loc[acc_strict, "t"], errors="coerce")
            dd2 = pd.to_numeric(df.loc[acc_strict, "dlog_abs"], errors="coerce")
            ss2 = pd.to_numeric(df.loc[acc_strict, "sigma"], errors="coerce") if "sigma" in df.columns else pd.Series([], dtype=float)
            ok2 = (
                np.isfinite(tt2.to_numpy(dtype=float, na_value=np.nan))
                & np.isfinite(dd2.to_numpy(dtype=float, na_value=np.nan))
                & np.isfinite(ss2.to_numpy(dtype=float, na_value=np.nan))
            )
            if ok2.any():
                j2 = int(np.nanargmin(dd2.to_numpy(dtype=float)[ok2]))
                best_t_strict = float(tt2.to_numpy(dtype=float)[ok2][j2])
                best_sigma_strict = float(ss2.to_numpy(dtype=float)[ok2][j2])
    except Exception:
        pass

    def _t_sorted(mask: pd.Series) -> np.ndarray:
        if "t" not in df.columns:
            return np.array([], dtype=float)
        tt = pd.to_numeric(df.loc[mask, "t"], errors="coerce")
        tt = tt[np.isfinite(tt.to_numpy(dtype=float, na_value=np.nan))]
        return np.sort(tt.to_numpy(dtype=float))

    t_strict = _t_sorted(acc_strict)
    t_frontier = _t_sorted(acc_frontier)

    nseg_s, wmax_s = _segments_width_max(t_strict, dt=float(dt))
    nseg_f, wmax_f = _segments_width_max(t_frontier, dt=float(dt))

    # Near-miss margins: phase margin computed among Tier01+logabs-ok.
    logabs_ok = tier01 & (dlog <= strict.tau_logabs)
    arg_ok = tier01 & (darg <= strict.tau_arg)
    min_darg_tier01_logabs = _min_finite(darg[logabs_ok])
    phase_margin_min = (
        float(min_darg_tier01_logabs) - float(strict.tau_arg)
        if (math.isfinite(float(min_darg_tier01_logabs)) and math.isfinite(float(strict.tau_arg)))
        else float("nan")
    )

    min_dlog_tier01_arg = _min_finite(dlog[arg_ok])
    logabs_margin_min = (
        float(min_dlog_tier01_arg) - float(strict.tau_logabs)
        if (math.isfinite(float(min_dlog_tier01_arg)) and math.isfinite(float(strict.tau_logabs)))
        else float("nan")
    )

    summary = _read_summary(_guess_delta_summary_path(aug_csv))

    # Mirror symmetry proxy: use a local sigma center when possible so the proxy
    # is meaningful even when the sweep isn't centered at 1.0.
    sigma_center_eff = float(sigma_center)
    try:
        if "sigma" in df.columns and n_tier01 >= 5:
            sig_tier01 = pd.to_numeric(df.loc[tier01, "sigma"], errors="coerce")
            sig_tier01 = sig_tier01[np.isfinite(sig_tier01.to_numpy(dtype=float, na_value=np.nan))]
            if not sig_tier01.empty:
                sigma_center_eff = float(np.nanmedian(sig_tier01.to_numpy(dtype=float)))
    except Exception:
        pass

    mirror = _mirror_sigma_stats(df, tier01=tier01, sigma_center=float(sigma_center_eff))
    mirror["mirror_sigma_center_used"] = float(sigma_center_eff)

    return {
        "file": str(aug_csv),
        "points": n_total,
        "sigma_median_all": float(sigma_all_med),
        "sigma_median_tier01": float(sigma_tier01_med),
        "median_dlog_abs_tier01": float(dlog_tier01_med),
        "median_darg_tier01": float(darg_tier01_med),
        "t_min": float(np.nanmin(tvals)) if tvals else float("nan"),
        "t_max": float(np.nanmax(tvals)) if tvals else float("nan"),
        "dt_est": float(dt),
        "tier01": n_tier01,
        "tier01_frac": (n_tier01 / n_total) if n_total else float("nan"),
        "strict_accept": int(acc_strict.sum()),
        "strict_accept_frac": (float(int(acc_strict.sum())) / n_total) if n_total else float("nan"),
        "strict_segments": int(nseg_s),
        "strict_width_max": float(wmax_s),
        "frontier_accept": int(acc_frontier.sum()),
        "frontier_accept_frac": (float(int(acc_frontier.sum())) / n_total) if n_total else float("nan"),
        "frontier_segments": int(nseg_f),
        "frontier_width_max": float(wmax_f),
        "diag_accept": int(acc_diag.sum()),
        "diag_gain": int(acc_diag.sum()) - int(acc_strict.sum()),
        "best_t_tier01_by_dlog": float(best_t_tier01),
        "best_t_strict_by_dlog": float(best_t_strict),
        "best_sigma_tier01_by_dlog": float(best_sigma_tier01),
        "best_sigma_strict_by_dlog": float(best_sigma_strict),
        **mirror,
        "min_dlog_abs_tier01": _min_finite(dlog[tier01]),
        "min_darg_tier01": _min_finite(darg[tier01]),
        "min_darg_tier01_logabs_ok": float(min_darg_tier01_logabs),
        "phase_margin_min": float(phase_margin_min),
        "min_dlog_abs_tier01_arg_ok": float(min_dlog_tier01_arg),
        "logabs_margin_min": float(logabs_margin_min),
        "N": summary.get("N"),
        "N2": summary.get("N2"),
        "quotient": summary.get("quotient") or _infer_obs_from_name(aug_csv),
        "fixed_gauge": _fixed_gauge_flag(summary),
        "cstar_source": ((summary.get("normalization") or {}).get("lsfit") or {}).get("source"),
    }


def _baseline_key(sigma: float, t: float) -> tuple[float, float]:
    return (float(np.round(float(sigma), 6)), float(np.round(float(t), 6)))


def _load_baseline_map(points_csv: Path) -> dict[tuple[float, float], dict[str, float]]:
    low = pd.read_csv(points_csv)
    low_sigma = pd.to_numeric(low.get("sigma"), errors="coerce")
    low_t = pd.to_numeric(low.get("t"), errors="coerce")
    low_dlog = pd.to_numeric(low.get("dlog_abs"), errors="coerce")

    m: dict[tuple[float, float], dict[str, float]] = {}
    for s_val, t_val, d_val in zip(low_sigma.tolist(), low_t.tolist(), low_dlog.tolist()):
        if not (math.isfinite(float(s_val)) and math.isfinite(float(t_val)) and math.isfinite(float(d_val))):
            continue
        k = _baseline_key(float(s_val), float(t_val))
        cur = m.get(k)
        if (cur is None) or (float(d_val) < float(cur.get("dlog_abs", float("inf")))):
            m[k] = {"dlog_abs": float(d_val)}
    return m


def _contraction_metrics(high_aug_csv: Path, *, side: str, gate: GateThresholds, baseline_map: dict[tuple[float, float], dict[str, float]]) -> dict[str, Any]:
    high = _load_aug_df(high_aug_csv, side=side)
    if high.empty:
        return {"contract_pairs": 0, "contract_n": 0, "contract_frac": float("nan"), "contract_best_improve": float("nan")}

    sigma = _coerce_num(high.get("sigma"))
    t = _coerce_num(high.get("t"))
    dlog_high = _coerce_num(high.get("dlog_abs"))

    tier01_high = _tier01_mask(high, gate)

    pairs = 0
    n_contract = 0
    best_improve = float("inf")

    for i in high.index.tolist():
        if not bool(tier01_high.loc[i]):
            continue
        if not (math.isfinite(float(sigma.loc[i])) and math.isfinite(float(t.loc[i])) and math.isfinite(float(dlog_high.loc[i]))):
            continue
        key = _baseline_key(float(sigma.loc[i]), float(t.loc[i]))
        low = baseline_map.get(key)
        if not low:
            continue
        dlog_low = low.get("dlog_abs")
        if dlog_low is None or (not math.isfinite(float(dlog_low))):
            continue

        pairs += 1
        improve = float(dlog_high.loc[i]) - float(dlog_low)
        if improve < 0.0:
            n_contract += 1
        if improve < best_improve:
            best_improve = improve

    return {
        "contract_pairs": int(pairs),
        "contract_n": int(n_contract),
        "contract_frac": (float(n_contract) / float(pairs)) if pairs else float("nan"),
        "contract_best_improve": float(best_improve) if pairs else float("nan"),
    }


# ---------- Basket selection + labeling ----------


def _case_category(row: dict[str, Any]) -> list[str]:
    tags: list[str] = []

    # Aggregate family-level strict accept.
    strict_sum = float(row.get("strict_accept_sum", 0.0) or 0.0)
    tier01_mean = float(row.get("tier01_frac_mean", float("nan")) or float("nan"))
    overlap3 = float(row.get("overlap_all3_n", float("nan")) or float("nan"))
    phase_block = float(row.get("phase_block_score", 0.0) or 0.0)
    diag_gain = float(row.get("diag_gain_sum", 0.0) or 0.0)
    contract_best = float(row.get("contract_best_improve", float("nan")) or float("nan"))

    if (not math.isfinite(tier01_mean)) or tier01_mean < 0.05:
        tags.append("dead")
    elif strict_sum <= 0.0 and (math.isfinite(overlap3) and overlap3 <= 0.0) and (diag_gain <= 0.0):
        tags.append("near_miss")

    if strict_sum <= 0.0 and diag_gain > 0.0:
        tags.append("phase_blocked")

    if math.isfinite(contract_best) and contract_best < 0.0 and strict_sum <= 0.0:
        tags.append("contraction_positive")

    if math.isfinite(overlap3) and overlap3 > 0.0:
        tags.append("overlap_positive")

    if phase_block > 0.0 and strict_sum > 0.0:
        tags.append("fragile")

    fixed = row.get("fixed_gauge_all")
    if fixed is False:
        tags.append("gauge_fragile")

    if not tags:
        tags.append("uncategorized")

    return tags


# ---------- Scoring ----------


def _nan0(x: float) -> float:
    return float(x) if math.isfinite(float(x)) else 0.0


def _zscore(v: np.ndarray) -> np.ndarray:
    v = v.astype(float)
    m = np.nanmean(v)
    s = np.nanstd(v)
    if not math.isfinite(float(s)) or s <= 0:
        return np.zeros_like(v)
    return (v - m) / s


def _compute_scores(df: pd.DataFrame, *, weights: dict[str, float]) -> pd.Series:
    # z-score continuous features; booleans as 0/1.
    feat = {}
    for k in weights.keys():
        if k not in df.columns:
            feat[k] = np.zeros(len(df), dtype=float)
            continue
        if df[k].dtype == bool:
            feat[k] = df[k].astype(float).to_numpy()
        else:
            feat[k] = pd.to_numeric(df[k], errors="coerce").to_numpy(dtype=float)
            feat[k] = _zscore(feat[k])
            feat[k] = np.nan_to_num(feat[k], nan=0.0, posinf=0.0, neginf=0.0)

    score = np.zeros(len(df), dtype=float)
    for k, w in weights.items():
        score += float(w) * feat[k]
    return pd.Series(score, index=df.index)


def _weight_perturbation(df: pd.DataFrame, *, base_weights: dict[str, float], n: int, frac: float, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    base = _compute_scores(df, weights=base_weights)
    base_rank = base.rank(method="average", ascending=False)

    topk = min(10, len(df))
    df0 = df.copy()
    df0["__score__"] = pd.to_numeric(base, errors="coerce")
    base_top = set(df0.sort_values(by="__score__", ascending=False).head(topk)["case"].tolist())

    corrs: list[float] = []
    topk_jacc: list[float] = []

    keys = list(base_weights.keys())
    for _ in range(n):
        w = {}
        for k in keys:
            delta = 1.0 + rng.uniform(-frac, frac)
            w[k] = float(base_weights[k]) * float(delta)
        s = _compute_scores(df, weights=w)
        r = s.rank(method="average", ascending=False)
        # Spearman correlation via rank correlation.
        corr = float(pd.Series(base_rank).corr(pd.Series(r), method="spearman"))
        corrs.append(corr)

        df1 = df.copy()
        df1["__score__"] = pd.to_numeric(s, errors="coerce")
        top = set(df1.sort_values(by="__score__", ascending=False).head(topk)["case"].tolist())
        j = len(top & base_top) / max(len(top | base_top), 1)
        topk_jacc.append(float(j))

    return {
        "n": int(n),
        "frac": float(frac),
        "spearman_rank_corr_p50": float(np.nanmedian(corrs)) if corrs else float("nan"),
        "spearman_rank_corr_p10": float(np.nanpercentile(corrs, 10)) if corrs else float("nan"),
        "spearman_rank_corr_p90": float(np.nanpercentile(corrs, 90)) if corrs else float("nan"),
        "topk": int(topk),
        "topk_jacc_p50": float(np.nanmedian(topk_jacc)) if topk_jacc else float("nan"),
        "topk_jacc_p10": float(np.nanpercentile(topk_jacc, 10)) if topk_jacc else float("nan"),
        "topk_jacc_p90": float(np.nanpercentile(topk_jacc, 90)) if topk_jacc else float("nan"),
    }


def _ablation(df: pd.DataFrame, *, base_weights: dict[str, float], blocks: dict[str, list[str]]) -> dict[str, Any]:
    base = _compute_scores(df, weights=base_weights)
    base_rank = base.rank(method="average", ascending=False)

    out: dict[str, Any] = {}
    for name, cols in blocks.items():
        w = dict(base_weights)
        for c in cols:
            w.pop(c, None)
        s = _compute_scores(df, weights=w)
        r = s.rank(method="average", ascending=False)
        corr = float(pd.Series(base_rank).corr(pd.Series(r), method="spearman"))
        out[name] = {"dropped": cols, "spearman_rank_corr": corr}
    return out


def _confusion_summary(df: pd.DataFrame, *, label_col: str, score_col: str) -> dict[str, Any]:
    topk = min(10, len(df))
    df2 = df.copy()
    df2[score_col] = pd.to_numeric(df2[score_col], errors="coerce")
    top = df2.sort_values(by=score_col, ascending=False).head(topk)

    def _has(lbl: str, key: str) -> bool:
        return key in str(lbl).split(";")

    labels = df2[label_col].astype(str).tolist() if label_col in df2.columns else []
    n_dead = sum(1 for x in labels if _has(x, "dead"))
    n_best = sum(1 for x in labels if _has(x, "best_current"))

    top_labels = top[label_col].astype(str).tolist() if label_col in top.columns else []
    top_dead = sum(1 for x in top_labels if _has(x, "dead"))
    top_best = sum(1 for x in top_labels if _has(x, "best_current"))

    return {
        "n_rows": int(len(df2)),
        "topk": int(topk),
        "dead_total": int(n_dead),
        "dead_in_topk": int(top_dead),
        "best_total": int(n_best),
        "best_in_topk": int(top_best),
        "topk_label_counts": dict(pd.Series(top_labels).value_counts().head(10)),
    }


def _json_default(o: Any) -> Any:
    # Coerce numpy/pandas scalars to builtin JSON-compatible types.
    try:
        import numpy as _np

        if isinstance(o, (_np.integer,)):
            return int(o)
        if isinstance(o, (_np.floating,)):
            return float(o)
        if isinstance(o, (_np.ndarray,)):
            return o.tolist()
    except Exception:
        pass

    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()

    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


# ---------- Main pipeline ----------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Artifact-only scorecard validation: build an ugly 20–30 row evaluation basket, compute features, "
            "assign simple hand-style labels (rule-based), compute structure/robustness scores, and emit stability/ablation/"
            "monotonicity/confusion reports. No projector emission."
        )
    )
    ap.add_argument("--run_dir", required=True, help="Output directory (and default scan root if --scan_root omitted)")
    ap.add_argument(
        "--scan_root",
        default=None,
        help="Root to scan for artifact CSVs (defaults to --run_dir). Historically this scanned only *__aug_lam.csv; now also includes *_delta* and *_samples*.",
    )
    ap.add_argument("--out_dir", default=None, help="Where to write outputs (defaults to --run_dir)")
    ap.add_argument("--side", default="right")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--sigma_center", type=float, default=1.0, help="Mirror symmetry center: sigma_m = 2*sigma_center - sigma")

    ap.add_argument("--gate_tin_inv_max", type=float, default=500.0)
    ap.add_argument("--gate_sep2_min", type=float, default=0.01)
    ap.add_argument("--gate_lam_cost_max", type=float, default=0.8)

    ap.add_argument("--baseline_points_csv", default="_sigma_sweep_win1_tier01_points_final.csv")

    ap.add_argument("--target_rows", type=int, default=80)
    ap.add_argument("--max_groups_scan", type=int, default=2000, help="Safety cap on number of groups to fully score")

    ap.add_argument(
        "--synthetic_smoothness_controls",
        type=int,
        default=0,
        help=(
            "If nonzero, score a few synthetic 'smoothness trap' controls against the basket (without changing basket ranks). "
            "These controls should ideally not rank near the best real rows."
        ),
    )

    ap.add_argument("--out_prefix", default="_bias_scorecard_v2")

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"run_dir not found: {run_dir}")

    scan_root = Path(args.scan_root) if args.scan_root else run_dir
    if not scan_root.exists():
        raise SystemExit(f"scan_root not found: {scan_root}")

    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    side = str(args.side)
    gate = GateThresholds(
        gate_tin_inv_max=float(args.gate_tin_inv_max),
        gate_sep2_min=float(args.gate_sep2_min),
        gate_lam_cost_max=float(args.gate_lam_cost_max),
    )

    baseline_path = Path(str(args.baseline_points_csv))
    if not baseline_path.is_absolute():
        baseline_path = out_dir / baseline_path
    baseline_map = _load_baseline_map(baseline_path) if baseline_path.exists() else {}

    # Discover artifacts.
    # We prefer augmented-lambda deltas, but to broaden support (esp. for cross-rung/divisor pairing)
    # we also include non-aug delta/samples artifacts.
    # Filter out this tool's own outputs to avoid self-ingestion.
    def _is_validator_output(p: Path) -> bool:
        n = p.name
        if n.startswith(str(args.out_prefix) + "_"):
            return True
        if n.startswith("_bias_scorecard_") or n.startswith("_bias_scorecard_v2"):
            return True
        return False

    artifact_paths: list[Path] = []
    artifact_paths.extend(list(scan_root.rglob("*__aug_lam.csv")))
    artifact_paths.extend(list(scan_root.rglob("*_delta*.csv")))
    artifact_paths.extend(list(scan_root.rglob("*_samples*.csv")))
    # De-dup while preserving order.
    seen = set()
    artifact_paths = [p for p in artifact_paths if (not _is_validator_output(p)) and (str(p) not in seen) and (not seen.add(str(p)))]

    if not artifact_paths:
        raise SystemExit(f"No artifact CSVs found under: {scan_root}")

    # Group.
    groups: dict[str, dict[str, Path]] = {}
    for p in artifact_paths:
        obs = _infer_obs_from_name(p)
        rel_parent = str(p.parent)
        try:
            rel_parent = str(p.parent.relative_to(scan_root))
        except Exception:
            pass
        gk = f"{rel_parent}/{_group_key(p)}"
        groups.setdefault(gk, {})[obs] = p

    # Prefer triads, but keep singles for contraction and dead/control.
    gitems = list(groups.items())

    # Deterministic shuffle for variety.
    rng = random.Random(int(args.seed))
    rng.shuffle(gitems)

    # Compute metrics for up to cap groups.
    rows: list[dict[str, Any]] = []

    def _frontier_for(obs: str) -> AcceptThresholds:
        if obs == "Q2_tr":
            return FRONTIER_Q2TR
        if obs == "Q2":
            return FRONTIER_Q2
        if obs == "Q2_phi2":
            return FRONTIER_Q2PHI2
        return FRONTIER_Q2

    # Always include anchors if present.
    def _want_anchor(gk: str) -> bool:
        name = gk.lower()
        return ("refine164" in name) or ("pocket162" in name)

    anchors = [(gk, obsmap) for (gk, obsmap) in gitems if _want_anchor(gk)]
    nonanchors = [(gk, obsmap) for (gk, obsmap) in gitems if not _want_anchor(gk)]
    scan_list = anchors + nonanchors

    for gk, obsmap in scan_list[: int(args.max_groups_scan)]:
        case = {
            "case": gk,
            "n_obs": int(len(obsmap)),
        }

        # Per-obs metrics.
        obs_m: dict[str, dict[str, Any]] = {}
        fixed_flags: list[bool] = []

        for obs, p in obsmap.items():
            m = _obs_metrics(
                p,
                side=side,
                gate=gate,
                strict=STRICT,
                diag=DIAG,
                frontier=_frontier_for(obs),
                sigma_center=float(args.sigma_center),
            )
            obs_m[obs] = m
            if isinstance(m.get("fixed_gauge"), bool):
                fixed_flags.append(bool(m["fixed_gauge"]))

            prefix = {
                "Q2_tr": "Q2tr",
                "Q2_phi2": "Q2phi2",
                "Q2": "Q2",
            }.get(obs, obs)
            for k, v in m.items():
                case[f"{prefix}_{k}"] = v

        case["fixed_gauge_all"] = (all(fixed_flags) if fixed_flags else None)

        # Aggregate features (structure/robustness building blocks).
        # Avoid RuntimeWarning("Mean of empty slice") when the list is non-empty but
        # contains only NaNs.
        tier01_fracs = np.array([float(obs_m[o].get("tier01_frac", float("nan"))) for o in obs_m], dtype=float)
        tier01_finite = tier01_fracs[np.isfinite(tier01_fracs)]
        case["tier01_frac_mean"] = float(np.mean(tier01_finite)) if tier01_finite.size else float("nan")

        strict_accepts = [float(obs_m[o].get("strict_accept", 0.0) or 0.0) for o in obs_m]
        frontier_accepts = [float(obs_m[o].get("frontier_accept", 0.0) or 0.0) for o in obs_m]
        diag_gains = [float(obs_m[o].get("diag_gain", 0.0) or 0.0) for o in obs_m]

        case["strict_accept_sum"] = float(np.nansum(strict_accepts))
        case["frontier_accept_sum"] = float(np.nansum(frontier_accepts))
        case["diag_gain_sum"] = float(np.nansum(diag_gains))

        strict_widths = [float(obs_m[o].get("strict_width_max", 0.0) or 0.0) for o in obs_m]
        frontier_widths = [float(obs_m[o].get("frontier_width_max", 0.0) or 0.0) for o in obs_m]
        case["strict_width_sum"] = float(np.nansum(strict_widths))
        case["frontier_width_sum"] = float(np.nansum(frontier_widths))

        # Disagreement proxy.
        if len(strict_accepts) >= 2:
            case["family_disagree"] = float(np.nanstd(np.array(strict_accepts, dtype=float)))
        else:
            case["family_disagree"] = float("nan")

        # Cross-observable agreement proxies ("zeta-structural" flavor):
        # When multiple observables exist for a case, check whether their best-T
        # and best-sigma locations align. Lower std/range is better.
        best_t_obs = np.array([
            float(obs_m[o].get("best_t_tier01_by_dlog", float("nan"))) for o in obs_m
        ], dtype=float)
        best_t_obs = best_t_obs[np.isfinite(best_t_obs)]
        if best_t_obs.size >= 2:
            case["best_t_obs_std"] = float(np.std(best_t_obs))
            case["best_t_obs_range"] = float(np.max(best_t_obs) - np.min(best_t_obs))
        else:
            case["best_t_obs_std"] = float("nan")
            case["best_t_obs_range"] = float("nan")

        best_sigma_obs = np.array([
            float(obs_m[o].get("best_sigma_tier01_by_dlog", float("nan"))) for o in obs_m
        ], dtype=float)
        best_sigma_obs = best_sigma_obs[np.isfinite(best_sigma_obs)]
        if best_sigma_obs.size >= 2:
            case["best_sigma_obs_std"] = float(np.std(best_sigma_obs))
            case["best_sigma_obs_range"] = float(np.max(best_sigma_obs) - np.min(best_sigma_obs))
        else:
            case["best_sigma_obs_std"] = float("nan")
            case["best_sigma_obs_range"] = float("nan")

        # Phase-block score: sum of positive phase margins across families.
        phase_margins = [float(obs_m[o].get("phase_margin_min", float("nan"))) for o in obs_m]
        case["phase_block_score"] = float(np.nansum([max(pm, 0.0) for pm in phase_margins if math.isfinite(pm)]))
        logabs_margins = [float(obs_m[o].get("logabs_margin_min", float("nan"))) for o in obs_m]
        case["logabs_block_score"] = float(np.nansum([max(lm, 0.0) for lm in logabs_margins if math.isfinite(lm)]))

        # Overlaps for triads (strict only): approximate by using identical t lists is expensive; use n only if available.
        # We compute overlap by intersecting strict accepted t-values across families when at least 2 obs exist.
        if "t" in WANT_COLS:
            strict_sets: list[set[float]] = []
            dt_list: list[float] = []
            for obs, p in obsmap.items():
                df = _load_aug_df(p, side=side)
                mask = _accept_mask(df, ThresholdBundle(gate=gate, accept=STRICT))
                tt = pd.to_numeric(df.loc[mask, "t"], errors="coerce") if "t" in df.columns else pd.Series([], dtype=float)
                tt = tt[np.isfinite(tt.to_numpy(dtype=float, na_value=np.nan))]
                keys = set(float(np.round(float(x), 12)) for x in tt.tolist())
                strict_sets.append(keys)
                dt_list.append(_estimate_dt(tt.tolist()))

            if len(strict_sets) >= 2:
                inter2 = strict_sets[0].intersection(*strict_sets[1:]) if strict_sets else set()
                case["overlap_all3_n"] = float(len(inter2)) if len(strict_sets) >= 3 else float("nan")
                if inter2:
                    dt = float(np.nanmin(np.array(dt_list, dtype=float)))
                    t_sorted = np.array(sorted(inter2), dtype=float)
                    _, wmax = _segments_width_max(t_sorted, dt=dt)
                    case["overlap_all3_width_max"] = float(wmax)
                else:
                    case["overlap_all3_width_max"] = 0.0
            else:
                case["overlap_all3_n"] = float("nan")
                case["overlap_all3_width_max"] = float("nan")

        # Contraction metrics for high-rung Q2tr.
        contract = {"contract_pairs": float("nan"), "contract_n": float("nan"), "contract_frac": float("nan"), "contract_best_improve": float("nan")}
        if "Q2_tr" in obsmap and baseline_map:
            m = obs_m.get("Q2_tr") or {}
            if m.get("N") == 4096 and m.get("N2") == 8192:
                contract = _contraction_metrics(obsmap["Q2_tr"], side=side, gate=gate, baseline_map=baseline_map)
        case.update(contract)

        # Provisional rule-based labels.
        labels = _case_category(case)
        case["label_provisional"] = ";".join(labels)

        rows.append(case)

    df = pd.DataFrame(rows)

    # Cross-rung proxy: within the scanned corpus, compute best-t shifts for rows
    # that share a similar window signature.
    def _window_id(case: str) -> str:
        s = str(case)
        s = re.sub(r"N\d+_\d+", "N{rung}", s)
        s = re.sub(r"N\d+_N2\d+", "N{rung}", s)
        s = re.sub(r"N\d+", "N{n}", s)
        s = re.sub(r"dt\d+p\d+", "dt{dt}", s)
        s = re.sub(r"_sigma\d+p\d+", "_sigma{sigma}", s)
        return s

    def _divisor_family_id(case: str) -> str:
        # Divisor/zero-stability needs *some* ladder support within families.
        # In practice, rungs often come with different "triad/pocket/refine" tags
        # while still sharing the same broad window semantics. A cheap but useful
        # family key is the suffix after `_{OBS}_` with t-window numeric ranges
        # normalized away.
        s = str(case)
        parts = re.split(r"_\{OBS\}_", s, maxsplit=1)
        if len(parts) == 2:
            s = parts[1]
        # Coarsen aggressively: we want some N->2N pairing support.
        # Many suffixes embed rung/dt tokens in inconsistent formats (e.g. dt0.25).
        s = re.sub(r"fixedcstar", "", s)
        s = re.sub(r"N\d+_\d+", "", s)
        s = re.sub(r"N\d+_N2\d+", "", s)
        s = re.sub(r"N\d+", "", s)
        s = re.sub(r"dt\d+p\d+", "", s)
        s = re.sub(r"dt\d+\.\d+", "", s)
        s = re.sub(r"_sigma\d+p\d+", "_sigma{sigma}", s)

        # Normalize time window tokens to avoid over-fragmentation.
        s = re.sub(r"t\d+p\d+_to_\d+p\d+", "t{range}", s)
        s = re.sub(r"t\d+p\d+", "t{t}", s)

        # Clean up separators.
        s = re.sub(r"__+", "_", s)
        s = re.sub(r"_+", "_", s)
        s = (s or "").strip("_ ")

        # Fall back to the stricter window id if we ended up with something empty.
        return s if s else _window_id(case)

    df["window_id"] = df["case"].astype(str).map(_window_id)
    df["divisor_family_id"] = df["case"].astype(str).map(_divisor_family_id)

    # --- Cross-artifact sigma-family mirror proxies ---
    def _sigma_family_features(df_in: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
        sig_col = f"{prefix}_sigma_median_all"
        dlog_col = f"{prefix}_median_dlog_abs_tier01"
        darg_col = f"{prefix}_median_darg_tier01"
        bestt_col = f"{prefix}_best_t_tier01_by_dlog"
        if sig_col not in df_in.columns:
            return df_in

        out = df_in.copy()
        fam_pairs = np.full(len(out), np.nan, dtype=float)
        fam_pair_frac = np.full(len(out), np.nan, dtype=float)
        fam_low_support = np.full(len(out), np.nan, dtype=float)
        fam_dlog_p50 = np.full(len(out), np.nan, dtype=float)
        fam_dlog_p90 = np.full(len(out), np.nan, dtype=float)
        fam_bestt_p50 = np.full(len(out), np.nan, dtype=float)
        fam_bestt_p90 = np.full(len(out), np.nan, dtype=float)

        sig_all = pd.to_numeric(out.get(sig_col), errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        dlog_all = pd.to_numeric(out.get(dlog_col), errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        darg_all = pd.to_numeric(out.get(darg_col), errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        bestt_all = pd.to_numeric(out.get(bestt_col), errors="coerce").to_numpy(dtype=float, na_value=np.nan) if bestt_col in out.columns else np.full(len(out), np.nan)

        for fam, idxs in out.groupby("window_id").groups.items():
            id_list = list(idxs)
            sigs = np.array([sig_all[i] for i in id_list], dtype=float)
            ok = np.isfinite(sigs)
            if int(ok.sum()) < 2:
                continue

            # Use unique sigma values for step/tolerance.
            uniq = np.unique(np.sort(sigs[ok]))
            if uniq.size < 2:
                continue
            diffs = np.diff(uniq)
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            if diffs.size == 0:
                continue
            step = float(np.nanmedian(diffs))
            tol = 0.75 * float(step)
            sigma_center = float(np.nanmedian(uniq))

            used: set[int] = set()
            pair_res_dlog: list[float] = []
            pair_res_bestt: list[float] = []
            pair_count = 0

            # Pair unique sigmas to their mirrored partner (one-to-one, no self matches).
            for i_local, s_i in enumerate(sigs.tolist()):
                if not math.isfinite(float(s_i)):
                    continue
                i_abs = id_list[i_local]
                if i_abs in used:
                    continue
                s_m = (2.0 * float(sigma_center)) - float(s_i)
                if abs(float(s_m) - float(s_i)) <= 1e-12:
                    continue
                # Find nearest sigma among remaining.
                best_j_local = None
                best_dist = float("inf")
                for j_local, s_j in enumerate(sigs.tolist()):
                    j_abs = id_list[j_local]
                    if j_abs == i_abs or j_abs in used:
                        continue
                    if not math.isfinite(float(s_j)):
                        continue
                    dist = abs(float(s_j) - float(s_m))
                    if dist < best_dist:
                        best_dist = dist
                        best_j_local = j_local
                if best_j_local is None:
                    continue
                if float(best_dist) > float(tol):
                    continue

                j_abs = id_list[int(best_j_local)]
                used.add(i_abs)
                used.add(j_abs)
                pair_count += 1

                # Residuals based on artifact-level aggregates.
                di = float(dlog_all[i_abs])
                dj = float(dlog_all[j_abs])
                if math.isfinite(di) and math.isfinite(dj):
                    pair_res_dlog.append(abs(di - dj))

                ti = float(bestt_all[i_abs])
                tj = float(bestt_all[j_abs])
                if math.isfinite(ti) and math.isfinite(tj):
                    pair_res_bestt.append(abs(ti - tj))

            # Family-level aggregates.
            uniq_n = int(uniq.size)
            possible_pairs = int(uniq_n // 2)
            frac = (float(pair_count) / float(possible_pairs)) if possible_pairs else float("nan")
            low = (pair_count < 1) or (math.isfinite(frac) and frac < 0.2)

            def _p(a: list[float], q: float) -> float:
                if not a:
                    return float("nan")
                x = np.array(a, dtype=float)
                x = x[np.isfinite(x)]
                if x.size == 0:
                    return float("nan")
                return float(np.nanpercentile(x, q))

            dlog_p50 = _p(pair_res_dlog, 50)
            dlog_p90 = _p(pair_res_dlog, 90)
            bestt_p50 = _p(pair_res_bestt, 50)
            bestt_p90 = _p(pair_res_bestt, 90)

            for i_abs in id_list:
                fam_pairs[i_abs] = float(pair_count)
                fam_pair_frac[i_abs] = float(frac) if math.isfinite(frac) else float("nan")
                fam_low_support[i_abs] = 1.0 if low else 0.0
                fam_dlog_p50[i_abs] = float(dlog_p50)
                fam_dlog_p90[i_abs] = float(dlog_p90)
                fam_bestt_p50[i_abs] = float(bestt_p50)
                fam_bestt_p90[i_abs] = float(bestt_p90)

        out[f"sigmafam_{prefix}_mirror_pairs"] = fam_pairs
        out[f"sigmafam_{prefix}_mirror_pair_frac"] = fam_pair_frac
        out[f"sigmafam_{prefix}_mirror_low_support"] = fam_low_support
        out[f"sigmafam_{prefix}_mirror_dlog_p50"] = fam_dlog_p50
        out[f"sigmafam_{prefix}_mirror_dlog_p90"] = fam_dlog_p90
        out[f"sigmafam_{prefix}_mirror_best_t_p50"] = fam_bestt_p50
        out[f"sigmafam_{prefix}_mirror_best_t_p90"] = fam_bestt_p90
        return out

    # Compute sigma-family mirror stats for every per-obs prefix that has a sigma median.
    prefixes = []
    for c in df.columns.tolist():
        if isinstance(c, str) and c.endswith("_sigma_median_all"):
            prefixes.append(c[: -len("_sigma_median_all")])
    prefixes = sorted(set(prefixes))
    for pfx in prefixes:
        df = _sigma_family_features(df, prefix=str(pfx))

    # Coalesce per-prefix sigmafam metrics into unified columns used by zeta scoring.
    def _coalesce(cols: list[str]) -> pd.Series:
        s = pd.Series([float("nan")] * len(df), index=df.index, dtype=float)
        for c in cols:
            if c in df.columns:
                s = s.fillna(pd.to_numeric(df[c], errors="coerce"))
        return s

    prio = ["Q2tr", "Q2", "Q2phi2"]
    other = [p for p in prefixes if p not in prio]
    ordered = prio + other

    df["sigmafam_mirror_pairs"] = _coalesce([f"sigmafam_{p}_mirror_pairs" for p in ordered]).fillna(0.0)
    df["sigmafam_mirror_pair_frac"] = _coalesce([f"sigmafam_{p}_mirror_pair_frac" for p in ordered]).fillna(0.0)
    df["sigmafam_mirror_low_support"] = _coalesce([f"sigmafam_{p}_mirror_low_support" for p in ordered]).fillna(1.0)
    df["sigmafam_mirror_dlog_p50"] = _coalesce([f"sigmafam_{p}_mirror_dlog_p50" for p in ordered])
    df["sigmafam_mirror_best_t_p50"] = _coalesce([f"sigmafam_{p}_mirror_best_t_p50" for p in ordered])

    best_t_cols = [
        "Q2tr_best_t_tier01_by_dlog",
        "Q2_best_t_tier01_by_dlog",
        "Q2phi2_best_t_tier01_by_dlog",
    ]

    def _row_best_t_any(r: pd.Series) -> float:
        for c in best_t_cols:
            if c in r and math.isfinite(float(r[c])):
                return float(r[c])
        return float("nan")

    df["best_t_any_tier01"] = df.apply(_row_best_t_any, axis=1)

    # Case-level aggregates across whatever observables exist for this case.
    sig_cols_any = [c for c in df.columns if isinstance(c, str) and c.endswith("_sigma_median_all")]
    dlog_cols_any = [c for c in df.columns if isinstance(c, str) and c.endswith("_median_dlog_abs_tier01")]

    def _row_nanmedian(r: pd.Series, cols: list[str]) -> float:
        vals: list[float] = []
        for c in cols:
            if c in r and math.isfinite(float(r[c])):
                vals.append(float(r[c]))
        if not vals:
            return float("nan")
        return float(np.nanmedian(np.array(vals, dtype=float)))

    df["sigma_any_median_all"] = df.apply(lambda r: _row_nanmedian(r, sig_cols_any), axis=1) if sig_cols_any else float("nan")
    df["median_dlog_abs_tier01_any"] = df.apply(lambda r: _row_nanmedian(r, dlog_cols_any), axis=1) if dlog_cols_any else float("nan")

    # Sigma-family mirror proxies across cases sharing a window signature (sigma-stripped).
    # This intentionally does NOT depend on an observable token in filenames.
    sig_any = pd.to_numeric(df.get("sigma_any_median_all"), errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    dlog_any = pd.to_numeric(df.get("median_dlog_abs_tier01_any"), errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    bestt_any = pd.to_numeric(df.get("best_t_any_tier01"), errors="coerce").to_numpy(dtype=float, na_value=np.nan)

    fam_pairs_any = np.full(len(df), np.nan, dtype=float)
    fam_pair_frac_any = np.full(len(df), np.nan, dtype=float)
    fam_low_support_any = np.full(len(df), np.nan, dtype=float)
    fam_dlog_p50_any = np.full(len(df), np.nan, dtype=float)
    fam_dlog_p90_any = np.full(len(df), np.nan, dtype=float)
    fam_bestt_p50_any = np.full(len(df), np.nan, dtype=float)
    fam_bestt_p90_any = np.full(len(df), np.nan, dtype=float)

    for wid, idxs in df.groupby("window_id").groups.items():
        id_list = list(idxs)
        sigs = np.array([sig_any[i] for i in id_list], dtype=float)
        ok = np.isfinite(sigs)
        if int(ok.sum()) < 2:
            continue
        uniq = np.unique(np.sort(sigs[ok]))
        if uniq.size < 2:
            continue
        diffs = np.diff(uniq)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size == 0:
            continue
        step = float(np.nanmedian(diffs))
        tol = 0.75 * float(step)
        sigma_center = float(np.nanmedian(uniq))

        used: set[int] = set()
        pair_res_dlog: list[float] = []
        pair_res_bestt: list[float] = []
        pair_count = 0

        # Pair within this family by mirrored sigma (one-to-one, no self matches).
        for i_local, s_i in enumerate(sigs.tolist()):
            i_abs = id_list[i_local]
            if i_abs in used:
                continue
            if not math.isfinite(float(s_i)):
                continue
            s_m = (2.0 * float(sigma_center)) - float(s_i)
            if abs(float(s_m) - float(s_i)) <= 1e-12:
                continue
            best_j_local = None
            best_dist = float("inf")
            for j_local, s_j in enumerate(sigs.tolist()):
                j_abs = id_list[j_local]
                if j_abs == i_abs or j_abs in used:
                    continue
                if not math.isfinite(float(s_j)):
                    continue
                dist = abs(float(s_j) - float(s_m))
                if dist < best_dist:
                    best_dist = dist
                    best_j_local = j_local
            if best_j_local is None:
                continue
            if float(best_dist) > float(tol):
                continue
            j_abs = id_list[int(best_j_local)]
            used.add(i_abs)
            used.add(j_abs)
            pair_count += 1

            di = float(dlog_any[i_abs])
            dj = float(dlog_any[j_abs])
            if math.isfinite(di) and math.isfinite(dj):
                pair_res_dlog.append(abs(di - dj))
            ti = float(bestt_any[i_abs])
            tj = float(bestt_any[j_abs])
            if math.isfinite(ti) and math.isfinite(tj):
                pair_res_bestt.append(abs(ti - tj))

        uniq_n = int(uniq.size)
        possible_pairs = int(uniq_n // 2)
        frac = (float(pair_count) / float(possible_pairs)) if possible_pairs else float("nan")
        low = (pair_count < 1) or (math.isfinite(frac) and frac < 0.2)

        def _p(a: list[float], q: float) -> float:
            if not a:
                return float("nan")
            x = np.array(a, dtype=float)
            x = x[np.isfinite(x)]
            if x.size == 0:
                return float("nan")
            return float(np.nanpercentile(x, q))

        dlog_p50 = _p(pair_res_dlog, 50)
        dlog_p90 = _p(pair_res_dlog, 90)
        bestt_p50 = _p(pair_res_bestt, 50)
        bestt_p90 = _p(pair_res_bestt, 90)

        for i_abs in id_list:
            fam_pairs_any[i_abs] = float(pair_count)
            fam_pair_frac_any[i_abs] = float(frac) if math.isfinite(frac) else float("nan")
            fam_low_support_any[i_abs] = 1.0 if low else 0.0
            fam_dlog_p50_any[i_abs] = float(dlog_p50)
            fam_dlog_p90_any[i_abs] = float(dlog_p90)
            fam_bestt_p50_any[i_abs] = float(bestt_p50)
            fam_bestt_p90_any[i_abs] = float(bestt_p90)

    # Override unified sigmafam columns with the case/window-based version (works for all schemas).
    df["sigmafam_mirror_pairs"] = pd.Series(fam_pairs_any, index=df.index).fillna(0.0)
    df["sigmafam_mirror_pair_frac"] = pd.Series(fam_pair_frac_any, index=df.index).fillna(0.0)
    df["sigmafam_mirror_low_support"] = pd.Series(fam_low_support_any, index=df.index).fillna(1.0)
    df["sigmafam_mirror_dlog_p50"] = pd.Series(fam_dlog_p50_any, index=df.index)
    df["sigmafam_mirror_best_t_p50"] = pd.Series(fam_bestt_p50_any, index=df.index)

    shift_min: list[float] = []
    for _, r in df.iterrows():
        t0 = float(r.get("best_t_any_tier01", float("nan")))
        if not math.isfinite(t0):
            shift_min.append(float("nan"))
            continue
        sub = df[(df["window_id"] == r["window_id"]) & (df["case"] != r["case"])].copy()
        if sub.empty:
            shift_min.append(float("nan"))
            continue
        tt = pd.to_numeric(sub["best_t_any_tier01"], errors="coerce")
        tt = tt[np.isfinite(tt.to_numpy(dtype=float, na_value=np.nan))]
        if tt.empty:
            shift_min.append(float("nan"))
            continue
        shift_min.append(float(np.nanmin(np.abs(tt.to_numpy(dtype=float) - float(t0)))))

    df["best_t_shift_min"] = shift_min

    # --- Cheap divisor/zero stability proxy: best-t shift from N -> 2N ---
    # Parse primary rung size N and optional sigma from the case string.
    def _parse_primary_n(case: str) -> int | None:
        m = re.search(r"N(\d{3,})", str(case))
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def _parse_sigma(case: str) -> float | None:
        m = re.search(r"_sigma(\d+)p(\d+)", str(case))
        if not m:
            return None
        try:
            a = int(m.group(1))
            b = int(m.group(2))
            return float(f"{a}.{b:0{len(m.group(2))}d}")
        except Exception:
            return None

    def _parse_n2(case: str) -> int | None:
        # Common rung encoding: N2048_4096, N4096_8192, etc.
        m = re.search(r"N(\d{3,})_(\d{3,})", str(case))
        if not m:
            return None
        try:
            return int(m.group(2))
        except Exception:
            return None

    df["N_primary"] = df["case"].astype(str).map(_parse_primary_n)
    df["N2_primary"] = df["case"].astype(str).map(_parse_n2)
    df["sigma_primary"] = df["case"].astype(str).map(_parse_sigma)

    # Estimate window width for scaling (median across per-obs t ranges).
    tmin_cols = [c for c in df.columns if isinstance(c, str) and c.endswith("_t_min")]
    tmax_cols = [c for c in df.columns if isinstance(c, str) and c.endswith("_t_max")]

    def _row_window_width(r: pd.Series) -> float:
        widths: list[float] = []
        for cmin in tmin_cols:
            cmax = cmin[:-len("_t_min")] + "_t_max"
            if cmax not in r:
                continue
            vmin = float(r.get(cmin, float("nan")))
            vmax = float(r.get(cmax, float("nan")))
            if math.isfinite(vmin) and math.isfinite(vmax) and vmax >= vmin:
                widths.append(float(vmax - vmin))
        if not widths:
            return float("nan")
        return float(np.nanmedian(np.array(widths, dtype=float)))

    df["window_width"] = df.apply(_row_window_width, axis=1)

    shift_n2n: list[float] = []
    shift_scaled: list[float] = []
    shift_support: list[int] = []

    for _, r in df.iterrows():
        n0 = r.get("N_primary")
        n2 = r.get("N2_primary")
        t0 = float(r.get("best_t_any_tier01", float("nan")))
        if (n0 is None) or (not math.isfinite(float(t0))):
            shift_n2n.append(float("nan"))
            shift_scaled.append(float("nan"))
            shift_support.append(0)
            continue
        try:
            n0i = int(n0)
        except Exception:
            shift_n2n.append(float("nan"))
            shift_scaled.append(float("nan"))
            shift_support.append(0)
            continue
        if n0i <= 0:
            shift_n2n.append(float("nan"))
            shift_scaled.append(float("nan"))
            shift_support.append(0)
            continue

        target_n = None
        if n2 is not None:
            try:
                n2i = int(n2)
                if n2i > 0:
                    target_n = n2i
            except Exception:
                target_n = None
        if target_n is None:
            target_n = int(2 * n0i)

        sub = df[(df["divisor_family_id"] == r["divisor_family_id"]) & (df["case"] != r["case"])].copy()
        sub = sub[pd.to_numeric(sub.get("N_primary"), errors="coerce") == float(target_n)]
        if sub.empty:
            shift_n2n.append(float("nan"))
            shift_scaled.append(float("nan"))
            shift_support.append(0)
            continue

        # Prefer nearest sigma match when sigma is parsable.
        sig0 = r.get("sigma_primary")
        if sig0 is not None and math.isfinite(float(sig0)) and ("sigma_primary" in sub.columns):
            sig_sub = pd.to_numeric(sub["sigma_primary"], errors="coerce")
            sig_sub = sig_sub[np.isfinite(sig_sub.to_numpy(dtype=float, na_value=np.nan))]
            if not sig_sub.empty:
                j = int(np.nanargmin(np.abs(sig_sub.to_numpy(dtype=float) - float(sig0))))
                sub = sub.loc[[sig_sub.index[j]]]

        tt = pd.to_numeric(sub["best_t_any_tier01"], errors="coerce")
        tt = tt[np.isfinite(tt.to_numpy(dtype=float, na_value=np.nan))]
        if tt.empty:
            shift_n2n.append(float("nan"))
            shift_scaled.append(float("nan"))
            shift_support.append(0)
            continue

        sh = float(np.nanmin(np.abs(tt.to_numpy(dtype=float) - float(t0))))
        shift_n2n.append(float(sh))
        w = float(r.get("window_width", float("nan")))
        if math.isfinite(w) and w > 0:
            shift_scaled.append(float(sh) / float(w))
        else:
            shift_scaled.append(float("nan"))
        shift_support.append(1)

    df["divisor_best_t_shift_n2n"] = shift_n2n
    df["divisor_best_t_shift_scaled"] = shift_scaled
    df["divisor_shift_pair_count"] = shift_support
    df["divisor_low_support"] = (pd.Series(shift_support, index=df.index) < 1).fillna(True).astype(bool)

    # Build selection basket.
    # Pick representatives from different categories and a few contradictory ones.
    # Contradictory: at least 2 observables and strict_accept std > 0.
    df["contradictory"] = pd.to_numeric(df.get("family_disagree"), errors="coerce") > 0.0

    def _pick(df_sub: pd.DataFrame, k: int, *, sort_by: str | None = None, asc: bool = True) -> pd.DataFrame:
        if df_sub.empty:
            return df_sub
        if sort_by and (sort_by in df_sub.columns):
            return df_sub.sort_values(by=sort_by, ascending=asc).head(k)
        return df_sub.head(k)

    target = int(args.target_rows)

    picked: list[pd.DataFrame] = []

    k_dead = max(10, int(0.20 * target))
    k_near = max(10, int(0.18 * target))
    k_phase = max(10, int(0.18 * target))
    k_contract = max(6, int(0.10 * target))
    k_overlap = max(10, int(0.18 * target))
    k_contra = max(10, int(0.16 * target))

    # Zeta-support enrichers (cheap, and only used if present).
    # These are intentionally small quotas so we don't distort the basket.
    k_divisor = max(6, int(0.06 * target))
    k_mirror = max(6, int(0.06 * target))

    divisor_ok = (pd.to_numeric(df.get("divisor_shift_pair_count"), errors="coerce").fillna(0.0) >= 1.0)
    mirror_ok = (pd.to_numeric(df.get("sigmafam_mirror_pair_frac"), errors="coerce").fillna(0.0) > 0.0)

    picked.append(_pick(df[divisor_ok], k_divisor, sort_by="divisor_best_t_shift_scaled", asc=False))
    picked.append(_pick(df[mirror_ok], k_mirror, sort_by="sigmafam_mirror_pair_frac", asc=False))

    picked.append(_pick(df[df["label_provisional"].str.contains("dead")], k_dead, sort_by="tier01_frac_mean", asc=True))
    picked.append(_pick(df[df["label_provisional"].str.contains("near_miss")], k_near, sort_by="phase_block_score", asc=True))
    picked.append(_pick(df[df["label_provisional"].str.contains("phase_blocked")], k_phase, sort_by="diag_gain_sum", asc=False))
    picked.append(_pick(df[df["label_provisional"].str.contains("contraction_positive")], k_contract, sort_by="contract_best_improve", asc=True))
    picked.append(_pick(df[df["label_provisional"].str.contains("overlap_positive")], k_overlap, sort_by="strict_width_sum", asc=False))
    picked.append(_pick(df[df["contradictory"]], k_contra, sort_by="family_disagree", asc=False))

    basket = pd.concat(picked, ignore_index=True).drop_duplicates(subset=["case"]).copy()

    # If still short, top up with random remaining.
    if len(basket) < target:
        remaining = df[~df["case"].isin(set(basket["case"].tolist()))].copy()
        remaining = remaining.sample(n=min(target - len(basket), len(remaining)), random_state=int(args.seed)) if not remaining.empty else remaining
        basket = pd.concat([basket, remaining], ignore_index=True)

    # If too many, trim deterministically by keeping anchors and then by category diversity.
    if len(basket) > target:
        anchors_mask = basket["case"].str.lower().str.contains("refine164|pocket162")
        anchors_df = basket[anchors_mask]
        rest = basket[~anchors_mask]
        rest = rest.head(max(target - len(anchors_df), 0))
        basket = pd.concat([anchors_df, rest], ignore_index=True).head(target)

    # Add a "best_current" tag to the most structured row (by strict_width_sum + overlap + frontier).
    basket = basket.copy()

    # Provide a human-editable label column. For this fast/lean pass it's initialized
    # from rule-based tags; treat it as a starting point, not authoritative.
    basket["hand_label"] = basket["label_provisional"].astype(str)

    # Define score features.
    basket["fixed_gauge_num"] = basket["fixed_gauge_all"].fillna(False).astype(bool).astype(int)
    basket["contract_bonus"] = basket["contract_best_improve"].apply(lambda x: float(-x) if math.isfinite(float(x)) and float(x) < 0.0 else 0.0)
    basket["overlap_all3_n_num"] = pd.to_numeric(basket.get("overlap_all3_n"), errors="coerce").fillna(0.0)
    basket["overlap_all3_width_num"] = pd.to_numeric(basket.get("overlap_all3_width_max"), errors="coerce").fillna(0.0)

    # Treat missing signal as "bad" (not neutral) so dead/empty rows don't float
    # into the top ranks due to NaN->0 coercion in z-scoring.
    basket["tier01_frac_mean"] = pd.to_numeric(basket.get("tier01_frac_mean"), errors="coerce").fillna(0.0)
    basket["dead_penalty"] = (basket["tier01_frac_mean"] < 0.05).astype(int)

    # Zeta-structure proxy features (artifact-only):
    # - cross-observable best-t alignment (lower std is better)
    # - sigma-family mirror symmetry across related sigma-sweep artifacts (higher pairing support is better;
    #   smaller mirror residuals are better)

    sigmafam_pair_frac = pd.to_numeric(basket.get("sigmafam_mirror_pair_frac"), errors="coerce")
    basket["sigmafam_mirror_pair_frac"] = sigmafam_pair_frac.fillna(0.0)

    sigmafam_low_support = pd.to_numeric(basket.get("sigmafam_mirror_low_support"), errors="coerce")
    basket["sigmafam_mirror_low_support"] = sigmafam_low_support.fillna(1.0).astype(float)

    sigmafam_dlog_p50 = pd.to_numeric(basket.get("sigmafam_mirror_dlog_p50"), errors="coerce")
    if sigmafam_dlog_p50.notna().any():
        sigmafam_dlog_p50 = sigmafam_dlog_p50.fillna(float(sigmafam_dlog_p50.max()))
    else:
        sigmafam_dlog_p50 = sigmafam_dlog_p50.fillna(0.0)
    basket["sigmafam_mirror_dlog_p50"] = sigmafam_dlog_p50

    sigmafam_bestt_p50 = pd.to_numeric(basket.get("sigmafam_mirror_best_t_p50"), errors="coerce")
    if sigmafam_bestt_p50.notna().any():
        sigmafam_bestt_p50 = sigmafam_bestt_p50.fillna(float(sigmafam_bestt_p50.max()))
    else:
        sigmafam_bestt_p50 = sigmafam_bestt_p50.fillna(0.0)
    basket["sigmafam_mirror_best_t_p50"] = sigmafam_bestt_p50

    basket["best_t_obs_std"] = pd.to_numeric(basket.get("best_t_obs_std"), errors="coerce")
    if basket["best_t_obs_std"].notna().any():
        basket["best_t_obs_std"] = basket["best_t_obs_std"].fillna(float(basket["best_t_obs_std"].max()))
    else:
        basket["best_t_obs_std"] = basket["best_t_obs_std"].fillna(0.0)

    # Synthetic smoothness trap safeguard:
    # Penalize rows that look "very clean" on generic gates/acceptance/width but have no extra structural evidence
    # (no mirror/divisor support and no overlap/contract signal). This is meant as a control against
    # feature-impostors, and is intentionally basket-relative.
    def _med(col: str) -> float:
        s = pd.to_numeric(basket.get(col, pd.Series([0.0] * len(basket))), errors="coerce")
        return float(s.median()) if s.notna().any() else 0.0

    med_tier = _med("tier01_frac_mean")
    med_strict_w = _med("strict_width_sum")
    med_strict_a = _med("strict_accept_sum")

    _sw = pd.to_numeric(basket.get("strict_width_sum", 0.0), errors="coerce").fillna(0.0)
    _sa = pd.to_numeric(basket.get("strict_accept_sum", 0.0), errors="coerce").fillna(0.0)
    _t01 = pd.to_numeric(basket.get("tier01_frac_mean", 0.0), errors="coerce").fillna(0.0)
    _mir = pd.to_numeric(basket.get("sigmafam_mirror_pair_frac", 0.0), errors="coerce").fillna(0.0)
    _div = pd.to_numeric(basket.get("divisor_shift_pair_count", 0.0), errors="coerce").fillna(0.0)
    _ov = pd.to_numeric(basket.get("overlap_all3_n_num", 0.0), errors="coerce").fillna(0.0)
    _cb = pd.to_numeric(basket.get("contract_bonus", 0.0), errors="coerce").fillna(0.0)
    basket["smoothness_trap"] = (
        (_t01 > med_tier)
        & (_sw > med_strict_w)
        & (_sa > med_strict_a)
        & (_mir <= 0.0)
        & (_div < 1.0)
        & (_ov <= 0.0)
        & (_cb <= 0.0)
    ).fillna(False).astype(bool)

    # Structure score weights.
    w_struct = {
        "dead_penalty": -4.0,
        "tier01_frac_mean": 0.8,
        "strict_width_sum": 0.9,
        "strict_accept_sum": 0.6,
        "frontier_width_sum": 0.4,
        "frontier_accept_sum": 0.3,
        "overlap_all3_n_num": 0.7,
        "overlap_all3_width_num": 0.4,
        "contract_bonus": 0.6,
        "smoothness_trap": -2.5,
        "family_disagree": -0.6,
    }

    # Robustness score weights.
    w_rob = {
        "dead_penalty": -5.0,
        "fixed_gauge_num": 0.8,
        "tier01_frac_mean": 0.7,
        "strict_accept_sum": 0.4,
        "strict_width_sum": 0.2,
        "overlap_all3_n_num": 0.2,
        "phase_block_score": -0.9,
        "logabs_block_score": -0.5,
        "diag_gain_sum": -0.6,
        "smoothness_trap": -2.0,
        "family_disagree": -0.4,
    }

    # Zeta-structure proxy score: mirror symmetry + cross-observable alignment + contraction.
    w_zeta = {
        "dead_penalty": -4.0,
        "best_t_obs_std": -1.0,
        "sigmafam_mirror_pair_frac": 0.9,
        "sigmafam_mirror_low_support": -0.7,
        "sigmafam_mirror_dlog_p50": -0.4,
        "sigmafam_mirror_best_t_p50": -0.3,
        "divisor_best_t_shift_scaled": -1.5,
        "divisor_low_support": -1.0,
        "contract_bonus": 0.6,
        "fixed_gauge_num": 0.2,
        "smoothness_trap": -1.0,
    }

    basket["structure_score"] = _compute_scores(basket, weights=w_struct)
    basket["robustness_score"] = _compute_scores(basket, weights=w_rob)
    basket["zeta_structure_score"] = _compute_scores(basket, weights=w_zeta)

    # Mark best_current as the top by (structure+robustness).
    basket["combo"] = (
        pd.to_numeric(basket["structure_score"], errors="coerce").fillna(0.0)
        + pd.to_numeric(basket["robustness_score"], errors="coerce").fillna(0.0)
        + 0.5 * pd.to_numeric(basket["zeta_structure_score"], errors="coerce").fillna(0.0)
    )
    best_idx = basket["combo"].idxmax() if len(basket) else None
    if best_idx is not None:
        basket.loc[best_idx, "hand_label"] = basket.loc[best_idx, "hand_label"] + ";best_current"

    # Reports.
    perturb_struct = _weight_perturbation(basket, base_weights=w_struct, n=200, frac=0.2, seed=int(args.seed) + 1)
    perturb_rob = _weight_perturbation(basket, base_weights=w_rob, n=200, frac=0.2, seed=int(args.seed) + 2)
    perturb_zeta = _weight_perturbation(basket, base_weights=w_zeta, n=200, frac=0.2, seed=int(args.seed) + 3)

    blocks_struct = {
        "drop_overlap": ["overlap_all3_n_num", "overlap_all3_width_num"],
        "drop_frontier": ["frontier_width_sum", "frontier_accept_sum"],
        "drop_contract": ["contract_bonus"],
        "drop_gates": ["tier01_frac_mean"],
        "drop_dead_penalty": ["dead_penalty"],
    }
    blocks_rob = {
        "drop_fragility": ["phase_block_score", "logabs_block_score", "diag_gain_sum"],
        "drop_gauge": ["fixed_gauge_num"],
        "drop_agreement": ["overlap_all3_n_num", "family_disagree"],
        "drop_dead_penalty": ["dead_penalty"],
    }

    ablate_struct = _ablation(basket, base_weights=w_struct, blocks=blocks_struct)
    ablate_rob = _ablation(basket, base_weights=w_rob, blocks=blocks_rob)
    ablate_zeta = _ablation(
        basket,
        base_weights=w_zeta,
        blocks={
            "drop_best_t_align": ["best_t_obs_std"],
            "drop_mirror": ["sigmafam_mirror_pair_frac", "sigmafam_mirror_low_support", "sigmafam_mirror_dlog_p50", "sigmafam_mirror_best_t_p50"],
            "drop_divisor": ["divisor_best_t_shift_scaled", "divisor_low_support"],
            "drop_contract": ["contract_bonus"],
            "drop_dead_penalty": ["dead_penalty"],
        },
    )

    confusion_struct = _confusion_summary(basket, label_col="hand_label", score_col="structure_score")
    confusion_rob = _confusion_summary(basket, label_col="hand_label", score_col="robustness_score")
    confusion_zeta = _confusion_summary(basket, label_col="hand_label", score_col="zeta_structure_score")

    monotonicity = {
        "structure_weights": w_struct,
        "robustness_weights": w_rob,
        "zeta_structure_weights": w_zeta,
        "sanity": {
            "overlap_positive_weight": (w_struct["overlap_all3_n_num"] > 0 and w_rob["overlap_all3_n_num"] > 0),
            "contract_positive_weight": (w_struct["contract_bonus"] > 0),
            "phase_fragility_penalized": (w_rob["phase_block_score"] < 0),
            "gauge_bonus_positive": (w_rob["fixed_gauge_num"] > 0),
            "best_t_alignment_rewarded": (w_zeta["best_t_obs_std"] < 0),
            "sigmafam_mirror_pair_frac_rewarded": (w_zeta["sigmafam_mirror_pair_frac"] > 0),
            "sigmafam_mirror_low_support_penalized": (w_zeta["sigmafam_mirror_low_support"] < 0),
            "divisor_shift_penalized": (w_zeta["divisor_best_t_shift_scaled"] < 0),
            "divisor_low_support_penalized": (w_zeta["divisor_low_support"] < 0),
        },
    }

    # Conclusion heuristic: require reasonable perturbation stability + dead separation.
    selective = (
        (perturb_struct.get("spearman_rank_corr_p10", 0.0) >= 0.6)
        and (perturb_rob.get("spearman_rank_corr_p10", 0.0) >= 0.6)
        and (confusion_struct.get("dead_in_topk", 0) <= max(1, int(0.2 * confusion_struct.get("topk", 10))))
    )

    conclusion = "features are selective" if selective else "features are descriptive but not yet selective"

    payload = {
        "run_dir": str(run_dir),
        "scan_root": str(scan_root),
        "out_dir": str(out_dir),
        "n_groups": int(len(groups)),
        "n_scored": int(len(df)),
        "n_basket": int(len(basket)),
        "weights": {"structure": w_struct, "robustness": w_rob, "zeta_structure": w_zeta},
        "weight_perturbation": {"structure": perturb_struct, "robustness": perturb_rob, "zeta_structure": perturb_zeta},
        "ablation": {"structure": ablate_struct, "robustness": ablate_rob, "zeta_structure": ablate_zeta},
        "monotonicity": monotonicity,
        "confusion": {"structure": confusion_struct, "robustness": confusion_rob, "zeta_structure": confusion_zeta},
        "conclusion": conclusion,
    }

    # ---- Freeze selector spec (selector_v1) ----
    selector_spec = {
        "version": str(SELECTOR_VERSION),
        "gate": {
            "gate_tin_inv_max": float(args.gate_tin_inv_max),
            "gate_sep2_min": float(args.gate_sep2_min),
            "gate_lam_cost_max": float(args.gate_lam_cost_max),
        },
        "accept": {
            "STRICT": {"tau_logabs": float(STRICT.tau_logabs), "tau_arg": float(STRICT.tau_arg)},
            "DIAG": {"tau_logabs": float(DIAG.tau_logabs), "tau_arg": float(DIAG.tau_arg)},
            "FRONTIER_Q2TR": {"tau_logabs": float(FRONTIER_Q2TR.tau_logabs), "tau_arg": float(FRONTIER_Q2TR.tau_arg)},
            "FRONTIER_Q2": {"tau_logabs": float(FRONTIER_Q2.tau_logabs), "tau_arg": float(FRONTIER_Q2.tau_arg)},
            "FRONTIER_Q2PHI2": {"tau_logabs": float(FRONTIER_Q2PHI2.tau_logabs), "tau_arg": float(FRONTIER_Q2PHI2.tau_arg)},
        },
        "weights": {"structure": w_struct, "robustness": w_rob, "zeta_structure": w_zeta},
        "features": {
            "structure": list(w_struct.keys()),
            "robustness": list(w_rob.keys()),
            "zeta_structure": list(w_zeta.keys()),
        },
    }
    selector_hash = hashlib.sha256(json.dumps(selector_spec, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    selector_spec["hash12"] = selector_hash
    payload["selector"] = selector_spec

    out_prefix = str(args.out_prefix)
    out_csv = out_dir / f"{out_prefix}_eval.csv"
    out_json = out_dir / f"{out_prefix}_report.json"
    out_txt = out_dir / f"{out_prefix}_report.txt"
    out_blind = out_dir / f"{out_prefix}_blind_top.csv"
    out_digest = out_dir / f"{out_prefix}_case_digest_topbot.csv"
    out_digest_full = out_dir / f"{out_prefix}_case_digest_full.csv"
    out_top10_why = out_dir / f"{out_prefix}_top10_why.csv"
    out_regime = out_dir / f"{out_prefix}_support_regime_summary.csv"
    out_syn = out_dir / f"{out_prefix}_synthetic_smoothness_controls.csv"
    out_selector = out_dir / f"{out_prefix}_{SELECTOR_VERSION}_spec.json"
    out_audit = out_dir / f"{out_prefix}_zeta_proxy_audit.csv"
    out_redundancy = out_dir / f"{out_prefix}_zeta_proxy_redundancy.csv"

    # Write frozen selector spec now that output paths are known.
    out_selector.write_text(json.dumps(selector_spec, indent=2), encoding="utf-8")

    # ---- Zeta proxy audit + redundancy reports ----
    zeta_cols = list(w_zeta.keys())
    audit_rows: list[dict[str, Any]] = []
    for c in zeta_cols:
        present = c in basket.columns
        s = basket[c] if present else pd.Series([0.0] * len(basket), dtype=float)
        if s.dtype == bool:
            s_num = s.astype(float)
        else:
            s_num = pd.to_numeric(s, errors="coerce")
        vals = s_num.to_numpy(dtype=float, na_value=np.nan)
        finite = vals[np.isfinite(vals)]
        row = {
            "proxy": str(c),
            "weight": float(w_zeta.get(c, 0.0)),
            "present": bool(present),
            "n": int(len(vals)),
            "non_nan": int(np.isfinite(vals).sum()),
            "nunique": int(pd.Series(finite).nunique()) if finite.size else 0,
            "mean": float(np.nanmean(vals)) if finite.size else float("nan"),
            "std": float(np.nanstd(vals)) if finite.size else float("nan"),
            "min": float(np.nanmin(vals)) if finite.size else float("nan"),
            "p25": float(np.nanpercentile(vals, 25)) if finite.size else float("nan"),
            "p50": float(np.nanpercentile(vals, 50)) if finite.size else float("nan"),
            "p75": float(np.nanpercentile(vals, 75)) if finite.size else float("nan"),
            "iqr": float(np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25)) if finite.size else float("nan"),
            "max": float(np.nanmax(vals)) if finite.size else float("nan"),
        }
        audit_rows.append(row)
    pd.DataFrame(audit_rows).to_csv(out_audit, index=False)

    red_rows: list[dict[str, Any]] = []
    for c in zeta_cols:
        if c not in basket.columns:
            red_rows.append({
                "proxy": str(c),
                "weight": float(w_zeta.get(c, 0.0)),
                "spearman_vs_structure": float("nan"),
                "spearman_vs_robustness": float("nan"),
                "spearman_vs_zeta": float("nan"),
            })
            continue
        s = basket[c]
        s_num = s.astype(float) if s.dtype == bool else pd.to_numeric(s, errors="coerce")
        red_rows.append({
            "proxy": str(c),
            "weight": float(w_zeta.get(c, 0.0)),
            "spearman_vs_structure": float(pd.Series(s_num).corr(pd.Series(basket["structure_score"]), method="spearman")),
            "spearman_vs_robustness": float(pd.Series(s_num).corr(pd.Series(basket["robustness_score"]), method="spearman")),
            "spearman_vs_zeta": float(pd.Series(s_num).corr(pd.Series(basket["zeta_structure_score"]), method="spearman")),
        })
    red_df = pd.DataFrame(red_rows)
    red_df.to_csv(out_redundancy, index=False)
    payload["zeta_proxy_redundancy"] = red_rows

    basket.to_csv(out_csv, index=False)

    # Blind ranking view: top rows by each score without labels.
    blind_cols = ["case", "structure_score", "robustness_score", "zeta_structure_score", "combo"]
    blind = basket.copy()
    for c in ["hand_label", "label_provisional"]:
        if c in blind.columns:
            blind = blind.drop(columns=[c])
    out_blind_df = pd.concat(
        [
            blind.sort_values("structure_score", ascending=False)[blind_cols].head(20).assign(rank_by="structure"),
            blind.sort_values("robustness_score", ascending=False)[blind_cols].head(20).assign(rank_by="robustness"),
            blind.sort_values("zeta_structure_score", ascending=False)[blind_cols].head(20).assign(rank_by="zeta_structure"),
        ],
        ignore_index=True,
    )
    out_blind_df.to_csv(out_blind, index=False)

    # ---- Support-regime analysis + top/bottom digest ----
    def _num(s: pd.Series) -> pd.Series:
        if s.dtype == bool:
            return s.astype(float)
        return pd.to_numeric(s, errors="coerce")

    mirror_active = _num(basket.get("sigmafam_mirror_pair_frac", pd.Series([np.nan] * len(basket)))).fillna(0.0) > 0.0
    divisor_active = (
        _num(basket.get("divisor_shift_pair_count", pd.Series([0.0] * len(basket)))).fillna(0.0) >= 1.0
    ) & (
        _num(basket.get("divisor_best_t_shift_scaled", pd.Series([np.nan] * len(basket)))).notna()
    )
    best_t_align_active = _num(basket.get("best_t_obs_std", pd.Series([np.nan] * len(basket)))).notna()

    # Lightweight per-row evidence/support scores.
    overlap_active = _num(basket.get("overlap_all3_n_num", pd.Series([0.0] * len(basket)))).fillna(0.0) > 0.0
    contract_active = _num(basket.get("contract_bonus", pd.Series([0.0] * len(basket)))).fillna(0.0) > 0.0
    fixed_gauge_active = _num(basket.get("fixed_gauge_num", pd.Series([0.0] * len(basket)))).fillna(0.0) > 0.0
    zeta_support_count = (mirror_active.astype(int) + divisor_active.astype(int)).astype(int)
    feature_support_score = (
        mirror_active.astype(int)
        + divisor_active.astype(int)
        + overlap_active.astype(int)
        + contract_active.astype(int)
        + fixed_gauge_active.astype(int)
    ).astype(int)

    # A disjoint label that helps quick audits.
    support_regime = np.where(
        divisor_active & mirror_active,
        "divisor+mirror",
        np.where(
            divisor_active,
            "divisor-only",
            np.where(mirror_active, "mirror-only", np.where(best_t_align_active, "best_t-only", "none")),
        ),
    )

    def _rank_desc(v: pd.Series) -> pd.Series:
        return (-pd.to_numeric(v, errors="coerce").fillna(float("-inf"))).rank(method="min")

    # ---- Anchor-collapse audit (mirror anchor) ----
    # Core question: is the selector more than its strongest anchor?
    anchor_name = "mirror"
    anchor_mask = mirror_active.astype(bool)
    mirror_keys = [
        "sigmafam_mirror_pair_frac",
        "sigmafam_mirror_low_support",
        "sigmafam_mirror_dlog_p50",
        "sigmafam_mirror_best_t_p50",
    ]

    def _label_mix_top(sub: pd.DataFrame, k: int) -> str:
        if sub.empty or ("hand_label" not in sub.columns):
            return ""
        vc = sub["hand_label"].astype(str).value_counts().head(k)
        return "; ".join([f"{lbl}={int(n)}" for lbl, n in vc.items()])

    anchor_audit: dict[str, Any] = {"anchor": anchor_name, "n_anchor_active": int(anchor_mask.sum())}

    # (1) Within-anchor ranking separation.
    sub_anchor = basket.loc[anchor_mask].copy()
    if sub_anchor.empty:
        anchor_audit["within_anchor"] = {
            "n": 0,
            "combo_p10": float("nan"),
            "combo_p50": float("nan"),
            "combo_p90": float("nan"),
            "combo_min": float("nan"),
            "combo_max": float("nan"),
            "combo_spread": float("nan"),
            "top5_label_mix": "",
            "bottom5_label_mix": "",
        }
    else:
        combo_a = pd.to_numeric(sub_anchor.get("combo", np.nan), errors="coerce")
        top5 = sub_anchor.sort_values("combo", ascending=False).head(min(5, len(sub_anchor)))
        bot5 = sub_anchor.sort_values("combo", ascending=True).head(min(5, len(sub_anchor)))
        anchor_audit["within_anchor"] = {
            "n": int(len(sub_anchor)),
            "combo_p10": float(combo_a.quantile(0.10)) if combo_a.notna().any() else float("nan"),
            "combo_p50": float(combo_a.quantile(0.50)) if combo_a.notna().any() else float("nan"),
            "combo_p90": float(combo_a.quantile(0.90)) if combo_a.notna().any() else float("nan"),
            "combo_min": float(combo_a.min()) if combo_a.notna().any() else float("nan"),
            "combo_max": float(combo_a.max()) if combo_a.notna().any() else float("nan"),
            "combo_spread": float(combo_a.max() - combo_a.min()) if combo_a.notna().any() else float("nan"),
            "top5_label_mix": _label_mix_top(top5, 6),
            "bottom5_label_mix": _label_mix_top(bot5, 6),
        }

    # (3) Anchor ablation impact: drop mirror block from zeta and recompute combo.
    w_zeta_no_anchor = dict(w_zeta)
    for k in mirror_keys:
        w_zeta_no_anchor.pop(k, None)
    zeta_no_anchor = _compute_scores(basket, weights=w_zeta_no_anchor)
    combo_no_anchor = (
        pd.to_numeric(basket.get("structure_score", 0.0), errors="coerce").fillna(0.0)
        + pd.to_numeric(basket.get("robustness_score", 0.0), errors="coerce").fillna(0.0)
        + 0.5 * pd.to_numeric(zeta_no_anchor, errors="coerce").fillna(0.0)
    )
    base_rank = pd.to_numeric(basket.get("combo", np.nan), errors="coerce").rank(method="average", ascending=False)
    abl_rank = pd.to_numeric(combo_no_anchor, errors="coerce").rank(method="average", ascending=False)
    spearman = float(pd.Series(base_rank).corr(pd.Series(abl_rank), method="spearman"))
    topk = min(10, len(basket))
    base_top = set(basket.sort_values("combo", ascending=False).head(topk)["case"].astype(str).tolist())
    abl_df = basket.copy()
    abl_df["combo_no_anchor"] = combo_no_anchor
    abl_top = set(abl_df.sort_values("combo_no_anchor", ascending=False).head(topk)["case"].astype(str).tolist())
    top10_overlap = float(len(base_top & abl_top) / max(len(base_top | abl_top), 1))
    abl_conf = _confusion_summary(abl_df, label_col="hand_label", score_col="combo_no_anchor")
    anchor_audit["anchor_ablation"] = {
        "dropped": mirror_keys,
        "spearman_rank_corr": spearman,
        "top10_overlap_jacc": top10_overlap,
        "dead_in_top10": int(abl_conf.get("dead_in_topk", 0)),
        "top10_label_counts": dict(abl_conf.get("topk_label_counts", {})),
    }

    # (4) Within-anchor contribution audit: do non-anchor pieces still matter when anchor is present?
    def _contrib_df(df_: pd.DataFrame, *, weights: dict[str, float]) -> pd.DataFrame:
        contrib: dict[str, np.ndarray] = {}
        for k, w in weights.items():
            if k not in df_.columns:
                contrib[k] = np.zeros(len(df_), dtype=float)
                continue
            if df_[k].dtype == bool:
                x = df_[k].astype(float).to_numpy()
            else:
                x = pd.to_numeric(df_[k], errors="coerce").to_numpy(dtype=float)
                x = _zscore(x)
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            contrib[k] = float(w) * x
        return pd.DataFrame(contrib, index=df_.index)

    zeta_contrib = _contrib_df(basket, weights=w_zeta)
    zeta_mirror_sum = zeta_contrib[[k for k in mirror_keys if k in zeta_contrib.columns]].sum(axis=1) if len(zeta_contrib) else pd.Series([], dtype=float)
    zeta_nonmirror_sum = zeta_contrib.drop(columns=[k for k in mirror_keys if k in zeta_contrib.columns], errors="ignore").sum(axis=1) if len(zeta_contrib) else pd.Series([], dtype=float)

    if sub_anchor.empty:
        anchor_audit["within_anchor_contrib"] = {
            "n": 0,
            "mean_latent_consistency": float("nan"),
            "mean_refinement_robustness": float("nan"),
            "mean_symmetry_spectral_prior": float("nan"),
            "mean_zeta_mirror_contrib": float("nan"),
            "mean_zeta_nonmirror_contrib": float("nan"),
        }
    else:
        idx = sub_anchor.index
        anchor_audit["within_anchor_contrib"] = {
            "n": int(len(idx)),
            "mean_latent_consistency": float(pd.to_numeric(basket.loc[idx, "structure_score"], errors="coerce").mean()),
            "mean_refinement_robustness": float(pd.to_numeric(basket.loc[idx, "robustness_score"], errors="coerce").mean()),
            "mean_symmetry_spectral_prior": float(pd.to_numeric(basket.loc[idx, "zeta_structure_score"], errors="coerce").mean()),
            "mean_zeta_mirror_contrib": float(pd.to_numeric(zeta_mirror_sum.loc[idx], errors="coerce").mean()),
            "mean_zeta_nonmirror_contrib": float(pd.to_numeric(zeta_nonmirror_sum.loc[idx], errors="coerce").mean()),
        }

    def _standardize_against_ref(
        target: pd.DataFrame,
        *,
        weights: dict[str, float],
        ref: pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        # Standardize each feature in `target` using mean/std computed on `ref`.
        feat: dict[str, np.ndarray] = {}
        for k in weights.keys():
            if k not in target.columns:
                feat[k] = np.zeros(len(target), dtype=float)
                continue
            if target[k].dtype == bool:
                feat[k] = target[k].astype(float).to_numpy()
                continue

            ref_s = pd.to_numeric(ref.get(k, pd.Series([0.0] * len(ref))), errors="coerce").to_numpy(dtype=float)
            m = float(np.nanmean(ref_s))
            s = float(np.nanstd(ref_s))
            if (not math.isfinite(s)) or s <= 0.0:
                feat[k] = np.zeros(len(target), dtype=float)
                continue

            tgt_s = pd.to_numeric(target[k], errors="coerce").to_numpy(dtype=float)
            z = (tgt_s - m) / s
            z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            feat[k] = z
        return feat

    def _score_and_contrib(
        target: pd.DataFrame,
        *,
        weights: dict[str, float],
        ref: pd.DataFrame,
    ) -> tuple[pd.Series, pd.DataFrame]:
        feat = _standardize_against_ref(target, weights=weights, ref=ref)
        score = np.zeros(len(target), dtype=float)
        contrib = {}
        for k, w in weights.items():
            c = float(w) * feat[k]
            contrib[k] = c
            score += c
        return pd.Series(score, index=target.index), pd.DataFrame(contrib, index=target.index)

    def _topk_contrib_str(contrib_row: pd.Series, *, k: int = 4) -> str:
        # Prefer positive contributions for a "why is it high" view; if none, fall back to most negative.
        s = contrib_row.copy()
        s = s[pd.to_numeric(s, errors="coerce").notna()]
        if s.empty:
            return ""
        s_num = pd.to_numeric(s, errors="coerce")
        pos = s_num[s_num > 0].sort_values(ascending=False)
        pick = pos.head(k) if not pos.empty else s_num.sort_values(ascending=True).head(k)
        parts: list[str] = []
        for name, val in pick.items():
            parts.append(f"{name}={float(val):+.3f}")
        return "; ".join(parts)

    # Digest: top/bottom by combo, with active subfeature flags.
    digest_cols: list[str] = [
        "case",
        "hand_label",
        "structure_score",
        "robustness_score",
        "zeta_structure_score",
        "combo",
        "rank_combo",
        "rank_structure",
        "rank_robustness",
        "rank_zeta",
        "mirror_active",
        "best_t_align_active",
        "divisor_active",
        "smoothness_trap",
        "zeta_support_count",
        "feature_support_score",
        "zeta_support_regime",
        "sigmafam_mirror_pair_frac",
        "best_t_obs_std",
        "divisor_shift_pair_count",
        "divisor_best_t_shift_scaled",
        "divisor_low_support",
    ]
    digest = basket.copy()
    if "hand_label" not in digest.columns:
        digest["hand_label"] = np.nan
    digest["rank_combo"] = _rank_desc(digest["combo"])
    digest["rank_structure"] = _rank_desc(digest["structure_score"])
    digest["rank_robustness"] = _rank_desc(digest["robustness_score"])
    digest["rank_zeta"] = _rank_desc(digest["zeta_structure_score"])
    digest["mirror_active"] = mirror_active
    digest["best_t_align_active"] = best_t_align_active
    digest["divisor_active"] = divisor_active
    digest["zeta_support_regime"] = support_regime
    digest["zeta_support_count"] = zeta_support_count
    digest["feature_support_score"] = feature_support_score

    for c in [
        "sigmafam_mirror_pair_frac",
        "best_t_obs_std",
        "divisor_shift_pair_count",
        "divisor_best_t_shift_scaled",
        "divisor_low_support",
    ]:
        if c not in digest.columns:
            digest[c] = np.nan

    digest_top = digest.sort_values("combo", ascending=False).head(10)
    digest_bot = digest.sort_values("combo", ascending=True).head(10)
    out_digest_df = pd.concat(
        [
            digest_top.assign(which="top10"),
            digest_bot.assign(which="bottom10"),
        ],
        ignore_index=True,
    )
    keep = [c for c in (digest_cols + ["which"]) if c in out_digest_df.columns]
    out_digest_df[keep].to_csv(out_digest, index=False)

    # Full sortable digest (all basket rows).
    digest_full_keep = [c for c in digest_cols if c in digest.columns]
    digest[digest_full_keep].sort_values("rank_combo", ascending=True).to_csv(out_digest_full, index=False)

    # Top-10 "why" breakdown: score block dominance + top feature contributions.
    top10 = digest.sort_values("combo", ascending=False).head(10).copy()
    struct_score, struct_contrib = _score_and_contrib(top10, weights=w_struct, ref=basket)
    rob_score, rob_contrib = _score_and_contrib(top10, weights=w_rob, ref=basket)
    zeta_score, zeta_contrib = _score_and_contrib(top10, weights=w_zeta, ref=basket)

    top10["structure_score_recalc"] = struct_score
    top10["robustness_score_recalc"] = rob_score
    top10["zeta_structure_score_recalc"] = zeta_score
    top10["combo_recalc"] = (
        pd.to_numeric(top10["structure_score_recalc"], errors="coerce").fillna(0.0)
        + pd.to_numeric(top10["robustness_score_recalc"], errors="coerce").fillna(0.0)
        + 0.5 * pd.to_numeric(top10["zeta_structure_score_recalc"], errors="coerce").fillna(0.0)
    )

    top10["combo_struct_part"] = pd.to_numeric(top10["structure_score_recalc"], errors="coerce").fillna(0.0)
    top10["combo_rob_part"] = pd.to_numeric(top10["robustness_score_recalc"], errors="coerce").fillna(0.0)
    top10["combo_zeta_part"] = 0.5 * pd.to_numeric(top10["zeta_structure_score_recalc"], errors="coerce").fillna(0.0)
    top10["dominant_block"] = top10[["combo_struct_part", "combo_rob_part", "combo_zeta_part"]].idxmax(axis=1)

    top10["why_structure_top"] = [
        _topk_contrib_str(struct_contrib.loc[idx]) for idx in top10.index
    ]
    top10["why_robustness_top"] = [
        _topk_contrib_str(rob_contrib.loc[idx]) for idx in top10.index
    ]
    top10["why_zeta_top"] = [
        _topk_contrib_str(zeta_contrib.loc[idx]) for idx in top10.index
    ]
    top10_out_cols = [
        "case",
        "hand_label",
        "rank_combo",
        "structure_score",
        "robustness_score",
        "zeta_structure_score",
        "combo",
        "mirror_active",
        "best_t_align_active",
        "divisor_active",
        "zeta_support_regime",
        "dominant_block",
        "combo_struct_part",
        "combo_rob_part",
        "combo_zeta_part",
        "why_structure_top",
        "why_robustness_top",
        "why_zeta_top",
    ]
    top10[top10_out_cols].to_csv(out_top10_why, index=False)

    # Regime summary stats.
    def _label_mix(sub: pd.DataFrame) -> str:
        if sub.empty or ("hand_label" not in sub.columns):
            return ""
        vc = sub["hand_label"].astype(str).value_counts().head(4)
        return "; ".join([f"{k}={int(v)}" for k, v in vc.items()])

    top_quartile_thr = float(np.nanpercentile(pd.to_numeric(digest["rank_combo"], errors="coerce"), 25))
    in_top_quartile = pd.to_numeric(digest["rank_combo"], errors="coerce") <= top_quartile_thr

    regime_rows: list[dict[str, Any]] = []
    for name, mask in [
        ("mirror_available", mirror_active),
        ("divisor_available", divisor_active),
        ("both_unavailable", (~mirror_active) & (~divisor_active)),
        ("only_best_t_active", best_t_align_active & (~mirror_active) & (~divisor_active)),
    ]:
        sub = digest.loc[mask].copy()
        if sub.empty:
            regime_rows.append({
                "regime": name,
                "n": 0,
                "avg_rank_combo": float("nan"),
                "median_zeta": float("nan"),
                "frac_top_quartile": float("nan"),
                "label_mix": "",
            })
            continue
        regime_rows.append({
            "regime": name,
            "n": int(len(sub)),
            "avg_rank_combo": float(pd.to_numeric(sub["rank_combo"], errors="coerce").mean()),
            "median_zeta": float(pd.to_numeric(sub["zeta_structure_score"], errors="coerce").median()),
            "frac_top_quartile": float(in_top_quartile.loc[sub.index].mean()),
            "label_mix": _label_mix(sub),
        })
    pd.DataFrame(regime_rows).to_csv(out_regime, index=False)

    # Synthetic smoothness control evaluation (optional): scored against basket normalization.
    syn_payload: dict[str, Any] | None = None
    if int(getattr(args, "synthetic_smoothness_controls", 0)):
        def _q(col: str, qv: float, default: float = 0.0) -> float:
            if col not in basket.columns:
                return float(default)
            s = pd.to_numeric(basket[col], errors="coerce")
            if s.notna().any():
                return float(s.quantile(qv))
            return float(default)

        base = {
            "hand_label": "synthetic_impostor",
            "label_provisional": "synthetic_impostor",
            "fixed_gauge_all": True,
            "fixed_gauge_num": 1,
            "dead_penalty": 0,
            "overlap_all3_n_num": 0.0,
            "overlap_all3_width_num": 0.0,
            "contract_bonus": 0.0,
            "family_disagree": 0.0,
            "phase_block_score": 0.0,
            "logabs_block_score": 0.0,
            "diag_gain_sum": 0.0,
            "sigmafam_mirror_pair_frac": 0.0,
            "sigmafam_mirror_low_support": 1.0,
            "sigmafam_mirror_dlog_p50": _q("sigmafam_mirror_dlog_p50", 0.50, default=0.0),
            "sigmafam_mirror_best_t_p50": _q("sigmafam_mirror_best_t_p50", 0.50, default=0.0),
            "divisor_shift_pair_count": 0.0,
            "divisor_best_t_shift_scaled": np.nan,
            "divisor_low_support": 1.0,
            "smoothness_trap": False,
        }

        def _mk(case: str, kind: str, suite: str, **kw: Any) -> dict[str, Any]:
            r = dict(base)
            r.update({
                "case": case,
                "synthetic_kind": kind,
                "synthetic_suite": suite,
            })
            r.update(kw)
            return r

        controls: list[dict[str, Any]] = []

        # 1) Smoothness trap (easy impostor): should be suppressed by smoothness_trap.
        controls.append(_mk(
            "__synthetic_smooth_trap_1",
            "smooth_trap",
            "plausible",
            tier01_frac_mean=_q("tier01_frac_mean", 0.90, default=0.20),
            strict_width_sum=_q("strict_width_sum", 0.90, default=0.0),
            strict_accept_sum=_q("strict_accept_sum", 0.90, default=0.0),
            frontier_width_sum=_q("frontier_width_sum", 0.80, default=0.0),
            frontier_accept_sum=_q("frontier_accept_sum", 0.80, default=0.0),
            best_t_obs_std=_q("best_t_obs_std", 0.50, default=_q("best_t_obs_std", 0.90, default=0.0)),
            smoothness_trap=True,
        ))

        # 2) Fake best-t consistency (harder): very low best_t_obs_std but not triggering smoothness_trap.
        controls.append(_mk(
            "__synthetic_fake_bestt_1",
            "fake_best_t",
            "plausible",
            tier01_frac_mean=_q("tier01_frac_mean", 0.75, default=0.20),
            strict_width_sum=_q("strict_width_sum", 0.55, default=0.0),
            strict_accept_sum=_q("strict_accept_sum", 0.55, default=0.0),
            frontier_width_sum=_q("frontier_width_sum", 0.55, default=0.0),
            frontier_accept_sum=_q("frontier_accept_sum", 0.55, default=0.0),
            best_t_obs_std=_q("best_t_obs_std", 0.05, default=0.0),
        ))

        # 3) Mirror-like impostor: turns on mirror support but with middling residual quality.
        controls.append(_mk(
            "__synthetic_mirrorish_1",
            "mirrorish",
            "adversarial_fabricated_evidence",
            tier01_frac_mean=_q("tier01_frac_mean", 0.75, default=0.20),
            strict_width_sum=_q("strict_width_sum", 0.55, default=0.0),
            strict_accept_sum=_q("strict_accept_sum", 0.55, default=0.0),
            best_t_obs_std=_q("best_t_obs_std", 0.25, default=0.0),
            sigmafam_mirror_pair_frac=0.5,
            sigmafam_mirror_low_support=0.0,
            sigmafam_mirror_dlog_p50=_q("sigmafam_mirror_dlog_p50", 0.60, default=0.0),
            sigmafam_mirror_best_t_p50=_q("sigmafam_mirror_best_t_p50", 0.60, default=0.0),
        ))

        # 4) Divisor-like impostor: claims divisor support with zero shift.
        controls.append(_mk(
            "__synthetic_divisorish_1",
            "divisorish",
            "adversarial_fabricated_evidence",
            tier01_frac_mean=_q("tier01_frac_mean", 0.75, default=0.20),
            strict_width_sum=_q("strict_width_sum", 0.55, default=0.0),
            strict_accept_sum=_q("strict_accept_sum", 0.55, default=0.0),
            best_t_obs_std=_q("best_t_obs_std", 0.25, default=0.0),
            divisor_shift_pair_count=1.0,
            divisor_best_t_shift_scaled=0.0,
            divisor_low_support=0.0,
        ))

        # 5) Mild oscillatory / phase variation: slight fragility penalties but otherwise smooth.
        controls.append(_mk(
            "__synthetic_mild_phase_1",
            "mild_phase",
            "plausible",
            tier01_frac_mean=_q("tier01_frac_mean", 0.75, default=0.20),
            strict_width_sum=_q("strict_width_sum", 0.55, default=0.0),
            strict_accept_sum=_q("strict_accept_sum", 0.55, default=0.0),
            best_t_obs_std=_q("best_t_obs_std", 0.25, default=0.0),
            phase_block_score=_q("phase_block_score", 0.70, default=0.2),
        ))

        # 6) Plausible hard impostor: strong Tier01/width + very consistent best-t, with mild phase variation.
        # Intended to test whether the selector rewards "looks-structured" even when zeta support is weak.
        controls.append(_mk(
            "__synthetic_hard_combo_1",
            "goodwidth_bestt_mildphase",
            "plausible_hard",
            tier01_frac_mean=_q("tier01_frac_mean", 0.85, default=0.20),
            strict_width_sum=_q("strict_width_sum", 0.85, default=0.0),
            strict_accept_sum=_q("strict_accept_sum", 0.85, default=0.0),
            frontier_width_sum=_q("frontier_width_sum", 0.70, default=0.0),
            frontier_accept_sum=_q("frontier_accept_sum", 0.70, default=0.0),
            best_t_obs_std=_q("best_t_obs_std", 0.10, default=0.0),
            phase_block_score=_q("phase_block_score", 0.65, default=0.15),
        ))

        # 7) "One zeta-like feature only" impostor (mirror-ish): turns on mirror pair fraction but keeps everything else generic.
        controls.append(_mk(
            "__synthetic_onezeta_mirror_1",
            "one_zeta_only_mirror",
            "plausible_hard",
            tier01_frac_mean=_q("tier01_frac_mean", 0.75, default=0.20),
            strict_width_sum=_q("strict_width_sum", 0.55, default=0.0),
            strict_accept_sum=_q("strict_accept_sum", 0.55, default=0.0),
            best_t_obs_std=_q("best_t_obs_std", 0.25, default=0.0),
            sigmafam_mirror_pair_frac=0.35,
            sigmafam_mirror_low_support=0.0,
        ))

        syn = pd.DataFrame(controls)
        syn_struct, syn_struct_c = _score_and_contrib(syn, weights=w_struct, ref=basket)
        syn_rob, syn_rob_c = _score_and_contrib(syn, weights=w_rob, ref=basket)
        syn_zeta, syn_zeta_c = _score_and_contrib(syn, weights=w_zeta, ref=basket)
        syn["structure_score"] = syn_struct
        syn["robustness_score"] = syn_rob
        syn["zeta_structure_score"] = syn_zeta
        syn["why_structure"] = syn_struct_c.apply(_topk_contrib_str, axis=1)
        syn["why_robustness"] = syn_rob_c.apply(_topk_contrib_str, axis=1)
        syn["why_zeta"] = syn_zeta_c.apply(_topk_contrib_str, axis=1)
        syn["combo"] = (
            pd.to_numeric(syn["structure_score"], errors="coerce").fillna(0.0)
            + pd.to_numeric(syn["robustness_score"], errors="coerce").fillna(0.0)
            + 0.5 * pd.to_numeric(syn["zeta_structure_score"], errors="coerce").fillna(0.0)
        )

        # Attribution: which component dominated the combo for this control.
        s_part = pd.to_numeric(syn["structure_score"], errors="coerce").fillna(0.0)
        r_part = pd.to_numeric(syn["robustness_score"], errors="coerce").fillna(0.0)
        z_part = 0.5 * pd.to_numeric(syn["zeta_structure_score"], errors="coerce").fillna(0.0)
        dom = []
        for a, b, c in zip(s_part.tolist(), r_part.tolist(), z_part.tolist()):
            if (a >= b) and (a >= c):
                dom.append("latent_consistency")
            elif (b >= a) and (b >= c):
                dom.append("refinement_robustness")
            else:
                dom.append("symmetry_spectral_prior")
        syn["promoting_component"] = dom

        syn_suite = syn.get("synthetic_suite", pd.Series([""] * len(syn))).astype(str)
        plausible_mask = syn_suite.str.contains("plausible", case=False, na=False)

        basket_combo = pd.to_numeric(basket["combo"], errors="coerce")
        syn_rank = []
        syn_pct = []
        for v in pd.to_numeric(syn["combo"], errors="coerce").tolist():
            better = int((basket_combo > float(v)).sum())
            rank = better + 1
            syn_rank.append(rank)
            syn_pct.append(float(better) / max(len(basket_combo), 1))
        syn["implied_rank_combo_vs_basket"] = syn_rank
        syn["implied_percentile_vs_basket"] = syn_pct

        # Anchor-matched synthetic audit: implied rank vs anchor-active subset only.
        anchor_combo = basket_combo.loc[anchor_mask] if hasattr(basket_combo, "loc") else basket_combo
        anchor_combo = pd.to_numeric(anchor_combo, errors="coerce")
        syn_rank_a = []
        syn_pct_a = []
        if int(anchor_combo.notna().sum()) == 0:
            syn_rank_a = [None for _ in range(len(syn))]
            syn_pct_a = [None for _ in range(len(syn))]
        else:
            denom = int(anchor_combo.notna().sum())
            for v in pd.to_numeric(syn["combo"], errors="coerce").tolist():
                better = int((anchor_combo > float(v)).sum())
                rank = better + 1
                syn_rank_a.append(rank)
                syn_pct_a.append(float(better) / max(denom, 1))
        syn["anchor_subset_n"] = int(anchor_mask.sum())
        syn["implied_rank_combo_vs_anchor"] = syn_rank_a
        syn["implied_percentile_vs_anchor"] = syn_pct_a

        # Summarize plausible-only anchor-matched outcomes.
        syn_rank_a_s = pd.Series(syn_rank_a, index=syn.index, dtype="object")
        plausible_anchor_ranks = syn_rank_a_s.loc[plausible_mask]
        anchor_audit["synthetic_anchor_matched"] = {
            "anchor": anchor_name,
            "anchor_subset_n": int(anchor_mask.sum()),
            "n_controls": int(len(syn)),
            "plausible_only": {
                "n": int(plausible_mask.sum()),
                "any_in_top10": bool((pd.to_numeric(plausible_anchor_ranks, errors="coerce") <= 10).any()) if len(plausible_anchor_ranks) else False,
                "n_in_top10": int((pd.to_numeric(plausible_anchor_ranks, errors="coerce") <= 10).sum()) if len(plausible_anchor_ranks) else 0,
                "min_rank": int(pd.to_numeric(plausible_anchor_ranks, errors="coerce").min()) if len(plausible_anchor_ranks) else None,
            },
        }
        syn.to_csv(out_syn, index=False)
        syn_rank_s = pd.Series(syn_rank, index=syn.index, dtype=int)
        plausible_ranks = syn_rank_s.loc[plausible_mask]

        syn_payload = {
            "n": int(len(syn)),
            "min_rank": int(min(syn_rank)) if syn_rank else None,
            "any_in_top10": bool(any(r <= 10 for r in syn_rank)),
            "n_in_top10": int(sum(1 for r in syn_rank if r <= 10)),
            "plausible": {
                "n": int(plausible_mask.sum()),
                "any_in_top10": bool((plausible_ranks <= 10).any()) if len(plausible_ranks) else False,
                "n_in_top10": int((plausible_ranks <= 10).sum()) if len(plausible_ranks) else 0,
                "min_rank": int(plausible_ranks.min()) if len(plausible_ranks) else None,
            },
            "rows": syn[[
                "case",
                "synthetic_kind",
                "synthetic_suite",
                "combo",
                "implied_rank_combo_vs_basket",
                "implied_percentile_vs_basket",
                "anchor_subset_n",
                "implied_rank_combo_vs_anchor",
                "implied_percentile_vs_anchor",
                "promoting_component",
                "why_structure",
                "why_robustness",
                "why_zeta",
            ]].to_dict(orient="records"),
        }

    lines = []
    lines.append(f"Selector: {SELECTOR_VERSION} hash={selector_hash}")
    lines.append(f"Basket rows: {len(basket)} (scored groups: {len(df)}; discovered groups: {len(groups)})")
    lines.append(f"Conclusion: {conclusion}")
    lines.append("")
    lines.append("Weight perturbation (±20%)")
    lines.append(f"  structure: spearman p10={payload['weight_perturbation']['structure']['spearman_rank_corr_p10']:.3f} p50={payload['weight_perturbation']['structure']['spearman_rank_corr_p50']:.3f} top10 jacc p50={payload['weight_perturbation']['structure']['topk_jacc_p50']:.3f}")
    lines.append(f"  robust  : spearman p10={payload['weight_perturbation']['robustness']['spearman_rank_corr_p10']:.3f} p50={payload['weight_perturbation']['robustness']['spearman_rank_corr_p50']:.3f} top10 jacc p50={payload['weight_perturbation']['robustness']['topk_jacc_p50']:.3f}")
    lines.append(f"  zeta    : spearman p10={payload['weight_perturbation']['zeta_structure']['spearman_rank_corr_p10']:.3f} p50={payload['weight_perturbation']['zeta_structure']['spearman_rank_corr_p50']:.3f} top10 jacc p50={payload['weight_perturbation']['zeta_structure']['topk_jacc_p50']:.3f}")
    lines.append("")
    lines.append("Ablation (spearman rank corr vs full)")
    for k, v in payload["ablation"]["structure"].items():
        lines.append(f"  structure {k}: {v['spearman_rank_corr']:.3f}")
    for k, v in payload["ablation"]["robustness"].items():
        lines.append(f"  robust   {k}: {v['spearman_rank_corr']:.3f}")
    for k, v in payload["ablation"]["zeta_structure"].items():
        lines.append(f"  zeta     {k}: {v['spearman_rank_corr']:.3f}")
    lines.append("")
    lines.append("Confusion (top10 composition)")
    lines.append(f"  structure: dead_in_top10={payload['confusion']['structure']['dead_in_topk']}/{payload['confusion']['structure']['topk']} best_in_top10={payload['confusion']['structure']['best_in_topk']}")
    lines.append(f"  robust  : dead_in_top10={payload['confusion']['robustness']['dead_in_topk']}/{payload['confusion']['robustness']['topk']} best_in_top10={payload['confusion']['robustness']['best_in_topk']}")
    lines.append(f"  zeta    : dead_in_top10={payload['confusion']['zeta_structure']['dead_in_topk']}/{payload['confusion']['zeta_structure']['topk']} best_in_top10={payload['confusion']['zeta_structure']['best_in_topk']}")

    # ---- Zeta support summary (support-limited vs variance-limited) ----
    def _support_status(*, non_nan: int, nunique: int, std: float, fam_count: int, n: int) -> str:
        support_limited = (non_nan < max(5, int(0.15 * n))) or (fam_count < 3)
        variance_limited = (nunique <= 1) or (not math.isfinite(float(std))) or (float(std) <= 0.0)
        if support_limited and variance_limited:
            return "support+variance-limited"
        if support_limited:
            return "support-limited"
        if variance_limited:
            return "variance-limited"
        return "ok"

    def _finite_stats(s: pd.Series) -> tuple[int, int, float]:
        s_num = s.astype(float) if s.dtype == bool else pd.to_numeric(s, errors="coerce")
        vals = s_num.to_numpy(dtype=float, na_value=np.nan)
        finite = vals[np.isfinite(vals)]
        return (int(np.isfinite(vals).sum()), int(pd.Series(finite).nunique()) if finite.size else 0, float(np.nanstd(vals)) if finite.size else float("nan"))

    n_basket = int(len(basket))

    # Mirror block: effective families are window_id where pairing support exists.
    mir = basket.get("sigmafam_mirror_pair_frac", pd.Series([np.nan] * n_basket))
    mir_non_nan, mir_nuniq, mir_std = _finite_stats(mir)
    mir_ok = pd.to_numeric(mir, errors="coerce").fillna(0.0) > 0.0
    mir_fams = int(basket.loc[mir_ok, "window_id"].nunique()) if ("window_id" in basket.columns) else 0

    # Best-t alignment block: families are window_id, value is best_t_obs_std.
    bta = basket.get("best_t_obs_std", pd.Series([np.nan] * n_basket))
    bta_non_nan, bta_nuniq, bta_std = _finite_stats(bta)
    bta_ok = pd.to_numeric(bta, errors="coerce").notna()
    bta_fams = int(basket.loc[bta_ok, "window_id"].nunique()) if ("window_id" in basket.columns) else 0

    # Divisor block: effective families are divisor_family_id where we got at least one N->2N pair.
    div = basket.get("divisor_best_t_shift_scaled", pd.Series([np.nan] * n_basket))
    div_non_nan, div_nuniq, div_std = _finite_stats(div)
    div_ok = pd.to_numeric(basket.get("divisor_shift_pair_count", 0.0), errors="coerce").fillna(0.0) >= 1.0
    div_fams = int(basket.loc[div_ok, "divisor_family_id"].nunique()) if ("divisor_family_id" in basket.columns) else 0

    lines.append("")
    lines.append("Zeta support summary")
    mir_status = _support_status(non_nan=mir_non_nan, nunique=mir_nuniq, std=mir_std, fam_count=mir_fams, n=n_basket)
    bta_status = _support_status(non_nan=bta_non_nan, nunique=bta_nuniq, std=bta_std, fam_count=bta_fams, n=n_basket)
    div_status = _support_status(non_nan=div_non_nan, nunique=div_nuniq, std=div_std, fam_count=div_fams, n=n_basket)
    lines.append(f"  mirror       : non_nan={mir_non_nan} fams={mir_fams} nunique={mir_nuniq} status={mir_status}")
    lines.append(f"  best_t_align : non_nan={bta_non_nan} fams={bta_fams} nunique={bta_nuniq} status={bta_status}")
    lines.append(f"  divisor      : non_nan={div_non_nan} fams={div_fams} nunique={div_nuniq} status={div_status}")

    # ---- Support-regime split (per-row availability) ----
    combo_num = pd.to_numeric(basket.get("combo", pd.Series([np.nan] * n_basket)), errors="coerce")
    top10_cases = set(basket.sort_values("combo", ascending=False).head(10)["case"].astype(str).tolist())

    def _subset_line(name: str, mask: pd.Series) -> list[str]:
        sub = basket.loc[mask].copy()
        n = int(len(sub))
        if n == 0:
            return [f"  {name:16s}: n=0"]
        sub_combo = pd.to_numeric(sub.get("combo", np.nan), errors="coerce")
        sub_struct = pd.to_numeric(sub.get("structure_score", np.nan), errors="coerce")
        sub_rob = pd.to_numeric(sub.get("robustness_score", np.nan), errors="coerce")
        sub_zeta = pd.to_numeric(sub.get("zeta_structure_score", np.nan), errors="coerce")
        in_top10 = int(sub["case"].astype(str).isin(top10_cases).sum())
        top_cases = sub.sort_values("combo", ascending=False)["case"].astype(str).head(3).tolist()
        top_cases_s = ", ".join(top_cases)
        return [
            f"  {name:16s}: n={n} mean(combo)={float(sub_combo.mean()):.3f} mean(struct)={float(sub_struct.mean()):.3f} mean(rob)={float(sub_rob.mean()):.3f} mean(zeta)={float(sub_zeta.mean()):.3f} in_global_top10={in_top10}",
            f"    top3_by_combo: {top_cases_s}",
        ]

    div_avail = divisor_active
    mir_avail = mirror_active
    both_unavail = (~div_avail) & (~mir_avail)
    only_best_t = best_t_align_active & both_unavail
    div_and_mir = div_avail & mir_avail

    lines.append("")
    lines.append("Support-regime split (per-row zeta subfeature availability)")
    lines.extend(_subset_line("divisor avail", div_avail))
    lines.extend(_subset_line("mirror avail", mir_avail))
    lines.extend(_subset_line("div+mir avail", div_and_mir))
    lines.extend(_subset_line("both unavail", both_unavail))
    lines.extend(_subset_line("only best_t", only_best_t))

    lines.append("")
    lines.append("Support-regime ranking summary")
    for r in regime_rows:
        lines.append(
            f"  {r['regime']:16s}: n={r['n']} avg_rank={r['avg_rank_combo'] if math.isfinite(float(r['avg_rank_combo'])) else float('nan'):.2f} median_zeta={r['median_zeta'] if math.isfinite(float(r['median_zeta'])) else float('nan'):.3f} frac_top_quartile={r['frac_top_quartile'] if math.isfinite(float(r['frac_top_quartile'])) else float('nan'):.3f}"
        )
        if r.get("label_mix"):
            lines.append(f"    labels: {r['label_mix']}")

    # ---- Anchor-collapse audit (mirror) ----
    lines.append("")
    lines.append("Anchor-collapse audit (mirror)")
    wa = anchor_audit.get("within_anchor", {}) if isinstance(anchor_audit, dict) else {}
    lines.append(
        f"  anchor_active_n={anchor_audit.get('n_anchor_active')} combo_p10={wa.get('combo_p10')} combo_p50={wa.get('combo_p50')} combo_p90={wa.get('combo_p90')} spread={wa.get('combo_spread')}"
    )
    if wa.get("top5_label_mix"):
        lines.append(f"  within_anchor_top5_labels: {wa.get('top5_label_mix')}")
    if wa.get("bottom5_label_mix"):
        lines.append(f"  within_anchor_bottom5_labels: {wa.get('bottom5_label_mix')}")
    ab = anchor_audit.get("anchor_ablation", {}) if isinstance(anchor_audit, dict) else {}
    if isinstance(ab, dict) and ab:
        lines.append(
            f"  ablate_mirror: spearman={ab.get('spearman_rank_corr')} top10_overlap_jacc={ab.get('top10_overlap_jacc')} dead_in_top10={ab.get('dead_in_top10')}"
        )
    wc = anchor_audit.get("within_anchor_contrib", {}) if isinstance(anchor_audit, dict) else {}
    if isinstance(wc, dict) and wc:
        lines.append(
            "  within_anchor_means: "
            f"latent={wc.get('mean_latent_consistency')} robust={wc.get('mean_refinement_robustness')} zeta={wc.get('mean_symmetry_spectral_prior')} "
            f"zeta_mirror={wc.get('mean_zeta_mirror_contrib')} zeta_nonmirror={wc.get('mean_zeta_nonmirror_contrib')}"
        )
    sa = anchor_audit.get("synthetic_anchor_matched", {}) if isinstance(anchor_audit, dict) else {}
    if isinstance(sa, dict) and isinstance(sa.get("plausible_only"), dict):
        p = sa["plausible_only"]
        lines.append(
            f"  synthetic_vs_anchor(plausible): any_in_top10={p.get('any_in_top10')} n_in_top10={p.get('n_in_top10')} min_rank={p.get('min_rank')} (n={p.get('n')})"
        )

    if syn_payload is not None:
        lines.append("")
        lines.append("Synthetic smoothness controls (scored vs basket normalization)")
        lines.append(
            f"  any_in_top10={syn_payload.get('any_in_top10')} n_in_top10={syn_payload.get('n_in_top10')} min_implied_rank={syn_payload.get('min_rank')}"
        )
        if isinstance(syn_payload.get("plausible"), dict):
            p = syn_payload["plausible"]
            lines.append(
                f"  plausible_only: any_in_top10={p.get('any_in_top10')} n_in_top10={p.get('n_in_top10')} min_implied_rank={p.get('min_rank')} (n={p.get('n')})"
            )
        for rr in syn_payload.get("rows", []):
            a_rank = rr.get("implied_rank_combo_vs_anchor")
            a_txt = f" anchor_rank={int(a_rank)}" if (a_rank is not None and str(a_rank) != "nan") else ""
            lines.append(
                f"  {rr['case']} ({rr.get('synthetic_kind','')}|{rr.get('synthetic_suite','')}): combo={float(rr['combo']):.3f} implied_rank={int(rr['implied_rank_combo_vs_basket'])} implied_percentile={float(rr['implied_percentile_vs_basket']):.3f}{a_txt}"
            )
    lines.append("")
    lines.append(f"Blind rankings: {out_blind}")
    lines.append(f"Case digest (top/bottom): {out_digest}")
    lines.append(f"Case digest (full): {out_digest_full}")
    lines.append(f"Top10 why-breakdown: {out_top10_why}")
    lines.append(f"Support regime summary: {out_regime}")
    lines.append(f"Frozen selector spec: {out_selector}")
    if syn_payload is not None:
        lines.append(f"Synthetic smoothness controls: {out_syn}")
    lines.append(f"Zeta proxy audit: {out_audit}")
    lines.append(f"Zeta proxy redundancy: {out_redundancy}")

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload["support_regime"] = {
        "mirror_block_status": str(mir_status),
        "best_t_align_block_status": str(bta_status),
        "divisor_block_status": str(div_status),
        "counts": {
            "divisor_available": int(div_avail.sum()),
            "mirror_available": int(mir_avail.sum()),
            "divisor_and_mirror": int(div_and_mir.sum()),
            "both_unavailable": int(both_unavail.sum()),
            "only_best_t_active": int(only_best_t.sum()),
        },
    }
    payload["support_regime_summary"] = regime_rows
    if syn_payload is not None:
        payload["synthetic_smoothness_controls"] = syn_payload
    payload["anchor_audit"] = anchor_audit

    # Write JSON report after all late-added diagnostics are attached.
    out_json.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_txt}")
    print(f"Wrote: {out_blind}")
    print(f"Wrote: {out_digest}")
    print(f"Wrote: {out_digest_full}")
    print(f"Wrote: {out_top10_why}")
    print(f"Wrote: {out_regime}")
    print(f"Wrote: {out_selector}")
    if syn_payload is not None:
        print(f"Wrote: {out_syn}")
    print(f"Wrote: {out_audit}")
    print(f"Wrote: {out_redundancy}")
    print(out_txt.read_text(encoding='utf-8'))


if __name__ == "__main__":
    main()
