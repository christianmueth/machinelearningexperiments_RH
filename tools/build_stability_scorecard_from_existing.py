from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Thresholds:
    gate_tin_inv_max: float = 500.0
    gate_sep2_min: float = 0.01
    gate_lam_cost_max: float = 0.8
    tau_logabs: float = 0.2
    tau_arg: float = 0.5


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


def _tier01_mask(df: pd.DataFrame, th: Thresholds) -> pd.Series:
    ok = pd.Series(True, index=df.index)

    if "Tin_inv_norm1_est_N2" in df.columns and math.isfinite(th.gate_tin_inv_max):
        ok &= _coerce_num(df["Tin_inv_norm1_est_N2"]) <= th.gate_tin_inv_max
    else:
        ok &= False

    if "sep2_N2" in df.columns and math.isfinite(th.gate_sep2_min):
        ok &= _coerce_num(df["sep2_N2"]) >= th.gate_sep2_min
    else:
        ok &= False

    if "lam_match_cost" in df.columns and math.isfinite(th.gate_lam_cost_max):
        ok &= _coerce_num(df["lam_match_cost"]) <= th.gate_lam_cost_max
    else:
        ok &= False

    return ok.fillna(False).astype(bool)


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
        # Fallback: treat any adjacent pair as contiguous if gap is small-ish.
        dt = float(np.nanmin(np.diff(t_sorted))) if t_sorted.size >= 2 else float("nan")
        if not math.isfinite(dt) or dt <= 0:
            return (1, 0.0)

    # Contiguous if gap <= 1.5*dt.
    gaps = np.diff(t_sorted)
    breaks = np.where(gaps > (1.5 * dt))[0]
    # Segment boundaries in indices.
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [t_sorted.size - 1]))

    widths = t_sorted[ends] - t_sorted[starts]
    return (int(len(starts)), float(np.nanmax(widths) if widths.size else 0.0))


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


def _format_obs_key(obs: str) -> str:
    obs = obs.strip()
    if obs == "Q2_tr":
        return "Q2tr"
    if obs == "Q2_phi2":
        return "Q2phi2"
    if obs == "Q2":
        return "Q2"
    return obs.replace(" ", "")


def _infer_obs_from_name(p: Path) -> str:
    raw = p.name.lower()
    tok = "_" + "".join(ch if ch.isalnum() else "_" for ch in raw) + "_"
    if "_q2tr_" in tok or "_q2_tr_" in tok:
        return "Q2_tr"
    if "_q2phi2_" in tok or "_q2_phi2_" in tok:
        return "Q2_phi2"
    if "_q2_" in tok:
        return "Q2"
    return p.stem


def _load_aug_df(path: Path, *, side: str = "right") -> pd.DataFrame:
    df = pd.read_csv(path)
    if side and "side" in df.columns:
        df = df[df["side"].astype(str) == side].copy()
    return df


def _accepted_t_keys(df: pd.DataFrame, th: Thresholds) -> tuple[list[float], set[float]]:
    t = pd.to_numeric(df.get("t"), errors="coerce").to_numpy(dtype=float)
    tier01 = _tier01_mask(df, th)

    ok = tier01.copy()
    if "dlog_abs" in df.columns and math.isfinite(th.tau_logabs):
        ok &= _coerce_num(df["dlog_abs"]) <= th.tau_logabs
    else:
        ok &= False

    if "darg" in df.columns and math.isfinite(th.tau_arg):
        ok &= _coerce_num(df["darg"]) <= th.tau_arg
    else:
        ok &= False

    t_accept = [float(x) for x in t[ok.to_numpy(dtype=bool, na_value=False)] if math.isfinite(float(x))]
    keys = set(float(np.round(x, 12)) for x in t_accept)
    return (sorted(set(float(np.round(x, 12)) for x in t_accept)), keys)


def _obs_metrics(aug_csv: Path, th: Thresholds, *, side: str) -> dict[str, Any]:
    df = _load_aug_df(aug_csv, side=side)

    t = pd.to_numeric(df.get("t"), errors="coerce")
    tvals = [float(x) for x in t.tolist() if math.isfinite(float(x))]
    dt = _estimate_dt(tvals)

    tier01 = _tier01_mask(df, th)
    n_total = int(len(df))
    n_tier01 = int(tier01.sum())

    dlog = _coerce_num(df.get("dlog_abs"))
    darg = _coerce_num(df.get("darg"))

    logabs_ok = tier01 & (dlog <= th.tau_logabs)
    arg_ok = tier01 & (darg <= th.tau_arg)

    accept = logabs_ok & (darg <= th.tau_arg)

    t_accept = pd.to_numeric(df.loc[accept, "t"], errors="coerce") if "t" in df.columns else pd.Series([], dtype=float)
    t_accept = t_accept[np.isfinite(t_accept.to_numpy(dtype=float, na_value=np.nan))]
    t_accept_sorted = np.sort(t_accept.to_numpy(dtype=float))

    n_seg, width_max = _segments_width_max(t_accept_sorted, dt=float(dt))

    min_darg_tier01_logabs = _min_finite(darg[logabs_ok])
    phase_margin_min = (
        float(min_darg_tier01_logabs) - float(th.tau_arg)
        if (math.isfinite(float(min_darg_tier01_logabs)) and math.isfinite(float(th.tau_arg)))
        else float("nan")
    )

    min_dlog_tier01_arg = _min_finite(dlog[arg_ok])
    logabs_margin_min = (
        float(min_dlog_tier01_arg) - float(th.tau_logabs)
        if (math.isfinite(float(min_dlog_tier01_arg)) and math.isfinite(float(th.tau_logabs)))
        else float("nan")
    )

    out: dict[str, Any] = {
        "file": str(aug_csv),
        "points": n_total,
        "t_min": float(np.nanmin(tvals)) if tvals else float("nan"),
        "t_max": float(np.nanmax(tvals)) if tvals else float("nan"),
        "dt_est": float(dt),
        "tier01": n_tier01,
        "tier01_frac": (n_tier01 / n_total) if n_total else float("nan"),
        "accept": int(accept.sum()),
        "accept_frac": (float(int(accept.sum())) / n_total) if n_total else float("nan"),
        "segments": int(n_seg),
        "width_max": float(width_max),
        "min_dlog_abs_tier01": _min_finite(dlog[tier01]),
        "min_darg_tier01": _min_finite(darg[tier01]),
        "tier01_logabs_ok": int(logabs_ok.sum()),
        "tier01_arg_ok": int(arg_ok.sum()),
        "tier01_logabs_ok_not_arg": int((logabs_ok & (~(darg <= th.tau_arg))).sum()),
        "tier01_arg_ok_not_logabs": int((arg_ok & (~(dlog <= th.tau_logabs))).sum()),
        "min_darg_tier01_logabs_ok": float(min_darg_tier01_logabs),
        "phase_margin_min": float(phase_margin_min),
        "min_dlog_abs_tier01_arg_ok": float(min_dlog_tier01_arg),
        "logabs_margin_min": float(logabs_margin_min),
    }

    summary_path = _guess_delta_summary_path(aug_csv)
    summary = _read_summary(summary_path)
    if summary:
        out.update(
            {
                "N": summary.get("N"),
                "N2": summary.get("N2"),
                "quotient": summary.get("quotient"),
                "fixed_gauge": _fixed_gauge_flag(summary),
                "cstar_source": ((summary.get("normalization") or {}).get("lsfit") or {}).get("source"),
                "c_star_abs": ((summary.get("normalization") or {}).get("lsfit") or {}).get("c_star_abs"),
            }
        )
    else:
        out.update({"N": None, "N2": None, "quotient": _infer_obs_from_name(aug_csv), "fixed_gauge": None})

    # Accepted t sets for overlap computation.
    accept_list, accept_keys = _accepted_t_keys(df, th)
    out["accept_t_list"] = accept_list
    out["accept_t_keys"] = accept_keys

    return out


def _case_row(
    *,
    label: str,
    th: Thresholds,
    obs_paths: dict[str, Path],
    side: str,
    baseline_map: dict[tuple[float, float], dict[str, float]] | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case": label,
        "side": side,
        "gate_tin_inv_max": th.gate_tin_inv_max,
        "gate_sep2_min": th.gate_sep2_min,
        "gate_lam_cost_max": th.gate_lam_cost_max,
        "tau_logabs": th.tau_logabs,
        "tau_arg": th.tau_arg,
    }

    # Per-observable metrics.
    obs_out: dict[str, dict[str, Any]] = {}
    for obs, p in obs_paths.items():
        obs_out[obs] = _obs_metrics(p, th, side=side)

        k = _format_obs_key(obs)
        for key, val in obs_out[obs].items():
            if key in {"accept_t_list", "accept_t_keys"}:
                continue
            row[f"{k}_{key}"] = val

    # Shared metadata: prefer Q2tr then Q2 then Q2phi2.
    for pref in ["Q2_tr", "Q2", "Q2_phi2"]:
        if pref in obs_out:
            row["N"] = obs_out[pref].get("N")
            row["N2"] = obs_out[pref].get("N2")
            row["fixed_gauge"] = obs_out[pref].get("fixed_gauge")
            row["cstar_source"] = obs_out[pref].get("cstar_source")
            row["c_star_abs"] = obs_out[pref].get("c_star_abs")
            break

    # Contraction diagnostics (high-rung vs baseline Tier01 point list).
    # This is designed specifically to make the 162 (4096→8192) reference case show up as a "good" pocket
    # even when strict boundary thresholds reject it.
    if baseline_map and ("Q2_tr" in obs_paths):
        q2tr_N = row.get("Q2tr_N")
        q2tr_N2 = row.get("Q2tr_N2")
        if (q2tr_N == 4096 or q2tr_N == 4096.0) and (q2tr_N2 == 8192 or q2tr_N2 == 8192.0):
            row.update(_contraction_metrics(obs_paths["Q2_tr"], th=th, side=side, baseline_map=baseline_map))

    # Overlaps (strict).
    def _overlap(a: str, b: str) -> tuple[int | None, float | None]:
        if a not in obs_out or b not in obs_out:
            return (None, None)
        keys = obs_out[a]["accept_t_keys"] & obs_out[b]["accept_t_keys"]
        if not keys:
            return (0, 0.0)
        t_sorted = np.array(sorted(keys), dtype=float)
        dt = min(
            float(obs_out[a].get("dt_est") or float("nan")),
            float(obs_out[b].get("dt_est") or float("nan")),
        )
        nseg, wmax = _segments_width_max(np.sort(t_sorted), dt=float(dt))
        return (int(len(keys)), float(wmax))

    def _overlap3(a: str, b: str, c: str) -> tuple[int | None, float | None]:
        if a not in obs_out or b not in obs_out or c not in obs_out:
            return (None, None)
        keys = obs_out[a]["accept_t_keys"] & obs_out[b]["accept_t_keys"] & obs_out[c]["accept_t_keys"]
        if not keys:
            return (0, 0.0)
        t_sorted = np.array(sorted(keys), dtype=float)
        dt = min(
            float(obs_out[a].get("dt_est") or float("nan")),
            float(obs_out[b].get("dt_est") or float("nan")),
            float(obs_out[c].get("dt_est") or float("nan")),
        )
        nseg, wmax = _segments_width_max(np.sort(t_sorted), dt=float(dt))
        return (int(len(keys)), float(wmax))

    n12, w12 = _overlap("Q2_tr", "Q2")
    n13, w13 = _overlap("Q2_tr", "Q2_phi2")
    n23, w23 = _overlap("Q2", "Q2_phi2")
    n123, w123 = _overlap3("Q2_tr", "Q2", "Q2_phi2")

    row.update(
        {
            "overlap_Q2tr_Q2_n": n12,
            "overlap_Q2tr_Q2_width_max": w12,
            "overlap_Q2tr_Q2phi2_n": n13,
            "overlap_Q2tr_Q2phi2_width_max": w13,
            "overlap_Q2_Q2phi2_n": n23,
            "overlap_Q2_Q2phi2_width_max": w23,
            "overlap_all3_n": n123,
            "overlap_all3_width_max": w123,
        }
    )

    return row


def _baseline_key(sigma: float, t: float) -> tuple[float, float]:
    return (float(np.round(float(sigma), 6)), float(np.round(float(t), 6)))


def _contraction_metrics(
    high_aug_csv: Path,
    *,
    th: Thresholds,
    side: str,
    baseline_map: dict[tuple[float, float], dict[str, float]],
) -> dict[str, Any]:
    high = _load_aug_df(high_aug_csv, side=side)
    if high.empty:
        return {
            "contract_pairs": 0,
            "contract_n": 0,
            "contract_frac": float("nan"),
            "contract_best_improve": float("nan"),
            "contract_best_t": float("nan"),
            "contract_best_sigma": float("nan"),
            "contract_best_low_dlog": float("nan"),
            "contract_best_high_dlog": float("nan"),
        }

    sigma = _coerce_num(high.get("sigma"))
    t = _coerce_num(high.get("t"))
    dlog_high = _coerce_num(high.get("dlog_abs"))

    # Require high-rung Tier01 gating for contraction accounting.
    tier01_high = _tier01_mask(high, th)

    pairs = 0
    n_contract = 0
    best_improve = float("inf")
    best_payload: dict[str, float] = {}

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
            best_payload = {
                "t": float(t.loc[i]),
                "sigma": float(sigma.loc[i]),
                "dlog_low": float(dlog_low),
                "dlog_high": float(dlog_high.loc[i]),
            }

    return {
        "contract_pairs": int(pairs),
        "contract_n": int(n_contract),
        "contract_frac": (float(n_contract) / float(pairs)) if pairs else float("nan"),
        "contract_best_improve": float(best_improve) if pairs else float("nan"),
        "contract_best_t": float(best_payload.get("t", float("nan"))),
        "contract_best_sigma": float(best_payload.get("sigma", float("nan"))),
        "contract_best_low_dlog": float(best_payload.get("dlog_low", float("nan"))),
        "contract_best_high_dlog": float(best_payload.get("dlog_high", float("nan"))),
    }


def _pick_one(paths: list[Path], *, prefer_contains: str | None = None) -> Path:
    if not paths:
        raise FileNotFoundError("No matching paths")
    if prefer_contains:
        for p in paths:
            if prefer_contains.lower() in p.name.lower():
                return p
    return sorted(paths)[0]


def _find_case_paths(run_dir: Path) -> dict[str, dict[str, Path]]:
    cases: dict[str, dict[str, Path]] = {}

    # Case: refine164 dt=0.01 (fixed gauge canonical postpassQ2tr) for all three families.
    q2 = list(run_dir.rglob("*_refine164p00_10_dt0p01*_Q2_*_delta_fixedcstar_postpassQ2tr__aug_lam.csv"))
    q2tr = list(run_dir.rglob("*_refine164p00_10_dt0p01*_Q2tr_*_delta_fixedcstar_postpassQ2tr__aug_lam.csv"))
    q2phi2 = list(run_dir.rglob("*_refine164p00_10_dt0p01*_Q2phi2_*_delta_fixedcstar_postpassQ2tr__aug_lam.csv"))

    if q2 and q2tr and q2phi2:
        cases["refine164_dt0p01"] = {
            "Q2": _pick_one(q2, prefer_contains="postpassq2tr"),
            "Q2_tr": _pick_one(q2tr, prefer_contains="postpassq2tr"),
            "Q2_phi2": _pick_one(q2phi2, prefer_contains="postpassq2tr"),
        }

    # Case: refine164 stage-1 dt=0.05 (fixed gauge via external_json c*; filenames differ).
    q2_s1 = list(run_dir.rglob("*_refine164_stage1_dt0p05_Q2_*_delta_fixedcstar__aug_lam.csv"))
    q2tr_s1 = list(run_dir.rglob("*_refine164_stage1_dt0p05_Q2tr_*_delta_fixedcstar__aug_lam.csv"))
    q2phi2_s1 = list(run_dir.rglob("*_refine164_stage1_dt0p05_Q2phi2_*_delta_fixedcstar__aug_lam.csv"))

    if q2_s1 and q2tr_s1 and q2phi2_s1:
        cases["refine164_stage1_dt0p05"] = {
            "Q2": _pick_one(q2_s1),
            "Q2_tr": _pick_one(q2tr_s1),
            "Q2_phi2": _pick_one(q2phi2_s1),
        }

    # Case: 162.08 high-rung pocket (Q2tr only). Prefer the non-fast run if present.
    pocket = list(run_dir.rglob("*_pocket162p08_N4096_8192_Q2tr*_delta__aug_lam.csv"))
    if pocket:
        nonfast = [p for p in pocket if "_fast_" not in p.name.lower()]
        cases["pocket162p08_N4096_8192"] = {"Q2_tr": _pick_one(nonfast or pocket)}

    return cases


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build a stability scorecard CSV from existing __aug_lam.csv artifacts only (no new projector emission). "
            "Auto-discovers the requested 164 (dt=0.05 and dt=0.01) and 162.08 (4096→8192) cases under --run_dir."
        )
    )
    ap.add_argument("--run_dir", required=True, help="Run directory containing the artifacts (searched recursively)")
    ap.add_argument("--out_csv", default="_stability_scorecard.csv")
    ap.add_argument("--side", default="right", help="Filter by side; set to '' for all")

    ap.add_argument(
        "--baseline_points_csv",
        default="_sigma_sweep_win1_tier01_points_final.csv",
        help="Optional baseline Tier01 point list (used for contraction scoring of 4096→8192 pockets).",
    )

    ap.add_argument("--gate_tin_inv_max", type=float, default=500.0)
    ap.add_argument("--gate_sep2_min", type=float, default=0.01)
    ap.add_argument("--gate_lam_cost_max", type=float, default=0.8)

    ap.add_argument("--tau_logabs", type=float, default=0.2)
    ap.add_argument("--tau_arg", type=float, default=0.5)
    ap.add_argument(
        "--tau_arg_diag",
        type=float,
        default=0.7,
        help="Also emit a diagnostic row for cases that include Q2 using this relaxed tau_arg",
    )

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"run_dir not found: {run_dir}")

    cases = _find_case_paths(run_dir)
    if not cases:
        raise SystemExit(f"No matching cases found under: {run_dir}")

    th_strict = Thresholds(
        gate_tin_inv_max=float(args.gate_tin_inv_max),
        gate_sep2_min=float(args.gate_sep2_min),
        gate_lam_cost_max=float(args.gate_lam_cost_max),
        tau_logabs=float(args.tau_logabs),
        tau_arg=float(args.tau_arg),
    )
    th_diag = Thresholds(
        gate_tin_inv_max=float(args.gate_tin_inv_max),
        gate_sep2_min=float(args.gate_sep2_min),
        gate_lam_cost_max=float(args.gate_lam_cost_max),
        tau_logabs=float(args.tau_logabs),
        tau_arg=float(args.tau_arg_diag),
    )

    rows: list[dict[str, Any]] = []
    side = str(args.side)

    # Baseline map for contraction diagnostics.
    baseline_map: dict[tuple[float, float], dict[str, float]] | None = None
    baseline_path = Path(str(args.baseline_points_csv))
    if not baseline_path.is_absolute():
        baseline_path = run_dir / baseline_path
    if baseline_path.exists():
        try:
            low = pd.read_csv(baseline_path)
            low_sigma = pd.to_numeric(low.get("sigma"), errors="coerce")
            low_t = pd.to_numeric(low.get("t"), errors="coerce")
            low_dlog = pd.to_numeric(low.get("dlog_abs"), errors="coerce")
            # For each (sigma,t) keep the minimal dlog_abs (best baseline point).
            baseline_map = {}
            for s_val, t_val, d_val in zip(low_sigma.tolist(), low_t.tolist(), low_dlog.tolist()):
                if not (math.isfinite(float(s_val)) and math.isfinite(float(t_val)) and math.isfinite(float(d_val))):
                    continue
                k = _baseline_key(float(s_val), float(t_val))
                cur = baseline_map.get(k)
                if (cur is None) or (float(d_val) < float(cur.get("dlog_abs", float("inf")))):
                    baseline_map[k] = {"dlog_abs": float(d_val)}
        except Exception:
            baseline_map = None

    for label, obs_paths in cases.items():
        rows.append(_case_row(label=label, th=th_strict, obs_paths=obs_paths, side=side, baseline_map=baseline_map))
        # For the 164 diagnostic case, include a relaxed tau_arg variant (useful for phase-block diagnosis).
        if "Q2" in obs_paths and math.isfinite(th_diag.tau_arg) and th_diag.tau_arg != th_strict.tau_arg:
            rows.append(
                _case_row(label=label + "__tauarg_diag", th=th_diag, obs_paths=obs_paths, side=side, baseline_map=baseline_map)
            )

    out_csv = Path(args.out_csv)
    if not out_csv.is_absolute():
        out_csv = run_dir / out_csv

    df = pd.DataFrame(rows)

    # Stable column order: core metadata first, then overlap, then per-observable.
    core_cols = [
        "case",
        "side",
        "N",
        "N2",
        "fixed_gauge",
        "c_star_abs",
        "cstar_source",
        "gate_tin_inv_max",
        "gate_sep2_min",
        "gate_lam_cost_max",
        "tau_logabs",
        "tau_arg",
        "overlap_all3_n",
        "overlap_all3_width_max",
        "overlap_Q2tr_Q2_n",
        "overlap_Q2tr_Q2_width_max",
        "overlap_Q2tr_Q2phi2_n",
        "overlap_Q2tr_Q2phi2_width_max",
        "overlap_Q2_Q2phi2_n",
        "overlap_Q2_Q2phi2_width_max",
    ]
    other_cols = [c for c in df.columns if c not in core_cols]
    df = df[[c for c in core_cols if c in df.columns] + sorted(other_cols)]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")
    print(df[[c for c in core_cols if c in df.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
