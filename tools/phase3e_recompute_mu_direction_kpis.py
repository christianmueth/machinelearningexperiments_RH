import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from phase3e_elambda_loop_suite import (
    direction_cosine_fro,
    effective_rank2,
    find_npz_files,
    key_for_pair,
    make_within_backend_anchor_offset_map,
    parse_rel_metadata,
)


def _resolve_path(p: str) -> str:
    p = str(p).strip()
    if not p:
        return p
    return os.path.abspath(p)


def _maybe_load(npz: np.lib.npyio.NpzFile, key: str) -> Optional[np.ndarray]:
    if key not in npz.files:
        return None
    return np.asarray(npz[key])


def _median_abs(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    a = np.asarray(xs, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(np.nanmedian(np.abs(a)))


def _median(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    a = np.asarray(xs, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(np.nanmedian(a))


def compute_mu_direction_kpis(
    *,
    out_root_a: str,
    out_root_b: str,
    out_root_b2: str,
    within_backend: str,
    within_backend_anchor_offset: int,
    direction_cosine_samples: int,
    restrict_keys: Optional[List[Tuple[int, int, float, float]]],
) -> Tuple[Dict[str, Any], int, int]:
    outA = str(out_root_a)
    outB = str(out_root_b)
    outB2 = str(out_root_b2).strip()

    filesA = find_npz_files(outA)
    filesB = find_npz_files(outB)
    filesB2 = find_npz_files(outB2) if outB2 else []

    mapA: Dict[Tuple[int, int, float, float], str] = {}
    mapB: Dict[Tuple[int, int, float, float], str] = {}
    mapB2: Dict[Tuple[int, int, float, float], str] = {}

    for p in filesA:
        _rel, _backend, seed, anchor, wlo, whi = parse_rel_metadata(p, outA)
        mapA[key_for_pair(seed, anchor, wlo, whi)] = p

    for p in filesB:
        _rel, _backend, seed, anchor, wlo, whi = parse_rel_metadata(p, outB)
        mapB[key_for_pair(seed, anchor, wlo, whi)] = p

    if outB2:
        for p in filesB2:
            _rel, _backend, seed, anchor, wlo, whi = parse_rel_metadata(p, outB2)
            mapB2[key_for_pair(seed, anchor, wlo, whi)] = p

    if restrict_keys:
        keep = set((int(s), int(a), float(wlo), float(whi)) for (s, a, wlo, whi) in restrict_keys)
        mapA = {k: v for k, v in mapA.items() if k in keep}
        mapB = {k: v for k, v in mapB.items() if k in keep}
        if outB2:
            mapB2 = {k: v for k, v in mapB2.items() if k in keep}

    within_backend = str(within_backend).strip()
    within_anchor_offset = int(within_backend_anchor_offset)

    if outB2 and within_backend != "off":
        raise SystemExit("within_backend controls are incompatible with out_root_b2")

    if within_backend == "A":
        outB = outA
        mapB = make_within_backend_anchor_offset_map(mapA, offset=within_anchor_offset)
        keys = sorted(mapA.keys())
    elif within_backend == "B":
        outA = outB
        mapA = dict(mapB)
        mapB = make_within_backend_anchor_offset_map(mapA, offset=within_anchor_offset)
        keys = sorted(mapA.keys())
    else:
        keys = sorted(set(mapA.keys()).intersection(set(mapB.keys())))
        if outB2:
            keys = sorted(set(keys).intersection(set(mapB2.keys())))

    failed = 0

    mu_cos_lam_med_per_key: List[float] = []
    mu_norm_ratio_med_per_key: List[float] = []
    mu_rank_med_per_key: List[float] = []

    mu_cos_S_med_per_key: List[float] = []
    mu_norm_ratio_S_med_per_key: List[float] = []
    mu_rank_S_med_per_key: List[float] = []

    mu_cos_trace_med_per_key: List[float] = []
    mu_norm_ratio_trace_med_per_key: List[float] = []
    mu_rank_trace_med_per_key: List[float] = []

    mu_cos_logdet_med_per_key: List[float] = []
    mu_norm_ratio_logdet_med_per_key: List[float] = []
    mu_rank_logdet_med_per_key: List[float] = []

    mu_cos_series_med_per_key: List[float] = []
    mu_norm_ratio_series_med_per_key: List[float] = []
    mu_rank_series_med_per_key: List[float] = []

    mu_cos_resid_med_per_key: List[float] = []
    mu_norm_ratio_resid_med_per_key: List[float] = []
    mu_rank_resid_med_per_key: List[float] = []

    nsamp = int(max(2, direction_cosine_samples))

    for k in keys:
        pA = mapA[k]
        pB = mapB[k]
        pB2 = mapB2[k] if outB2 else ""

        dA = np.load(pA)
        dB = np.load(pB)
        dB2 = np.load(pB2) if outB2 else None

        if "lambda_snap" not in dA.files or "lambda_snap" not in dB.files:
            failed += 1
            continue
        if outB2 and (dB2 is not None) and ("lambda_snap" not in dB2.files):
            failed += 1
            continue

        lamA = np.asarray(dA["lambda_snap"], dtype=np.complex128)
        lamB = np.asarray(dB["lambda_snap"], dtype=np.complex128)
        lamB2 = np.asarray(dB2["lambda_snap"], dtype=np.complex128) if (outB2 and (dB2 is not None)) else None

        if lamA.shape != lamB.shape:
            failed += 1
            continue
        if (lamB2 is not None) and (lamA.shape != lamB2.shape):
            failed += 1
            continue

        nE = int(lamA.shape[0])
        if nE < 2:
            continue

        if lamB2 is None:
            continue

        idx = [0, nE - 1]
        if nsamp >= 3:
            idx.append(int(nE // 2))
        if nsamp >= 4:
            idx.append(int(nE // 3))
        if nsamp >= 5:
            idx.append(int((2 * nE) // 3))
        idx = [int(np.clip(int(i), 0, nE - 1)) for i in idx]
        idx = list(dict.fromkeys(idx))[:nsamp]

        cos_mu_lam: List[float] = []
        norm_ratio: List[float] = []
        ranks: List[int] = []

        cos_mu_lam_S: List[float] = []
        norm_ratio_S: List[float] = []
        ranks_S: List[int] = []

        cos_mu_lam_trace: List[float] = []
        norm_ratio_trace: List[float] = []
        ranks_trace: List[int] = []

        cos_mu_lam_logdet: List[float] = []
        norm_ratio_logdet: List[float] = []
        ranks_logdet: List[int] = []

        cos_mu_lam_series: List[float] = []
        norm_ratio_series: List[float] = []
        ranks_series: List[int] = []

        cos_mu_lam_resid: List[float] = []
        norm_ratio_resid: List[float] = []
        ranks_resid: List[int] = []

        SA = _maybe_load(dA, "S_snap")
        SB = _maybe_load(dB, "S_snap")
        SB2 = _maybe_load(dB2, "S_snap") if (dB2 is not None) else None
        has_S = (SA is not None) and (SB is not None) and (SB2 is not None)

        trA = _maybe_load(dA, "trace_powers")
        trB = _maybe_load(dB, "trace_powers")
        trB2 = _maybe_load(dB2, "trace_powers") if (dB2 is not None) else None
        has_trace = (trA is not None) and (trB is not None) and (trB2 is not None)

        ldA = _maybe_load(dA, "logdet_IminusS")
        ldB = _maybe_load(dB, "logdet_IminusS")
        ldB2 = _maybe_load(dB2, "logdet_IminusS") if (dB2 is not None) else None
        has_logdet = (ldA is not None) and (ldB is not None) and (ldB2 is not None)

        spA = _maybe_load(dA, "series_partial")
        spB = _maybe_load(dB, "series_partial")
        spB2 = _maybe_load(dB2, "series_partial") if (dB2 is not None) else None
        has_series = (spA is not None) and (spB is not None) and (spB2 is not None)

        rpA = _maybe_load(dA, "resid_partial")
        rpB = _maybe_load(dB, "resid_partial")
        rpB2 = _maybe_load(dB2, "resid_partial") if (dB2 is not None) else None
        has_resid = (rpA is not None) and (rpB is not None) and (rpB2 is not None)

        for ei in idx:
            dl = np.asarray(lamB[int(ei)], dtype=np.complex128) - np.asarray(lamA[int(ei)], dtype=np.complex128)
            dm = np.asarray(lamB2[int(ei)], dtype=np.complex128) - np.asarray(lamB[int(ei)], dtype=np.complex128)
            cos_mu_lam.append(direction_cosine_fro(dl, dm))
            nl = float(np.linalg.norm(dl.ravel()))
            nm = float(np.linalg.norm(dm.ravel()))
            if np.isfinite(nl) and nl > 0 and np.isfinite(nm):
                norm_ratio.append(float(nm / nl))
            ranks.append(effective_rank2(dl, dm))

            if has_S and (SA is not None) and (SB is not None) and (SB2 is not None):
                dlS = np.asarray(SB[int(ei)], dtype=np.complex128) - np.asarray(SA[int(ei)], dtype=np.complex128)
                dmS = np.asarray(SB2[int(ei)], dtype=np.complex128) - np.asarray(SB[int(ei)], dtype=np.complex128)
                cos_mu_lam_S.append(direction_cosine_fro(dlS, dmS))
                nlS = float(np.linalg.norm(dlS.ravel()))
                nmS = float(np.linalg.norm(dmS.ravel()))
                if np.isfinite(nlS) and nlS > 0 and np.isfinite(nmS):
                    norm_ratio_S.append(float(nmS / nlS))
                ranks_S.append(effective_rank2(dlS, dmS))

            if has_trace and (trA is not None) and (trB is not None) and (trB2 is not None):
                if (trA.shape == trB.shape) and (trA.shape == trB2.shape) and (int(ei) < int(trA.shape[0])):
                    dlT = np.asarray(trB[int(ei)], dtype=np.complex128) - np.asarray(trA[int(ei)], dtype=np.complex128)
                    dmT = np.asarray(trB2[int(ei)], dtype=np.complex128) - np.asarray(trB[int(ei)], dtype=np.complex128)
                    cos_mu_lam_trace.append(direction_cosine_fro(dlT, dmT))
                    nlT = float(np.linalg.norm(dlT.ravel()))
                    nmT = float(np.linalg.norm(dmT.ravel()))
                    if np.isfinite(nlT) and nlT > 0 and np.isfinite(nmT):
                        norm_ratio_trace.append(float(nmT / nlT))
                    ranks_trace.append(effective_rank2(dlT, dmT))

            if has_logdet and (ldA is not None) and (ldB is not None) and (ldB2 is not None):
                if (ldA.shape == ldB.shape) and (ldA.shape == ldB2.shape) and (int(ei) < int(ldA.shape[0])):
                    dlL = np.asarray(ldB[int(ei)], dtype=np.complex128) - np.asarray(ldA[int(ei)], dtype=np.complex128)
                    dmL = np.asarray(ldB2[int(ei)], dtype=np.complex128) - np.asarray(ldB[int(ei)], dtype=np.complex128)
                    cos_mu_lam_logdet.append(direction_cosine_fro(dlL, dmL))
                    nlL = float(np.linalg.norm(np.asarray(dlL).ravel()))
                    nmL = float(np.linalg.norm(np.asarray(dmL).ravel()))
                    if np.isfinite(nlL) and nlL > 0 and np.isfinite(nmL):
                        norm_ratio_logdet.append(float(nmL / nlL))
                    ranks_logdet.append(effective_rank2(dlL, dmL))

            if has_series and (spA is not None) and (spB is not None) and (spB2 is not None):
                if (spA.shape == spB.shape) and (spA.shape == spB2.shape) and (int(ei) < int(spA.shape[0])):
                    dlP = np.asarray(spB[int(ei)], dtype=np.complex128) - np.asarray(spA[int(ei)], dtype=np.complex128)
                    dmP = np.asarray(spB2[int(ei)], dtype=np.complex128) - np.asarray(spB[int(ei)], dtype=np.complex128)
                    cos_mu_lam_series.append(direction_cosine_fro(dlP, dmP))
                    nlP = float(np.linalg.norm(dlP.ravel()))
                    nmP = float(np.linalg.norm(dmP.ravel()))
                    if np.isfinite(nlP) and nlP > 0 and np.isfinite(nmP):
                        norm_ratio_series.append(float(nmP / nlP))
                    ranks_series.append(effective_rank2(dlP, dmP))

            if has_resid and (rpA is not None) and (rpB is not None) and (rpB2 is not None):
                if (rpA.shape == rpB.shape) and (rpA.shape == rpB2.shape) and (int(ei) < int(rpA.shape[0])):
                    dlR = np.asarray(rpB[int(ei)], dtype=np.complex128) - np.asarray(rpA[int(ei)], dtype=np.complex128)
                    dmR = np.asarray(rpB2[int(ei)], dtype=np.complex128) - np.asarray(rpB[int(ei)], dtype=np.complex128)
                    cos_mu_lam_resid.append(direction_cosine_fro(dlR, dmR))
                    nlR = float(np.linalg.norm(dlR.ravel()))
                    nmR = float(np.linalg.norm(dmR.ravel()))
                    if np.isfinite(nlR) and nlR > 0 and np.isfinite(nmR):
                        norm_ratio_resid.append(float(nmR / nlR))
                    ranks_resid.append(effective_rank2(dlR, dmR))

        mu_cos_lam_med_per_key.append(_median_abs(cos_mu_lam))
        mu_norm_ratio_med_per_key.append(_median(norm_ratio))
        mu_rank_med_per_key.append(_median([float(x) for x in ranks]))

        if cos_mu_lam_S:
            mu_cos_S_med_per_key.append(_median_abs(cos_mu_lam_S))
        if norm_ratio_S:
            mu_norm_ratio_S_med_per_key.append(_median(norm_ratio_S))
        if ranks_S:
            mu_rank_S_med_per_key.append(_median([float(x) for x in ranks_S]))

        if cos_mu_lam_trace:
            mu_cos_trace_med_per_key.append(_median_abs(cos_mu_lam_trace))
        if norm_ratio_trace:
            mu_norm_ratio_trace_med_per_key.append(_median(norm_ratio_trace))
        if ranks_trace:
            mu_rank_trace_med_per_key.append(_median([float(x) for x in ranks_trace]))

        if cos_mu_lam_logdet:
            mu_cos_logdet_med_per_key.append(_median_abs(cos_mu_lam_logdet))
        if norm_ratio_logdet:
            mu_norm_ratio_logdet_med_per_key.append(_median(norm_ratio_logdet))
        if ranks_logdet:
            mu_rank_logdet_med_per_key.append(_median([float(x) for x in ranks_logdet]))

        if cos_mu_lam_series:
            mu_cos_series_med_per_key.append(_median_abs(cos_mu_lam_series))
        if norm_ratio_series:
            mu_norm_ratio_series_med_per_key.append(_median(norm_ratio_series))
        if ranks_series:
            mu_rank_series_med_per_key.append(_median([float(x) for x in ranks_series]))

        if cos_mu_lam_resid:
            mu_cos_resid_med_per_key.append(_median_abs(cos_mu_lam_resid))
        if norm_ratio_resid:
            mu_norm_ratio_resid_med_per_key.append(_median(norm_ratio_resid))
        if ranks_resid:
            mu_rank_resid_med_per_key.append(_median([float(x) for x in ranks_resid]))

    kpis: Dict[str, Any] = {
        "lambda_snap": {
            "n_keys": int(len(mu_cos_lam_med_per_key)),
            "mu_cos_to_lambda_abs_median": _median(mu_cos_lam_med_per_key),
            "mu_norm_ratio_median": _median(mu_norm_ratio_med_per_key),
            "mu_effective_rank_median": _median(mu_rank_med_per_key),
        },
        "S_snap": {
            "n_keys": int(len(mu_cos_S_med_per_key)),
            "mu_cos_to_lambda_abs_median": _median(mu_cos_S_med_per_key),
            "mu_norm_ratio_median": _median(mu_norm_ratio_S_med_per_key),
            "mu_effective_rank_median": _median(mu_rank_S_med_per_key),
        },
        "trace_powers": {
            "n_keys": int(len(mu_cos_trace_med_per_key)),
            "mu_cos_to_lambda_abs_median": _median(mu_cos_trace_med_per_key),
            "mu_norm_ratio_median": _median(mu_norm_ratio_trace_med_per_key),
            "mu_effective_rank_median": _median(mu_rank_trace_med_per_key),
        },
        "logdet_IminusS": {
            "n_keys": int(len(mu_cos_logdet_med_per_key)),
            "mu_cos_to_lambda_abs_median": _median(mu_cos_logdet_med_per_key),
            "mu_norm_ratio_median": _median(mu_norm_ratio_logdet_med_per_key),
            "mu_effective_rank_median": _median(mu_rank_logdet_med_per_key),
        },
        "series_partial": {
            "n_keys": int(len(mu_cos_series_med_per_key)),
            "mu_cos_to_lambda_abs_median": _median(mu_cos_series_med_per_key),
            "mu_norm_ratio_median": _median(mu_norm_ratio_series_med_per_key),
            "mu_effective_rank_median": _median(mu_rank_series_med_per_key),
        },
        "resid_partial": {
            "n_keys": int(len(mu_cos_resid_med_per_key)),
            "mu_cos_to_lambda_abs_median": _median(mu_cos_resid_med_per_key),
            "mu_norm_ratio_median": _median(mu_norm_ratio_resid_med_per_key),
            "mu_effective_rank_median": _median(mu_rank_resid_med_per_key),
        },
    }

    return kpis, int(len(keys)), int(failed)


def load_keys_from_suite_rows_csv(suite_out_root: str) -> List[Tuple[int, int, float, float]]:
    csv_path = os.path.join(suite_out_root, "phase3e_elambda_suite_rows.csv")
    if not os.path.exists(csv_path):
        return []

    keys: List[Tuple[int, int, float, float]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                seed = int(row.get("seed", "-1"))
                anchor = int(row.get("anchor_seed", "-1"))
                wlo = float(row.get("wlo", "nan"))
                whi = float(row.get("whi", "nan"))
            except Exception:
                continue
            if seed < 0 or anchor < 0 or (not np.isfinite(wlo)) or (not np.isfinite(whi)):
                continue
            keys.append((seed, anchor, wlo, whi))

    # unique, stable order
    return sorted(set(keys))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite_out_root", required=True, help="Folder containing phase3e_elambda_suite_summary.json")
    ap.add_argument("--inplace", action="store_true", help="Update suite summary JSON in place")
    args = ap.parse_args()

    suite_out_root = _resolve_path(args.suite_out_root)
    summary_path = os.path.join(suite_out_root, "phase3e_elambda_suite_summary.json")
    if not os.path.exists(summary_path):
        raise SystemExit(f"Missing suite summary: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        summ = json.load(f)

    out_root_a = _resolve_path(summ.get("out_root_a", ""))
    out_root_b = _resolve_path(summ.get("out_root_b", ""))
    out_root_b2 = _resolve_path(summ.get("out_root_b2", ""))

    within_backend = str(summ.get("within_backend", "off"))
    within_backend_anchor_offset = int(summ.get("within_backend_anchor_offset", 1))
    direction_cosine_samples = int(summ.get("direction_cosine_samples", 3))

    restrict_keys = load_keys_from_suite_rows_csv(suite_out_root)

    kpis, n_pairs, n_failed = compute_mu_direction_kpis(
        out_root_a=out_root_a,
        out_root_b=out_root_b,
        out_root_b2=out_root_b2,
        within_backend=within_backend,
        within_backend_anchor_offset=within_backend_anchor_offset,
        direction_cosine_samples=direction_cosine_samples,
        restrict_keys=restrict_keys,
    )

    summ["mu_direction_kpis"] = kpis
    summ["mu_direction_kpis_recomputed"] = True
    summ["mu_direction_kpis_recomputed_n_pairs"] = int(n_pairs)
    summ["mu_direction_kpis_recomputed_n_failed"] = int(n_failed)

    if args.inplace:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summ, f, indent=2)
            f.write("\n")
        print(f"Updated: {summary_path}")
    else:
        print(json.dumps({"summary_path": summary_path, "mu_direction_kpis": kpis}, indent=2))


if __name__ == "__main__":
    main()
