import argparse
import os
from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FileDiffStats:
    rel_path: str
    d_lambda_max_abs: float
    d_lambda_rel_fro: float
    d_S_max_abs: float
    d_S_rel_fro: float


ArtifactKey = Tuple[int, int, str, str, str]


_RE_SEED_ANCHOR = re.compile(r"seed(?P<seed>\d+)_anchor(?P<anchor>\d+)")
_RE_WINDOW = re.compile(r"window_(?P<wlo>-?\d+(?:\.\d+)?)_(?P<whi>-?\d+(?:\.\d+)?)")


def _parse_key_from_relpath(rel_path: str) -> Optional[ArtifactKey]:
    """Extract (seed, anchor, wlo, whi, filename) from a Phase-3D artifact relative path."""
    parts = rel_path.replace("\\", "/").split("/")
    if len(parts) < 4:
        return None
    m1 = _RE_SEED_ANCHOR.search(rel_path)
    m2 = _RE_WINDOW.search(rel_path)
    if not m1 or not m2:
        return None
    seed = int(m1.group("seed"))
    anchor = int(m1.group("anchor"))
    wlo = str(m2.group("wlo"))
    whi = str(m2.group("whi"))
    fname = parts[-1]
    return (seed, anchor, wlo, whi, fname)


def _iter_npz_files(root: str) -> Iterable[str]:
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".npz"):
                yield os.path.join(dirpath, fn)


def _iter_npz_files_filtered(root: str, filename: str) -> Iterable[str]:
    fn_want = str(filename).strip()
    if not fn_want:
        yield from _iter_npz_files(root)
        return
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn != fn_want:
                continue
            if fn.endswith(".npz"):
                yield os.path.join(dirpath, fn)


def _reservoir_sample_rel_paths(root: str, n: int, rng: np.random.Generator, filename: str = "") -> List[str]:
    """Sample up to n relative .npz paths from root without materializing the full list."""
    n = int(max(1, n))
    reservoir: List[str] = []
    seen = 0
    for p in _iter_npz_files_filtered(root, filename):
        rel = _safe_relpath(p, root)
        seen += 1
        if len(reservoir) < n:
            reservoir.append(rel)
        else:
            j = int(rng.integers(0, seen))
            if j < n:
                reservoir[j] = rel
    return sorted(set(reservoir))


def _reservoir_sample_keys(root: str, n: int, rng: np.random.Generator, filename: str = "") -> List[Tuple[ArtifactKey, str]]:
    """Sample up to n artifact keys from root without materializing the full list."""
    n = int(max(1, n))
    reservoir: List[Tuple[ArtifactKey, str]] = []
    seen = 0
    for p in _iter_npz_files_filtered(root, filename):
        rel = _safe_relpath(p, root)
        k = _parse_key_from_relpath(rel)
        if k is None:
            continue
        if filename and k[-1] != filename:
            continue
        seen += 1
        if len(reservoir) < n:
            reservoir.append((k, rel))
        else:
            j = int(rng.integers(0, seen))
            if j < n:
                reservoir[j] = (k, rel)
    # De-dupe by key (keep first)
    out: Dict[ArtifactKey, str] = {}
    for k, rel in reservoir:
        out.setdefault(k, rel)
    return sorted(out.items(), key=lambda t: t[1])


def _safe_relpath(path: str, root: str) -> str:
    rel = os.path.relpath(path, root)
    return rel.replace("\\", "/")


def _complex_to_real_vec(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex128).ravel()
    return np.concatenate([np.real(x), np.imag(x)])


def _fro_rel(diff: np.ndarray, base: np.ndarray, eps: float = 1e-18) -> float:
    nd = float(np.linalg.norm(np.asarray(diff).ravel()))
    nb = float(np.linalg.norm(np.asarray(base).ravel()))
    return float(nd / max(nb, eps))


def _load_arrays(npz_path: str, keys: Sequence[str]) -> Dict[str, np.ndarray]:
    with np.load(npz_path) as d:
        out: Dict[str, np.ndarray] = {}
        for k in keys:
            if k in d.files:
                out[k] = np.asarray(d[k], dtype=np.complex128)
        return out


def _diff_one(ref_root: str, cand_root: str, rel_path: str, keys: Sequence[str]) -> Optional[FileDiffStats]:
    pref = os.path.join(ref_root, rel_path)
    pcand = os.path.join(cand_root, rel_path)
    if not os.path.exists(pref) or not os.path.exists(pcand):
        return None

    ref = _load_arrays(pref, keys)
    cand = _load_arrays(pcand, keys)

    if "lambda_snap" not in ref or "lambda_snap" not in cand:
        return None

    lam_ref = ref["lambda_snap"]
    lam_cand = cand["lambda_snap"]
    if lam_ref.shape != lam_cand.shape:
        return None

    dlam = lam_cand - lam_ref
    d_lambda_max_abs = float(np.max(np.abs(dlam)))
    d_lambda_rel_fro = _fro_rel(dlam, lam_ref)

    d_S_max_abs = float("nan")
    d_S_rel_fro = float("nan")
    if "S_snap" in ref and "S_snap" in cand:
        S_ref = ref["S_snap"]
        S_cand = cand["S_snap"]
        if S_ref.shape == S_cand.shape:
            dS = S_cand - S_ref
            d_S_max_abs = float(np.max(np.abs(dS)))
            d_S_rel_fro = _fro_rel(dS, S_ref)

    return FileDiffStats(
        rel_path=rel_path,
        d_lambda_max_abs=d_lambda_max_abs,
        d_lambda_rel_fro=d_lambda_rel_fro,
        d_S_max_abs=d_S_max_abs,
        d_S_rel_fro=d_S_rel_fro,
    )


def _build_candidate_paths_for_keys(cand_root: str, wanted: Sequence[ArtifactKey], filename: str = "") -> Dict[ArtifactKey, str]:
    wanted_set = set(wanted)
    out: Dict[ArtifactKey, str] = {}
    if not wanted_set:
        return out

    # Fast path: try direct existence checks under top-level backend_* directories.
    try:
        backend_dirs = [
            d
            for d in os.listdir(cand_root)
            if d.startswith("backend_") and os.path.isdir(os.path.join(cand_root, d))
        ]
    except Exception:
        backend_dirs = []
    backend_dirs = sorted(set(backend_dirs))

    for k in wanted_set:
        seed, anchor, wlo, whi, fname = k
        if filename and fname != filename:
            continue
        for bd in backend_dirs:
            rel = f"{bd}/seed{seed}_anchor{anchor}/window_{wlo}_{whi}/{fname}"
            p = os.path.join(cand_root, rel)
            if os.path.exists(p):
                out[k] = rel
                break
        if len(out) >= len(wanted_set):
            break

    # Fallback: if some keys still missing, do a filtered walk but stop early.
    missing = wanted_set.difference(out.keys())
    if missing:
        for p in _iter_npz_files_filtered(cand_root, filename):
            rel = _safe_relpath(p, cand_root)
            kk = _parse_key_from_relpath(rel)
            if kk is None:
                continue
            if filename and kk[-1] != filename:
                continue
            if kk in missing and kk not in out:
                out[kk] = rel
                missing.remove(kk)
                if not missing:
                    break
    return out


def _cos_abs_and_rank_and_ratio(dl: np.ndarray, dm: np.ndarray) -> Tuple[float, float, float]:
    dl = np.asarray(dl, dtype=np.complex128)
    dm = np.asarray(dm, dtype=np.complex128)
    # cosine
    n1 = float(np.linalg.norm(dl.ravel()))
    n2 = float(np.linalg.norm(dm.ravel()))
    if (not np.isfinite(n1)) or (not np.isfinite(n2)) or n1 <= 0.0 or n2 <= 0.0:
        cos_abs = float("nan")
        ratio = float("nan")
        rk = float("nan")
    else:
        ip = np.vdot(dl.ravel(), dm.ravel())
        cos_abs = float(abs(np.real(ip) / (n1 * n2)))
        ratio = float(n2 / n1)
        rk = float(effective_rank2(dl, dm))
    return cos_abs, rk, ratio


def _complex_flatten_realimag(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex128).ravel()
    return np.concatenate([np.real(x), np.imag(x)])


def effective_rank2(v1: np.ndarray, v2: np.ndarray, rtol: float = 1e-6) -> int:
    a = _complex_flatten_realimag(v1)
    b = _complex_flatten_realimag(v2)
    if a.size != b.size or a.size == 0:
        return 0
    M = np.stack([a, b], axis=0).astype(np.float64, copy=False)
    try:
        s = np.linalg.svd(M, compute_uv=False)
    except Exception:
        return 0
    if s.size == 0 or (not np.isfinite(s).all()):
        return 0
    s0 = float(np.max(s))
    if (not np.isfinite(s0)) or s0 <= 0.0:
        return 0
    thr = float(rtol) * s0
    return int(np.sum(s > thr))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Rank candidate Phase-3D roots by how different they are from a reference root, "
            "based on matched NPZ artifacts (same relative path). Useful for finding a non-degenerate B2 endpoint." 
        )
    )
    ap.add_argument("--ref_root", required=True, help="Reference Phase-3D root (A or B1)")
    ap.add_argument(
        "--candidates",
        default="out_phase3D_channel_diag_*",
        help="Glob (relative to --root) selecting candidate roots (default out_phase3D_channel_diag_*)",
    )
    ap.add_argument("--root", default=".", help="Workspace root containing the Phase-3D folders (default .)")
    ap.add_argument("--n_files", type=int, default=12, help="Number of matched artifact files to sample (default 12)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for sampling matched files (default 0)")
    ap.add_argument(
        "--keys",
        default="lambda_snap,S_snap",
        help="Comma list of NPZ keys to compare (default lambda_snap,S_snap)",
    )
    ap.add_argument(
        "--match_mode",
        default="key",
        choices=["relpath", "key"],
        help="relpath: require identical relative paths; key: match by (seed,anchor,window,filename) (default key)",
    )
    ap.add_argument(
        "--out_csv",
        default="b2_independence_scan.csv",
        help="Where to write the ranking CSV (default b2_independence_scan.csv)",
    )
    ap.add_argument(
        "--a_root",
        default="",
        help="Optional A-root. When set and --match_mode=key, also computes µ-direction KPIs using dl=(B1-A), dm=(B2-B1).",
    )
    ap.add_argument(
        "--filename",
        default="channel_diag__E3_K8.npz",
        help="Only consider artifacts with this filename (default channel_diag__E3_K8.npz). Empty disables filter.",
    )
    ap.add_argument(
        "--e_samples",
        type=int,
        default=3,
        help="Number of E snapshots to sample for KPI cos/rank/ratio (default 3).",
    )
    ap.add_argument(
        "--limit_candidates",
        type=int,
        default=0,
        help="If >0, only scan the first N candidate roots after glob expansion (useful for quick iterations).",
    )

    args = ap.parse_args()

    workspace_root = os.path.abspath(str(args.root))
    ref_root = os.path.abspath(os.path.join(workspace_root, str(args.ref_root)))
    keys = [k.strip() for k in str(args.keys).split(",") if k.strip()]
    filename = str(args.filename).strip()

    a_root = str(args.a_root).strip()
    a_root_abs = os.path.abspath(os.path.join(workspace_root, a_root)) if a_root else ""
    if a_root and (not os.path.isdir(a_root_abs)):
        raise SystemExit(f"a_root not found: {a_root_abs}")

    if not os.path.isdir(ref_root):
        raise SystemExit(f"ref_root not found: {ref_root}")

    rng = np.random.default_rng(int(args.seed))
    n_files = int(max(1, args.n_files))
    match_mode = str(args.match_mode)
    if match_mode == "relpath":
        sample_rel_paths = _reservoir_sample_rel_paths(ref_root, n_files, rng, filename=filename)
        if not sample_rel_paths:
            raise SystemExit(f"No .npz files found under ref_root: {ref_root}")
        sample_keys: List[ArtifactKey] = []
        ref_key_to_rel: Dict[ArtifactKey, str] = {}
    else:
        pairs = _reservoir_sample_keys(ref_root, n_files, rng, filename=filename)
        if not pairs:
            raise SystemExit(f"No .npz files found under ref_root: {ref_root}")
        ref_key_to_rel = dict(pairs)
        sample_keys = list(ref_key_to_rel.keys())
        sample_rel_paths = []

    # Enumerate candidates.
    import glob

    cand_dirs = [
        os.path.abspath(p)
        for p in glob.glob(os.path.join(workspace_root, str(args.candidates)))
        if os.path.isdir(p)
    ]
    cand_dirs = sorted(set(cand_dirs))
    lim = int(args.limit_candidates)
    if lim > 0:
        cand_dirs = cand_dirs[:lim]

    rows: List[Dict[str, object]] = []

    # If computing KPIs, pre-build A mapping once.
    a_key_to_rel: Dict[ArtifactKey, str] = {}
    if match_mode == "key" and a_root_abs:
        a_key_to_rel = _build_candidate_paths_for_keys(a_root_abs, sample_keys, filename=filename)

    for cand in cand_dirs:
        if os.path.samefile(cand, ref_root):
            continue

        cand_key_to_rel: Dict[ArtifactKey, str] = {}
        if match_mode == "key":
            cand_key_to_rel = _build_candidate_paths_for_keys(cand, sample_keys, filename=filename)

        stats: List[FileDiffStats] = []
        if match_mode == "relpath":
            for rel in sample_rel_paths:
                s = _diff_one(ref_root, cand, rel, keys=keys)
                if s is not None:
                    stats.append(s)
        else:
            for k in sample_keys:
                rel_ref = ref_key_to_rel.get(k)
                rel_cand = cand_key_to_rel.get(k)
                if (rel_ref is None) or (rel_cand is None):
                    continue
                # Diffing requires file paths; use the reference rel_path field just for provenance.
                pref = os.path.join(ref_root, rel_ref)
                pcand = os.path.join(cand, rel_cand)
                ref = _load_arrays(pref, keys)
                cand_arr = _load_arrays(pcand, keys)
                if "lambda_snap" not in ref or "lambda_snap" not in cand_arr:
                    continue
                lam_ref = ref["lambda_snap"]
                lam_cand = cand_arr["lambda_snap"]
                if lam_ref.shape != lam_cand.shape:
                    continue
                dlam = lam_cand - lam_ref
                d_lambda_max_abs = float(np.max(np.abs(dlam)))
                d_lambda_rel_fro = _fro_rel(dlam, lam_ref)
                d_S_max_abs = float("nan")
                d_S_rel_fro = float("nan")
                if "S_snap" in ref and "S_snap" in cand_arr:
                    S_ref = ref["S_snap"]
                    S_cand = cand_arr["S_snap"]
                    if S_ref.shape == S_cand.shape:
                        dS = S_cand - S_ref
                        d_S_max_abs = float(np.max(np.abs(dS)))
                        d_S_rel_fro = _fro_rel(dS, S_ref)
                stats.append(
                    FileDiffStats(
                        rel_path=rel_ref,
                        d_lambda_max_abs=d_lambda_max_abs,
                        d_lambda_rel_fro=d_lambda_rel_fro,
                        d_S_max_abs=d_S_max_abs,
                        d_S_rel_fro=d_S_rel_fro,
                    )
                )

        # Optional µ-direction KPIs.
        kpi_cos_lam: List[float] = []
        kpi_rank_lam: List[float] = []
        kpi_ratio_lam: List[float] = []
        kpi_cos_S: List[float] = []
        kpi_rank_S: List[float] = []
        kpi_ratio_S: List[float] = []

        if match_mode == "key" and a_root_abs:
            nsamp = int(max(2, int(args.e_samples)))
            for k in sample_keys:
                relA = a_key_to_rel.get(k)
                relB1 = ref_key_to_rel.get(k)
                relB2 = cand_key_to_rel.get(k)
                if (relA is None) or (relB1 is None) or (relB2 is None):
                    continue
                pref = os.path.join(a_root_abs, relA)
                pb1 = os.path.join(ref_root, relB1)
                pb2 = os.path.join(cand, relB2)
                if not (os.path.exists(pref) and os.path.exists(pb1) and os.path.exists(pb2)):
                    continue
                A = _load_arrays(pref, ["lambda_snap", "S_snap"])
                B1 = _load_arrays(pb1, ["lambda_snap", "S_snap"])
                B2 = _load_arrays(pb2, ["lambda_snap", "S_snap"])
                if ("lambda_snap" not in A) or ("lambda_snap" not in B1) or ("lambda_snap" not in B2):
                    continue
                lamA = A["lambda_snap"]
                lamB1 = B1["lambda_snap"]
                lamB2 = B2["lambda_snap"]
                if lamA.shape != lamB1.shape or lamA.shape != lamB2.shape:
                    continue
                nE = int(lamA.shape[0])
                if nE < 2:
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

                cos_vals: List[float] = []
                rk_vals: List[float] = []
                ratio_vals: List[float] = []
                cosS_vals: List[float] = []
                rkS_vals: List[float] = []
                ratioS_vals: List[float] = []

                has_S = ("S_snap" in A) and ("S_snap" in B1) and ("S_snap" in B2)
                SA = A.get("S_snap")
                SB1 = B1.get("S_snap")
                SB2 = B2.get("S_snap")

                for ei in idx:
                    dl = np.asarray(lamB1[int(ei)], dtype=np.complex128) - np.asarray(lamA[int(ei)], dtype=np.complex128)
                    dm = np.asarray(lamB2[int(ei)], dtype=np.complex128) - np.asarray(lamB1[int(ei)], dtype=np.complex128)
                    c, rk, rr = _cos_abs_and_rank_and_ratio(dl, dm)
                    cos_vals.append(c)
                    rk_vals.append(rk)
                    ratio_vals.append(rr)

                    if has_S and SA is not None and SB1 is not None and SB2 is not None:
                        if SA.shape == SB1.shape and SA.shape == SB2.shape and int(ei) < int(SA.shape[0]):
                            dlS = np.asarray(SB1[int(ei)], dtype=np.complex128) - np.asarray(SA[int(ei)], dtype=np.complex128)
                            dmS = np.asarray(SB2[int(ei)], dtype=np.complex128) - np.asarray(SB1[int(ei)], dtype=np.complex128)
                            cS, rkS, rrS = _cos_abs_and_rank_and_ratio(dlS, dmS)
                            cosS_vals.append(cS)
                            rkS_vals.append(rkS)
                            ratioS_vals.append(rrS)

                cA = np.asarray(cos_vals, dtype=np.float64)
                cA = cA[np.isfinite(cA)]
                if cA.size:
                    kpi_cos_lam.append(float(np.nanmedian(cA)))
                rA = np.asarray(rk_vals, dtype=np.float64)
                rA = rA[np.isfinite(rA)]
                if rA.size:
                    kpi_rank_lam.append(float(np.nanmedian(rA)))
                rrA = np.asarray(ratio_vals, dtype=np.float64)
                rrA = rrA[np.isfinite(rrA)]
                if rrA.size:
                    kpi_ratio_lam.append(float(np.nanmedian(rrA)))

                if cosS_vals:
                    cS = np.asarray(cosS_vals, dtype=np.float64)
                    cS = cS[np.isfinite(cS)]
                    if cS.size:
                        kpi_cos_S.append(float(np.nanmedian(cS)))
                if rkS_vals:
                    rS = np.asarray(rkS_vals, dtype=np.float64)
                    rS = rS[np.isfinite(rS)]
                    if rS.size:
                        kpi_rank_S.append(float(np.nanmedian(rS)))
                if ratioS_vals:
                    rrS = np.asarray(ratioS_vals, dtype=np.float64)
                    rrS = rrS[np.isfinite(rrS)]
                    if rrS.size:
                        kpi_ratio_S.append(float(np.nanmedian(rrS)))

        if not stats:
            rows.append(
                {
                    "candidate": os.path.basename(cand),
                    "n_matched": 0,
                    "d_lambda_max_abs_max": float("nan"),
                    "d_lambda_rel_fro_median": float("nan"),
                    "d_S_rel_fro_median": float("nan"),
                    "d_S_max_abs_max": float("nan"),
                    "mu_cos_to_lambda_abs_median": float("nan"),
                    "mu_effective_rank_median": float("nan"),
                    "mu_norm_ratio_median": float("nan"),
                    "mu_cos_to_lambda_abs_median_S": float("nan"),
                    "mu_effective_rank_median_S": float("nan"),
                    "mu_norm_ratio_median_S": float("nan"),
                }
            )
            continue

        dlam_max = np.asarray([s.d_lambda_max_abs for s in stats], dtype=np.float64)
        dlam_rel = np.asarray([s.d_lambda_rel_fro for s in stats], dtype=np.float64)
        dS_rel = np.asarray([s.d_S_rel_fro for s in stats], dtype=np.float64)
        dS_max = np.asarray([s.d_S_max_abs for s in stats], dtype=np.float64)

        rows.append(
            {
                "candidate": os.path.basename(cand),
                "n_matched": int(len(stats)),
                "d_lambda_max_abs_max": float(np.nanmax(dlam_max)),
                "d_lambda_rel_fro_median": float(np.nanmedian(dlam_rel)),
                "d_S_rel_fro_median": float(np.nanmedian(dS_rel)),
                "d_S_max_abs_max": float(np.nanmax(dS_max)),
                "mu_cos_to_lambda_abs_median": float(np.nanmedian(np.asarray(kpi_cos_lam, dtype=np.float64))) if kpi_cos_lam else float("nan"),
                "mu_effective_rank_median": float(np.nanmedian(np.asarray(kpi_rank_lam, dtype=np.float64))) if kpi_rank_lam else float("nan"),
                "mu_norm_ratio_median": float(np.nanmedian(np.asarray(kpi_ratio_lam, dtype=np.float64))) if kpi_ratio_lam else float("nan"),
                "mu_cos_to_lambda_abs_median_S": float(np.nanmedian(np.asarray(kpi_cos_S, dtype=np.float64))) if kpi_cos_S else float("nan"),
                "mu_effective_rank_median_S": float(np.nanmedian(np.asarray(kpi_rank_S, dtype=np.float64))) if kpi_rank_S else float("nan"),
                "mu_norm_ratio_median_S": float(np.nanmedian(np.asarray(kpi_ratio_S, dtype=np.float64))) if kpi_ratio_S else float("nan"),
            }
        )

    df = pd.DataFrame(rows)
    # If KPIs exist, prioritize finding true 2D direction: high rank + low cosine.
    if "mu_effective_rank_median" in df.columns and df["mu_effective_rank_median"].notna().any():
        df["sort_rank"] = df["mu_effective_rank_median"].copy()
        df["sort_cos"] = df["mu_cos_to_lambda_abs_median"].copy()
        df = df.sort_values(["sort_rank", "sort_cos", "d_S_rel_fro_median", "n_matched"], ascending=[False, True, False, False]).drop(
            columns=["sort_rank", "sort_cos"]
        )
    else:
        # Prefer S_snap difference when available; fall back to lambda_snap.
        df["sort_key"] = df["d_S_rel_fro_median"].copy()
        df.loc[~np.isfinite(df["sort_key"]), "sort_key"] = df.loc[~np.isfinite(df["sort_key"]), "d_lambda_rel_fro_median"]
        df = df.sort_values(["sort_key", "n_matched"], ascending=[False, False]).drop(columns=["sort_key"])

    out_csv = os.path.abspath(os.path.join(workspace_root, str(args.out_csv)))
    df.to_csv(out_csv, index=False)

    # Print a short leaderboard.
    top = df.head(10)
    print(f"[scan_b2_independence] wrote: {out_csv}")
    with pd.option_context("display.max_columns", 999, "display.width", 140):
        print(top.to_string(index=False))


if __name__ == "__main__":
    main()
