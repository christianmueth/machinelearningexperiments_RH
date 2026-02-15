import argparse
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import phase3e_elambda_loop_suite as s


def _resolve_artifact_root(out_root: str, p: str) -> str:
    """Resolve artifact roots referenced in a Phase-3E summary.

    Summaries sometimes store out_root_a/out_root_b/out_root_b2 relative to the
    workspace root, while out_root can be nested (e.g. out_phase3E.../refine_*).
    Try a few reasonable bases.
    """
    if not p:
        return ""
    p = str(p)
    if os.path.isabs(p):
        return p

    out_root = os.path.abspath(str(out_root))
    ws_root = os.path.abspath(os.path.join(out_root, os.pardir))
    ws_root2 = os.path.abspath(os.path.join(ws_root, os.pardir))
    ws_root3 = os.path.abspath(os.path.join(ws_root2, os.pardir))
    here = os.path.abspath(os.getcwd())
    script_dir = os.path.abspath(os.path.dirname(__file__))

    candidates = [
        os.path.join(ws_root, p),
        os.path.join(ws_root2, p),
        os.path.join(ws_root3, p),
        os.path.join(here, p),
        os.path.join(script_dir, p),
        os.path.join(os.path.abspath(os.path.join(script_dir, os.pardir)), p),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    return os.path.join(ws_root, p)



def angle_wrap_pi(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return ((x + np.pi) % (2.0 * np.pi)) - np.pi


def load_v_minus1(npz_path: str) -> Optional[np.ndarray]:
    try:
        z = np.load(npz_path, allow_pickle=True)
        v = np.asarray(z["v_minus1"], dtype=np.complex128)
    except Exception:
        return None
    vn = float(np.linalg.norm(v))
    if not np.isfinite(vn) or vn <= 0:
        return None
    return v / vn


def _resolve_dump_path(dump_root: str, holonomy_dump: str) -> str:
    p = str(holonomy_dump).replace("\\", "/")
    if not p:
        return ""
    if os.path.isabs(p):
        return p
    return os.path.join(dump_root, p)


def _merge_dump_index(df: pd.DataFrame, dump_index_csv: str) -> pd.DataFrame:
    if not dump_index_csv:
        return df
    if not os.path.exists(dump_index_csv):
        return df

    di = pd.read_csv(dump_index_csv)
    merge_cols = [
        "seed",
        "anchor_seed",
        "wlo",
        "whi",
        "loop_e0",
        "loop_e1",
        "lambda_j0",
        "lambda_j1",
        "block",
        "dim",
    ]

    keep = [c for c in merge_cols if c in df.columns and c in di.columns]
    if not keep or ("holonomy_dump" not in di.columns):
        return df

    out = df.merge(di[keep + ["holonomy_dump"]], on=keep, how="left", suffixes=("", "_dump"))
    if "holonomy_dump_dump" in out.columns:
        out["holonomy_dump"] = out["holonomy_dump_dump"].where(out["holonomy_dump_dump"].notna(), out.get("holonomy_dump", ""))
        out.drop(columns=["holonomy_dump_dump"], inplace=True)
    return out


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(int(n)))
        self.rank = [0 for _ in range(int(n))]

    def find(self, x: int) -> int:
        p = self.parent[int(x)]
        if p != int(x):
            self.parent[int(x)] = self.find(p)
        return self.parent[int(x)]

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _quantile(vals: Sequence[float], q: float) -> float:
    v = np.asarray([float(x) for x in vals], dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.nanquantile(v, q))


@dataclass(frozen=True)
class CellKey:
    seed: int
    anchor_seed: int
    wlo: float
    whi: float
    block: int
    dim: int
    loop_e0: int
    loop_e1: int
    lambda0: float
    lambda1: float


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True, help="Phase-3E run folder containing phase3e_elambda_suite_rows.csv")
    ap.add_argument(
        "--dump_index_csv",
        default="",
        help="Optional CSV (phase3f_pi_dump_index.csv) to merge holonomy_dump paths onto rows when absent.",
    )
    ap.add_argument(
        "--dump_root",
        default="",
        help="Base directory for resolving relative holonomy_dump paths (defaults to --out_root).",
    )
    ap.add_argument("--tol_pi", type=float, default=0.25, help="| |wrap(det_phase)| - pi | <= tol (radians)")
    ap.add_argument(
        "--tol_pi_discovery",
        type=float,
        default=float("nan"),
        help="If finite: use this tol for forming candidate pi-cells (default: use --tol_pi).",
    )
    ap.add_argument(
        "--tol_pi_confirm",
        type=float,
        default=float("nan"),
        help="If finite: use this tol for confirming a cell actually has pi rows (default: use --tol_pi).",
    )
    ap.add_argument("--min_pi_rows_per_cell", type=int, default=3, help="Only cluster cells with at least this many pi rows (discovery tol)")
    ap.add_argument(
        "--min_pi_rows_per_cell_auto_frac",
        type=float,
        default=0.01,
        help=(
            "If --min_pi_rows_per_cell <= 0: set it to max(1, floor(frac * N_pi_total)) using discovery tol (default frac=0.01)."
        ),
    )
    ap.add_argument(
        "--min_pi_rows_per_cell_confirm",
        type=int,
        default=1,
        help="Require at least this many pi rows under confirm tol for a cell to be kept (default 1).",
    )

    ap.add_argument(
        "--use_lasso",
        type=int,
        default=0,
        help=(
            "If 1: compute basepoint-aligned holonomies using lasso loops (shared i_base/j_base) and derive flip vectors from those. "
            "This makes cross-E flip overlaps meaningful. If 0: use v_minus1 from rectangle-loop dumps/indices."
        ),
    )
    ap.add_argument(
        "--i_base",
        type=int,
        default=-1,
        help="Lasso base energy index (default: min(loop_e0,loop_e1) for each pair).",
    )
    ap.add_argument(
        "--j_base",
        type=int,
        default=-1,
        help="Lasso base lambda index (default: min(lambda_j0,lambda_j1) for each pair).",
    )
    ap.add_argument(
        "--refine_steps",
        type=int,
        default=-1,
        help="Refine steps along E connectors (default: use loop_refine_steps from rows when available).",
    )
    ap.add_argument(
        "--eta",
        type=float,
        default=-1.0,
        help="If >0 fixed eta for lasso recompute; else use min(cayley_eta_a,cayley_eta_b).",
    )
    ap.add_argument(
        "--lambda_tol",
        type=float,
        default=1e-6,
        help="Max |lambda_mid - lambda_mid'| for allowing merges (default very strict; increase for refined grids).",
    )
    ap.add_argument(
        "--lambda_tol_no_vec",
        type=float,
        default=1e-12,
        help="When flip vectors are missing, only merge if lambda mids match within this tighter tolerance.",
    )
    ap.add_argument(
        "--flip_overlap_tol",
        type=float,
        default=0.90,
        help="Require |<v_minus1, v_minus1'>| >= tol to merge cells across E.",
    )
    ap.add_argument(
        "--max_dim_for_flip",
        type=int,
        default=32,
        help="Only attempt to load/compare flip-mode vectors when dim <= this.",
    )
    ap.add_argument(
        "--max_cells_per_pair",
        type=int,
        default=500,
        help="Safety cap: max event cells per (seed,anchor,wlo,whi,block,dim) for clustering.",
    )
    args = ap.parse_args()

    out_root = str(args.out_root)
    dump_root = str(args.dump_root).strip() or out_root

    rows_csv = os.path.join(out_root, "phase3e_elambda_suite_rows.csv")
    if not os.path.exists(rows_csv):
        raise SystemExit(f"Missing rows CSV: {rows_csv}")

    df = pd.read_csv(rows_csv)
    df = _merge_dump_index(df, str(args.dump_index_csv).strip())

    # Optional: suite summary used to locate artifact roots and lambdas_eff for lasso recompute.
    summ: Dict[str, Any] = {}
    summ_path = os.path.join(out_root, "phase3e_elambda_suite_summary.json")
    if os.path.exists(summ_path):
        try:
            import json

            with open(summ_path, "r", encoding="utf-8") as f:
                summ = json.load(f)
        except Exception:
            summ = {}

    fwd = df[
        (df.get("loop_type") == "rectangle")
        & (df.get("loop_direction") == "fwd")
        & (df.get("loop_double") == False)
        & (df.get("valid", True) == True)
    ].copy()

    if not len(fwd):
        raise SystemExit("No canonical fwd-rectangle valid rows found")

    lam0_col = "lambda_eff0" if "lambda_eff0" in fwd.columns else "lambda0"
    lam1_col = "lambda_eff1" if "lambda_eff1" in fwd.columns else "lambda1"

    tol_disc = float(args.tol_pi) if (not np.isfinite(float(args.tol_pi_discovery))) else float(args.tol_pi_discovery)
    tol_conf = float(args.tol_pi) if (not np.isfinite(float(args.tol_pi_confirm))) else float(args.tol_pi_confirm)

    ph = angle_wrap_pi(fwd["holonomy_det_phase"].values.astype(np.float64))
    aph = np.abs(ph)
    is_pi_disc = np.abs(aph - math.pi) <= float(tol_disc)
    is_pi_conf = np.abs(aph - math.pi) <= float(tol_conf)
    fwd["_is_pi_confirm"] = is_pi_conf
    pi = fwd[is_pi_disc].copy()

    if "holonomy_dump" not in pi.columns:
        pi["holonomy_dump"] = ""

    if bool(int(args.use_lasso) != 0):
        # Need these columns to recompute lassos.
        need_cols = ["artifact_a", "artifact_b", "lambda_j0", "lambda_j1", "blocks"]
        for c in need_cols:
            if c not in pi.columns:
                raise SystemExit(f"--use_lasso=1 requires column {c} in rows CSV")

        lambdas_eff = summ.get("lambdas_eff", summ.get("lambdas", []))
        if not lambdas_eff:
            raise SystemExit("--use_lasso=1 requires phase3e_elambda_suite_summary.json with lambdas_eff/lambdas")
        lambdas_eff = [float(x) for x in lambdas_eff]

        outA = str(summ.get("out_root_a", ""))
        outB = str(summ.get("out_root_b", ""))
        outB2 = str(summ.get("out_root_b2", "") or "").strip()
        if not outA or not outB:
            raise SystemExit("--use_lasso=1 requires out_root_a/out_root_b in suite summary")
        outA = _resolve_artifact_root(out_root, outA)
        outB = _resolve_artifact_root(out_root, outB)
        outB2 = _resolve_artifact_root(out_root, outB2) if outB2 else ""

        if not os.path.exists(outA) or not os.path.exists(outB):
            raise SystemExit(f"--use_lasso=1 could not resolve artifact roots: outA={outA} outB={outB}")
        if outB2 and (not os.path.exists(outB2)):
            raise SystemExit(f"--use_lasso=1 could not resolve out_root_b2: outB2={outB2}")

        if outB2:
            for c in ["mu", "artifact_b2"]:
                if c not in pi.columns:
                    raise SystemExit(f"--use_lasso=1 with out_root_b2 requires column {c} in rows CSV")

        # Cache loaded artifacts per pair key (seed,anchor,wlo,whi).
        loaded: Dict[Tuple[int, int, float, float], Dict[str, Any]] = {}

        def lasso_points(
            *,
            i_base: int,
            j_base: int,
            i0: int,
            i1: int,
            j0: int,
            j1: int,
            nE: int,
            refine_steps: int,
        ) -> List[Tuple[int, int]]:
            # Connector base -> (i_base,j0) in lambda
            pts: List[Tuple[int, int]] = []
            step = 1 if j0 >= j_base else -1
            for j in range(int(j_base), int(j0) + step, step):
                pts.append((int(i_base), int(j)))

            # Connector in energy: (i_base,j0) -> (i0,j0)
            ip = s.refine_path_indices(int(i_base), int(i0), int(nE), refine_steps=int(refine_steps))
            for ii in ip[1:]:
                pts.append((int(ii), int(j0)))

            # Rectangle at (i0,j0)
            i_path = s.refine_path_indices(int(i0), int(i1), int(nE), refine_steps=int(refine_steps))
            rect = s.loop_points_elambda(list(i_path), int(min(j0, j1)), int(max(j0, j1)), s.LoopSpec(loop_type="rectangle", direction="fwd"))
            if pts[-1] != rect[0]:
                raise ValueError("lasso connector did not meet rectangle")
            pts.extend(rect[1:])

            # Return connector: (i0,j0)->(i_base,j0)->(i_base,j_base)
            ip_back = s.refine_path_indices(int(i0), int(i_base), int(nE), refine_steps=int(refine_steps))
            for ii in ip_back[1:]:
                pts.append((int(ii), int(j0)))

            step2 = 1 if j_base >= j0 else -1
            for j in range(int(j0), int(j_base) + step2, step2):
                pts.append((int(i_base), int(j)))

            if pts[0] != pts[-1]:
                raise ValueError("lasso_points did not close")
            return pts

        def minus1_mode_vec(M: np.ndarray) -> Optional[np.ndarray]:
            M = np.asarray(M, dtype=np.complex128)
            vals, vecs = np.linalg.eig(M)
            dists = np.abs(vals - (-1.0 + 0.0j))
            k = int(np.argmin(dists))
            v = np.asarray(vecs[:, k], dtype=np.complex128)
            nrm = float(np.linalg.norm(v))
            if not np.isfinite(nrm) or nrm <= 0:
                return None
            return v / nrm

    # Build event-cell list: group by (pair,block,dim,loop_e0,loop_e1,lambda0,lambda1)
    group_cols = [
        "seed",
        "anchor_seed",
        "wlo",
        "whi",
        "mu",
        "block",
        "dim",
        "loop_e0",
        "loop_e1",
        lam0_col,
        lam1_col,
    ]

    if "mu" not in pi.columns:
        group_cols = [c for c in group_cols if c != "mu"]

    # ensure required cols exist
    for c in group_cols:
        if c not in pi.columns:
            raise SystemExit(f"rows CSV missing required column: {c}")

    # Auto min pi rows per cell (based on discovery tol) if requested.
    min_pi_rows = int(args.min_pi_rows_per_cell)
    if min_pi_rows <= 0:
        n_pi_total = int(np.sum(is_pi_disc))
        min_pi_rows = max(1, int(math.floor(float(args.min_pi_rows_per_cell_auto_frac) * float(n_pi_total))))
        if min_pi_rows <= 0:
            min_pi_rows = 1

    gb = pi.groupby(group_cols, dropna=False)

    cell_rows: List[Dict[str, Any]] = []

    for key, g in gb:
        if "mu" in group_cols:
            (
                seed,
                anchor,
                wlo,
                whi,
                mu0,
                block,
                dim,
                e0,
                e1,
                l0,
                l1,
            ) = key
        else:
            (
                seed,
                anchor,
                wlo,
                whi,
                block,
                dim,
                e0,
                e1,
                l0,
                l1,
            ) = key
            mu0 = float("nan")

        n_pi = int(len(g))
        n_pi_confirm = int(np.sum(g.get("_is_pi_confirm", False)))
        if n_pi < int(min_pi_rows):
            continue
        if n_pi_confirm < int(args.min_pi_rows_per_cell_confirm):
            continue

        # Representative row: prefer smallest dist-to-minus1 if available
        g2 = g.copy()
        if "holonomy_min_dist_to_minus1" in g2.columns:
            g2["_mind"] = g2["holonomy_min_dist_to_minus1"].astype(float)
        else:
            g2["_mind"] = float("nan")

        g2 = g2.sort_values(["_mind"], ascending=True)
        rep = g2.iloc[0]

        l0f = float(rep[lam0_col])
        l1f = float(rep[lam1_col])
        lmid = 0.5 * (l0f + l1f)

        dump_rel = str(rep.get("holonomy_dump", ""))
        dump_path = _resolve_dump_path(dump_root, dump_rel)
        have_dump = bool(dump_rel) and os.path.exists(dump_path)

        rep_v: Optional[np.ndarray] = None
        rep_dist = float(rep.get("holonomy_min_dist_to_minus1", float("nan")))
        if (not bool(int(args.use_lasso) != 0)) and int(dim) <= int(args.max_dim_for_flip) and have_dump:
            rep_v = load_v_minus1(dump_path)

        cell_rows.append(
            {
                "seed": int(seed),
                "anchor_seed": int(anchor),
                "wlo": float(wlo),
                "whi": float(whi),
                "mu": float(mu0),
                "block": int(block),
                "dim": int(dim),
                "loop_e0": int(e0),
                "loop_e1": int(e1),
                "lambda0": float(l0f),
                "lambda1": float(l1f),
                "lambda_mid": float(lmid),
                "lambda_j0": int(rep.get("lambda_j0", -1)),
                "lambda_j1": int(rep.get("lambda_j1", -1)),
                "loop_refine_steps": int(rep.get("loop_refine_steps", 0)),
                "blocks": int(rep.get("blocks", summ.get("blocks", 4))),
                "artifact_a": str(rep.get("artifact_a", "")),
                "artifact_b": str(rep.get("artifact_b", "")),
                "artifact_b2": str(rep.get("artifact_b2", "")),
                "n_pi": int(n_pi),
                "n_pi_confirm": int(n_pi_confirm),
                "rep_holonomy_min_dist_to_minus1": float(rep_dist),
                "rep_holonomy_dump": str(dump_rel),
                "rep_holonomy_dump_abs": str(dump_path) if have_dump else "",
                "rep_has_v_minus1": bool(rep_v is not None),
                "_rep_v": rep_v,
            }
        )

    if not cell_rows:
        raise SystemExit(
            "No event cells after thresholds; try loosening --tol_pi_discovery, lowering --min_pi_rows_per_cell (or <=0 for auto), "
            "or lowering --min_pi_rows_per_cell_confirm / loosening --tol_pi_confirm."
        )

    cells = pd.DataFrame(cell_rows)

    # If requested, compute basepoint-aligned flip vectors via lasso holonomies.
    # This is done once per (seed,anchor,wlo,whi,dim) and per rectangle key, and then reused across blocks.
    if bool(int(args.use_lasso) != 0):
        cells["_rep_v"] = None
        cells["rep_has_v_minus1"] = False

        lasso_group_cols = ["seed", "anchor_seed", "wlo", "whi", "dim"]
        if "mu" in cells.columns:
            lasso_group_cols.insert(4, "mu")
        for lasso_key, gg in cells.groupby(lasso_group_cols, dropna=False):
            if isinstance(lasso_key, tuple) and (len(lasso_key) == 6):
                seed, anchor, wlo, whi, mu0, dim = lasso_key
            else:
                seed, anchor, wlo, whi, dim = lasso_key
                mu0 = float("nan")
            gg = gg.copy().reset_index(drop=False)  # keep original row index

            # Shared basepoint per lasso group.
            i0s = gg[["loop_e0", "loop_e1"]].astype(int).values
            i_base_shared = int(args.i_base) if int(args.i_base) >= 0 else int(np.min(i0s))

            j0s = gg[["lambda_j0", "lambda_j1"]].astype(int).values
            j0s_ok = j0s[j0s >= 0]
            if j0s_ok.size == 0:
                continue
            j_base_shared = int(args.j_base) if int(args.j_base) >= 0 else int(np.min(j0s_ok))

            refine_steps_use = int(args.refine_steps)
            if refine_steps_use < 0:
                refine_steps_use = int(np.max(gg["loop_refine_steps"].astype(int).values))

            blocks_use = int(np.max(gg["blocks"].astype(int).values))

            pair_key = (int(seed), int(anchor), float(wlo), float(whi))
            if pair_key in loaded:
                pair = loaded[pair_key]
                lamA = pair["lamA"]
                lamB = pair["lamB"]
                lamB2 = pair.get("lamB2", None)
                etaA = pair["etaA"]
                etaB = pair["etaB"]
            else:
                relA = str(gg.loc[0, "artifact_a"]).replace("\\", "/")
                relB = str(gg.loc[0, "artifact_b"]).replace("\\", "/")
                pA = os.path.join(outA, relA)
                pB = os.path.join(outB, relB)
                if not (os.path.exists(pA) and os.path.exists(pB)):
                    continue
                dA = np.load(pA)
                dB = np.load(pB)
                lamA = np.asarray(dA["lambda_snap"], dtype=np.complex128)
                lamB = np.asarray(dB["lambda_snap"], dtype=np.complex128)
                lamB2 = None
                if outB2 and ("artifact_b2" in gg.columns):
                    relB2 = str(gg.loc[0, "artifact_b2"]).replace("\\", "/")
                    if relB2:
                        pB2 = os.path.join(outB2, relB2)
                        if os.path.exists(pB2):
                            dB2 = np.load(pB2)
                            if "lambda_snap" in dB2.files:
                                lamB2 = np.asarray(dB2["lambda_snap"], dtype=np.complex128)
                etaA = float(dA["cayley_eta"]) if ("cayley_eta" in dA.files) else float("nan")
                etaB = float(dB["cayley_eta"]) if ("cayley_eta" in dB.files) else float("nan")
                loaded[pair_key] = {"lamA": lamA, "lamB": lamB, "lamB2": lamB2, "etaA": etaA, "etaB": etaB}

            eta_use = float(args.eta)
            if (not np.isfinite(eta_use)) or eta_use <= 0:
                eta_use = float(min(float(etaA), float(etaB)))

            nE = int(lamA.shape[0])
            if not (nE > 1 and np.isfinite(eta_use) and eta_use > 0):
                continue

            i_base_n = int(s.normalize_index(int(i_base_shared), int(nE)))

            # Reuse basis cache within this lasso group for speed.
            basis_cache: Dict[Tuple[Any, ...], Tuple[List[np.ndarray], List[int], Dict[str, Any]]] = {}
            rect_cache: Dict[Tuple[Any, ...], List[Optional[np.ndarray]]] = {}

            for _row in gg.itertuples(index=False):
                row_idx = int(getattr(_row, "index"))
                i0i = int(s.normalize_index(int(getattr(_row, "loop_e0")), int(nE)))
                i1i = int(s.normalize_index(int(getattr(_row, "loop_e1")), int(nE)))
                j0 = int(getattr(_row, "lambda_j0"))
                j1 = int(getattr(_row, "lambda_j1"))
                b = int(getattr(_row, "block"))
                if j0 < 0 or j1 < 0 or b < 0:
                    continue
                jlo = int(min(j0, j1))
                jhi = int(max(j0, j1))

                rect_key = (
                    int(i_base_n),
                    int(j_base_shared),
                    int(i0i),
                    int(i1i),
                    int(jlo),
                    int(jhi),
                    int(refine_steps_use),
                    float(eta_use),
                    float(mu0) if np.isfinite(float(mu0)) else float("nan"),
                )

                if rect_key not in rect_cache:
                    pts = lasso_points(
                        i_base=int(i_base_n),
                        j_base=int(j_base_shared),
                        i0=int(i0i),
                        i1=int(i1i),
                        j0=int(jlo),
                        j1=int(jhi),
                        nE=int(nE),
                        refine_steps=int(refine_steps_use),
                    )
                    Ms, _diags = s.compute_monodromy_elambda(
                        lamA,
                        lamB,
                        lambdas_eff,
                        pts=pts,
                        eta=float(eta_use),
                        blocks=int(blocks_use),
                        basis_cache=basis_cache,
                        lamB2=lamB2,
                        mu=float(mu0) if np.isfinite(float(mu0)) else 0.0,
                    )

                    v_by_block: List[Optional[np.ndarray]] = []
                    for bb in range(int(blocks_use)):
                        if 0 <= bb < len(Ms) and Ms[bb] is not None:
                            v_by_block.append(minus1_mode_vec(np.asarray(Ms[bb], dtype=np.complex128)))
                        else:
                            v_by_block.append(None)
                    rect_cache[rect_key] = v_by_block

                v = rect_cache[rect_key][b] if b < len(rect_cache[rect_key]) else None
                if v is not None and int(dim) <= int(args.max_dim_for_flip):
                    cells.at[row_idx, "_rep_v"] = v
                    cells.at[row_idx, "rep_has_v_minus1"] = True

    # Cluster within each (pair,block,dim)
    clus_cols = ["seed", "anchor_seed", "wlo", "whi", "block", "dim"]
    if "mu" in cells.columns:
        clus_cols.insert(4, "mu")

    out_cells: List[pd.DataFrame] = []
    gen_rows: List[Dict[str, Any]] = []

    for clus_key, g in cells.groupby(clus_cols, dropna=False):
        if "mu" in clus_cols:
            seed, anchor, wlo, whi, mu0, block, dim = clus_key
        else:
            seed, anchor, wlo, whi, block, dim = clus_key
            mu0 = float("nan")
        g = g.copy().reset_index(drop=True)
        if len(g) > int(args.max_cells_per_pair):
            g = g.sort_values(["n_pi"], ascending=False).head(int(args.max_cells_per_pair)).reset_index(drop=True)

        n = int(len(g))
        uf = UnionFind(n)

        # Pre-pull vectors (either from dumps or from lasso precompute).
        vecs: List[Optional[np.ndarray]] = [None for _ in range(n)]

        for _i in range(n):
            v = g.loc[_i, "_rep_v"]
            if isinstance(v, np.ndarray):
                vecs[_i] = np.asarray(v, dtype=np.complex128)

        mids = g["lambda_mid"].astype(float).values

        for i in range(n):
            for j in range(i + 1, n):
                dl = float(abs(float(mids[i]) - float(mids[j])))
                vi = vecs[i]
                vj = vecs[j]

                if (vi is not None) and (vj is not None):
                    if dl > float(args.lambda_tol):
                        continue
                    ov = float(abs(np.vdot(vi, vj)))
                    if ov >= float(args.flip_overlap_tol):
                        uf.union(i, j)
                else:
                    # Conservative: only merge without vectors if the lambda mids are essentially identical.
                    if dl <= float(args.lambda_tol_no_vec):
                        uf.union(i, j)

        # Assign component ids
        root_to_gid: Dict[int, int] = {}
        gids: List[int] = []
        next_gid = 0
        for i in range(n):
            r = uf.find(i)
            if r not in root_to_gid:
                root_to_gid[r] = next_gid
                next_gid += 1
            gids.append(int(root_to_gid[r]))

        g["gen_local_id"] = gids

        # Summarize each generator
        for gen_local_id, gg in g.groupby("gen_local_id", dropna=False):
            if isinstance(gen_local_id, tuple) and len(gen_local_id) == 1:
                gen_local_id = gen_local_id[0]
            gen_local_id = int(gen_local_id)
            lm = gg["lambda_mid"].astype(float).values
            n_cells = int(len(gg))
            n_pi_total = int(np.sum(gg["n_pi"].astype(int).values))

            # Overlaps among available rep vectors
            idxs = gg.index.tolist()
            ovs: List[float] = []
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    va = vecs[int(idxs[a])]
                    vb = vecs[int(idxs[b])]
                    if (va is None) or (vb is None):
                        continue
                    ovs.append(float(abs(np.vdot(va, vb))))

            gen_rows.append(
                {
                    "seed": int(seed),
                    "anchor_seed": int(anchor),
                    "wlo": float(wlo),
                    "whi": float(whi),
                    "mu": float(mu0) if ("mu" in clus_cols and np.isfinite(float(mu0))) else 0.0,
                    "block": int(block),
                    "dim": int(dim),
                    "gen_local_id": int(gen_local_id),
                    "n_cells": int(n_cells),
                    "n_pi_total": int(n_pi_total),
                    "lambda_mid_min": float(np.min(lm)) if lm.size else float("nan"),
                    "lambda_mid_median": float(np.median(lm)) if lm.size else float("nan"),
                    "lambda_mid_max": float(np.max(lm)) if lm.size else float("nan"),
                    "rep_holonomy_min_dist_to_minus1_median": float(np.nanmedian(gg["rep_holonomy_min_dist_to_minus1"].astype(float).values)),
                    "rep_has_v_minus1_frac": float(np.mean(gg["rep_has_v_minus1"].astype(bool).values)),
                    "within_gen_flip_overlap_n": int(len(ovs)),
                    "within_gen_flip_overlap_min": float(np.min(ovs)) if ovs else float("nan"),
                    "within_gen_flip_overlap_p10": _quantile(ovs, 0.10) if ovs else float("nan"),
                    "within_gen_flip_overlap_median": float(np.median(ovs)) if ovs else float("nan"),
                }
            )

        out_cells.append(g.drop(columns=["_rep_v"]))

    cells_out = pd.concat(out_cells, ignore_index=True)

    # Add a stable-ish string id
    mu_tag = ""
    if "mu" in cells_out.columns:
        mu_tag = ":mu" + cells_out["mu"].astype(float).map(lambda x: f"{float(x):.6g}")

    cells_out["gen_id"] = (
        cells_out["seed"].astype(int).astype(str)
        + ":"
        + cells_out["anchor_seed"].astype(int).astype(str)
        + ":"
        + cells_out["wlo"].astype(float).map(lambda x: f"{x:.10g}")
        + ":"
        + cells_out["whi"].astype(float).map(lambda x: f"{x:.10g}")
        + mu_tag
        + ":b"
        + cells_out["block"].astype(int).astype(str)
        + ":d"
        + cells_out["dim"].astype(int).astype(str)
        + ":g"
        + cells_out["gen_local_id"].astype(int).map(lambda x: f"{x:04d}")
    )

    gens = pd.DataFrame(gen_rows)

    gens_mu_tag = ""
    if "mu" in gens.columns:
        gens_mu_tag = ":mu" + gens["mu"].astype(float).map(lambda x: f"{float(x):.6g}")

    gens["gen_id"] = (
        gens["seed"].astype(int).astype(str)
        + ":"
        + gens["anchor_seed"].astype(int).astype(str)
        + ":"
        + gens["wlo"].astype(float).map(lambda x: f"{x:.10g}")
        + ":"
        + gens["whi"].astype(float).map(lambda x: f"{x:.10g}")
        + gens_mu_tag
        + ":b"
        + gens["block"].astype(int).astype(str)
        + ":d"
        + gens["dim"].astype(int).astype(str)
        + ":g"
        + gens["gen_local_id"].astype(int).map(lambda x: f"{x:04d}")
    )

    # Generator recurrence across loop pairs (how often each (e0,e1) sees this gen)
    rec = (
        cells_out.groupby(["gen_id", "loop_e0", "loop_e1"], dropna=False)
        .agg(n_cells=("lambda_mid", "count"), n_pi_total=("n_pi", "sum"))
        .reset_index()
        .sort_values(["gen_id", "n_pi_total"], ascending=[True, False])
        .reset_index(drop=True)
    )

    cells_out = cells_out.sort_values(["seed", "anchor_seed", "wlo", "whi", "block", "dim", "gen_local_id", "lambda_mid"], ascending=True).reset_index(drop=True)
    gens = gens.sort_values(["seed", "anchor_seed", "wlo", "whi", "block", "dim", "n_pi_total"], ascending=[True, True, True, True, True, True, False]).reset_index(drop=True)

    cells_out.to_csv(os.path.join(out_root, "phase3f_event_cells_with_gen.csv"), index=False)
    gens.to_csv(os.path.join(out_root, "phase3f_generators.csv"), index=False)
    rec.to_csv(os.path.join(out_root, "phase3f_generator_recurrence.csv"), index=False)


if __name__ == "__main__":
    main()
