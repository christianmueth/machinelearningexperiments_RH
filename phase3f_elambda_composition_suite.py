import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import phase3e_elambda_loop_suite as s


def angle_wrap_pi(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return ((x + np.pi) % (2.0 * np.pi)) - np.pi


def _abs_det_phase(M: np.ndarray) -> float:
    ph = float(np.angle(np.linalg.det(np.asarray(M, dtype=np.complex128))))
    ph = float(angle_wrap_pi(np.asarray([ph], dtype=np.float64))[0])
    return float(abs(ph))


def _closest_index(vals: Sequence[float], target: float, *, tol: float = 1e-10) -> int:
    v = np.asarray([float(x) for x in vals], dtype=np.float64)
    t = float(target)
    i = int(np.argmin(np.abs(v - t)))
    if not np.isfinite(v[i]) or abs(float(v[i]) - t) > float(tol):
        raise ValueError(f"Could not match target={t} within tol={tol}; nearest={float(v[i])} at idx={i}")
    return i


def path_E(
    *,
    i_from: int,
    i_to: int,
    nE: int,
    refine_steps: int,
    j_fixed: int,
) -> List[Tuple[int, int]]:
    i_path = s.refine_path_indices(int(i_from), int(i_to), int(nE), refine_steps=int(refine_steps))
    return [(int(i), int(j_fixed)) for i in i_path]


def path_lam(ei: int, j_from: int, j_to: int) -> List[Tuple[int, int]]:
    if int(j_from) == int(j_to):
        return [(int(ei), int(j_from))]
    step = 1 if int(j_to) > int(j_from) else -1
    pts: List[Tuple[int, int]] = []
    for j in range(int(j_from), int(j_to) + step, step):
        pts.append((int(ei), int(j)))
    return pts


def lasso_loop(
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
    """A lasso loop based at (i_base,j_base) enclosing the rectangle spanning i0->i1 and j0->j1.

    Connector path: (i_base,j_base) -> (i_base,j0) -> (i0,j0).
    Then traverse the rectangle (fwd), then return along the connector reversed.
    """

    i_base = int(i_base)
    j_base = int(j_base)
    i0 = int(i0)
    i1 = int(i1)
    j0 = int(j0)
    j1 = int(j1)
    nE = int(nE)
    refine_steps = int(refine_steps)

    # base -> (i_base,j0)
    pts: List[Tuple[int, int]] = []
    pts.extend(path_lam(i_base, j_base, j0))

    # (i_base,j0) -> (i0,j0)
    e_conn = path_E(i_from=i_base, i_to=i0, nE=nE, refine_steps=refine_steps, j_fixed=j0)
    pts = concat_paths(pts, e_conn)

    # rectangle at (i0,j0)
    i_path = s.refine_path_indices(i0, i1, nE, refine_steps=refine_steps)
    rect = s.loop_points_elambda(list(i_path), j0, j1, s.LoopSpec(loop_type="rectangle", direction="fwd"))
    pts = concat_paths(pts, rect)

    # return: (i0,j0)->(i_base,j0)->(i_base,j_base)
    e_back = path_E(i_from=i0, i_to=i_base, nE=nE, refine_steps=refine_steps, j_fixed=j0)
    pts = concat_paths(pts, e_back)
    lam_back = path_lam(i_base, j0, j_base)
    pts = concat_paths(pts, lam_back)

    # ensure loop closure
    if pts[0] != pts[-1]:
        raise ValueError("lasso_loop did not close")
    return pts


def concat_paths(a: Sequence[Tuple[int, int]], b: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not a:
        return list(b)
    if not b:
        return list(a)
    if a[-1] != b[0]:
        raise ValueError("paths do not connect")
    return list(a) + list(b[1:])


def load_summary(out_root: str) -> Dict[str, Any]:
    p = os.path.join(out_root, "phase3e_elambda_suite_summary.json")
    if not os.path.exists(p):
        raise SystemExit(f"Missing summary JSON: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def select_pairs(
    rows_csv: str,
    *,
    e0: int,
    e1: int,
    lam0: float,
    lam1: float,
    block: int,
    dim: int,
    max_pairs: int,
    tol_pi: float,
    require_pi: bool,
) -> List[Tuple[int, int, float, float]]:
    df = pd.read_csv(rows_csv)
    fwd = df[
        (df.get("loop_type") == "rectangle")
        & (df.get("loop_direction") == "fwd")
        & (df.get("loop_double") == False)
        & (df.get("valid", True) == True)
    ].copy()

    lam0_col = "lambda_eff0" if "lambda_eff0" in fwd.columns else "lambda0"
    lam1_col = "lambda_eff1" if "lambda_eff1" in fwd.columns else "lambda1"

    g = fwd[
        (fwd["loop_e0"].astype(int) == int(e0))
        & (fwd["loop_e1"].astype(int) == int(e1))
        & (np.abs(fwd[lam0_col].astype(float) - float(lam0)) < 1e-12)
        & (np.abs(fwd[lam1_col].astype(float) - float(lam1)) < 1e-12)
        & (fwd["block"].astype(int) == int(block))
        & (fwd["dim"].astype(int) == int(dim))
    ]

    if not len(g):
        raise SystemExit("No matching rows for requested event cell; check lam0/lam1 and source run")

    if require_pi:
        ph = angle_wrap_pi(g["holonomy_det_phase"].values.astype(np.float64))
        aph = np.abs(ph)
        is_pi = np.abs(aph - math.pi) <= float(tol_pi)
        g = g[is_pi]
        if not len(g):
            raise SystemExit("No rows near pi found for this event cell; try --require_pi 0 or increase --tol_pi")

    pairs = (
        g[["seed", "anchor_seed", "wlo", "whi"]]
        .drop_duplicates()
        .sort_values(["seed", "anchor_seed", "wlo", "whi"], ascending=True)
    )

    out: List[Tuple[int, int, float, float]] = []
    for _idx, r in pairs.iterrows():
        out.append((int(r["seed"]), int(r["anchor_seed"]), float(r["wlo"]), float(r["whi"])))
        if len(out) >= int(max_pairs):
            break
    return out


def minus1_stats(M: np.ndarray) -> Tuple[int, float]:
    n, d = s.minus1_eig_stats(np.asarray(M, dtype=np.complex128))
    return int(n), float(d)


def minus1_mode_vec(M: np.ndarray) -> Tuple[float, np.ndarray]:
    M = np.asarray(M, dtype=np.complex128)
    vals, vecs = np.linalg.eig(M)
    vals = np.asarray(vals, dtype=np.complex128)
    vecs = np.asarray(vecs, dtype=np.complex128)
    dists = np.abs(vals - (-1.0 + 0.0j))
    k = int(np.argmin(dists))
    v = vecs[:, k]
    nrm = float(np.linalg.norm(v))
    if not np.isfinite(nrm) or nrm <= 0:
        return float(dists[k]), v
    return float(dists[k]), (v / nrm)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_run", required=True, help="Existing Phase-3E run folder (provides lambdas_eff + rows for selecting pairs)")
    ap.add_argument("--out_root_a", required=True, help="Artifact root A (e.g. legacy channel_diag tree)")
    ap.add_argument("--out_root_b", required=True, help="Artifact root B (e.g. geom_v9 channel_diag tree)")
    ap.add_argument("--out_root", required=True, help="Where to write Phase-3F composition outputs")

    ap.add_argument("--event1", required=True, help="Event cell as 'e0,e1,lam0,lam1,block,dim' using lambda_eff coordinates")
    ap.add_argument("--event2", default="", help="Optional second event cell in same format")

    ap.add_argument("--tol_pi", type=float, default=0.25, help="Tolerance for selecting pi rows (radians)")
    ap.add_argument("--require_pi", type=int, default=1, help="1 selects only pairs exhibiting pi in the event cell; 0 selects arbitrary pairs")
    ap.add_argument(
        "--require_pi_both",
        type=int,
        default=0,
        help="If event2 is provided and require_pi=1: 1 requires pairs to be pi in BOTH event cells (intersection); 0 uses event1-only selection.",
    )

    ap.add_argument("--refine_steps", type=int, default=0, help="Match Phase-3E refine_steps along E edges")
    ap.add_argument("--eta", type=float, default=-1.0, help="if >0 fixed eta; else use min(cayley_eta_a,cayley_eta_b)")
    ap.add_argument("--blocks", type=int, default=4)

    ap.add_argument("--j_base", type=int, default=-1, help="Base lambda index for lassos; default uses min(j0 of events)")
    ap.add_argument("--i_base", type=int, default=-1, help="Base energy index for lassos; default uses event1 e0")
    ap.add_argument("--max_pairs", type=int, default=32, help="Max (seed,anchor,window) pairs to evaluate")
    ap.add_argument(
        "--pair",
        default="",
        help="Optional override: run only one pair specified as 'seed,anchor,wlo,whi' (floats for wlo/whi).",
    )

    args = ap.parse_args()

    os.makedirs(str(args.out_root), exist_ok=True)

    def parse_event(s_ev: str) -> Tuple[int, int, float, float, int, int]:
        parts = [p.strip() for p in str(s_ev).split(",") if p.strip()]
        if len(parts) != 6:
            raise SystemExit("event format must be 'e0,e1,lam0,lam1,block,dim'")
        return (int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3]), int(parts[4]), int(parts[5]))

    e1 = parse_event(str(args.event1))
    e2: Optional[Tuple[int, int, float, float, int, int]] = parse_event(str(args.event2)) if str(args.event2).strip() else None

    # Load lambdas grid from source run
    summ = load_summary(str(args.source_run))
    lambdas_eff = [float(x) for x in summ.get("lambdas_eff", summ.get("lambdas", []))]
    if len(lambdas_eff) < 2:
        raise SystemExit("source_run summary missing lambdas_eff/lambdas")

    # Map lam0/lam1 to lambda indices
    j0_1 = _closest_index(lambdas_eff, float(e1[2]))
    j1_1 = _closest_index(lambdas_eff, float(e1[3]))
    if j0_1 == j1_1:
        raise SystemExit("event1 lam0 and lam1 matched same lambda index")

    j0_2 = j1_2 = -1
    if e2 is not None:
        j0_2 = _closest_index(lambdas_eff, float(e2[2]))
        j1_2 = _closest_index(lambdas_eff, float(e2[3]))
        if j0_2 == j1_2:
            raise SystemExit("event2 lam0 and lam1 matched same lambda index")

    j_base = int(args.j_base)
    if j_base < 0:
        j_base = int(min(j0_1, j0_2 if e2 is not None else j0_1))

    # Base energy index
    i_base = int(args.i_base)
    if i_base < 0:
        i_base = int(e1[0])

    # Select which (seed,anchor,window) pairs to run from the source rows (or use --pair override).
    pairs: List[Tuple[int, int, float, float]]
    if str(args.pair).strip():
        parts = [p.strip() for p in str(args.pair).split(",") if p.strip()]
        if len(parts) != 4:
            raise SystemExit("--pair must be 'seed,anchor,wlo,whi'")
        pairs = [(int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3]))]
    else:
        rows_csv = os.path.join(str(args.source_run), "phase3e_elambda_suite_rows.csv")
        pairs = select_pairs(
            rows_csv,
            e0=int(e1[0]),
            e1=int(e1[1]),
            lam0=float(e1[2]),
            lam1=float(e1[3]),
            block=int(e1[4]),
            dim=int(e1[5]),
            max_pairs=int(args.max_pairs),
            tol_pi=float(args.tol_pi),
            require_pi=bool(int(args.require_pi) != 0),
        )

    if (not str(args.pair).strip()) and (e2 is not None) and bool(int(args.require_pi) != 0) and bool(int(args.require_pi_both) != 0):
        pairs2 = select_pairs(
            rows_csv,
            e0=int(e2[0]),
            e1=int(e2[1]),
            lam0=float(e2[2]),
            lam1=float(e2[3]),
            block=int(e2[4]),
            dim=int(e2[5]),
            max_pairs=int(args.max_pairs) * 10,
            tol_pi=float(args.tol_pi),
            require_pi=True,
        )
        s1 = set(pairs)
        s2 = set(pairs2)
        inter = sorted(list(s1.intersection(s2)))
        if not inter:
            raise SystemExit("No pairs satisfy pi in both event cells; try --require_pi_both 0 or increase --tol_pi")
        pairs = inter[: int(args.max_pairs)]

    # Prepare artifact lookup
    outA = str(args.out_root_a)
    outB = str(args.out_root_b)

    filesA = s.find_npz_files(outA)
    filesB = s.find_npz_files(outB)
    if not filesA or not filesB:
        raise SystemExit("No channel_diag*.npz found under one of the roots")

    mapA: Dict[Tuple[int, int, float, float], str] = {}
    mapB: Dict[Tuple[int, int, float, float], str] = {}
    for p in filesA:
        _rel, _backend, seed, anchor, wlo, whi = s.parse_rel_metadata(p, outA)
        mapA[s.key_for_pair(seed, anchor, wlo, whi)] = p
    for p in filesB:
        _rel, _backend, seed, anchor, wlo, whi = s.parse_rel_metadata(p, outB)
        mapB[s.key_for_pair(seed, anchor, wlo, whi)] = p

    rows_out: List[Dict[str, Any]] = []

    for (seed, anchor, wlo, whi) in pairs:
        kk = s.key_for_pair(int(seed), int(anchor), float(wlo), float(whi))
        if kk not in mapA or kk not in mapB:
            continue

        pA = mapA[kk]
        pB = mapB[kk]
        dA = np.load(pA)
        dB = np.load(pB)

        lamA = np.asarray(dA["lambda_snap"], dtype=np.complex128)
        lamB = np.asarray(dB["lambda_snap"], dtype=np.complex128)
        EA = np.asarray(dA["energies_snap"], dtype=np.float64)

        if lamA.shape != lamB.shape:
            continue

        etaA = float(dA["cayley_eta"]) if ("cayley_eta" in dA.files) else float("nan")
        etaB = float(dB["cayley_eta"]) if ("cayley_eta" in dB.files) else float("nan")

        eta = float(args.eta)
        if (not np.isfinite(eta)) or eta <= 0:
            eta = float(min(etaA, etaB))
        if (not np.isfinite(eta)) or eta <= 0:
            continue

        nE = int(lamA.shape[0])
        i_base_n = s.normalize_index(int(i_base), nE)
        i0_1 = s.normalize_index(int(e1[0]), nE)
        i1_1 = s.normalize_index(int(e1[1]), nE)

        pts1 = lasso_loop(
            i_base=i_base_n,
            j_base=int(j_base),
            i0=i0_1,
            i1=i1_1,
            j0=int(min(j0_1, j1_1)),
            j1=int(max(j0_1, j1_1)),
            nE=nE,
            refine_steps=int(args.refine_steps),
        )

        # Locality controls: adjacent lambda cells (same E edge), intended to be non-enclosing.
        j0_ev = int(min(j0_1, j1_1))
        j1_ev = int(max(j0_1, j1_1))
        pts1_ctrl_left: Optional[List[Tuple[int, int]]] = None
        pts1_ctrl_right: Optional[List[Tuple[int, int]]] = None
        if j0_ev - 1 >= 0:
            pts1_ctrl_left = lasso_loop(
                i_base=i_base_n,
                j_base=int(j_base),
                i0=i0_1,
                i1=i1_1,
                j0=int(j0_ev - 1),
                j1=int(j0_ev),
                nE=nE,
                refine_steps=int(args.refine_steps),
            )
        if j1_ev + 1 < len(lambdas_eff):
            pts1_ctrl_right = lasso_loop(
                i_base=i_base_n,
                j_base=int(j_base),
                i0=i0_1,
                i1=i1_1,
                j0=int(j1_ev),
                j1=int(j1_ev + 1),
                nE=nE,
                refine_steps=int(args.refine_steps),
            )

        pts2: Optional[List[Tuple[int, int]]] = None
        if e2 is not None:
            i0_2 = s.normalize_index(int(e2[0]), nE)
            i1_2 = s.normalize_index(int(e2[1]), nE)
            pts2 = lasso_loop(
                i_base=i_base_n,
                j_base=int(j_base),
                i0=i0_2,
                i1=i1_2,
                j0=int(min(j0_2, j1_2)),
                j1=int(max(j0_2, j1_2)),
                nE=nE,
                refine_steps=int(args.refine_steps),
            )

        cache: Dict[Tuple[int, int], Tuple[List[np.ndarray], List[int], Dict[str, Any]]] = {}

        Ms1, diags1 = s.compute_monodromy_elambda(
            lamA,
            lamB,
            lambdas_eff,
            pts=pts1,
            eta=float(eta),
            blocks=int(args.blocks),
            basis_cache=cache,
        )

        # compute W1 twice (cancellation control)
        pts1_twice = concat_paths(pts1, pts1)
        Ms1tw, _diags1tw = s.compute_monodromy_elambda(
            lamA,
            lamB,
            lambdas_eff,
            pts=pts1_twice,
            eta=float(eta),
            blocks=int(args.blocks),
            basis_cache=cache,
        )

        Ms1cL = [None for _ in range(int(args.blocks))]
        Ms1cR = [None for _ in range(int(args.blocks))]
        if pts1_ctrl_left is not None:
            Ms1cL, _ = s.compute_monodromy_elambda(
                lamA,
                lamB,
                lambdas_eff,
                pts=pts1_ctrl_left,
                eta=float(eta),
                blocks=int(args.blocks),
                basis_cache=cache,
            )
        if pts1_ctrl_right is not None:
            Ms1cR, _ = s.compute_monodromy_elambda(
                lamA,
                lamB,
                lambdas_eff,
                pts=pts1_ctrl_right,
                eta=float(eta),
                blocks=int(args.blocks),
                basis_cache=cache,
            )

        if pts2 is None:
            Ms2 = [None for _ in range(int(args.blocks))]
            Ms2tw = [None for _ in range(int(args.blocks))]
            Ms12 = [None for _ in range(int(args.blocks))]
            Ms21 = [None for _ in range(int(args.blocks))]
        else:
            Ms2, _diags2 = s.compute_monodromy_elambda(
                lamA,
                lamB,
                lambdas_eff,
                pts=pts2,
                eta=float(eta),
                blocks=int(args.blocks),
                basis_cache=cache,
            )

            pts2_twice = concat_paths(pts2, pts2)
            Ms2tw, _ = s.compute_monodromy_elambda(
                lamA,
                lamB,
                lambdas_eff,
                pts=pts2_twice,
                eta=float(eta),
                blocks=int(args.blocks),
                basis_cache=cache,
            )

            pts12 = concat_paths(pts1, pts2)
            Ms12, _diags12 = s.compute_monodromy_elambda(
                lamA,
                lamB,
                lambdas_eff,
                pts=pts12,
                eta=float(eta),
                blocks=int(args.blocks),
                basis_cache=cache,
            )

            pts21 = concat_paths(pts2, pts1)
            Ms21, _diags21 = s.compute_monodromy_elambda(
                lamA,
                lamB,
                lambdas_eff,
                pts=pts21,
                eta=float(eta),
                blocks=int(args.blocks),
                basis_cache=cache,
            )

        for b in range(int(args.blocks)):
            M1 = Ms1[b]
            M1t = Ms1tw[b]
            if M1 is None or M1t is None:
                continue

            row: Dict[str, Any] = {
                "seed": int(seed),
                "anchor_seed": int(anchor),
                "wlo": float(wlo),
                "whi": float(whi),
                "loop_e0": int(e1[0]),
                "loop_e1": int(e1[1]),
                "event1_e0": int(e1[0]),
                "event1_e1": int(e1[1]),
                "event2_e0": int(e2[0]) if e2 is not None else int(-1),
                "event2_e1": int(e2[1]) if e2 is not None else int(-1),
                "refine_steps": int(args.refine_steps),
                "eta": float(eta),
                "j_base": int(j_base),
                "i_base": int(i_base),
                "blocks": int(args.blocks),
                "block": int(b),
                "dim": int(np.asarray(M1).shape[0]),
                "event1_lam0": float(e1[2]),
                "event1_lam1": float(e1[3]),
                "event2_lam0": float(e2[2]) if e2 is not None else float("nan"),
                "event2_lam1": float(e2[3]) if e2 is not None else float("nan"),
                "M1_abs_det_phase": float(_abs_det_phase(M1)),
                "M1_num_eigs_near_minus1": int(minus1_stats(M1)[0]),
                "M1_min_dist_to_minus1": float(minus1_stats(M1)[1]),
                "M1_twice_abs_det_phase": float(_abs_det_phase(M1t)),
                "M1_twice_norm_Iminus": float(np.linalg.norm(np.eye(M1t.shape[0]) - np.asarray(M1t), ord="fro")),
                "M1_ctrl_left_abs_det_phase": float(_abs_det_phase(Ms1cL[b])) if (Ms1cL[b] is not None) else float("nan"),
                "M1_ctrl_right_abs_det_phase": float(_abs_det_phase(Ms1cR[b])) if (Ms1cR[b] is not None) else float("nan"),
            }

            if pts2 is not None:
                M2 = Ms2[b]
                M2t = Ms2tw[b]
                M12 = Ms12[b]
                M21 = Ms21[b]
                if (M2 is not None) and (M2t is not None) and (M12 is not None) and (M21 is not None):
                    _d1, v1 = minus1_mode_vec(M1)
                    _d2, v2 = minus1_mode_vec(M2)
                    flip_overlap = float(abs(np.vdot(v1, v2)))
                    row.update(
                        {
                            "M2_abs_det_phase": float(_abs_det_phase(M2)),
                            "M2_num_eigs_near_minus1": int(minus1_stats(M2)[0]),
                            "M2_min_dist_to_minus1": float(minus1_stats(M2)[1]),
                            "M2_twice_abs_det_phase": float(_abs_det_phase(M2t)),
                            "M2_twice_norm_Iminus": float(np.linalg.norm(np.eye(M2t.shape[0]) - np.asarray(M2t), ord="fro")),
                            "flip_overlap_M1_M2": flip_overlap,
                            "M12_abs_det_phase": float(_abs_det_phase(M12)),
                            "M21_abs_det_phase": float(_abs_det_phase(M21)),
                            "order_detphase_diff": float(abs(_abs_det_phase(M12) - _abs_det_phase(M21))),
                            "seq12_norm_diff_from_M1M2": float(
                                np.linalg.norm(np.asarray(M12) - (np.asarray(M1) @ np.asarray(M2)), ord="fro")
                            ),
                            "seq21_norm_diff_from_M2M1": float(
                                np.linalg.norm(np.asarray(M21) - (np.asarray(M2) @ np.asarray(M1)), ord="fro")
                            ),
                        }
                    )

            rows_out.append(row)

    out_csv = os.path.join(str(args.out_root), "phase3f_elambda_composition_rows.csv")
    pd.DataFrame(rows_out).to_csv(out_csv, index=False)

    with open(os.path.join(str(args.out_root), "phase3f_elambda_composition_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_run": str(args.source_run),
                "out_root_a": str(args.out_root_a),
                "out_root_b": str(args.out_root_b),
                "event1": str(args.event1),
                "event2": str(args.event2),
                "max_pairs": int(args.max_pairs),
                "n_rows": int(len(rows_out)),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
