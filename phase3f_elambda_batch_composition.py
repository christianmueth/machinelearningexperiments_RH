import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _resolve_artifact_root(out_root: str, p: str) -> str:
    """Resolve artifact roots referenced in a Phase-3E summary.

    Historically summaries may store out_root_a/out_root_b/out_root_b2 as relative
    paths w.r.t. the *workspace* root, while Phase-3F tools run on nested
    out_root folders (e.g. refinement runs). Try a small set of reasonable bases.
    """

    if not p:
        return ""
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

    # Fall back to original behavior.
    return os.path.join(ws_root, p)

import phase3e_elambda_loop_suite as s


def angle_wrap_pi(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return ((x + np.pi) % (2.0 * np.pi)) - np.pi


def _closest_index(vals: Sequence[float], target: float, *, tol: float = 1e-10) -> int:
    v = np.asarray([float(x) for x in vals], dtype=np.float64)
    t = float(target)
    i = int(np.argmin(np.abs(v - t)))
    if not np.isfinite(v[i]) or abs(float(v[i]) - t) > float(tol):
        raise ValueError(f"Could not match target={t} within tol={tol}; nearest={float(v[i])} at idx={i}")
    return i


def path_E(*, i_from: int, i_to: int, nE: int, refine_steps: int, j_fixed: int) -> List[Tuple[int, int]]:
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


def concat_paths(a: Sequence[Tuple[int, int]], b: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not a:
        return list(b)
    if not b:
        return list(a)
    if a[-1] != b[0]:
        raise ValueError("paths do not connect")
    return list(a) + list(b[1:])


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
    """Lasso loop based at (i_base,j_base) enclosing rectangle spanning i0->i1 and j0->j1."""
    i_base = int(i_base)
    j_base = int(j_base)
    i0 = int(i0)
    i1 = int(i1)
    j0 = int(j0)
    j1 = int(j1)
    nE = int(nE)
    refine_steps = int(refine_steps)

    pts: List[Tuple[int, int]] = []
    pts.extend(path_lam(i_base, j_base, j0))

    e_conn = path_E(i_from=i_base, i_to=i0, nE=nE, refine_steps=refine_steps, j_fixed=j0)
    pts = concat_paths(pts, e_conn)

    i_path = s.refine_path_indices(i0, i1, nE, refine_steps=refine_steps)
    rect = s.loop_points_elambda(list(i_path), j0, j1, s.LoopSpec(loop_type="rectangle", direction="fwd"))
    pts = concat_paths(pts, rect)

    e_back = path_E(i_from=i0, i_to=i_base, nE=nE, refine_steps=refine_steps, j_fixed=j0)
    pts = concat_paths(pts, e_back)
    lam_back = path_lam(i_base, j0, j_base)
    pts = concat_paths(pts, lam_back)

    if pts[0] != pts[-1]:
        raise ValueError("lasso_loop did not close")
    return pts


def frob_norm(M: np.ndarray) -> float:
    M = np.asarray(M)
    return float(np.linalg.norm(M))


def minus1_mode_vec(M: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
    M = np.asarray(M, dtype=np.complex128)
    vals, vecs = np.linalg.eig(M)
    vals = np.asarray(vals, dtype=np.complex128)
    vecs = np.asarray(vecs, dtype=np.complex128)
    dists = np.abs(vals - (-1.0 + 0.0j))
    k = int(np.argmin(dists))
    v = vecs[:, k]
    nrm = float(np.linalg.norm(v))
    if not np.isfinite(nrm) or nrm <= 0:
        return float(dists[k]), None
    return float(dists[k]), (v / nrm)


@dataclass(frozen=True)
class PairKey:
    seed: int
    anchor_seed: int
    wlo: float
    whi: float
    mu: float
    dim: int


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True, help="Phase-3E run folder containing phase3e_elambda_suite_rows.csv and phase3e_elambda_suite_summary.json")
    ap.add_argument("--cells_csv", default="", help="Optional path to phase3f_event_cells_with_gen.csv (default: inside out_root)")
    ap.add_argument("--gens_csv", default="", help="Optional path to phase3f_generators.csv (default: inside out_root)")

    ap.add_argument("--top_k_gens", type=int, default=8, help="Per (pair,block,dim): keep top K generators by n_pi_total then n_cells")
    ap.add_argument("--max_pairs", type=int, default=16, help="Max distinct (seed,anchor,wlo,whi,dim) pairs to process")

    ap.add_argument("--i_base", type=int, default=-1, help="Base energy index; default min(loop_e0,loop_e1) among selected gens")
    ap.add_argument("--j_base", type=int, default=-1, help="Base lambda index; default min(lambda_j0,lambda_j1) among selected gens")

    ap.add_argument("--refine_steps", type=int, default=-1, help="Refine steps along E connectors; default max(loop_refine_steps) among selected gens")
    ap.add_argument("--eta", type=float, default=-1.0, help="If >0 fixed eta; else use min(cayley_eta_a,cayley_eta_b)")

    ap.add_argument(
        "--do_concat_check",
        type=int,
        default=1,
        help="If 1: for each generator pair (A,B) also compute M(AB) from the concatenated lasso loop and report ||M(AB) - MA MB||.",
    )

    ap.add_argument("--max_blocks", type=int, default=-1, help="Override blocks count; default from rows/gens")
    ap.add_argument("--max_dim_for_flip", type=int, default=32, help="Compute flip vectors only when dim <= this")

    args = ap.parse_args()

    out_root = str(args.out_root)
    rows_csv = os.path.join(out_root, "phase3e_elambda_suite_rows.csv")
    summ_path = os.path.join(out_root, "phase3e_elambda_suite_summary.json")
    if not os.path.exists(rows_csv):
        raise SystemExit(f"Missing rows CSV: {rows_csv}")
    if not os.path.exists(summ_path):
        raise SystemExit(f"Missing summary JSON: {summ_path}")

    with open(summ_path, "r", encoding="utf-8") as f:
        summ = json.load(f)

    lambdas_eff = [float(x) for x in summ.get("lambdas_eff", summ.get("lambdas", []))]
    if len(lambdas_eff) < 2:
        raise SystemExit("Summary missing lambdas_eff/lambdas")

    outA = str(summ.get("out_root_a", ""))
    outB = str(summ.get("out_root_b", ""))
    outB2 = str(summ.get("out_root_b2", "") or "").strip()
    if not outA or not outB:
        raise SystemExit("Summary missing out_root_a/out_root_b")

    outA = _resolve_artifact_root(out_root, outA)
    outB = _resolve_artifact_root(out_root, outB)
    outB2 = _resolve_artifact_root(out_root, outB2) if outB2 else ""

    if not (os.path.exists(outA) and os.path.exists(outB)):
        raise SystemExit(f"Could not resolve artifact roots: outA={outA} outB={outB}")
    if outB2 and (not os.path.exists(outB2)):
        raise SystemExit(f"Could not resolve out_root_b2: outB2={outB2}")

    cells_csv = str(args.cells_csv).strip() or os.path.join(out_root, "phase3f_event_cells_with_gen.csv")
    gens_csv = str(args.gens_csv).strip() or os.path.join(out_root, "phase3f_generators.csv")
    if not os.path.exists(cells_csv):
        raise SystemExit(f"Missing cells CSV: {cells_csv} (run phase3f_elambda_generator_registry.py first)")
    if not os.path.exists(gens_csv):
        raise SystemExit(f"Missing generators CSV: {gens_csv} (run phase3f_elambda_generator_registry.py first)")

    cells = pd.read_csv(cells_csv)
    gens = pd.read_csv(gens_csv)

    # Pick top generators per (pair,block,dim)
    gcols = ["seed", "anchor_seed", "wlo", "whi", "block", "dim", "gen_id"]
    if "mu" in gens.columns:
        gcols.insert(4, "mu")
    need = set(gcols + ["n_pi_total", "n_cells"])
    missing = [c for c in need if c not in gens.columns]
    if missing:
        raise SystemExit(f"gens CSV missing columns: {missing}")

    # Limit number of distinct pairs processed.
    pair_cols = ["seed", "anchor_seed", "wlo", "whi", "dim"]
    if "mu" in gens.columns:
        pair_cols.insert(4, "mu")
    pair_list = (
        gens[pair_cols]
        .drop_duplicates()
        .sort_values(pair_cols, ascending=True)
        .head(int(args.max_pairs))
        .itertuples(index=False)
    )
    if "mu" in gens.columns:
        pair_keys = [PairKey(int(p.seed), int(p.anchor_seed), float(p.wlo), float(p.whi), float(p.mu), int(p.dim)) for p in pair_list]
    else:
        pair_keys = [PairKey(int(p.seed), int(p.anchor_seed), float(p.wlo), float(p.whi), float("nan"), int(p.dim)) for p in pair_list]
    if not pair_keys:
        raise SystemExit("No pairs found in generators table")

    # Representative cell for each gen_id: choose minimal rep_holonomy_min_dist_to_minus1 when available.
    if "rep_holonomy_min_dist_to_minus1" in cells.columns:
        cells["_mind"] = pd.to_numeric(cells["rep_holonomy_min_dist_to_minus1"], errors="coerce")
    else:
        cells["_mind"] = float("nan")

    # Cache loaded lambda snapshots per pair.
    lam_cache: Dict[Tuple[int, int, float, float], Dict[str, Any]] = {}

    out_rows: List[Dict[str, Any]] = []

    for pk in pair_keys:
        gens_pk = gens[
            (gens.seed.astype(int) == pk.seed)
            & (gens.anchor_seed.astype(int) == pk.anchor_seed)
            & (gens.wlo.astype(float) == pk.wlo)
            & (gens.whi.astype(float) == pk.whi)
            & (gens.dim.astype(int) == pk.dim)
        ].copy()
        if "mu" in gens_pk.columns:
            gens_pk = gens_pk[gens_pk.mu.astype(float) == float(pk.mu)]
        if not len(gens_pk):
            continue

        for (block, dim), gg in gens_pk.groupby(["block", "dim"], dropna=False):
            gg = gg.sort_values(["n_pi_total", "n_cells"], ascending=[False, False]).head(int(args.top_k_gens)).reset_index(drop=True)
            if len(gg) < 1:
                continue

            # Get representative cell rows for these generators.
            rep_cells: List[pd.Series] = []
            for gen_id in gg.gen_id.tolist():
                cc = cells[cells.gen_id == gen_id].copy()
                if not len(cc):
                    continue
                cc = cc.sort_values(["_mind"], ascending=True)
                rep_cells.append(cc.iloc[0])

            if not rep_cells:
                continue

            # Determine basepoint and parameters for this pair/block/dim.
            loop_e_mins = [int(min(int(r.loop_e0), int(r.loop_e1))) for r in rep_cells]
            lam_j_mins = [int(min(int(r.lambda_j0), int(r.lambda_j1))) for r in rep_cells if int(r.lambda_j0) >= 0 and int(r.lambda_j1) >= 0]

            i_base = int(args.i_base) if int(args.i_base) >= 0 else int(min(loop_e_mins))
            if not lam_j_mins:
                continue
            j_base = int(args.j_base) if int(args.j_base) >= 0 else int(min(lam_j_mins))

            refine_steps = int(args.refine_steps)
            if refine_steps < 0:
                if "loop_refine_steps" in cells.columns:
                    refine_steps = int(np.max(pd.to_numeric(cells.loop_refine_steps, errors="coerce").fillna(0).astype(int).values))
                else:
                    refine_steps = 0

            blocks_use = int(args.max_blocks) if int(args.max_blocks) > 0 else int(np.max(pd.to_numeric(gg.block, errors="coerce").astype(int).values) + 1)
            blocks_use = max(blocks_use, int(np.max(pd.to_numeric(cells.blocks, errors="coerce").fillna(blocks_use).astype(int).values)))

            pair_key = (pk.seed, pk.anchor_seed, pk.wlo, pk.whi)
            if pair_key not in lam_cache:
                # All representative cells should share artifact paths.
                relA = str(rep_cells[0].artifact_a).replace("\\", "/")
                relB = str(rep_cells[0].artifact_b).replace("\\", "/")
                pA = os.path.join(outA, relA)
                pB = os.path.join(outB, relB)
                if not (os.path.exists(pA) and os.path.exists(pB)):
                    continue
                dA = np.load(pA)
                dB = np.load(pB)
                lamA = np.asarray(dA["lambda_snap"], dtype=np.complex128)
                lamB = np.asarray(dB["lambda_snap"], dtype=np.complex128)
                lamB2 = None
                if outB2 and ("artifact_b2" in rep_cells[0].index):
                    relB2 = str(rep_cells[0].get("artifact_b2", "")).replace("\\", "/")
                    if relB2:
                        pB2 = os.path.join(outB2, relB2)
                        if os.path.exists(pB2):
                            dB2 = np.load(pB2)
                            if "lambda_snap" in dB2.files:
                                lamB2 = np.asarray(dB2["lambda_snap"], dtype=np.complex128)
                etaA = float(dA["cayley_eta"]) if ("cayley_eta" in dA.files) else float("nan")
                etaB = float(dB["cayley_eta"]) if ("cayley_eta" in dB.files) else float("nan")
                lam_cache[pair_key] = {"lamA": lamA, "lamB": lamB, "lamB2": lamB2, "etaA": etaA, "etaB": etaB}

            lamA = lam_cache[pair_key]["lamA"]
            lamB = lam_cache[pair_key]["lamB"]
            lamB2 = lam_cache[pair_key].get("lamB2", None)
            etaA = lam_cache[pair_key]["etaA"]
            etaB = lam_cache[pair_key]["etaB"]

            eta_use = float(args.eta)
            if (not np.isfinite(eta_use)) or eta_use <= 0:
                eta_use = float(min(float(etaA), float(etaB)))

            nE = int(lamA.shape[0])
            if nE < 2:
                continue

            i_base_n = int(s.normalize_index(i_base, nE))

            # Build each generator matrix once (lasso at shared basepoint).
            basis_cache: Dict[Tuple[Any, ...], Tuple[List[np.ndarray], List[int], Dict[str, Any]]] = {}
            M_by_gen: Dict[str, np.ndarray] = {}
            pts_by_gen: Dict[str, List[Tuple[int, int]]] = {}
            v_by_gen: Dict[str, Optional[np.ndarray]] = {}
            dist_by_gen: Dict[str, float] = {}

            for r in rep_cells:
                gen_id = str(r.gen_id)
                mu_use = float(getattr(r, "mu", pk.mu))
                i0 = int(s.normalize_index(int(r.loop_e0), nE))
                i1 = int(s.normalize_index(int(r.loop_e1), nE))
                j0 = int(r.lambda_j0)
                j1 = int(r.lambda_j1)
                if j0 < 0 or j1 < 0:
                    # fall back to matching by lambda_eff values
                    lam0 = float(r.lambda0)
                    lam1 = float(r.lambda1)
                    j0 = _closest_index(lambdas_eff, lam0)
                    j1 = _closest_index(lambdas_eff, lam1)

                jlo = int(min(j0, j1))
                jhi = int(max(j0, j1))
                pts = lasso_loop(
                    i_base=i_base_n,
                    j_base=j_base,
                    i0=i0,
                    i1=i1,
                    j0=jlo,
                    j1=jhi,
                    nE=nE,
                    refine_steps=refine_steps,
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
                    mu=float(mu_use) if np.isfinite(mu_use) else 0.0,
                )
                b = int(block)
                if not (0 <= b < len(Ms)) or Ms[b] is None:
                    continue
                M = np.asarray(Ms[b], dtype=np.complex128)
                M_by_gen[gen_id] = M
                pts_by_gen[gen_id] = pts
                if int(dim) <= int(args.max_dim_for_flip):
                    dist, vv = minus1_mode_vec(M)
                    v_by_gen[gen_id] = vv
                    dist_by_gen[gen_id] = dist

            gen_ids = list(M_by_gen.keys())
            if len(gen_ids) < 1:
                continue

            # Per-gen invariants
            for gen_id in gen_ids:
                M = M_by_gen[gen_id]
                I = np.eye(M.shape[0], dtype=np.complex128)
                out_rows.append(
                    {
                        "kind": "gen",
                        "seed": pk.seed,
                        "anchor_seed": pk.anchor_seed,
                        "wlo": pk.wlo,
                        "whi": pk.whi,
                        "mu": float(pk.mu),
                        "block": int(block),
                        "dim": int(dim),
                        "i_base": int(i_base_n),
                        "j_base": int(j_base),
                        "eta": float(eta_use),
                        "refine_steps": int(refine_steps),
                        "genA": gen_id,
                        "genB": "",
                        "norm_M2_minus_I": frob_norm(M @ M - I),
                        "abs_det_phase": float(abs(angle_wrap_pi(np.asarray([np.angle(np.linalg.det(M))], dtype=np.float64))[0])),
                        "min_dist_to_minus1": float(dist_by_gen.get(gen_id, float("nan"))),
                    }
                )

            # Pairwise composition + commutator
            for i in range(len(gen_ids)):
                for j in range(i, len(gen_ids)):
                    ga = gen_ids[i]
                    gb = gen_ids[j]
                    MA = M_by_gen[ga]
                    MB = M_by_gen[gb]
                    I = np.eye(MA.shape[0], dtype=np.complex128)

                    Mprod = MA @ MB
                    comm = MA @ MB - MB @ MA

                    norm_concat_minus_prod = float("nan")
                    if bool(int(args.do_concat_check) != 0) and (ga in pts_by_gen) and (gb in pts_by_gen):
                        ptsAB = concat_paths(pts_by_gen[ga], pts_by_gen[gb])
                        MsAB, _ = s.compute_monodromy_elambda(
                            lamA,
                            lamB,
                            lambdas_eff,
                            pts=ptsAB,
                            eta=float(eta_use),
                            blocks=int(blocks_use),
                            basis_cache=basis_cache,
                            lamB2=lamB2,
                            mu=float(pk.mu) if np.isfinite(float(pk.mu)) else 0.0,
                        )
                        if 0 <= int(block) < len(MsAB) and MsAB[int(block)] is not None:
                            MAB = np.asarray(MsAB[int(block)], dtype=np.complex128)
                            norm_concat_minus_prod = frob_norm(MAB - Mprod)

                    ov = float("nan")
                    if (ga in v_by_gen) and (gb in v_by_gen) and (v_by_gen[ga] is not None) and (v_by_gen[gb] is not None):
                        ov = float(abs(np.vdot(v_by_gen[ga], v_by_gen[gb])))

                    out_rows.append(
                        {
                            "kind": "pair",
                            "seed": pk.seed,
                            "anchor_seed": pk.anchor_seed,
                            "wlo": pk.wlo,
                            "whi": pk.whi,
                            "mu": float(pk.mu),
                            "block": int(block),
                            "dim": int(dim),
                            "i_base": int(i_base_n),
                            "j_base": int(j_base),
                            "eta": float(eta_use),
                            "refine_steps": int(refine_steps),
                            "genA": ga,
                            "genB": gb,
                            "norm_commutator": frob_norm(comm),
                            "norm_prod2_minus_I": frob_norm(Mprod @ Mprod - I),
                            "norm_concat_minus_prod": norm_concat_minus_prod,
                            "flip_overlap": ov,
                        }
                    )

    if not out_rows:
        raise SystemExit("No batch composition rows produced; check inputs")

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(os.path.join(out_root, "phase3f_batch_composition.csv"), index=False)

    # Small summary table
    if "kind" in out_df.columns:
        summ_rows = []
        for (kind, block, dim), g in out_df.groupby(["kind", "block", "dim"], dropna=False):
            row: Dict[str, Any] = {"kind": kind, "block": int(block), "dim": int(dim), "n": int(len(g))}
            for col in ["norm_M2_minus_I", "norm_commutator", "norm_prod2_minus_I", "flip_overlap"]:
                if col in g.columns:
                    v = pd.to_numeric(g[col], errors="coerce")
                    finite = v.values[np.isfinite(v.values)]
                    row[col + "_median"] = float(np.median(finite)) if finite.size else float("nan")
                    row[col + "_p10"] = float(np.quantile(finite, 0.10)) if finite.size else float("nan")
                    row[col + "_max"] = float(np.max(finite)) if finite.size else float("nan")
            summ_rows.append(row)
        pd.DataFrame(summ_rows).to_csv(os.path.join(out_root, "phase3f_batch_composition_summary.csv"), index=False)


if __name__ == "__main__":
    main()
