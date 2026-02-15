import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import phase3e_elambda_loop_suite as s


def angle_wrap_pi(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return ((x + np.pi) % (2.0 * np.pi)) - np.pi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_run", required=True, help="Existing Phase-3E run folder (has rows + summary)")
    ap.add_argument("--out_root", required=True, help="Where to write dumps + index CSV")
    ap.add_argument("--tol_pi", type=float, default=0.25)
    ap.add_argument("--max_rows", type=int, default=2000)
    ap.add_argument("--max_dim", type=int, default=16)
    ap.add_argument("--eta", type=float, default=-1.0, help="if >0 fixed eta; else use min(cayley_eta_a,cayley_eta_b)")
    args = ap.parse_args()

    src = str(args.source_run)
    os.makedirs(str(args.out_root), exist_ok=True)

    rows_csv = os.path.join(src, "phase3e_elambda_suite_rows.csv")
    summ_json = os.path.join(src, "phase3e_elambda_suite_summary.json")
    if not os.path.exists(rows_csv) or not os.path.exists(summ_json):
        raise SystemExit("source_run must contain phase3e_elambda_suite_rows.csv and phase3e_elambda_suite_summary.json")

    with open(summ_json, "r", encoding="utf-8") as f:
        summ = json.load(f)

    outA = os.path.join(os.path.dirname(src), str(summ.get("out_root_a", ""))) if not os.path.isabs(str(summ.get("out_root_a", ""))) else str(summ.get("out_root_a"))
    outB = os.path.join(os.path.dirname(src), str(summ.get("out_root_b", ""))) if not os.path.isabs(str(summ.get("out_root_b", ""))) else str(summ.get("out_root_b"))

    lambdas_eff = [float(x) for x in summ.get("lambdas_eff", summ.get("lambdas", []))]
    if len(lambdas_eff) < 2:
        raise SystemExit("summary missing lambdas_eff/lambdas")

    df = pd.read_csv(rows_csv)

    fwd = df[
        (df.get("loop_type") == "rectangle")
        & (df.get("loop_direction") == "fwd")
        & (df.get("loop_double") == False)
        & (df.get("valid", True) == True)
    ].copy()

    if not len(fwd):
        raise SystemExit("No canonical fwd-rectangle valid rows found")

    ph = angle_wrap_pi(fwd["holonomy_det_phase"].values.astype(np.float64))
    aph = np.abs(ph)
    is_pi = np.abs(aph - math.pi) <= float(args.tol_pi)
    pi = fwd[is_pi].copy()

    if not len(pi):
        raise SystemExit("No pi rows at this tol")

    pi = pi.sort_values(["holonomy_min_dist_to_minus1"], ascending=True).head(int(args.max_rows)).reset_index(drop=True)

    dump_dir = os.path.join(str(args.out_root), "phase3f_holonomy_dumps")
    os.makedirs(dump_dir, exist_ok=True)

    index_rows: List[Dict[str, Any]] = []
    dump_id = 0

    for _idx, r in pi.iterrows():
        dim = int(r["dim"])
        if dim > int(args.max_dim):
            continue

        seed = int(r["seed"])
        anchor = int(r["anchor_seed"])
        wlo = float(r["wlo"])
        whi = float(r["whi"])

        relA = str(r.get("artifact_a", ""))
        relB = str(r.get("artifact_b", ""))
        pA = os.path.join(outA, relA)
        pB = os.path.join(outB, relB)
        if not os.path.exists(pA) or not os.path.exists(pB):
            continue

        dA = np.load(pA)
        dB = np.load(pB)

        lamA = np.asarray(dA["lambda_snap"], dtype=np.complex128)
        lamB = np.asarray(dB["lambda_snap"], dtype=np.complex128)
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
        i0 = s.normalize_index(int(r["loop_e0"]), nE)
        i1 = s.normalize_index(int(r["loop_e1"]), nE)
        refine_steps = int(r.get("loop_refine_steps", r.get("refine_steps", 0)))
        i_path = s.refine_path_indices(i0, i1, nE, refine_steps=int(refine_steps))

        j0 = int(r["lambda_j0"])
        j1 = int(r["lambda_j1"])
        pts = s.loop_points_elambda(i_path, j0, j1, s.LoopSpec(loop_type="rectangle", direction="fwd"))

        cache: Dict[Tuple[int, int], Tuple[List[np.ndarray], List[int], Dict[str, Any]]] = {}
        Ms, diags = s.compute_monodromy_elambda(
            lamA,
            lamB,
            lambdas_eff,
            pts=pts,
            eta=float(eta),
            blocks=int(r.get("blocks", summ.get("blocks", 4))),
            basis_cache=cache,
        )

        b = int(r["block"])
        if b < 0 or b >= len(Ms):
            continue
        M = Ms[b]
        if M is None:
            continue

        meta = {
            "source_run": os.path.basename(src.rstrip("/\\")),
            "seed": int(seed),
            "anchor_seed": int(anchor),
            "wlo": float(wlo),
            "whi": float(whi),
            "loop_id": int(r.get("loop_id", -1)),
            "loop_e0": int(i0),
            "loop_e1": int(i1),
            "lambda_eff0": float(r.get("lambda_eff0", r.get("lambda0", float("nan")))),
            "lambda_eff1": float(r.get("lambda_eff1", r.get("lambda1", float("nan")))),
            "lambda_j0": int(j0),
            "lambda_j1": int(j1),
            "block": int(b),
            "dim": int(dim),
            "eta": float(eta),
            "holonomy_det_phase": float(r.get("holonomy_det_phase", float("nan"))),
            "holonomy_min_dist_to_minus1": float(r.get("holonomy_min_dist_to_minus1", float("nan"))),
        }

        dump_path = s.dump_holonomy_npz(dump_dir, dump_id=int(dump_id), M=np.asarray(M, dtype=np.complex128), meta=meta)
        dump_rel = os.path.relpath(dump_path, str(args.out_root)).replace("\\", "/")

        index_rows.append(
            {
                "seed": int(seed),
                "anchor_seed": int(anchor),
                "wlo": float(wlo),
                "whi": float(whi),
                "loop_e0": int(i0),
                "loop_e1": int(i1),
                "lambda_eff0": float(meta["lambda_eff0"]),
                "lambda_eff1": float(meta["lambda_eff1"]),
                "lambda_j0": int(j0),
                "lambda_j1": int(j1),
                "block": int(b),
                "dim": int(dim),
                "holonomy_dump": str(dump_rel),
            }
        )
        dump_id += 1

    out_csv = os.path.join(str(args.out_root), "phase3f_pi_dump_index.csv")
    pd.DataFrame(index_rows).to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
