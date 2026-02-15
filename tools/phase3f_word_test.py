import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _canon_unit_vector(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v).reshape(-1)
    if v.size == 0:
        return v
    idx = int(np.argmax(np.abs(v)))
    phase = np.angle(v[idx])
    v = v * np.exp(-1j * phase)
    nrm = float(np.linalg.norm(v))
    if nrm == 0.0:
        return v
    return v / nrm


def _load_M(npz_abs: Path) -> np.ndarray:
    with np.load(str(npz_abs), allow_pickle=True) as d:
        return np.asarray(d["M"], dtype=np.complex128)


def _unit_eigs(M: np.ndarray) -> np.ndarray:
    vals = np.linalg.eigvals(M)
    mags = np.abs(vals)
    mags = np.where(mags == 0, 1.0, mags)
    vals = vals / mags
    order = np.argsort(np.angle(vals))
    return vals[order]


def _eig_dist(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    dr = np.real(a) - np.real(b)
    di = np.imag(a) - np.imag(b)
    return float(np.sqrt(np.mean(dr * dr + di * di)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True, help="Phase-3E run folder containing phase3f_batch_composition.csv and phase3f_event_cells_with_gen.csv")
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--anchor", type=int, default=-1)
    ap.add_argument("--wlo", type=float, default=float("nan"))
    ap.add_argument("--whi", type=float, default=float("nan"))
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--block", type=int, default=1)
    ap.add_argument("--dim", type=int, default=2)
    ap.add_argument("--flip_overlap_min", type=float, default=0.6)
    ap.add_argument("--flip_overlap_max", type=float, default=0.9)
    ap.add_argument("--out_json", default="phase3f_word_test.json")
    args = ap.parse_args()

    out_root = Path(str(args.out_root)).resolve()
    comp_path = out_root / "phase3f_batch_composition.csv"
    cells_path = out_root / "phase3f_event_cells_with_gen.csv"
    if not comp_path.exists():
        raise SystemExit(f"Missing: {comp_path}")
    if not cells_path.exists():
        raise SystemExit(f"Missing: {cells_path}")

    comp = pd.read_csv(comp_path)
    cells = pd.read_csv(cells_path)

    # Filter to group
    def in_group(df: pd.DataFrame) -> pd.Series:
        m = (df.get("mu") == float(args.mu)) & (df.get("block") == int(args.block)) & (df.get("dim") == int(args.dim))
        if int(args.seed) >= 0:
            m &= df.get("seed") == int(args.seed)
        if int(args.anchor) >= 0:
            m &= df.get("anchor_seed") == int(args.anchor)
        if np.isfinite(float(args.wlo)):
            m &= df.get("wlo") == float(args.wlo)
        if np.isfinite(float(args.whi)):
            m &= df.get("whi") == float(args.whi)
        return m

    pairs = comp[(comp.get("kind") == "pair") & in_group(comp)].copy()
    if len(pairs) == 0:
        raise SystemExit("No pair rows matched group filter")

    pairs = pairs[pairs.get("genA") != pairs.get("genB")]
    if len(pairs) == 0:
        # Common when a mu-slice only has one generator (so no A!=B pairs exist).
        n_g = int(cells[in_group(cells)].get("gen_id").nunique()) if "gen_id" in cells.columns else 0
        raise SystemExit(f"No distinct generator pairs (A!=B) in group; n_gens={n_g}")
    pairs = pairs[pd.to_numeric(pairs.get("flip_overlap"), errors="coerce").between(float(args.flip_overlap_min), float(args.flip_overlap_max), inclusive="both")]
    if len(pairs) == 0:
        # fall back to top commutator regardless of overlap
        pairs = comp[(comp.get("kind") == "pair") & in_group(comp)].copy()
        pairs = pairs[pairs.get("genA") != pairs.get("genB")]

    if len(pairs) == 0:
        n_g = int(cells[in_group(cells)].get("gen_id").nunique()) if "gen_id" in cells.columns else 0
        raise SystemExit(f"No distinct generator pairs (A!=B) in group (after fallback); n_gens={n_g}")

    pairs["_comm"] = pd.to_numeric(pairs.get("norm_commutator"), errors="coerce")
    pairs = pairs.sort_values("_comm", ascending=False)
    pick = pairs.iloc[0]
    genA = str(pick.get("genA"))
    genB = str(pick.get("genB"))

    # Representative holonomy dumps for each generator: choose minimal dist-to-minus1
    reps = (
        cells[cells.get("gen_id").isin([genA, genB])]
        .sort_values(["gen_id", "rep_holonomy_min_dist_to_minus1"], ascending=[True, True])
        .groupby("gen_id", as_index=False)
        .head(1)
        .copy()
    )

    if set(reps.get("gen_id", []).astype(str).tolist()) != {genA, genB}:
        raise SystemExit("Could not locate representative dumps for both generators")

    def dump_abs(gen_id: str) -> Path:
        r = reps[reps.get("gen_id") == gen_id].iloc[0]
        p = Path(str(r.get("rep_holonomy_dump_abs", "")))
        if p.exists():
            return p
        rel = str(r.get("rep_holonomy_dump", ""))
        if rel:
            pr = (out_root / rel).resolve()
            if pr.exists():
                return pr
        raise FileNotFoundError(f"Missing holonomy dump for {gen_id}")

    MA = _load_M(dump_abs(genA))
    MB = _load_M(dump_abs(genB))

    M12 = MA @ MB
    M21 = MB @ MA

    # Report KPIs
    k: Dict[str, Any] = {}
    k["genA"] = genA
    k["genB"] = genB
    k["norm_commutator_reported"] = float(pick.get("norm_commutator")) if np.isfinite(float(pick.get("norm_commutator"))) else None
    k["flip_overlap_reported"] = float(pick.get("flip_overlap")) if np.isfinite(float(pick.get("flip_overlap"))) else None

    k["norm_M12_minus_M21"] = float(np.linalg.norm(M12 - M21))
    k["trace_MA"] = complex(np.trace(MA))
    k["trace_MB"] = complex(np.trace(MB))
    k["trace_M12"] = complex(np.trace(M12))
    k["trace_M21"] = complex(np.trace(M21))

    eigA = _unit_eigs(MA)
    eigB = _unit_eigs(MB)
    eig12 = _unit_eigs(M12)
    eig21 = _unit_eigs(M21)

    k["eig_dist_M12_vs_MA"] = _eig_dist(eig12, eigA)
    k["eig_dist_M12_vs_MB"] = _eig_dist(eig12, eigB)
    k["eig_dist_M21_vs_MA"] = _eig_dist(eig21, eigA)
    k["eig_dist_M21_vs_MB"] = _eig_dist(eig21, eigB)
    k["eig_dist_M12_vs_M21"] = _eig_dist(eig12, eig21)

    # Serialize complex numbers safely
    def enc(x: Any) -> Any:
        if isinstance(x, complex):
            return {"re": float(np.real(x)), "im": float(np.imag(x))}
        if isinstance(x, (np.floating, np.integer)):
            return float(x)
        return x

    out_json = out_root / str(args.out_json)
    import json

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({k_: enc(v) if not isinstance(v, (dict, list)) else v for k_, v in k.items()}, f, indent=2)

    print(f"Wrote: {out_json}")
    print(f"Picked pair: {genA} vs {genB}")
    print(f"||M12 - M21|| = {k['norm_M12_minus_M21']:.6g} (loop-level noncommutativity proxy)")


if __name__ == "__main__":
    main()
