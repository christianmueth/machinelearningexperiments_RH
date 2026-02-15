import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class CanonicalClass:
    canon_id: str
    dim: int
    eigvals_unit: np.ndarray  # shape (dim,)
    v_minus1_unit: Optional[np.ndarray]  # shape (dim,)
    trace: complex
    det: complex
    n_members: int = 0


def _unit_eigs(M: np.ndarray) -> np.ndarray:
    vals = np.linalg.eigvals(M)
    mags = np.abs(vals)
    mags = np.where(mags == 0, 1.0, mags)
    vals = vals / mags
    order = np.argsort(np.angle(vals))
    return vals[order]


def _eig_dist(a: np.ndarray, b: np.ndarray) -> float:
    # Compare complex numbers as 2D points.
    if a.shape != b.shape:
        return float("inf")
    dr = np.real(a) - np.real(b)
    di = np.imag(a) - np.imag(b)
    return float(np.sqrt(np.mean(dr * dr + di * di)))


def _canon_unit_vector(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v).reshape(-1)
    if v.size == 0:
        return v

    # Fix arbitrary complex phase: rotate so the largest-magnitude entry is real-positive.
    idx = int(np.argmax(np.abs(v)))
    phase = np.angle(v[idx])
    v = v * np.exp(-1j * phase)

    # Normalize.
    nrm = float(np.linalg.norm(v))
    if nrm == 0.0:
        return v
    return v / nrm


def _load_holonomy(npz_path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    # NPZ contains an object array meta_json; allow_pickle avoids load errors.
    with np.load(str(npz_path), allow_pickle=True) as d:
        M = np.asarray(d["M"])
        v = None
        if "v_minus1" in d.files:
            try:
                v = _canon_unit_vector(np.asarray(d["v_minus1"]))
            except Exception:
                v = None
    return M, v


def _fmt_complex(z: complex, digits: int) -> str:
    return f"{float(np.real(z)):.{digits}f}+{float(np.imag(z)):.{digits}f}f*i".replace("f*i", "j")


def _sig_from_eigs(eigs_unit: np.ndarray, digits: int) -> str:
    parts = []
    for z in eigs_unit:
        parts.append(f"{float(np.real(z)):.{digits}f}{float(np.imag(z)):+.{digits}f}j")
    return "|".join(parts)


def _assign_class(
    classes: List[CanonicalClass],
    dim: int,
    eigs_unit: np.ndarray,
    v_minus1_unit: Optional[np.ndarray],
    trace: complex,
    det: complex,
    tol: float,
) -> Tuple[CanonicalClass, bool, float]:
    # Try matching against existing classes; allow global sign flip (-M) equivalence
    # as an option because some gauge choices can flip overall sign.
    best: Optional[Tuple[CanonicalClass, bool, float]] = None

    for c in classes:
        if c.dim != dim:
            continue

        # Eigenvalues alone often collapse for involutions; include v_minus1 direction when available.
        d_e0 = _eig_dist(eigs_unit, c.eigvals_unit)
        d_e1 = _eig_dist(-eigs_unit, c.eigvals_unit)

        d_v = 0.0
        if v_minus1_unit is not None and c.v_minus1_unit is not None:
            # Invariant to overall sign/phase: use abs inner-product.
            ip = complex(np.vdot(v_minus1_unit, c.v_minus1_unit))
            d_v = float(1.0 - min(1.0, abs(ip)))

        d0 = float(d_e0 + d_v)
        d1 = float(d_e1 + d_v)
        if best is None or min(d0, d1) < best[2]:
            best = (c, bool(d1 < d0), float(min(d0, d1)))

    if best is not None and best[2] <= tol:
        return best

    canon_id = f"C{len(classes):04d}_d{dim}"
    cnew = CanonicalClass(
        canon_id=canon_id,
        dim=dim,
        eigvals_unit=eigs_unit,
        v_minus1_unit=v_minus1_unit,
        trace=trace,
        det=det,
    )
    classes.append(cnew)
    return (cnew, False, 0.0)


def _iter_run_dirs(out_root: Path, include_glob: str, include_base: bool) -> List[Path]:
    run_dirs: List[Path] = []
    if include_base:
        run_dirs.append(out_root)
    if include_glob:
        run_dirs.extend([p for p in sorted(out_root.glob(include_glob)) if p.is_dir()])
    return run_dirs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True, help="Phase-3E run folder (smoke run) containing phase3f_*.csv")
    ap.add_argument(
        "--include_glob",
        default="refine_prime_*",
        help="Glob for child refinement runs to include (default: refine_prime_*)",
    )
    ap.add_argument("--include_base", type=int, default=1, help="Include the base out_root itself (default: 1)")
    ap.add_argument("--tol", type=float, default=2e-2, help="Eigenvalue signature matching tolerance (default: 2e-2)")
    ap.add_argument("--sig_digits", type=int, default=4, help="Digits for pretty signatures (default: 4)")
    ap.add_argument("--max_pairs_per_group", type=int, default=25, help="Relations: cap per group (default: 25)")

    args = ap.parse_args()

    out_root = Path(str(args.out_root)).resolve()
    run_dirs = _iter_run_dirs(out_root, str(args.include_glob), bool(int(args.include_base)))

    classes: List[CanonicalClass] = []
    member_rows: List[Dict[str, object]] = []
    gen_to_canon: Dict[str, str] = {}

    for run_dir in run_dirs:
        cells_csv = run_dir / "phase3f_event_cells_with_gen.csv"
        gens_csv = run_dir / "phase3f_generators.csv"
        if not (cells_csv.exists() and gens_csv.exists()):
            continue

        try:
            df_cells = pd.read_csv(cells_csv)
            df_gens = pd.read_csv(gens_csv)
        except Exception:
            continue

        # Choose one representative holonomy dump per generator.
        # Prefer the smallest dist-to-minus1 (closest to a clean -1 eigenvector).
        if "gen_id" not in df_cells.columns or "rep_holonomy_dump_abs" not in df_cells.columns:
            continue

        rep = (
            df_cells.sort_values(["gen_id", "rep_holonomy_min_dist_to_minus1"], ascending=[True, True])
            .groupby("gen_id", as_index=False)
            .head(1)
            .copy()
        )

        run_rel = os.path.relpath(str(run_dir), str(out_root))

        for _, row in rep.iterrows():
            gen_id = str(row["gen_id"])
            npz_abs = Path(str(row["rep_holonomy_dump_abs"]))
            if not npz_abs.exists():
                # Fall back to relative path inside run_dir
                npz_rel = Path(str(row.get("rep_holonomy_dump", "")))
                if npz_rel:
                    npz_abs = (run_dir / npz_rel).resolve()
            if not npz_abs.exists():
                continue

            try:
                M, v_minus1_unit = _load_holonomy(npz_abs)
            except Exception:
                continue

            if M.ndim != 2 or M.shape[0] != M.shape[1]:
                continue

            dim = int(M.shape[0])
            tr = complex(np.trace(M))
            det = complex(np.linalg.det(M))
            eigs_unit = _unit_eigs(M)

            c, flipped, dist = _assign_class(classes, dim, eigs_unit, v_minus1_unit, tr, det, float(args.tol))
            c.n_members += 1
            gen_to_canon[gen_id] = c.canon_id

            member_rows.append(
                {
                    "canon_id": c.canon_id,
                    "run_rel": run_rel,
                    "gen_id": gen_id,
                    "seed": int(row.get("seed", -1)) if "seed" in row else None,
                    "anchor_seed": int(row.get("anchor_seed", -1)) if "anchor_seed" in row else None,
                    "wlo": float(row.get("wlo", float("nan"))) if "wlo" in row else None,
                    "whi": float(row.get("whi", float("nan"))) if "whi" in row else None,
                    "mu": float(row.get("mu", float("nan"))) if "mu" in row else None,
                    "block": int(row.get("block", -1)) if "block" in row else None,
                    "dim": dim,
                    "lambda_mid": float(row.get("lambda_mid", float("nan"))) if "lambda_mid" in row else None,
                    "rep_holonomy_dump": str(row.get("rep_holonomy_dump", "")),
                    "rep_holonomy_dump_abs": str(npz_abs),
                    "trace": tr,
                    "det": det,
                    "eigs_sig": _sig_from_eigs(eigs_unit, int(args.sig_digits)),
                    "v_minus1_sig": (
                        "" if v_minus1_unit is None else _sig_from_eigs(np.asarray(v_minus1_unit).reshape(-1), int(args.sig_digits))
                    ),
                    "flipped": bool(flipped),
                    "match_dist": float(dist),
                }
            )

    # Write dictionary outputs
    df_members = pd.DataFrame(member_rows)
    dict_csv = out_root / "phase3f_primitive_dictionary.csv"
    df_members.to_csv(dict_csv, index=False)

    class_rows: List[Dict[str, object]] = []
    for c in classes:
        class_rows.append(
            {
                "canon_id": c.canon_id,
                "dim": c.dim,
                "n_members": c.n_members,
                "trace": c.trace,
                "det": c.det,
                "eigs_sig": _sig_from_eigs(c.eigvals_unit, int(args.sig_digits)),
                "v_minus1_sig": (
                    "" if c.v_minus1_unit is None else _sig_from_eigs(np.asarray(c.v_minus1_unit).reshape(-1), int(args.sig_digits))
                ),
            }
        )
    df_classes = pd.DataFrame(class_rows).sort_values(["dim", "n_members"], ascending=[True, False])
    classes_csv = out_root / "phase3f_primitive_classes.csv"
    df_classes.to_csv(classes_csv, index=False)

    # Build relations table from batch composition outputs
    rel_rows: List[Dict[str, object]] = []
    for run_dir in run_dirs:
        comp_csv = run_dir / "phase3f_batch_composition.csv"
        if not comp_csv.exists():
            continue
        try:
            comp = pd.read_csv(comp_csv)
        except Exception:
            continue

        if "kind" not in comp.columns:
            continue
        comp = comp[comp["kind"] == "pair"].copy()
        if len(comp) == 0:
            continue

        # For each (seed, anchor, wlo, whi, mu, block, dim), keep top commutators
        group_cols = [c for c in ["seed", "anchor_seed", "wlo", "whi", "mu", "block", "dim"] if c in comp.columns]
        if "norm_commutator" in comp.columns:
            comp = comp.sort_values("norm_commutator", ascending=False)

        run_rel = os.path.relpath(str(run_dir), str(out_root))

        if group_cols:
            for _, sub in comp.groupby(group_cols, dropna=False):
                sub = sub.head(int(args.max_pairs_per_group))
                for _, r in sub.iterrows():
                    ga = str(r.get("genA", ""))
                    gb = str(r.get("genB", ""))
                    if ga not in gen_to_canon or gb not in gen_to_canon:
                        continue
                    rel_rows.append(
                        {
                            "run_rel": run_rel,
                            "seed": r.get("seed"),
                            "anchor_seed": r.get("anchor_seed"),
                            "wlo": r.get("wlo"),
                            "whi": r.get("whi"),
                            "mu": r.get("mu"),
                            "block": r.get("block"),
                            "dim": r.get("dim"),
                            "canonA": gen_to_canon[ga],
                            "canonB": gen_to_canon[gb],
                            "genA": ga,
                            "genB": gb,
                            "norm_commutator": r.get("norm_commutator"),
                            "flip_overlap": r.get("flip_overlap"),
                            "norm_concat_minus_prod": r.get("norm_concat_minus_prod"),
                        }
                    )
        else:
            # No grouping columns: just dump top rows
            for _, r in comp.head(int(args.max_pairs_per_group)).iterrows():
                ga = str(r.get("genA", ""))
                gb = str(r.get("genB", ""))
                if ga not in gen_to_canon or gb not in gen_to_canon:
                    continue
                rel_rows.append(
                    {
                        "run_rel": run_rel,
                        "canonA": gen_to_canon[ga],
                        "canonB": gen_to_canon[gb],
                        "genA": ga,
                        "genB": gb,
                        "norm_commutator": r.get("norm_commutator"),
                        "flip_overlap": r.get("flip_overlap"),
                        "norm_concat_minus_prod": r.get("norm_concat_minus_prod"),
                    }
                )

    df_rel = pd.DataFrame(rel_rows)
    rel_csv = out_root / "phase3f_primitive_relations.csv"
    df_rel.to_csv(rel_csv, index=False)

    # Minimal console summary
    print(f"Wrote: {dict_csv}")
    print(f"Wrote: {classes_csv}")
    print(f"Wrote: {rel_csv}")
    print(f"Classes: {len(classes)} | Members: {len(df_members)} | Relations: {len(df_rel)}")


if __name__ == "__main__":
    main()
