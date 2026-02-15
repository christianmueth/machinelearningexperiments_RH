import argparse
import glob
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResolvedPath:
    requested: str
    resolved: str
    method: str


def _as_path_str(p: str) -> str:
    return str(Path(p).expanduser().resolve())


def _parse_csv_list(s: str) -> list[str]:
    items = []
    for part in (s or "").split(","):
        part = part.strip()
        if part:
            items.append(part)
    return items


def _norm_fro(A: np.ndarray) -> float:
    return float(np.linalg.norm(A, ord="fro"))


def _unitarity_defect(S: np.ndarray) -> float:
    S = np.asarray(S, dtype=np.complex128)
    n = int(S.shape[0])
    I = np.eye(n, dtype=np.complex128)
    return _norm_fro(S.conj().T @ S - I)


def _hermitian_defect(Lam: np.ndarray) -> float:
    Lam = np.asarray(Lam, dtype=np.complex128)
    return _norm_fro(Lam - Lam.conj().T)


def _svmin(Lam: np.ndarray) -> float:
    Lam = np.asarray(Lam, dtype=np.complex128)
    svals = np.linalg.svd(Lam, compute_uv=False)
    return float(svals[-1])


def _eigmin_abs(Lam: np.ndarray) -> float:
    Lam = np.asarray(Lam, dtype=np.complex128)
    eig = np.linalg.eigvals(Lam)
    return float(np.min(np.abs(eig)))


def _min_dist_to_minus1(S: np.ndarray) -> float:
    S = np.asarray(S, dtype=np.complex128)
    eig = np.linalg.eigvals(S)
    return float(np.min(np.abs(eig + 1)))


def _num_eigs_near_minus1(S: np.ndarray, tol: float) -> int:
    S = np.asarray(S, dtype=np.complex128)
    eig = np.linalg.eigvals(S)
    return int(np.sum(np.abs(eig + 1) <= float(tol)))


def _det_phase(S: np.ndarray) -> float:
    S = np.asarray(S, dtype=np.complex128)
    return float(np.angle(np.linalg.det(S)))


def _cayley_from_lambda(Lam: np.ndarray, eta: float) -> np.ndarray:
    Lam = np.asarray(Lam, dtype=np.complex128)
    eta = float(eta)
    n = int(Lam.shape[0])
    I = np.eye(n, dtype=np.complex128)

    # S = (Lam - i eta I) (Lam + i eta I)^{-1}
    A = Lam + 1j * eta * I
    B = Lam - 1j * eta * I
    invA = np.linalg.solve(A, I)
    return B @ invA


def _resolve_artifact_path(workspace_root: str, requested: str) -> ResolvedPath | None:
    if requested is None:
        return None
    requested = str(requested).strip()
    if not requested:
        return None

    # 1) direct path relative to workspace root
    direct = Path(workspace_root) / Path(requested)
    if direct.exists():
        return ResolvedPath(requested=requested, resolved=str(direct.resolve()), method="workspace_rel")

    # 2) common Phase-3D output roots
    # The hotspot CSVs often store paths like "backend_legacy/.../channel_diag__E3_K8.npz"
    pat1 = str(Path(workspace_root) / "out_phase3D_channel_diag_*" / Path(requested))
    matches = glob.glob(pat1)
    if matches:
        return ResolvedPath(requested=requested, resolved=_as_path_str(matches[0]), method="out_phase3D_channel_diag_glob")

    # 3) fallback: recursive glob for the full relative path
    # (This is slower, but we only do it for a small number of artifacts.)
    requested_os = requested.replace("/", os.sep).replace("\\", os.sep)
    pat2 = os.path.join(workspace_root, "**", requested_os)
    matches = glob.glob(pat2, recursive=True)
    if matches:
        return ResolvedPath(requested=requested, resolved=_as_path_str(matches[0]), method="recursive_glob")

    return None


def analyze_channel_diag_npz(npz_path: str, *, minus1_tol: float, eta_override: float | None) -> list[dict]:
    d = np.load(npz_path)
    required = ["energies_snap", "lambda_snap", "S_snap"]
    for k in required:
        if k not in d.files:
            raise ValueError(f"{npz_path}: missing required key {k}")

    E = np.asarray(d["energies_snap"], dtype=np.float64)
    lam = np.asarray(d["lambda_snap"], dtype=np.complex128)
    S = np.asarray(d["S_snap"], dtype=np.complex128)

    if lam.ndim != 3 or S.ndim != 3:
        raise ValueError(f"{npz_path}: expected lambda_snap and S_snap to be (nE,n,n)")

    if lam.shape[0] != E.shape[0] or S.shape[0] != E.shape[0]:
        raise ValueError(f"{npz_path}: inconsistent energies_snap vs snapshots")

    eta = float(eta_override) if (eta_override is not None) else float(d["cayley_eta"]) if ("cayley_eta" in d.files) else float("nan")

    rows: list[dict] = []
    for i in range(int(E.shape[0])):
        Lam_i = lam[i]
        S_i = S[i]

        recompute_ok = np.isfinite(eta)
        S_re = None
        S_re_diff = float("nan")
        if recompute_ok:
            S_re = _cayley_from_lambda(Lam_i, eta=eta)
            S_re_diff = _norm_fro(S_re - S_i)

        rows.append(
            {
                "npz_path": npz_path,
                "energy_index": int(i),
                "energy": float(E[i]),
                "n": int(S_i.shape[0]),
                "eta": float(eta),
                "lambda_hermitian_defect": _hermitian_defect(Lam_i),
                "lambda_svmin": _svmin(Lam_i),
                "lambda_eigmin_abs": _eigmin_abs(Lam_i),
                "S_unitarity_defect": _unitarity_defect(S_i),
                "S_det_phase": _det_phase(S_i),
                "S_min_dist_to_minus1": _min_dist_to_minus1(S_i),
                "S_num_eigs_near_minus1": _num_eigs_near_minus1(S_i, tol=minus1_tol),
                "S_recompute_diff_fro": float(S_re_diff),
            }
        )

    return rows


def _df_to_markdown_table(df: pd.DataFrame, *, max_rows: int = 200) -> str:
    """Render a DataFrame as a Markdown table without optional dependencies."""
    if df is None or len(df) == 0:
        return "(empty)"

    orig_len = int(len(df))
    df = df.head(int(max_rows)).copy()
    cols = list(df.columns)

    def fmt(v: object) -> str:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return ""
        if isinstance(v, float):
            # Compact but readable for tiny defects.
            if abs(v) < 1e-3:
                return f"{v:.3e}"
            return f"{v:.6g}"
        return str(v)

    header = "| " + " | ".join(map(str, cols)) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]).replace("\n", " ") for c in cols) + " |")
    if int(len(df)) < orig_len:
        lines.append(f"(truncated to {max_rows} rows)")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase-3E/3D scattering-spine Version-0 diagnostics on saved channel_diag NPZ files.")

    ap.add_argument("--workspace_root", type=str, default=str(Path(__file__).resolve().parents[1]), help="Workspace root (used to resolve artifact paths)")

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--npz", nargs="+", help="One or more channel_diag__*.npz files to analyze")
    src.add_argument("--hotspots_csv", type=str, help="Hotspot CSV containing artifact paths (e.g. phase3e_elambda_pi_hotspots.csv)")

    ap.add_argument("--artifact_cols", type=str, default="artifact_a,artifact_b", help="Comma-separated artifact columns to load from hotspots CSV")
    ap.add_argument("--max_rows", type=int, default=50, help="Max hotspots rows to process")

    ap.add_argument("--minus1_tol", type=float, default=1e-2, help="Tolerance for counting eig(S) near -1")
    ap.add_argument("--eta_override", type=float, default=float("nan"), help="If finite, override eta used for Cayley recompute check")

    ap.add_argument("--out_csv", type=str, required=True, help="Output CSV path")
    ap.add_argument("--out_md", type=str, default="", help="Optional Markdown summary output")

    args = ap.parse_args()

    workspace_root = _as_path_str(args.workspace_root)
    eta_override = None if not np.isfinite(float(args.eta_override)) else float(args.eta_override)

    all_rows: list[dict] = []

    if args.npz:
        for p in args.npz:
            npz_path = _as_path_str(p)
            rows = analyze_channel_diag_npz(npz_path, minus1_tol=float(args.minus1_tol), eta_override=eta_override)
            all_rows.extend(rows)

    if args.hotspots_csv:
        hs_path = _as_path_str(args.hotspots_csv)
        df = pd.read_csv(hs_path)
        cols = _parse_csv_list(args.artifact_cols)
        if not cols:
            raise ValueError("--artifact_cols must list at least one column")
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"hotspots_csv missing columns: {missing}")

        df_head = df.head(int(args.max_rows)).copy()
        for ridx, r in df_head.iterrows():
            base_meta = {k: r[k] for k in df.columns if k not in cols}
            for c in cols:
                rp = _resolve_artifact_path(workspace_root, str(r[c]))
                if rp is None:
                    all_rows.append(
                        {
                            **base_meta,
                            "artifact_col": c,
                            "artifact_requested": r[c],
                            "artifact_resolved": "",
                            "artifact_resolve_method": "",
                            "error": f"could_not_resolve:{r[c]}",
                        }
                    )
                    continue

                try:
                    rows = analyze_channel_diag_npz(rp.resolved, minus1_tol=float(args.minus1_tol), eta_override=eta_override)
                    for rr in rows:
                        all_rows.append(
                            {
                                **base_meta,
                                "artifact_col": c,
                                "artifact_requested": rp.requested,
                                "artifact_resolved": rp.resolved,
                                "artifact_resolve_method": rp.method,
                                "error": "",
                                **rr,
                            }
                        )
                except Exception as e:  # noqa: BLE001
                    all_rows.append(
                        {
                            **base_meta,
                            "artifact_col": c,
                            "artifact_requested": rp.requested,
                            "artifact_resolved": rp.resolved,
                            "artifact_resolve_method": rp.method,
                            "error": f"analyze_failed:{type(e).__name__}:{e}",
                        }
                    )

    if not all_rows:
        raise RuntimeError("No rows produced.")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(out_csv, index=False)

    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)

        # Make a compact summary table per (artifact_resolved)
        if "artifact_resolved" in out_df.columns:
            key = "artifact_resolved"
        else:
            key = "npz_path"

        df_ok = out_df[out_df.get("error", "") == ""].copy() if "error" in out_df.columns else out_df.copy()
        if len(df_ok) == 0:
            summary = pd.DataFrame([{key: "(none)", "note": "no analyzable rows"}])
        else:
            summary = (
                df_ok.groupby(key, dropna=False)
                .agg(
                    n_rows=("energy", "count"),
                    min_lambda_svmin=("lambda_svmin", "min"),
                    min_lambda_eigmin_abs=("lambda_eigmin_abs", "min"),
                    max_S_unitarity_defect=("S_unitarity_defect", "max"),
                    min_S_min_dist_to_minus1=("S_min_dist_to_minus1", "min"),
                    max_S_num_eigs_near_minus1=("S_num_eigs_near_minus1", "max"),
                )
                .reset_index()
                .sort_values(["min_lambda_svmin", "min_S_min_dist_to_minus1"], ascending=[True, True])
            )

        md_lines = [
            "# Scattering-spine v0 diagnostics",
            "",
            f"- minus1_tol: `{float(args.minus1_tol)}`",
            f"- eta_override: `{eta_override}`",
            "",
            "## Per-file summary",
            "",
            _df_to_markdown_table(summary),
            "",
        ]
        out_md.write_text("\n".join(md_lines), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
