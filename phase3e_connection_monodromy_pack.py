import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd


def find_npz_files(out_root: str):
    hits = []
    for root, _dirs, files in os.walk(out_root):
        for fn in files:
            if fn.endswith(".npz") and fn.startswith("channel_diag"):
                hits.append(os.path.join(root, fn))
    return sorted(hits)


def cayley_from_lambda(L, eta: float):
    L = np.asarray(L, dtype=np.complex128)
    n = int(L.shape[0])
    I = np.eye(n, dtype=np.complex128)
    e = float(eta)
    return (L - 1j * e * I) @ np.linalg.inv(L + 1j * e * I)


def unitary_procrustes(A, B):
    """Find unitary U minimizing ||A - U B||_F (complex Procrustes)."""
    A = np.asarray(A, dtype=np.complex128)
    B = np.asarray(B, dtype=np.complex128)
    M = A @ np.conjugate(B.T)
    U, _s, Vh = np.linalg.svd(M)
    return U @ Vh


def principal_angles(Qa, Qb):
    if Qa.shape[1] == 0 or Qb.shape[1] == 0:
        return np.asarray([], dtype=np.float64)
    X = np.conjugate(Qa.T) @ Qb
    s = np.linalg.svd(X, compute_uv=False)
    s = np.clip(np.real(s), 0.0, 1.0)
    return np.arccos(s)


def basis_blocks_from_S(S, *, blocks: int, mode: str):
    """Return list of orthonormal bases Q[b] that partition eigenvectors.

    mode:
      - 'equalcount': sort eigenphases and split into B contiguous equal-size blocks
      - 'phasebands': fixed phase intervals on [0,2pi)
    """
    S = np.asarray(S, dtype=np.complex128)
    eigvals, eigvecs = np.linalg.eig(S)
    angles = np.mod(np.angle(eigvals), 2.0 * math.pi)
    n = int(angles.size)
    B = int(max(1, min(int(blocks), n)))

    if str(mode) == "phasebands":
        band_w = 2.0 * math.pi / B
        labels = np.floor(angles / band_w).astype(int)
        labels = np.clip(labels, 0, B - 1)
        order = [np.where(labels == b)[0] for b in range(B)]
    else:
        # equalcount
        idx_sorted = np.argsort(angles)
        sizes = [n // B for _ in range(B)]
        for i in range(n - sum(sizes)):
            sizes[i] += 1
        order = []
        start = 0
        for b in range(B):
            end = start + sizes[b]
            order.append(idx_sorted[start:end])
            start = end

    Qs = []
    dims = []
    for b in range(B):
        idx = np.asarray(order[b], dtype=np.int64)
        if idx.size == 0:
            Qs.append(np.zeros((n, 0), dtype=np.complex128))
            dims.append(0)
            continue
        V = eigvecs[:, idx]
        Q, _ = np.linalg.qr(V)
        Qs.append(np.asarray(Q, dtype=np.complex128))
        dims.append(int(Q.shape[1]))
    return Qs, dims


def parse_rel_metadata(path: str, out_root: str):
    rel = os.path.relpath(path, out_root).replace("\\", "/")
    parts = rel.split("/")
    backend_label = parts[0].replace("backend_", "") if parts else ""
    seed = -1
    anchor = -1
    wlo = float("nan")
    whi = float("nan")
    for p in parts:
        if p.startswith("seed") and "_anchor" in p:
            a, b = p.split("_anchor")
            seed = int(a.replace("seed", ""))
            anchor = int(b)
        if p.startswith("window_"):
            w = p.replace("window_", "")
            a, b = w.split("_", 1)
            wlo = float(a)
            whi = float(b)
    return rel, backend_label, seed, anchor, wlo, whi


def monodromy_for_artifact(npz_path: str, out_root: str, *, blocks: int, mode: str, eta_scale: float, loop_e0: int, loop_e1: int):
    rel, backend_label, seed, anchor, wlo, whi = parse_rel_metadata(npz_path, out_root)
    d = np.load(npz_path)

    if "lambda_snap" not in d.files:
        raise ValueError("missing lambda_snap")
    lam_snap = np.asarray(d["lambda_snap"], dtype=np.complex128)
    E_snap = np.asarray(d["energies_snap"], dtype=np.float64)
    eta0 = float(d["cayley_eta"]) if ("cayley_eta" in d.files) else None
    if eta0 is None or not np.isfinite(eta0) or eta0 <= 0:
        raise ValueError("missing/invalid cayley_eta")

    nE = int(lam_snap.shape[0])
    if nE < 2:
        return []

    i0 = int(loop_e0) if loop_e0 >= 0 else int(nE + loop_e0)
    i1 = int(loop_e1) if loop_e1 >= 0 else int(nE + loop_e1)
    i0 = int(np.clip(i0, 0, nE - 1))
    i1 = int(np.clip(i1, 0, nE - 1))
    if i0 == i1:
        i1 = 0 if i0 != 0 else nE - 1

    eta1 = float(eta0)
    eta2 = float(eta_scale) * float(eta0)

    # Rectangle loop in (E, eta):
    #  P0: (E[i0], eta1)
    #  P1: (E[i1], eta1)
    #  P2: (E[i1], eta2)
    #  P3: (E[i0], eta2)
    #  P4: (E[i0], eta1)
    pts = [
        (i0, eta1),
        (i1, eta1),
        (i1, eta2),
        (i0, eta2),
        (i0, eta1),
    ]

    bases = []
    dims0 = None
    for (ei, eta) in pts:
        S = cayley_from_lambda(lam_snap[ei], eta=float(eta))
        Qs, dims = basis_blocks_from_S(S, blocks=int(blocks), mode=str(mode))
        if dims0 is None:
            dims0 = dims
        bases.append(Qs)

    B = int(len(bases[0]))
    rows = []

    for b in range(B):
        # Procrustes transport along loop
        valid = True
        M = None
        step_angles = []
        step_overlaps = []
        step_smin = []

        for k in range(len(pts) - 1):
            U = np.asarray(bases[k][b], dtype=np.complex128)
            V = np.asarray(bases[k + 1][b], dtype=np.complex128)
            if U.shape[1] != V.shape[1]:
                valid = False
                break
            if U.shape[1] == 0:
                continue

            G = np.conjugate(U.T) @ V
            # singular values reflect overlap health
            s = np.linalg.svd(G, compute_uv=False)
            smin = float(np.min(np.real(s))) if s.size else float("nan")
            step_smin.append(smin)

            R = unitary_procrustes(U, V)
            M = R if (M is None) else (M @ R)

            ang = principal_angles(U, V)
            ang_max = float(np.max(ang)) if ang.size else float("nan")
            step_angles.append(ang_max)
            # overlap = Tr(PQ)/d = ||U^*V||_F^2 / d
            ov = float((np.linalg.norm(G, ord="fro") ** 2) / (float(U.shape[1]) + 1e-12))
            step_overlaps.append(ov)

        dim = int(bases[0][b].shape[1])
        if (not valid) or (dim == 0) or (M is None):
            rows.append(
                {
                    "backend_label": str(backend_label),
                    "seed": int(seed),
                    "anchor_seed": int(anchor),
                    "wlo": float(wlo),
                    "whi": float(whi),
                    "artifact": rel,
                    "block_mode": str(mode),
                    "blocks": int(blocks),
                    "block": int(b),
                    "dim": int(dim),
                    "loop_E0": float(E_snap[i0]),
                    "loop_E1": float(E_snap[i1]),
                    "eta1": float(eta1),
                    "eta2": float(eta2),
                    "valid": False,
                    "holonomy_det_phase": float("nan"),
                    "holonomy_trace": float("nan"),
                    "holonomy_norm_IminusM": float("nan"),
                    "holonomy_eigphase_meanabs": float("nan"),
                    "step_angle_median": float(np.nanmedian(step_angles)) if step_angles else float("nan"),
                    "step_angle_worst": float(np.nanmax(step_angles)) if step_angles else float("nan"),
                    "step_overlap_median": float(np.nanmedian(step_overlaps)) if step_overlaps else float("nan"),
                    "step_smin_median": float(np.nanmedian(step_smin)) if step_smin else float("nan"),
                }
            )
            continue

        # Holonomy invariants
        eig = np.linalg.eigvals(M)
        phases = np.angle(eig)
        det_phase = float(np.angle(np.linalg.det(M)))
        tr = np.trace(M)
        I = np.eye(M.shape[0], dtype=np.complex128)
        norm_im = float(np.linalg.norm(I - M, ord="fro"))

        rows.append(
            {
                "backend_label": str(backend_label),
                "seed": int(seed),
                "anchor_seed": int(anchor),
                "wlo": float(wlo),
                "whi": float(whi),
                "artifact": rel,
                "block_mode": str(mode),
                "blocks": int(blocks),
                "block": int(b),
                "dim": int(dim),
                "loop_E0": float(E_snap[i0]),
                "loop_E1": float(E_snap[i1]),
                "eta1": float(eta1),
                "eta2": float(eta2),
                "valid": True,
                "holonomy_det_phase": float(det_phase),
                "holonomy_trace": complex(tr).real,  # store real part only for quick scanning
                "holonomy_norm_IminusM": float(norm_im),
                "holonomy_eigphase_meanabs": float(np.mean(np.abs(phases))) if phases.size else float("nan"),
                "step_angle_median": float(np.nanmedian(step_angles)) if step_angles else float("nan"),
                "step_angle_worst": float(np.nanmax(step_angles)) if step_angles else float("nan"),
                "step_overlap_median": float(np.nanmedian(step_overlaps)) if step_overlaps else float("nan"),
                "step_smin_median": float(np.nanmedian(step_smin)) if step_smin else float("nan"),
            }
        )

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--block_mode", default="equalcount", choices=["equalcount", "phasebands"])
    ap.add_argument("--eta_scale", type=float, default=2.0)
    ap.add_argument("--loop_e0", type=int, default=0)
    ap.add_argument("--loop_e1", type=int, default=-1)
    args = ap.parse_args()

    out_root = str(args.out_root)
    npz_files = find_npz_files(out_root)
    if not npz_files:
        raise SystemExit(f"No channel_diag*.npz found under {out_root}")

    all_rows = []
    bad = 0
    for p in npz_files:
        try:
            all_rows.extend(
                monodromy_for_artifact(
                    p,
                    out_root,
                    blocks=int(args.blocks),
                    mode=str(args.block_mode),
                    eta_scale=float(args.eta_scale),
                    loop_e0=int(args.loop_e0),
                    loop_e1=int(args.loop_e1),
                )
            )
        except Exception:
            bad += 1

    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(out_root, "phase3e_monodromy_rows.csv")
    df.to_csv(out_csv, index=False)

    summary = {
        "out_root": out_root,
        "blocks": int(args.blocks),
        "block_mode": str(args.block_mode),
        "eta_scale": float(args.eta_scale),
        "loop_e0": int(args.loop_e0),
        "loop_e1": int(args.loop_e1),
        "n_artifacts": int(len(npz_files)),
        "n_rows": int(len(df)),
        "n_artifacts_failed": int(bad),
    }
    if len(df):
        valid = df[df["valid"] == True]
        summary["valid_fraction"] = float(len(valid) / max(1, len(df)))
        if len(valid):
            summary["holonomy_norm_IminusM_median"] = float(np.nanmedian(valid["holonomy_norm_IminusM"].values))
            summary["holonomy_eigphase_meanabs_median"] = float(np.nanmedian(valid["holonomy_eigphase_meanabs"].values))
            summary["step_angle_median_median"] = float(np.nanmedian(valid["step_angle_median"].values))
            summary["step_smin_median_median"] = float(np.nanmedian(valid["step_smin_median"].values))

    with open(os.path.join(out_root, "phase3e_monodromy_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[phase3e_connection_monodromy_pack] wrote phase3e_monodromy_rows.csv and phase3e_monodromy_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
