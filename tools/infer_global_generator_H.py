from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Import simulator module from sibling file (tools/ is not a package).
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
import six_by_six_prime_tower_sim as sim  # type: ignore


def _parse_int_list(s: str) -> list[int]:
    return sim._parse_int_list(str(s))


def _parse_str_list(s: str) -> list[str]:
    out: list[str] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(part)
    return out


def _first_n_primes(n: int) -> list[int]:
    n = int(n)
    if n <= 0:
        return []
    out: list[int] = []
    x = 2
    while len(out) < n:
        is_p = True
        r = int(math.isqrt(x))
        for q in out:
            if q > r:
                break
            if x % q == 0:
                is_p = False
                break
        if is_p:
            out.append(int(x))
        x += 1 if x == 2 else 2
    return out


def _pauli() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    I = np.eye(2, dtype=np.complex128)
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sy = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    return I, sx, sy, sz


def _unitary_polar(U: np.ndarray) -> np.ndarray:
    """Project a 2x2 matrix to the nearest unitary by polar decomposition."""

    U = np.asarray(U, dtype=np.complex128)
    M = U.conj().T @ U
    # invsqrt(M) via eigendecomposition
    w, V = np.linalg.eigh(M)
    w = np.maximum(np.real(w), 0.0)
    invsqrt = V @ np.diag(1.0 / np.sqrt(w + 1e-300)) @ V.conj().T
    return (U @ invsqrt).astype(np.complex128)


def _normalize_det1(U: np.ndarray) -> np.ndarray:
    U = np.asarray(U, dtype=np.complex128)
    det = complex(np.linalg.det(U))
    # Choose principal sqrt(det)
    s = complex(np.exp(0.5 * complex(np.log(det)))) if det != 0 else 1.0 + 0j
    if abs(s) < 1e-300:
        return U
    return (U / s).astype(np.complex128)


def _project_to_su2(S: np.ndarray) -> np.ndarray:
    """Project an approximate packet to SU(2): unitary + det=1."""

    U = _unitary_polar(S)
    U = _normalize_det1(U)
    # Re-unitarize after det normalization (small correction)
    U = _unitary_polar(U)
    U = _normalize_det1(U)
    return U.astype(np.complex128)


def _log_su2(U: np.ndarray) -> tuple[np.ndarray, float]:
    """Return anti-Hermitian log(U) for U in SU(2) and the principal angle phi in [0,pi]."""

    U = np.asarray(U, dtype=np.complex128)
    tr = complex(np.trace(U))
    # For SU(2), tr is real = 2 cos(phi). Use real part and clamp.
    c = float(np.real(tr) / 2.0)
    c = max(-1.0, min(1.0, c))
    phi = float(math.acos(c))

    # Compute skew-Hermitian part
    K = 0.5 * (U - U.conj().T)

    if phi < 1e-10:
        # For small angles, log(U) ~ K
        return K.astype(np.complex128), phi

    s = float(math.sin(phi))
    if abs(s) < 1e-12:
        # Near pi: axis is ill-conditioned; fall back to eigen decomposition.
        w, V = np.linalg.eig(U)
        angles = np.angle(w)
        angles = np.unwrap(angles)
        D = np.diag(1j * angles)
        A = V @ D @ np.linalg.inv(V)
        A = 0.5 * (A - A.conj().T)
        return A.astype(np.complex128), phi

    # For SU(2): U - U^* = 2 i sin(phi) (n·sigma)
    # so log(U) = i phi (n·sigma) = (phi/sin(phi)) * (U-U^*)/2
    A = (phi / s) * K
    # ensure anti-Hermitian
    A = 0.5 * (A - A.conj().T)
    return A.astype(np.complex128), phi


def _exp_su2_from_H(H: np.ndarray, logp: float) -> np.ndarray:
    """Compute exp(i logp H) for 2x2 Hermitian traceless H."""

    H = np.asarray(H, dtype=np.complex128)
    A = float(logp) * H
    # For traceless Hermitian 2x2: eigenvalues are ±a where a = sqrt(0.5 tr(A^2)) (real).
    a2 = 0.5 * float(np.real(np.trace(A @ A)))
    a = float(math.sqrt(max(a2, 0.0)))
    I, *_ = _pauli()
    if a < 1e-12:
        return I.astype(np.complex128)
    return (math.cos(a) * I + 1j * (math.sin(a) / a) * A).astype(np.complex128)


def _length_values(name: str, p: np.ndarray, logp: np.ndarray) -> np.ndarray:
    name = str(name).strip().lower()
    if name in {"logp", "log"}:
        return logp
    if name in {"logp_over_p", "logp/p", "log_over_p"}:
        return logp / (p + 1e-15)
    if name in {"invp", "1/p"}:
        return 1.0 / (p + 1e-15)
    raise ValueError(f"unknown length '{name}'")


def _unitarity_defect(U: np.ndarray) -> float:
    U = np.asarray(U, dtype=np.complex128)
    I = np.eye(2, dtype=np.complex128)
    return float(np.linalg.norm(U.conj().T @ U - I, ord="fro") / (np.linalg.norm(I, ord="fro") + 1e-300))


def _det_defect(U: np.ndarray) -> float:
    det = complex(np.linalg.det(np.asarray(U, dtype=np.complex128)))
    return float(abs(det - 1.0))


@dataclass(frozen=True)
class Gauge:
    boundary: tuple[int, int]
    sign: str


def _choose_gauge_two_point(
    *,
    primes_seed: list[int],
    sharp_mode: str,
    x_mode: str,
    x_gamma: float,
    scattering_mode: str,
    u_probe: float,
    hd_tol: float = 1e-6,
    ud_tol: float = 1e-6,
    sens_min: float = 1e-6,
    den_floor: float = 1e-9,
) -> tuple[Gauge, dict[str, float]]:
    """Copy of the two-point gauge selector (u=0 and u=u_probe), minimizing degeneracy."""

    pairs: list[tuple[int, int]] = [(i, j) for i in range(6) for j in range(i + 1, 6)]
    u_list = [0.0, float(u_probe)]

    def herm_def_clamped(Lam: np.ndarray) -> float:
        num = float(np.linalg.norm(Lam - Lam.conj().T, ord="fro"))
        den = float(np.linalg.norm(Lam, ord="fro"))
        return float(num / max(float(den_floor), den))

    candidates: list[tuple[float, float, float, tuple[int, int], str]] = []

    for boundary in pairs:
        for sgn in ["-", "+"]:
            max_hd = 0.0
            max_ud = 0.0
            sens_acc = 0.0
            sens_n = 0
            ok_any = False
            for p in primes_seed:
                S0 = None
                for u in u_list:
                    try:
                        _, A, Ash = sim._local_blocks_for_prime_power(
                            int(p),
                            1,
                            sharp_mode=str(sharp_mode),
                            x_mode=str(x_mode),
                            x_gamma=float(x_gamma),
                            x_shear=float(u),
                            x_lower=float(-u),
                        )
                        B = sim._bulk_B_from_A(A, Ash)
                        Lam = sim._schur_complement_Lambda(B, boundary=boundary, sign=str(sgn))
                        S = sim._scattering_from_Lambda(Lam, mode=str(scattering_mode))
                    except Exception:
                        continue

                    ok_any = True
                    max_hd = max(max_hd, herm_def_clamped(Lam))
                    max_ud = max(max_ud, float(sim._unitarity_defect(S)))

                    if float(u) == 0.0:
                        S0 = S
                    elif S0 is not None:
                        sens_acc += float(np.linalg.norm(S - S0, ord="fro"))
                        sens_n += 1

            if not ok_any:
                continue
            sensitivity = float(sens_acc / max(1, sens_n))
            candidates.append((sensitivity, max_hd, max_ud, boundary, str(sgn)))

    if not candidates:
        raise RuntimeError("gauge selection failed")

    good = [c for c in candidates if (c[1] <= float(hd_tol) and c[2] <= float(ud_tol) and c[0] >= float(sens_min))]
    if not good:
        good = [c for c in candidates if (c[1] <= float(hd_tol) and c[2] <= float(ud_tol))]
    if not good:
        good = candidates

    good.sort(key=lambda x: (-x[0], x[1], x[2]))
    sens, hd, ud, boundary, sgn = good[0]
    return Gauge(boundary=tuple(boundary), sign=str(sgn)), {"seed_sens": float(sens), "seed_hd": float(hd), "seed_ud": float(ud)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Infer whether local packets S_p(u) are generated by a single global Hermitian generator H via S_p ≈ exp(i log p * H). "
            "Computes H_p = (-i/log p) log(S_p) using a stable SU(2) axis-angle log and summarizes dispersion across primes."
        )
    )
    ap.add_argument("--primes", default="2,3,5,7,11,13,17,19,23,29,31,37")
    ap.add_argument("--n_primes", type=int, default=0, help="If >0, ignore --primes and use the first n primes.")
    ap.add_argument("--u", type=float, default=0.0)
    ap.add_argument("--u_probe", type=float, default=0.2)

    ap.add_argument("--sharp", choices=["transpose", "conj_transpose"], default="conj_transpose")
    ap.add_argument("--X_mode", choices=["raw", "det1"], default="det1")
    ap.add_argument("--X_gamma", type=float, default=1.0)
    ap.add_argument("--scattering", choices=["lambda_pm_i", "i_pm_lambda"], default="i_pm_lambda")

    ap.add_argument(
        "--local_model",
        choices=["sixby6", "satake"],
        default="sixby6",
        help="Which local packet model to use: sixby6 (from 6x6 bulk) or satake (direct injection).",
    )
    ap.add_argument(
        "--satake_family",
        choices=["trivial", "phase_hash", "phase_logp", "table"],
        default="trivial",
        help="Only used when --local_model satake.",
    )
    ap.add_argument(
        "--satake_matrix",
        choices=["diag", "companion"],
        default="diag",
        help="Only used when --local_model satake.",
    )
    ap.add_argument("--theta_scale", type=float, default=1.0, help="Only used for satake_family phase_*.")
    ap.add_argument("--seed", type=int, default=0, help="Only used for satake_family phase_hash.")
    ap.add_argument("--satake_table_csv", default="", help="Only used when satake_family=table.")

    ap.add_argument(
        "--lengths",
        default="logp,logp_over_p,invp",
        help="Comma list of prime length candidates L(p) to test: logp, logp_over_p, invp.",
    )
    ap.add_argument("--out_length_csv", default="", help="Optional CSV summary for the length-function comparison.")

    ap.add_argument("--out_csv", required=True)

    args = ap.parse_args()

    if int(args.n_primes) > 0:
        primes = _first_n_primes(int(args.n_primes))
    else:
        primes = _parse_int_list(args.primes)
    if not primes:
        raise SystemExit("--primes must be non-empty")

    u = float(args.u)

    # Gauge selection from seed primes (ends).
    p_seed_n = min(8, len(primes))
    half = max(1, p_seed_n // 2)
    primes_seed = []
    for p in list(primes[:half]) + list(primes[-(p_seed_n - half) :]):
        if p not in primes_seed:
            primes_seed.append(int(p))

    gauge, gauge_stats = _choose_gauge_two_point(
        primes_seed=primes_seed,
        sharp_mode=str(args.sharp),
        x_mode=str(args.X_mode),
        x_gamma=float(args.X_gamma),
        scattering_mode=str(args.scattering),
        u_probe=float(args.u_probe),
    )

    print(
        "gauge | "
        + " | ".join(
            [
                f"boundary={gauge.boundary}",
                f"sign={gauge.sign}",
                f"u={u:g}",
                f"u_probe={float(args.u_probe):g}",
                f"seed_sens={gauge_stats['seed_sens']:.3e}",
                f"seed_hd={gauge_stats['seed_hd']:.3e}",
                f"seed_ud={gauge_stats['seed_ud']:.3e}",
            ]
        )
    )

    # Build packets at requested u.
    satake_table = None
    if str(args.local_model).strip().lower() == "satake" and str(args.satake_family).strip().lower() == "table":
        if not str(args.satake_table_csv).strip():
            raise SystemExit("--satake_table_csv is required when --satake_family table")
        satake_table = sim._load_satake_table_csv(str(args.satake_table_csv))

    packets = sim._build_packets(
        primes,
        1,
        local_model=str(args.local_model),
        boundary=gauge.boundary,
        sign=gauge.sign,
        sharp_mode=str(args.sharp),
        x_mode=str(args.X_mode),
        x_gamma=float(args.X_gamma),
        x_shear=float(u),
        x_lower=float(-u),
        scattering_mode=str(args.scattering),
        satake_family=str(args.satake_family),
        satake_matrix=str(args.satake_matrix),
        theta_scale=float(args.theta_scale),
        seed=int(args.seed),
        satake_table=satake_table,
    )
    by_p = {int(pkt.p): pkt for pkt in packets}

    I, sx, sy, sz = _pauli()

    rows: list[dict] = []
    H_list: list[np.ndarray] = []
    Uproj_by_p: dict[int, np.ndarray] = {}
    for p in primes:
        pkt = by_p.get(int(p))
        if pkt is None:
            continue
        S = np.asarray(pkt.S, dtype=np.complex128)
        logp = float(math.log(float(p)))

        U = _project_to_su2(S)
        Uproj_by_p[int(p)] = U
        logU, phi = _log_su2(U)
        H = (-1j / logp) * logU  # Hermitian traceless if hypothesis holds.
        H = 0.5 * (H + H.conj().T)

        # Pauli coordinates: H = hx*sx + hy*sy + hz*sz.
        hx = float(np.real(0.5 * np.trace(H @ sx)))
        hy = float(np.real(0.5 * np.trace(H @ sy)))
        hz = float(np.real(0.5 * np.trace(H @ sz)))
        trH = float(np.real(np.trace(H)))

        rows.append(
            {
                "p": int(p),
                "logp": float(logp),
                "phi": float(phi),
                "unitarity_def_S": float(sim._unitarity_defect(S)),
                "det_def_S": float(abs(complex(np.linalg.det(S)) - 1.0)),
                "unitarity_def_Uproj": float(_unitarity_defect(U)),
                "det_def_Uproj": float(_det_defect(U)),
                "trH": float(trH),
                "hx": float(hx),
                "hy": float(hy),
                "hz": float(hz),
                "H_fro": float(np.linalg.norm(H, ord="fro")),
            }
        )
        H_list.append(H)

    if not H_list:
        raise SystemExit("no packets produced")

    H_mean = sum(H_list) / float(len(H_list))
    # Force traceless Hermitian mean
    H_mean = 0.5 * (H_mean + H_mean.conj().T)
    H_mean = H_mean - 0.5 * np.trace(H_mean) * I

    # Reconstruction error
    errs = []
    for r in rows:
        p = int(r["p"])
        logp = float(r["logp"])
        S_obs = np.asarray(by_p[p].S, dtype=np.complex128)
        S_hat = _exp_su2_from_H(H_mean, logp)
        errs.append(float(np.linalg.norm(S_hat - _project_to_su2(S_obs), ord="fro")))
    err_mean = float(np.mean(errs))
    err_med = float(np.median(errs))

    # Dispersion of H_p around H_mean
    dH = [float(np.linalg.norm(H - H_mean, ord="fro")) for H in H_list]
    dH_mean = float(np.mean(dH))
    dH_med = float(np.median(dH))

    df = pd.DataFrame(rows).sort_values("p")

    # Axis / collinearity diagnostics in Pauli-vector coordinates.
    V = df[["hx", "hy", "hz"]].to_numpy(float)
    vnorm = np.linalg.norm(V, axis=1)
    Vn = V / (vnorm[:, None] + 1e-15)
    mean_dir = np.mean(Vn, axis=0)
    R = float(np.linalg.norm(mean_dir))  # 1.0 means perfect alignment
    axis = mean_dir / (R + 1e-15)
    C = np.cov(V.T)
    w, _ = np.linalg.eigh(C)
    w = np.sort(w)[::-1]
    ev1_frac = float((w[0] / w.sum()) if float(w.sum()) > 0 else 0.0)

    coef = df["phi"].to_numpy(float) / (df["logp"].to_numpy(float) + 1e-15)  # phi/logp
    invp = 1.0 / df["p"].to_numpy(float)
    logp = df["logp"].to_numpy(float)

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 2:
            return float("nan")
        aa = a - float(np.mean(a))
        bb = b - float(np.mean(b))
        den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
        return float(float(aa @ bb) / den) if den > 0 else float("nan")

    corr_coef_logp = corr(coef, logp)
    corr_coef_invp = corr(coef, invp)
    a_over_p = coef / (invp + 1e-15)  # ~= constant if coef ~ a/p
    a_p = df["phi"].to_numpy(float) * df["p"].to_numpy(float) / (df["logp"].to_numpy(float) + 1e-15)  # phi*p/logp

    # Length-function comparison: does there exist H0 such that S_p ≈ exp(i L(p) H0)?
    lengths = _parse_str_list(args.lengths)
    p_arr = df["p"].to_numpy(float)
    logp_arr = df["logp"].to_numpy(float)

    length_rows: list[dict] = []
    for lname in lengths:
        Lp = _length_values(lname, p_arr, logp_arr)
        x = Lp / (logp_arr + 1e-15)  # since V_p = (Lp/logp) * v0
        s = V @ axis  # scalar along the common axis
        den = float(x @ x) + 1e-30
        a_hat = float((x @ s) / den)
        v0 = a_hat * axis
        V_hat = x[:, None] * v0[None, :]
        rmse_V = float(np.sqrt(np.mean(np.sum((V - V_hat) ** 2, axis=1))))

        H0 = (v0[0] * sx + v0[1] * sy + v0[2] * sz).astype(np.complex128)

        S_errs = []
        for p_i, L_i in zip(df["p"].to_numpy(int), Lp.tolist()):
            U_obs = Uproj_by_p.get(int(p_i))
            if U_obs is None:
                continue
            S_hat = _exp_su2_from_H(H0, float(L_i))
            S_errs.append(float(np.linalg.norm(S_hat - U_obs, ord="fro")))
        mean_S_err = float(np.mean(S_errs)) if S_errs else float("nan")
        med_S_err = float(np.median(S_errs)) if S_errs else float("nan")

        phi_arr = df["phi"].to_numpy(float)
        phi_over_L = phi_arr / (Lp + 1e-15)

        length_rows.append(
            {
                "length": str(lname),
                "a_hat": float(a_hat),
                "rmse_V": float(rmse_V),
                "mean_S_err": float(mean_S_err),
                "med_S_err": float(med_S_err),
                "phi_over_L_med": float(np.median(phi_over_L)),
                "phi_over_L_std": float(np.std(phi_over_L)),
            }
        )

    length_df = pd.DataFrame(length_rows)
    if not length_df.empty:
        length_df = length_df.sort_values(["mean_S_err", "rmse_V"], ascending=[True, True])
        best = dict(length_df.iloc[0])
        print(
            "length_best | "
            + " | ".join(
                [
                    f"length={best['length']}",
                    f"mean_S_err={best['mean_S_err']:.3e}",
                    f"rmse_V={best['rmse_V']:.3e}",
                    f"phi_over_L_med={best['phi_over_L_med']:.3e}",
                    f"phi_over_L_std={best['phi_over_L_std']:.3e}",
                ]
            )
        )
        if len(length_df) > 1:
            print("length_table | " + ", ".join([f"{r['length']}:{r['mean_S_err']:.3e}" for r in length_rows]))

    if str(args.out_length_csv).strip():
        outL = Path(str(args.out_length_csv))
        outL.parent.mkdir(parents=True, exist_ok=True)
        length_df.to_csv(outL, index=False)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(
        "summary | "
        + " | ".join(
            [
                f"n={len(H_list)}",
                f"||H_mean||_F={float(np.linalg.norm(H_mean, ord='fro')):.3e}",
                f"mean||H_p-H_mean||_F={dH_mean:.3e}",
                f"med||H_p-H_mean||_F={dH_med:.3e}",
                f"mean||S_hat-S_obs||_F={err_mean:.3e}",
                f"med||S_hat-S_obs||_F={err_med:.3e}",
                f"axis_R={R:.6f}",
                f"pca_ev1={ev1_frac:.6f}",
                f"coef(phi/logp)_med={float(np.median(coef)):.3e}",
                f"corr(coef,1/p)={corr_coef_invp:.3f}",
                f"corr(coef,logp)={corr_coef_logp:.3f}",
                f"a_p=phi*p/logp_med={float(np.median(a_p)):.3e}",
                f"a_p_std={float(np.std(a_p)):.3e}",
                f"out={out_path}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
