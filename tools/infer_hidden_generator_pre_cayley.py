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


def _parse_str_list(s: str) -> list[str]:
    out: list[str] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(part)
    return out


def _pauli() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    I = np.eye(2, dtype=np.complex128)
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sy = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    return I, sx, sy, sz


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
    """Two-point gauge selector (u=0 and u=u_probe) with u-sensitivity."""

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


def _length_values(name: str, p: np.ndarray, logp: np.ndarray) -> np.ndarray:
    name = str(name).strip().lower()
    if name in {"logp", "log"}:
        return logp
    if name in {"logp_over_p", "logp/p", "log_over_p"}:
        return logp / (p + 1e-15)
    if name in {"invp", "1/p"}:
        return 1.0 / (p + 1e-15)
    if name in {"log1p_invp", "log(1+1/p)", "log1p(1/p)"}:
        return np.log1p(1.0 / (p + 1e-15))
    if name in {"logp1_minus_logp", "log(p+1)-logp"}:
        return np.log(p + 1.0) - logp
    raise ValueError(f"unknown length '{name}'")


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    aa = a - float(np.mean(a))
    bb = b - float(np.mean(b))
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(float(aa @ bb) / den) if den > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Hidden-generator diagnostic at the pre-Cayley layer: compute the 2x2 Schur-complement boundary operator Λ_p "
            "(Hermitian approx), then test whether its traceless part aligns to a single axis and scales as Λ_p^0 ≈ ℓ(p) * H0. "
            "Compares multiple prime length laws ℓ(p) (logp, logp/p, 1/p, log(1+1/p), ...)."
        )
    )
    ap.add_argument("--primes", default="2,3,5,7,11,13,17,19,23,29,31,37")
    ap.add_argument("--n_primes", type=int, default=0)
    ap.add_argument("--u", type=float, default=0.2)
    ap.add_argument("--u_probe", type=float, default=0.2)

    ap.add_argument("--sharp", choices=["transpose", "conj_transpose"], default="conj_transpose")
    ap.add_argument("--X_mode", choices=["raw", "det1"], default="det1")
    ap.add_argument("--X_gamma", type=float, default=1.0)
    ap.add_argument("--scattering", choices=["lambda_pm_i", "i_pm_lambda"], default="i_pm_lambda")
    ap.add_argument(
        "--p_mode",
        default="p",
        choices=["p", "p1_over_p", "p_over_p1", "invp", "p_minus1_over_p"],
        help="How to feed prime information into the local diagonal factor D (length-law replacement test).",
    )

    ap.add_argument(
        "--lengths",
        default="logp,logp_over_p,invp,log1p_invp,logp1_minus_logp",
        help="Comma list of length laws to compare.",
    )
    ap.add_argument("--out_csv", required=True, help="Per-prime Λ_p diagnostics.")
    ap.add_argument("--out_length_csv", required=True, help="Length-law comparison summary.")

    args = ap.parse_args()

    if int(args.n_primes) > 0:
        primes = _first_n_primes(int(args.n_primes))
    else:
        primes = sim._parse_int_list(str(args.primes))
    if not primes:
        raise SystemExit("no primes")

    u = float(args.u)

    # Gauge selection from seed primes.
    p_seed_n = min(10, len(primes))
    half = max(1, p_seed_n // 2)
    primes_seed: list[int] = []
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

    I, sx, sy, sz = _pauli()

    rows: list[dict] = []

    for p in primes:
        try:
            _, A, Ash = sim._local_blocks_for_prime_power(
                int(p),
                1,
                sharp_mode=str(args.sharp),
                x_mode=str(args.X_mode),
                x_gamma=float(args.X_gamma),
                x_shear=float(u),
                x_lower=float(-u),
                p_mode=str(args.p_mode),
            )
            B = sim._bulk_B_from_A(A, Ash)
            Lam = sim._schur_complement_Lambda(B, boundary=gauge.boundary, sign=str(gauge.sign))
        except Exception:
            continue

        # Also compute the explicit Schur pieces: B_bb and Q := B_bi B_ii^{-1} B_ib.
        bidx = tuple(int(i) for i in gauge.boundary)
        all_idx = list(range(6))
        iidx = [i for i in all_idx if i not in bidx]
        Bbb = B[np.ix_(bidx, bidx)]
        Bbi = B[np.ix_(bidx, iidx)]
        Bib = B[np.ix_(iidx, bidx)]
        Bii = B[np.ix_(iidx, iidx)]
        used_pinv = False
        try:
            Bii_inv = np.linalg.inv(Bii)
        except Exception:
            used_pinv = True
            Bii_inv = np.linalg.pinv(Bii, rcond=1e-12)
        Q = (Bbi @ Bii_inv @ Bib).astype(np.complex128)
        # Condition estimate (can be inf if singular)
        try:
            cond_Bii = float(np.linalg.cond(Bii))
        except Exception:
            cond_Bii = float("inf")

        def herm_and_traceless(M: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
            MH = 0.5 * (M + M.conj().T)
            trM = float(np.real(np.trace(MH)))
            M0 = MH - 0.5 * np.trace(MH) * I
            return MH, M0, trM

        LamH, Lam0, trLam = herm_and_traceless(Lam)
        BbbH, Bbb0, trBbb = herm_and_traceless(Bbb)
        QH, Q0, trQ = herm_and_traceless(Q)

        def pauli_coords(M0: np.ndarray) -> tuple[float, float, float, float]:
            hx_ = float(np.real(0.5 * np.trace(M0 @ sx)))
            hy_ = float(np.real(0.5 * np.trace(M0 @ sy)))
            hz_ = float(np.real(0.5 * np.trace(M0 @ sz)))
            fro_ = float(np.linalg.norm(M0, ord="fro"))
            return hx_, hy_, hz_, fro_

        hx, hy, hz, Lam0_f = pauli_coords(Lam0)
        b_hx, b_hy, b_hz, Bbb0_f = pauli_coords(Bbb0)
        q_hx, q_hy, q_hz, Q0_f = pauli_coords(Q0)

        herm_def = float(np.linalg.norm(Lam - Lam.conj().T, ord="fro") / (np.linalg.norm(Lam, ord="fro") + 1e-300))
        herm_def_bbb = float(np.linalg.norm(Bbb - Bbb.conj().T, ord="fro") / (np.linalg.norm(Bbb, ord="fro") + 1e-300))
        herm_def_q = float(np.linalg.norm(Q - Q.conj().T, ord="fro") / (np.linalg.norm(Q, ord="fro") + 1e-300))

        rows.append(
            {
                "p": int(p),
                "logp": float(math.log(float(p))),
                "herm_def": float(herm_def),
                "herm_def_Bbb": float(herm_def_bbb),
                "herm_def_Q": float(herm_def_q),
                "cond_Bii": float(cond_Bii),
                "used_pinv": int(1 if used_pinv else 0),
                "tr_LamH": float(trLam),
                "tr_BbbH": float(trBbb),
                "tr_QH": float(trQ),
                "hx": float(hx),
                "hy": float(hy),
                "hz": float(hz),
                "Lam0_fro": float(Lam0_f),
                "Bbb_hx": float(b_hx),
                "Bbb_hy": float(b_hy),
                "Bbb_hz": float(b_hz),
                "Bbb0_fro": float(Bbb0_f),
                "Q_hx": float(q_hx),
                "Q_hy": float(q_hy),
                "Q_hz": float(q_hz),
                "Q0_fro": float(Q0_f),
            }
        )

    if not rows:
        raise SystemExit("no Λ_p computed")

    df = pd.DataFrame(rows).sort_values("p")

    # Evaluate length candidates on multiple targets: Lambda0, Bbb0, Q0.
    p_arr = df["p"].to_numpy(float)
    logp_arr = df["logp"].to_numpy(float)
    lengths = _parse_str_list(args.lengths)

    def axis_and_pca(V: np.ndarray) -> tuple[np.ndarray, float, float]:
        vnorm = np.linalg.norm(V, axis=1)
        Vn = V / (vnorm[:, None] + 1e-15)
        mean_dir = np.mean(Vn, axis=0)
        R_ = float(np.linalg.norm(mean_dir))
        axis_ = mean_dir / (R_ + 1e-15)
        C_ = np.cov(V.T)
        w_, _ = np.linalg.eigh(C_)
        w_ = np.sort(w_)[::-1]
        ev1_frac_ = float((w_[0] / w_.sum()) if float(w_.sum()) > 0 else 0.0)
        return axis_, R_, ev1_frac_

    targets = {
        "Lam0": df[["hx", "hy", "hz"]].to_numpy(float),
        "Bbb0": df[["Bbb_hx", "Bbb_hy", "Bbb_hz"]].to_numpy(float),
        "Q0": df[["Q_hx", "Q_hy", "Q_hz"]].to_numpy(float),
    }

    length_rows: list[dict] = []
    for tname, V in targets.items():
        axis, R, ev1_frac = axis_and_pca(V)
        vnorm = np.linalg.norm(V, axis=1)
        vnorm_med = float(np.median(vnorm))
        for lname in lengths:
            Lp = _length_values(lname, p_arr, logp_arr)
            s = V @ axis
            den = float(Lp @ Lp) + 1e-30
            a_hat = float((Lp @ s) / den)
            V_hat = (a_hat * Lp)[:, None] * axis[None, :]
            rmse_V = float(np.sqrt(np.mean(np.sum((V - V_hat) ** 2, axis=1))))
            rel_rmse_V = float(rmse_V / (vnorm_med + 1e-15))
            kappa = s / (Lp + 1e-15)
            length_rows.append(
                {
                    "target": str(tname),
                    "length": str(lname),
                    "axis_R": float(R),
                    "pca_ev1": float(ev1_frac),
                    "a_hat": float(a_hat),
                    "rmse_V": float(rmse_V),
                    "rel_rmse_V": float(rel_rmse_V),
                    "vnorm_med": float(vnorm_med),
                    "kappa_med": float(np.median(kappa)),
                    "kappa_std": float(np.std(kappa)),
                    "corr_kappa_1/p": float(_corr(kappa, 1.0 / (p_arr + 1e-15))),
                    "corr_kappa_logp": float(_corr(kappa, logp_arr)),
                }
            )

    length_df = pd.DataFrame(length_rows)
    length_df = length_df.sort_values(["target", "rel_rmse_V", "kappa_std"], ascending=[True, True, True])

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    out_len = Path(args.out_length_csv)
    out_len.parent.mkdir(parents=True, exist_ok=True)
    length_df.to_csv(out_len, index=False)

    print(
        "diag | "
        + " | ".join(
            [
                f"n={len(df)}",
                f"herm_def_med={float(df['herm_def'].median()):.3e}",
                f"herm_def_Bbb_med={float(df['herm_def_Bbb'].median()):.3e}",
                f"herm_def_Q_med={float(df['herm_def_Q'].median()):.3e}",
                f"cond_Bii_med={float(df['cond_Bii'].median()):.3e}",
                f"used_pinv_frac={float(df['used_pinv'].mean()):.3f}",
            ]
        )
    )

    fro_col = {"Lam0": "Lam0_fro", "Bbb0": "Bbb0_fro", "Q0": "Q0_fro"}

    for tname in ["Lam0", "Bbb0", "Q0"]:
        sub = length_df[length_df["target"] == tname]
        best = dict(sub.iloc[0])
        print(
            f"axis_{tname} | "
            + " | ".join(
                [
                    f"axis_R={best['axis_R']:.6f}",
                    f"pca_ev1={best['pca_ev1']:.6f}",
                    f"fro_med={float(df[fro_col[tname]].median()):.3e}",
                ]
            )
        )
        print(
            f"best_{tname} | "
            + " | ".join(
                [
                    f"length={best['length']}",
                    f"rmse_V={best['rmse_V']:.3e}",
                    f"rel_rmse_V={best['rel_rmse_V']:.3e}",
                    f"kappa_med={best['kappa_med']:.3e}",
                    f"kappa_std={best['kappa_std']:.3e}",
                ]
            )
        )
        # Print a short ranking line
        top = sub.sort_values(["rel_rmse_V", "kappa_std"]).head(5)
        print(
            f"rank_{tname} | "
            + ", ".join([f"{r.length}:{float(r.rel_rmse_V):.3e}" for r in top.itertuples(index=False)])
        )

    print(f"out | per_prime={out_csv} | lengths={out_len}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
