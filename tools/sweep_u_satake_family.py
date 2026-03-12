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


def _trapz_weights(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float).ravel()
    n = int(t.size)
    if n <= 0:
        return np.asarray([], dtype=float)
    if n == 1:
        return np.asarray([1.0], dtype=float)
    dt = np.diff(t)
    w = np.zeros(n, dtype=float)
    w[0] = float(dt[0]) * 0.5
    w[-1] = float(dt[-1]) * 0.5
    if n > 2:
        w[1:-1] = 0.5 * (dt[:-1] + dt[1:])
    s = float(np.sum(w))
    if not math.isfinite(s) or s <= 0:
        return np.full(n, 1.0 / float(n), dtype=float)
    return (w / s).astype(float)


def _weighted_norm(w: np.ndarray, z: np.ndarray) -> float:
    w = np.asarray(w, dtype=float).ravel()
    z = np.asarray(z, dtype=np.complex128).ravel()
    if w.size != z.size:
        raise ValueError("w and z must have same length")
    return float(math.sqrt(float(np.sum(w * (np.abs(z) ** 2)))))


def _matched_filter_coeffs(t: np.ndarray, g: np.ndarray, *, sigma: float, n_list: list[int]) -> dict[int, complex]:
    t = np.asarray(t, dtype=float).ravel()
    g = np.asarray(g, dtype=np.complex128).ravel()
    if t.size != g.size:
        raise ValueError("t and g length mismatch")
    if t.size == 0:
        return {}

    w = _trapz_weights(t)
    out: dict[int, complex] = {}
    for n in n_list:
        ln = float(math.log(float(n)))
        ph = np.exp(1j * t * ln)
        est = (float(n) ** float(sigma)) * complex(np.sum(w * g * ph))
        out[int(n)] = complex(est)
    return out


def _reconstruct_from_coeffs(t: np.ndarray, *, sigma: float, coeffs: dict[int, complex]) -> np.ndarray:
    t = np.asarray(t, dtype=float).ravel()
    if t.size == 0:
        return np.asarray([], dtype=np.complex128)

    out = np.zeros_like(t, dtype=np.complex128)
    for n, a in coeffs.items():
        ln = float(math.log(float(n)))
        out += complex(a) * (float(n) ** (-float(sigma))) * np.exp(-1j * t * ln)
    return out.astype(np.complex128)


def _prime_powers_upto(n_max: int, primes: list[int]) -> list[int]:
    out: set[int] = set()
    for p in primes:
        pk = int(p)
        while pk <= int(n_max):
            out.add(pk)
            if pk > int(n_max) // int(p):
                break
            pk *= int(p)
    return sorted(out)


def _theta_from_eig(eig: complex) -> float:
    eig = complex(eig)
    return float(math.atan2(float(np.imag(eig)), float(np.real(eig))))


@dataclass(frozen=True)
class ThetaFit:
    a_over_p: float
    rmse: float
    median_abs_resid: float


def _fit_theta_a_over_p(primes: list[int], theta: np.ndarray) -> ThetaFit:
    p = np.asarray(primes, dtype=float).ravel()
    th = np.asarray(theta, dtype=float).ravel()
    if p.size != th.size or p.size == 0:
        return ThetaFit(float("nan"), float("nan"), float("nan"))

    x = (1.0 / p).reshape(-1, 1)
    y = th.reshape(-1, 1)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    a = float(coef.reshape(-1)[0])
    th_fit = (x.reshape(-1) * a).astype(float)
    resid = (th - th_fit).astype(float)
    rmse = float(np.sqrt(np.mean(resid**2)))
    med_abs = float(np.median(np.abs(resid)))
    return ThetaFit(a_over_p=a, rmse=rmse, median_abs_resid=med_abs)


def _eig_deviation_stats(S: np.ndarray) -> tuple[float, float]:
    w = np.linalg.eigvals(np.asarray(S, dtype=np.complex128))
    dev = np.max(np.abs(w - 1.0))
    return float(dev), float(sim._unitarity_defect(S))


def _satake_template_coeffs(
    *,
    packets: list[sim.Packet],
    sigma: float,
    n_max: int,
    r_max: int,
) -> dict[int, complex]:
    # Build a_n for n=p^r: a_n = log p * Tr(S_p^r)
    # using the packet's S (k=1) per prime.
    # NOTE: Our Phi(s) in the simulator is defined as Phi(s) = -d/ds log det(I-K(s)).
    # For det(I - p^{-s} S_p), this gives coefficients a_{p^r} = -log p * Tr(S_p^r).
    by_p: dict[int, np.ndarray] = {}
    for pkt in packets:
        if int(pkt.k) != 1:
            continue
        by_p[int(pkt.p)] = np.asarray(pkt.S, dtype=np.complex128)

    out: dict[int, complex] = {}
    for p, S in by_p.items():
        ln_p = float(math.log(float(p)))
        P = np.eye(2, dtype=np.complex128)
        pk = int(p)
        for r in range(1, int(r_max) + 1):
            P = (P @ S).astype(np.complex128)
            tr = complex(np.trace(P))
            if pk > int(n_max):
                break
            out[int(pk)] = complex(-ln_p * tr)
            if pk > int(n_max) // int(p):
                break
            pk *= int(p)
    return out


def _compute_global_phi(
    *,
    packets: list[sim.Packet],
    sigma: float,
    t_grid: np.ndarray,
) -> np.ndarray:
    out = np.zeros_like(t_grid, dtype=np.complex128)
    for i, t in enumerate(t_grid.tolist()):
        s = complex(float(sigma), float(t))
        K, Kp = sim._global_K(s, packets)
        I = np.eye(K.shape[0], dtype=np.complex128)
        M = (I - K).astype(np.complex128)
        Minv = np.linalg.inv(M)
        out[i] = complex(np.trace(Minv @ Kp))
    return out.astype(np.complex128)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Sweep mixed-sign SL(2) deformation u with v=-u for the sixby6 local model. "
            "For each u: infer local eigen-angle law theta_p, compute deviation from trivial packet, "
            "and score global Phi(s) against (i) matched-filter full basis, (ii) prime-power-only basis, "
            "and (iii) the generalized Satake ladder template log p * Tr(S_p^r)."
        )
    )
    ap.add_argument("--u_list", default="0,0.02,0.05,0.1,0.2", help="Comma list of u values to sweep")
    ap.add_argument("--primes_infer", default="2,3,5,7,11,13,17,19,23,29,31,37,41,43,47")
    ap.add_argument("--primes_global", default="2,3,5,7,11,13")

    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--t_min", type=float, default=10.0)
    ap.add_argument("--t_max", type=float, default=30.0)
    ap.add_argument("--n_t", type=int, default=401)

    ap.add_argument("--n_max", type=int, default=128)
    ap.add_argument("--r_max", type=int, default=8)

    ap.add_argument("--out_csv", required=True)

    # Fixed conventions (kept explicit to avoid ambiguity)
    ap.add_argument("--sharp", choices=["transpose", "conj_transpose"], default="conj_transpose")
    ap.add_argument("--X_mode", choices=["raw", "det1"], default="det1")
    ap.add_argument("--X_gamma", type=float, default=1.0)
    ap.add_argument("--scattering", choices=["lambda_pm_i", "i_pm_lambda"], default="i_pm_lambda")

    args = ap.parse_args()

    u_list = sim._parse_float_list(str(args.u_list))
    if not u_list:
        raise SystemExit("--u_list must be non-empty")

    primes_infer = sim._parse_int_list(str(args.primes_infer))
    primes_global = sim._parse_int_list(str(args.primes_global))
    if not primes_infer or not primes_global:
        raise SystemExit("--primes_infer and --primes_global must be non-empty")

    sigma = float(args.sigma)
    t_grid = np.linspace(float(args.t_min), float(args.t_max), int(args.n_t), dtype=float)

    n_max = int(args.n_max)
    n_list = list(range(2, n_max + 1))

    rows: list[dict] = []

    # Use the same seed sets for boundary search per u.
    p_seed = [int(primes_global[0]), int(primes_global[-1])] if len(primes_global) >= 2 else [int(primes_global[0])]
    k_seed = [1]

    for u in u_list:
        u = float(u)
        v = -float(u)

        boundary, schur_sign, seed_hd, seed_ud = sim._boundary_search(
            p_seed,
            k_seed,
            sharp_mode=str(args.sharp),
            x_mode=str(args.X_mode),
            x_gamma=float(args.X_gamma),
            x_shear=float(u),
            x_lower=float(v),
            scattering_mode=str(args.scattering),
        )

        # Local inference packets (k=1) on the broader prime set.
        infer_packets = sim._build_packets(
            primes_infer,
            1,
            local_model="sixby6",
            boundary=boundary,
            sign=str(schur_sign),
            sharp_mode=str(args.sharp),
            x_mode=str(args.X_mode),
            x_gamma=float(args.X_gamma),
            x_shear=float(u),
            x_lower=float(v),
            scattering_mode=str(args.scattering),
            satake_family="trivial",
            satake_matrix="diag",
            theta_scale=1.0,
            seed=0,
        )

        # Extract theta_p from eig1.
        infer_by_p = {pkt.p: pkt for pkt in infer_packets}
        thetas: list[float] = []
        devs: list[float] = []
        uds: list[float] = []
        primes_used: list[int] = []
        for p in primes_infer:
            pkt = infer_by_p.get(int(p))
            if pkt is None:
                continue
            S = np.asarray(pkt.S, dtype=np.complex128)
            w = np.linalg.eigvals(S)
            # order by argument
            w = sorted([complex(z) for z in w], key=lambda z: float(math.atan2(float(np.imag(z)), float(np.real(z)))))
            eig1 = w[0]
            thetas.append(_theta_from_eig(eig1))
            d, ud = _eig_deviation_stats(S)
            devs.append(float(d))
            uds.append(float(ud))
            primes_used.append(int(p))

        theta_fit = _fit_theta_a_over_p(primes_used, np.asarray(thetas, dtype=float))

        # Global packets (k=1) on the smaller prime set.
        global_packets = sim._build_packets(
            primes_global,
            1,
            local_model="sixby6",
            boundary=boundary,
            sign=str(schur_sign),
            sharp_mode=str(args.sharp),
            x_mode=str(args.X_mode),
            x_gamma=float(args.X_gamma),
            x_shear=float(u),
            x_lower=float(v),
            scattering_mode=str(args.scattering),
            satake_family="trivial",
            satake_matrix="diag",
            theta_scale=1.0,
            seed=0,
        )

        # Compute global Phi(t)
        Phi = _compute_global_phi(packets=global_packets, sigma=sigma, t_grid=t_grid)

        # Full matched filter coefficients and recon
        coeff_full = _matched_filter_coeffs(t_grid, Phi, sigma=sigma, n_list=n_list)
        Phi_full = _reconstruct_from_coeffs(t_grid, sigma=sigma, coeffs=coeff_full)

        # Prime-power-only recon (using the same coeff estimates but zeroing non-pp)
        pp_list = _prime_powers_upto(n_max, primes=primes_global)
        coeff_pp = {n: coeff_full[n] for n in pp_list if n in coeff_full}
        Phi_pp = _reconstruct_from_coeffs(t_grid, sigma=sigma, coeffs=coeff_pp)

        # Generalized Satake ladder template recon
        coeff_tpl = _satake_template_coeffs(packets=global_packets, sigma=sigma, n_max=n_max, r_max=int(args.r_max))
        Phi_tpl = _reconstruct_from_coeffs(t_grid, sigma=sigma, coeffs=coeff_tpl)

        w = _trapz_weights(t_grid)
        y_norm = _weighted_norm(w, Phi)
        rel_full = _weighted_norm(w, Phi - Phi_full) / (y_norm + 1e-300)
        rel_pp = _weighted_norm(w, Phi - Phi_pp) / (y_norm + 1e-300)
        rel_tpl = _weighted_norm(w, Phi - Phi_tpl) / (y_norm + 1e-300)

        # Energy split of coeff_full between pp and non-pp
        pp_set = set(pp_list)
        e_pp = float(sum(abs(coeff_full[n]) ** 2 for n in coeff_full.keys() if n in pp_set))
        e_non = float(sum(abs(coeff_full[n]) ** 2 for n in coeff_full.keys() if n not in pp_set))
        frac_pp = float(e_pp / (e_pp + e_non + 1e-300))

        row = {
            "u": float(u),
            "v": float(v),
            "sigma": float(sigma),
            "boundary": f"{boundary[0]},{boundary[1]}",
            "schur_sign": str(schur_sign),
            "seed_max_herm_def": float(seed_hd),
            "seed_max_unit_def": float(seed_ud),
            "n_primes_infer": int(len(primes_used)),
            "median_max_abs_eig_minus1": float(np.median(np.asarray(devs, dtype=float))) if devs else float("nan"),
            "max_max_abs_eig_minus1": float(np.max(np.asarray(devs, dtype=float))) if devs else float("nan"),
            "median_unitarity_defect": float(np.median(np.asarray(uds, dtype=float))) if uds else float("nan"),
            "max_unitarity_defect": float(np.max(np.asarray(uds, dtype=float))) if uds else float("nan"),
            "theta_fit_a_over_p": float(theta_fit.a_over_p),
            "theta_fit_rmse": float(theta_fit.rmse),
            "theta_fit_median_abs_resid": float(theta_fit.median_abs_resid),
            "rel_rmse_full": float(rel_full),
            "rel_rmse_pp": float(rel_pp),
            "rel_rmse_satake_template": float(rel_tpl),
            "coeff_energy_frac_primepowers": float(frac_pp),
            "n_max": int(n_max),
            "r_max": int(args.r_max),
            "t_min": float(args.t_min),
            "t_max": float(args.t_max),
            "n_t": int(args.n_t),
            "primes_global": str(args.primes_global),
        }
        rows.append(row)
        print(
            f"u={u:g} v={v:g}  median_dev={row['median_max_abs_eig_minus1']:.3e}  "
            f"rel_tpl={row['rel_rmse_satake_template']:.3e}  rel_pp={row['rel_rmse_pp']:.3e}"
        )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["u"]).to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
