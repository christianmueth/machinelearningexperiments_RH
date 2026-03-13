import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.mangoldt_logderivative_probe import _central_dlog_dt, _design_matrix, _matched_filter_coeffs, _ridge_solve  # noqa: E402


def primes_upto(n: int):
    if n < 2:
        return []
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, int(n**0.5) + 1):
        if sieve[p]:
            start = p * p
            sieve[start : n + 1 : p] = b"\x00" * (((n - start) // p) + 1)
    return [i for i in range(2, n + 1) if sieve[i]]


def von_mangoldt(n: int, primes):
    for p in primes:
        if p * p > n:
            break
        if n % p == 0:
            m = n
            while m % p == 0:
                m //= p
            return math.log(p) if m == 1 else 0.0
    return math.log(n)


def main():
    df = pd.read_csv("tools/_smoke_mangoldt_qtrack.csv")
    t = df["t"].to_numpy(float)
    D = (df["q_track_re"].to_numpy(float) + 1j * df["q_track_im"].to_numpy(float)).astype(np.complex128)

    t_mid, dlog = _central_dlog_dt(t, D, 1e-30)
    g_est = 1j * dlog

    sigma = float(df["sigma"].iloc[0])
    n_max = 128

    pr = primes_upto(n_max)
    logs = np.log(np.arange(2, n_max + 1, dtype=float))
    lam = np.array([von_mangoldt(n, pr) for n in range(2, n_max + 1)], dtype=float)
    amp = np.exp(-sigma * logs)
    phase = np.exp(-1j * t_mid.reshape(-1, 1) * logs.reshape(1, -1))
    g_true = phase @ (lam * amp)

    rel_rmse = float(np.sqrt(np.mean(np.abs(g_est - g_true) ** 2)) / (np.sqrt(np.mean(np.abs(g_true) ** 2)) + 1e-300))

    print("rel_rmse(g_est vs g_true)=", rel_rmse)
    print("rms(g_true)=", float(np.sqrt(np.mean(np.abs(g_true) ** 2))))
    print("rms(g_est)=", float(np.sqrt(np.mean(np.abs(g_est) ** 2))))
    print("max|D|=", float(np.max(np.abs(D))))
    print("min|D|=", float(np.min(np.abs(D))))

    n_list = list(range(2, n_max + 1))

    # Matched filter coefficients (intended to approximate Lambda(n)).
    mf = _matched_filter_coeffs(t_mid, g_est, sigma=sigma, n_list=n_list)
    mf_vec = np.array([mf[n] for n in n_list], dtype=np.complex128)
    lam_vec = lam.astype(np.complex128)
    rel_mf = float(np.sqrt(np.sum(np.abs(mf_vec - lam_vec) ** 2)) / (np.sqrt(np.sum(np.abs(lam_vec) ** 2)) + 1e-300))
    print("rel_l2_err(matched_filter vs Lambda)=", rel_mf)

    # Least squares coefficients on the full basis.
    X = _design_matrix(t_mid, sigma=sigma, n_list=n_list)
    a = _ridge_solve(X, g_est, ridge=0.0)
    rel_ls = float(np.sqrt(np.sum(np.abs(a - lam_vec) ** 2)) / (np.sqrt(np.sum(np.abs(lam_vec) ** 2)) + 1e-300))
    print("rel_l2_err(least_squares vs Lambda)=", rel_ls)

    # Show a few sample coefficients
    for n in [2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 25, 27, 32, 64, 81, 125]:
        if n > n_max:
            continue
        idx = n - 2
        print(f"n={n:3d}  Lambda={lam_vec[idx].real:8.5f}  mf={mf_vec[idx]: .5f}  ls={a[idx]: .5f}")


if __name__ == "__main__":
    main()
