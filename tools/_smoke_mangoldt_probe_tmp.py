import math
from pathlib import Path

import numpy as np
import pandas as pd


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
    out = Path("tools/_smoke_mangoldt_qtrack.csv")
    sigma = 2.0
    n_max = 128
    t = np.linspace(10.0, 30.0, 401)

    primes = primes_upto(n_max)
    # g(t) = sum_{n<=N} Lambda(n) n^{-sigma} e^{-i t log n}
    logs = np.log(np.arange(2, n_max + 1, dtype=float))
    lam = np.array([von_mangoldt(n, primes) for n in range(2, n_max + 1)], dtype=float)
    amp = np.exp(-sigma * logs)

    phase = np.exp(-1j * t.reshape(-1, 1) * logs.reshape(1, -1))
    g = phase @ (lam * amp)

    # d/dt log D = -i g
    dlog = (-1j * g).astype(np.complex128)

    logD = np.zeros_like(dlog)
    for k in range(1, t.size):
        dt = float(t[k] - t[k - 1])
        logD[k] = logD[k - 1] + 0.5 * (dlog[k] + dlog[k - 1]) * dt

    D = np.exp(logD)

    df = pd.DataFrame(
        {
            "sigma": sigma,
            "t": t,
            "side": "right",
            "q_track_re": np.real(D).astype(float),
            "q_track_im": np.imag(D).astype(float),
            "Tin_inv_norm1_est_N2": 1.0,
            "sep2_N2": 1.0,
            "lam_match_cost": 0.0,
        }
    )
    df.to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
