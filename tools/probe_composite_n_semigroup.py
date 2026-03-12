import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import simulator module from sibling file (tools/ is not a package).
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
import six_by_six_prime_tower_sim as sim  # type: ignore


def _prime_factorization(n: int) -> dict[int, int]:
    n0 = int(n)
    if n0 <= 0:
        raise ValueError("n must be positive")
    out: dict[int, int] = {}
    m = n0

    e = 0
    while m % 2 == 0:
        m //= 2
        e += 1
    if e:
        out[2] = e

    p = 3
    while p * p <= m:
        e = 0
        while m % p == 0:
            m //= p
            e += 1
        if e:
            out[int(p)] = int(e)
        p += 2

    if m > 1:
        out[int(m)] = int(out.get(int(m), 0) + 1)

    return out


def _divisors_from_factorization(f: dict[int, int]) -> list[int]:
    divs = [1]
    for p, e in f.items():
        cur = []
        pe = 1
        for _ in range(e + 1):
            for d in divs:
                cur.append(int(d * pe))
            pe *= int(p)
        divs = cur
    return sorted(set(int(d) for d in divs))


def _mobius_mu(n: int) -> int:
    f = _prime_factorization(int(n))
    for e in f.values():
        if e >= 2:
            return 0
    return -1 if (len(f) % 2 == 1) else 1


def _packet_observable_from_SL(pkt_S: np.ndarray, pkt_Lam: np.ndarray, *, mode: str) -> float:
    mode = str(mode).strip().lower()
    if mode == "devs_pm_i":
        S = np.asarray(pkt_S, dtype=np.complex128)
        I2 = np.eye(2, dtype=np.complex128)
        dev = float(min(np.linalg.norm(S - I2, ord="fro"), np.linalg.norm(S + I2, ord="fro")))
        return float(dev)
    if mode == "frolam":
        Lam = np.asarray(pkt_Lam, dtype=np.complex128)
        return float(np.linalg.norm(Lam, ord="fro"))
    if mode == "trlam":
        Lam = np.asarray(pkt_Lam, dtype=np.complex128)
        return float(np.real(np.trace(Lam)))
    if mode in {"logdet_i_minus_s", "logdet(i-s)", "logdet(i_minus_s)"}:
        S = np.asarray(pkt_S, dtype=np.complex128)
        I2 = np.eye(2, dtype=np.complex128)
        det = complex(np.linalg.det(I2 - S))
        return float(math.log(max(1e-300, abs(det))))
    if mode in {"logdetlam", "logdet_lam", "logdet(lambda)", "logdet_lambda"}:
        Lam = np.asarray(pkt_Lam, dtype=np.complex128)
        det = complex(np.linalg.det(Lam))
        return float(math.log(max(1e-300, abs(det))))
    raise ValueError("obs_mode must be one of: devS_pm_I, froLam, trLam, logdet_I_minus_S, logdetLam")


@dataclass(frozen=True)
class Row:
    n: int
    V: float
    P: float


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Probe composite-n observables by building a semigroup-represented X_n in GL2: X_n = prod_p X_{p,1}^{v_p(n)} (in increasing p). "
            "Then run the usual Cayley->bulk->Schur->scattering pipeline to get an observable V(n), and compute Dirichlet primitive P(n)=sum_{d|n} mu(d)V(n/d).\n\n"
            "This is an MVP composite-n composition that is richer than log-additive union-of-primes, but still principled (semigroup homomorphism at the generator level)."
        )
    )

    ap.add_argument("--ns", default="2,3,5,6,10,15,30", help="Comma-list of n to evaluate")
    ap.add_argument("--u", type=float, default=0.2)
    ap.add_argument("--boundary", default="0,3")
    ap.add_argument("--schur_sign", choices=["-", "+"], default="+")

    ap.add_argument("--sharp", choices=["transpose", "conj_transpose"], default="conj_transpose")
    ap.add_argument("--X_mode", choices=["raw", "det1"], default="det1")
    ap.add_argument("--X_gamma", type=float, default=1.0)
    ap.add_argument(
        "--p_mode",
        choices=["p", "logp", "p1_over_p", "p_over_p1", "invp", "p_minus1_over_p"],
        default="p",
    )
    ap.add_argument("--scattering", choices=["lambda_pm_i", "i_pm_lambda"], default="i_pm_lambda")

    ap.add_argument(
        "--obs_mode",
        choices=["devS_pm_I", "froLam", "trLam", "logdet_I_minus_S", "logdetLam"],
        default="devS_pm_I",
    )
    ap.add_argument("--out_csv", default="", help="Optional output CSV")

    args = ap.parse_args()

    ns = [int(x.strip()) for x in str(args.ns).split(",") if x.strip()]
    if not ns:
        raise SystemExit("--ns must be non-empty")

    boundary_parts = sim._parse_int_list(str(args.boundary))
    if len(boundary_parts) != 2:
        raise SystemExit("--boundary must be i,j")
    boundary = (int(boundary_parts[0]), int(boundary_parts[1]))

    u = float(args.u)
    v = -float(u)

    # Cache prime generators X_{p,1}.
    X_cache: dict[int, np.ndarray] = {}

    def X_p1(p: int) -> np.ndarray:
        p = int(p)
        if p not in X_cache:
            X, _, _ = sim._local_blocks_for_prime_power(
                int(p),
                1,
                sharp_mode=str(args.sharp),
                x_mode=str(args.X_mode),
                x_gamma=float(args.X_gamma),
                x_shear=float(u),
                x_lower=float(v),
                p_mode=str(args.p_mode),
            )
            X_cache[p] = np.asarray(X, dtype=np.complex128)
        return X_cache[p]

    def build_X_n(n: int) -> np.ndarray:
        f = _prime_factorization(int(n))
        Xn = np.eye(2, dtype=np.complex128)
        for p in sorted(f.keys()):
            e = int(f[p])
            Xn = (Xn @ np.linalg.matrix_power(X_p1(p), e)).astype(np.complex128)
        return Xn

    def V_of_n(n: int) -> float:
        Xn = build_X_n(int(n))
        A = sim._cayley(Xn)
        Ash = sim._symplectic_partner(A, mode=str(args.sharp))
        B = sim._bulk_B_from_A(A, Ash)
        Lam = sim._schur_complement_Lambda(B, boundary=boundary, sign=str(args.schur_sign))
        S = sim._scattering_from_Lambda(Lam, mode=str(args.scattering))
        return float(_packet_observable_from_SL(S, Lam, mode=str(args.obs_mode)))

    V_map: dict[int, float] = {}

    def P_of_n(n: int) -> float:
        f = _prime_factorization(int(n))
        divs = _divisors_from_factorization(f)
        acc = 0.0
        for d in divs:
            mu = _mobius_mu(int(d))
            if mu == 0:
                continue
            acc += float(mu) * float(V_map[int(n // d)])
        return float(acc)

    # Evaluate V on all divisors needed for Dirichlet inversion.
    all_needed = set()
    for n in ns:
        all_needed.add(int(n))
        for d in _divisors_from_factorization(_prime_factorization(int(n))):
            all_needed.add(int(d))
    all_needed.add(1)

    # Define V(1)=0 in this MVP (identity element has no packet content).
    V_map[1] = 0.0

    for n in sorted(all_needed):
        if n == 1:
            continue
        V_map[n] = float(V_of_n(n))

    rows: list[Row] = []
    for n in sorted(set(int(n) for n in ns)):
        rows.append(Row(n=int(n), V=float(V_map[n]), P=float(P_of_n(n))))

    out = pd.DataFrame([r.__dict__ for r in rows]).sort_values(["n"])

    # Coprime multiplicativity diagnostics for P.
    P_map = {int(r.n): float(r.P) for r in rows}

    rel_mult_errs: list[float] = []
    logadd_errs: list[float] = []

    def safe_log_abs(x: float, floor: float = 1e-12) -> float:
        return float(math.log(max(float(floor), abs(float(x)))))

    for a in P_map.keys():
        for b in P_map.keys():
            if a <= 1 or b <= 1:
                continue
            if math.gcd(int(a), int(b)) != 1:
                continue
            ab = int(a * b)
            if ab not in P_map:
                continue
            Pa = float(P_map[a])
            Pb = float(P_map[b])
            Pab = float(P_map[ab])
            denom = max(1e-12, abs(Pa) * abs(Pb))
            rel_mult_errs.append(float(abs(Pab - Pa * Pb) / denom))
            logadd_errs.append(float(abs(safe_log_abs(Pab) - (safe_log_abs(Pa) + safe_log_abs(Pb)))))

    def fmt(x: float) -> str:
        if not math.isfinite(float(x)):
            return "nan"
        return f"{float(x):.4g}"

    print("composite-n semigroup probe")
    print(f"  u={u:g} boundary={boundary} sign={args.schur_sign} p_mode={args.p_mode} X_mode={args.X_mode} gamma={float(args.X_gamma):g}")
    print(f"  obs_mode={args.obs_mode} scattering={args.scattering}")
    print(f"  ns={','.join(str(n) for n in sorted(set(ns)))}")

    if rel_mult_errs:
        arr = np.asarray(rel_mult_errs, dtype=float)
        print("  P multiplicativity relerr median:", fmt(float(np.median(arr))), "max:", fmt(float(np.max(arr))), "n:", int(arr.size))
    else:
        print("  P multiplicativity relerr: (no coprime pairs ab present)")

    if logadd_errs:
        arr = np.asarray(logadd_errs, dtype=float)
        print("  log|P| additivity abs err median:", fmt(float(np.median(arr))), "max:", fmt(float(np.max(arr))), "n:", int(arr.size))
    else:
        print("  log|P| additivity abs err: (no coprime pairs ab present)")

    if str(args.out_csv).strip():
        pth = Path(str(args.out_csv))
        pth.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(str(pth), index=False)
        print(f"wrote {pth}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
