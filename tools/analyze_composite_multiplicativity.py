import argparse
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _prime_factorization(n: int) -> dict[int, int]:
    """Return prime factorization n = prod p^e as {p: e}."""

    n0 = int(n)
    if n0 <= 0:
        raise ValueError("n must be positive")
    out: dict[int, int] = {}
    m = n0

    # Factor 2
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
                cur.append(d * pe)
            pe *= p
        divs = cur
    return sorted(set(int(d) for d in divs))


def _mobius_mu(n: int) -> int:
    f = _prime_factorization(int(n))
    for e in f.values():
        if e >= 2:
            return 0
    # squarefree
    return -1 if (len(f) % 2 == 1) else 1


def _safe_log_abs(x: float, *, floor: float = 1e-300) -> float:
    return float(math.log(max(float(floor), abs(float(x)))))


@dataclass(frozen=True)
class CompositeRow:
    n: int
    V: float
    P: float


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "MVP composite-n arithmetic probe using a log-additive composition rule.\n\n"
            "Given per-prime per-k observable data (from tools/fe_defect_perturbation_u0.py --k_obs_emit_per_prime 1), "
            "define an additive composite function V(n) by summing the prime-power contributions for each exact factor p^e || n. "
            "Then compute the Dirichlet primitive P(n) via Möbius inversion over divisors of n: P(n)=sum_{d|n} mu(d)V(n/d).\n\n"
            "This enables first-pass tests on squarefree composites (n=pq) of coprime multiplicativity / log-additivity."
        )
    )

    ap.add_argument("--kobs_csv", required=True, help="CSV from fe_defect_perturbation_u0.py --k_obs_out_csv ...")
    ap.add_argument("--label", default="u_0.2", help="Which label to use from kobs CSV (e.g. u_0.2)")
    ap.add_argument("--u", type=float, default=0.2, help="Target u value")
    ap.add_argument("--u_tol", type=float, default=1e-9, help="Tolerance for matching u")

    ap.add_argument(
        "--value_col",
        choices=["F_k_abs", "F_k_re", "F_k_im"],
        default="F_k_abs",
        help=(
            "Which per-prime per-k value to treat as local prime-power contribution v(p,k). "
            "(Use abs for robustness.)"
        ),
    )
    ap.add_argument(
        "--k_max",
        type=int,
        default=7,
        help="Max k to use (requires that the per-prime table contains these k values)",
    )

    ap.add_argument(
        "--ns",
        default="2,3,5,6,10,15,30",
        help="Comma-list of n values to evaluate (suggest squarefree composites first)",
    )

    ap.add_argument("--out_csv", default="", help="Optional output CSV for (n,V(n),P(n))")

    args = ap.parse_args()

    df = pd.read_csv(str(args.kobs_csv))
    if "k_obs_scope" not in df.columns:
        raise SystemExit("kobs_csv missing k_obs_scope; re-run with updated FE tool")

    df = df[df["k_obs_scope"] == "per_prime"].copy()
    df = df[df["label"].astype(str) == str(args.label)].copy()
    df = df[np.abs(df["u"].astype(float) - float(args.u)) <= float(args.u_tol)].copy()
    if df.empty:
        raise SystemExit("No matching per_prime rows for given --label/--u")

    k_max = int(args.k_max)
    if k_max <= 0:
        raise SystemExit("--k_max must be positive")

    # Build lookup v[p][k] from the per-prime table.
    v: dict[int, dict[int, float]] = {}
    for r in df.itertuples(index=False):
        p = int(getattr(r, "p"))
        k = int(getattr(r, "k"))
        if k < 1 or k > k_max:
            continue
        val = float(getattr(r, str(args.value_col)))
        v.setdefault(p, {})[k] = val

    # Verify we have at least v[p][1] for all primes.
    primes = sorted(v.keys())
    if not primes:
        raise SystemExit("No primes found in per_prime rows")

    def V_of_n(n: int) -> float:
        f = _prime_factorization(int(n))
        acc = 0.0
        for p, e in f.items():
            if p not in v:
                raise KeyError(f"prime {p} not present in kobs table")
            if e not in v[p]:
                raise KeyError(f"missing k={e} for prime {p} in kobs table")
            acc += float(v[p][e])
        return float(acc)

    def P_of_n(n: int) -> float:
        f = _prime_factorization(int(n))
        divs = _divisors_from_factorization(f)
        acc = 0.0
        for d in divs:
            mu = _mobius_mu(int(d))
            if mu == 0:
                continue
            acc += float(mu) * float(V_of_n(int(n // d)))
        return float(acc)

    ns = [int(x.strip()) for x in str(args.ns).split(",") if x.strip()]
    if not ns:
        raise SystemExit("--ns must be non-empty")

    rows: list[CompositeRow] = []
    for n in ns:
        rows.append(CompositeRow(n=int(n), V=float(V_of_n(n)), P=float(P_of_n(n))))

    out = pd.DataFrame([r.__dict__ for r in rows]).sort_values(["n"])

    # Coprime multiplicativity diagnostics on provided ns.
    P_map = {int(r.n): float(r.P) for r in rows}

    mult_errs: list[float] = []
    logadd_errs: list[float] = []

    for a in ns:
        for b in ns:
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
            mult_errs.append(float(abs(Pab - Pa * Pb) / denom))

            la = _safe_log_abs(Pa)
            lb = _safe_log_abs(Pb)
            lab = _safe_log_abs(Pab)
            logadd_errs.append(float(abs(lab - (la + lb))))

    def fmt(x: float) -> str:
        if not math.isfinite(float(x)):
            return "nan"
        return f"{float(x):.4g}"

    print("composite-n multiplicativity probe")
    print(f"  kobs_csv: {args.kobs_csv}")
    print(f"  label/u: {args.label} @ u={float(args.u):g}")
    print(f"  value_col: {args.value_col}")
    print(f"  k_max: {k_max}")
    print(f"  ns: {','.join(str(n) for n in ns)}")

    if mult_errs:
        arr = np.asarray(mult_errs, dtype=float)
        print("  multiplicativity relerr (|P(ab)-P(a)P(b)|/(|P(a)P(b)|)):")
        print("    median:", fmt(float(np.median(arr))), " max:", fmt(float(np.max(arr))), " n:", int(arr.size))
    else:
        print("  multiplicativity relerr: (no coprime pairs ab present in --ns)")

    if logadd_errs:
        arr = np.asarray(logadd_errs, dtype=float)
        print("  log-additivity abs err (|log|P(ab)|-(log|P(a)|+log|P(b)|)|):")
        print("    median:", fmt(float(np.median(arr))), " max:", fmt(float(np.max(arr))), " n:", int(arr.size))
    else:
        print("  log-additivity abs err: (no coprime pairs ab present in --ns)")

    if str(args.out_csv).strip():
        out.to_csv(str(args.out_csv), index=False)
        print(f"wrote {args.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
