import argparse
import itertools
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


def _parse_float_csv(s: str) -> list[float]:
    out: list[float] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _parse_prime_sets(s: str) -> list[list[int]]:
    # Format: "2,3,5;2,3,5,7" (semicolon separates sets)
    sets: list[list[int]] = []
    for chunk in str(s).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        ps: list[int] = []
        for part in chunk.split(","):
            part = part.strip()
            if not part:
                continue
            ps.append(int(part))
        ps = sorted(set(int(p) for p in ps))
        if len(ps) < 2:
            raise ValueError("each prime set must have >=2 primes")
        sets.append(ps)
    if not sets:
        raise ValueError("--prime_sets must be non-empty")
    return sets


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


def _safe_log_abs(x: float, floor: float = 1e-12) -> float:
    return float(math.log(max(float(floor), abs(float(x)))))


def _fmt(x: float) -> str:
    if not math.isfinite(float(x)):
        return "nan"
    return f"{float(x):.4g}"


def _build_squarefree_ns(primes: list[int], *, max_omega: int) -> list[int]:
    primes = sorted(set(int(p) for p in primes))
    ns: set[int] = set()
    for r in range(1, max_omega + 1):
        for comb in itertools.combinations(primes, r):
            n = 1
            for p in comb:
                n *= int(p)
            ns.add(int(n))
    # Ensure pairwise products exist (max_omega>=2 typically).
    return sorted(ns)


@dataclass(frozen=True)
class Row:
    prime_set: str
    n_primes: int
    max_omega: int
    u: float
    gated_out_p: int
    gated_out_invp: int
    logadd_med_p: float
    logadd_med_invp: float
    p_wins: int
    invp_wins: int


@dataclass(frozen=True)
class SummaryRow:
    prime_set: str
    n_primes: int
    max_omega: int
    min_gen_devI_median: float
    n_u: int
    n_u_ungated: int
    p_wins: int
    invp_wins: int
    ties: int
    median_margin_p_minus_invp: float


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Prime-set sensitivity check for the screened composite arithmetic window test.\n\n"
            "For each prime set, constructs a squarefree composite set n consisting of products of up to --max_omega distinct primes, "
            "then compares p_mode=p vs p_mode=invp on the composite log-additivity score (Dirichlet primitive of logdet(I-S)).\n\n"
            "A generator nontriviality gate is applied using median ||X_{p,1}-I||_F across the same prime set."
        )
    )

    ap.add_argument(
        "--prime_sets",
        default="2,3,5,7,11,13,17;2,3,5,7,11,13,17,19;2,3,5,7,11,13,17,19,23",
        help="Semicolon-separated prime sets, each comma-separated, e.g. '2,3,5;2,3,5,7'",
    )
    ap.add_argument("--max_omega", type=int, default=3, help="Max number of distinct primes per n in the composite set")

    ap.add_argument("--us", default="0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24")
    ap.add_argument("--min_gen_devI_median", type=float, default=0.8)
    ap.add_argument("--min_abs_P", type=float, default=0.0)

    ap.add_argument("--boundary", default="0,3")
    ap.add_argument("--schur_sign", choices=["-", "+"], default="+")
    ap.add_argument("--scattering", choices=["lambda_pm_i", "i_pm_lambda"], default="i_pm_lambda")

    ap.add_argument("--sharp", choices=["transpose", "conj_transpose"], default="conj_transpose")
    ap.add_argument("--X_mode", choices=["raw", "det1"], default="det1")
    ap.add_argument("--X_gamma", type=float, default=1.0)

    ap.add_argument("--out_csv", default="out/prime_set_sensitivity_rows.csv")
    ap.add_argument("--out_summary_csv", default="out/prime_set_sensitivity_summary.csv")
    args = ap.parse_args()

    prime_sets = _parse_prime_sets(str(args.prime_sets))
    us = _parse_float_csv(str(args.us))
    if not us:
        raise SystemExit("--us must be non-empty")

    if int(args.max_omega) < 1:
        raise SystemExit("--max_omega must be >=1")

    boundary_parts = sim._parse_int_list(str(args.boundary))
    if len(boundary_parts) != 2:
        raise SystemExit("--boundary must be i,j")
    boundary = (int(boundary_parts[0]), int(boundary_parts[1]))

    def eval_logadd(*, u: float, p_mode: str, primes_for_gate: list[int], ns: list[int]) -> tuple[float, int]:
        u = float(u)
        v = -float(u)

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
                    p_mode=str(p_mode),
                )
                X_cache[p] = np.asarray(X, dtype=np.complex128)
            return X_cache[p]

        I2 = np.eye(2, dtype=np.complex128)
        devs = []
        for p in primes_for_gate:
            Xp = X_p1(int(p))
            devs.append(float(np.linalg.norm(Xp - I2, ord="fro")))
        devs_arr = np.asarray(devs, dtype=float)
        devs_arr = devs_arr[np.isfinite(devs_arr)]
        dev_med = float(np.median(devs_arr)) if devs_arr.size else float("nan")

        if (not math.isfinite(dev_med)) or dev_med < float(args.min_gen_devI_median):
            return float("nan"), 1

        # Dirichlet primitive of logdet(I-S).
        all_needed = set([1])
        for n in ns:
            all_needed.add(int(n))
            for d in _divisors_from_factorization(_prime_factorization(int(n))):
                all_needed.add(int(d))

        def build_X_n(n: int) -> np.ndarray:
            f = _prime_factorization(int(n))
            Xn = np.eye(2, dtype=np.complex128)
            for p in sorted(f.keys()):
                e = int(f[p])
                Xn = (Xn @ np.linalg.matrix_power(X_p1(int(p)), e)).astype(np.complex128)
            return Xn

        def V_of_n(n: int) -> float:
            Xn = build_X_n(int(n))
            A = sim._cayley(Xn)
            Ash = sim._symplectic_partner(A, mode=str(args.sharp))
            B = sim._bulk_B_from_A(A, Ash)
            Lam = sim._schur_complement_Lambda(B, boundary=boundary, sign=str(args.schur_sign))
            S = sim._scattering_from_Lambda(Lam, mode=str(args.scattering))
            det = complex(np.linalg.det(I2 - np.asarray(S, dtype=np.complex128)))
            return float(math.log(max(1e-300, abs(det))))

        V_map: dict[int, float] = {1: 0.0}
        for n in sorted(set(int(x) for x in all_needed)):
            if n == 1:
                continue
            V_map[n] = float(V_of_n(int(n)))

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

        P_map = {int(n): float(P_of_n(int(n))) for n in ns}
        thrP = float(args.min_abs_P)

        errs: list[float] = []
        for a in ns:
            for b in ns:
                if a <= 1 or b <= 1:
                    continue
                if math.gcd(int(a), int(b)) != 1:
                    continue
                ab = int(a * b)
                if ab not in P_map:
                    continue
                Pa = float(P_map[int(a)])
                Pb = float(P_map[int(b)])
                Pab = float(P_map[int(ab)])
                if thrP > 0 and (abs(Pa) < thrP or abs(Pb) < thrP or abs(Pab) < thrP):
                    continue
                errs.append(float(abs(_safe_log_abs(Pab) - (_safe_log_abs(Pa) + _safe_log_abs(Pb)))))

        if not errs:
            return float("nan"), 0
        arr = np.asarray(errs, dtype=float)
        return float(np.median(arr)), 0

    rows: list[Row] = []
    summaries: list[SummaryRow] = []

    for primes in prime_sets:
        prime_set_label = ",".join(str(p) for p in primes)
        ns = _build_squarefree_ns(primes, max_omega=int(args.max_omega))

        margins = []
        n_ungated = 0
        p_wins = 0
        invp_wins = 0
        ties = 0

        for u in us:
            lp, gp = eval_logadd(u=float(u), p_mode="p", primes_for_gate=primes, ns=ns)
            li, gi = eval_logadd(u=float(u), p_mode="invp", primes_for_gate=primes, ns=ns)

            # record per-u row
            pw = 0
            iw = 0
            if math.isfinite(lp) and math.isfinite(li) and gp == 0 and gi == 0:
                n_ungated += 1
                margins.append(float(lp - li))
                if lp < li:
                    pw = 1
                    p_wins += 1
                elif li < lp:
                    iw = 1
                    invp_wins += 1
                else:
                    ties += 1
            rows.append(
                Row(
                    prime_set=prime_set_label,
                    n_primes=int(len(primes)),
                    max_omega=int(args.max_omega),
                    u=float(u),
                    gated_out_p=int(gp),
                    gated_out_invp=int(gi),
                    logadd_med_p=float(lp),
                    logadd_med_invp=float(li),
                    p_wins=int(pw),
                    invp_wins=int(iw),
                )
            )

            print(
                "prime_set=", prime_set_label,
                "u=", _fmt(float(u)),
                "p:", _fmt(float(lp)), "gated" if gp else "",
                "| invp:", _fmt(float(li)), "gated" if gi else "",
            )

        med_margin = float(np.median(np.asarray(margins, dtype=float))) if margins else float("nan")
        summaries.append(
            SummaryRow(
                prime_set=prime_set_label,
                n_primes=int(len(primes)),
                max_omega=int(args.max_omega),
                min_gen_devI_median=float(args.min_gen_devI_median),
                n_u=int(len(us)),
                n_u_ungated=int(n_ungated),
                p_wins=int(p_wins),
                invp_wins=int(invp_wins),
                ties=int(ties),
                median_margin_p_minus_invp=float(med_margin),
            )
        )

        print("summary for prime_set", prime_set_label)
        print("  ungated u:", n_ungated, "/", int(len(us)))
        print("  p_wins:", p_wins, "invp_wins:", invp_wins, "ties:", ties)
        print("  median(p_minus_invp):", _fmt(float(med_margin)))

    out_path = Path(str(args.out_csv))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.__dict__ for r in rows]).to_csv(str(out_path), index=False)
    print(f"wrote {out_path}")

    out_sum = Path(str(args.out_summary_csv))
    out_sum.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.__dict__ for r in summaries]).to_csv(str(out_sum), index=False)
    print(f"wrote {out_sum}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
