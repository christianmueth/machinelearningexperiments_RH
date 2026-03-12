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

from src.arithmetic.mobius import mobius_mu, divisors, mobius_invert_divisor_sum


def _parse_int_csv(s: str) -> list[int]:
    out: list[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _fmt(x: float) -> str:
    if not math.isfinite(float(x)):
        return "nan"
    return f"{float(x):.6g}"


def _safe_log_abs(x: float, *, floor: float) -> float:
    return float(math.log(max(float(floor), abs(float(x)))))


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


def _euler_transform_from_ghost(g: list[float], *, k_max: int) -> list[float]:
    """Compute A_0..A_k_max from ghost sequence g_1..g_k_max.

    L_loc(u) = exp( sum_{m>=1} g_m/m * u^m ) = sum_{k>=0} A_k u^k

    Recursion: k A_k = sum_{j=1..k} g_j A_{k-j}, A_0 = 1.

    g must be indexed 0..k_max with g[0] ignored.
    """

    if len(g) < k_max + 1:
        raise ValueError("g must have length >= k_max+1")

    A = [0.0] * (k_max + 1)
    A[0] = 1.0
    for k in range(1, k_max + 1):
        acc = 0.0
        for j in range(1, k + 1):
            acc += float(g[j]) * float(A[k - j])
        A[k] = float(acc) / float(k)
    return A


@dataclass(frozen=True)
class NRow:
    n: int
    cls: str
    omega: int
    emp_A: float
    lift_a: float
    log_abs_emp: float
    log_abs_lift: float
    log_ratio_abs: float


@dataclass(frozen=True)
class SummaryRow:
    p_mode: str
    ghost_mode: str
    u: float
    gated_out: int
    gen_devI_median: float

    # dyadic ghost extraction
    dyadic_k_max: int
    ghost_t: float
    g1: float
    g2: float
    g3: float

    # lift coefficients
    A1: float
    A2: float
    A3: float

    # support ratios on empirical primitive
    C_sf_emp_median: float
    C_pp_emp_median: float

    # support ratios on lifted model
    C_sf_lift_median: float
    C_pp_lift_median: float

    # fit-to-empirical error by class (median |log ratio|)
    med_abs_logratio_primes: float
    med_abs_logratio_primepowers: float
    med_abs_logratio_squarefree_pairs: float
    med_abs_logratio_mixed: float


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Inverse ghost-map / Euler-transform lift probe.\n\n"
            "Frozen setup: X-semigroup composition, observable V(n)=log|det(I-S_n)|.\n"
            "Empirical arithmetic object: A_emp(n)=Dirichlet primitive of V(n).\n\n"
            "Lift: extract a dyadic ghost sequence g_m from the dyadic corridor n=2^m:\n"
            "  F(m)=V(2^m), then g = Möbius-invert(F) on the divisor poset of m (period primitive).\n"
            "Then build L_loc(u)=exp(sum g_m/m u^m) and coefficients A_k by Euler transform recursion:\n"
            "  k A_k = sum_{j=1..k} g_j A_{k-j}, A_0=1.\n"
            "Declare lifted prime-power coefficients a_{p^k}=A_k (independent of p) and extend multiplicatively.\n\n"
            "This script compares lifted a_n to empirical A_emp(n) over: primes, prime powers, squarefree pq, and mixed p^a q^b." 
        )
    )

    ap.add_argument("--ps", default="2,3,5,7,11,13,17", help="Comma-list of primes")
    ap.add_argument("--u", type=float, default=0.2)
    ap.add_argument("--p_modes", default="p,invp", help="Comma-list, usually 'p,invp'")

    ap.add_argument("--min_gen_devI_median", type=float, default=0.8)
    ap.add_argument("--min_abs", type=float, default=1e-12, help="Floor for log|.| and skipping tiny values")

    ap.add_argument("--dyadic_k_max", type=int, default=12, help="Max m for dyadic corridor n=2^m")
    ap.add_argument("--max_power", type=int, default=3, help="Max exponent used for prime powers and mixed terms")

    ap.add_argument(
        "--ghost_mode",
        choices=["mobius_primitive", "aggregate_as_ghost", "delta_as_ghost"],
        default="mobius_primitive",
        help=(
            "How to turn the dyadic corridor sequence F(m)=V(2^m) into ghost coordinates g_m. "
            "mobius_primitive: g = Mobius-invert(F) on m (assumes divisor-sum aggregate). "
            "aggregate_as_ghost: g_m = F(m). "
            "delta_as_ghost: g_m = F(m)-F(m-1)."
        ),
    )

    ap.add_argument(
        "--ghost_t",
        type=float,
        default=1.0,
        help="Scale ghost sequence as g_m <- t^m g_m (equivalently u->t u in L_loc).",
    )
    ap.add_argument(
        "--auto_scale_A1_to_emp_prime_median",
        type=int,
        default=0,
        help="If 1, overrides --ghost_t by choosing t so that |A1| matches median_p |A_emp(p)|.",
    )

    ap.add_argument("--boundary", default="0,3")
    ap.add_argument("--schur_sign", choices=["-", "+"], default="+")
    ap.add_argument("--scattering", choices=["lambda_pm_i", "i_pm_lambda"], default="i_pm_lambda")

    ap.add_argument("--sharp", choices=["transpose", "conj_transpose"], default="conj_transpose")
    ap.add_argument("--X_mode", choices=["raw", "det1"], default="det1")
    ap.add_argument("--X_gamma", type=float, default=1.0)

    ap.add_argument("--out_rows_csv", default="out/ghostmap_euler_lift_rows.csv")
    ap.add_argument("--out_summary_csv", default="out/ghostmap_euler_lift_summary.csv")

    args = ap.parse_args()

    ps = sorted(set(int(p) for p in _parse_int_csv(str(args.ps))))
    if len(ps) < 2:
        raise SystemExit("--ps must contain at least two primes")

    u = float(args.u)
    v = -float(u)

    boundary_parts = sim._parse_int_list(str(args.boundary))
    if len(boundary_parts) != 2:
        raise SystemExit("--boundary must be i,j")
    boundary = (int(boundary_parts[0]), int(boundary_parts[1]))

    p_modes = [m.strip() for m in str(args.p_modes).split(",") if m.strip()]
    if not p_modes:
        raise SystemExit("--p_modes must be non-empty")

    dyadic_k_max = int(args.dyadic_k_max)
    if dyadic_k_max < 3:
        raise SystemExit("--dyadic_k_max must be >=3")

    max_power = int(args.max_power)
    if max_power < 1:
        raise SystemExit("--max_power must be >=1")

    min_abs = max(0.0, float(args.min_abs))

    I2 = np.eye(2, dtype=np.complex128)

    def summarize_median(xs: list[float]) -> float:
        arr = np.asarray(xs, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("nan")
        return float(np.median(arr))

    def med_abs_logratio(rows: list[NRow], cls: str) -> float:
        xs = [abs(r.log_ratio_abs) for r in rows if r.cls == cls and math.isfinite(r.log_ratio_abs)]
        return summarize_median(xs)

    all_rows: list[NRow] = []
    summaries: list[SummaryRow] = []

    for p_mode in p_modes:
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
                    p_mode=str(p_mode),
                )
                X_cache[p] = np.asarray(X, dtype=np.complex128)
            return X_cache[p]

        devs = [float(np.linalg.norm(X_p1(int(p)) - I2, ord="fro")) for p in ps]
        devs_arr = np.asarray(devs, dtype=float)
        devs_arr = devs_arr[np.isfinite(devs_arr)]
        gen_dev_med = float(np.median(devs_arr)) if devs_arr.size else float("nan")

        gated_out = 0
        if (not math.isfinite(gen_dev_med)) or gen_dev_med < float(args.min_gen_devI_median):
            gated_out = 1

        # Build semigroup X_n for arbitrary n.
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

        # Precompute empirical A_emp(p) magnitude target for optional normalization.
        # We use A_emp defined from Dirichlet primitive of V.
        # For primes p, A_emp(p) = V(p) - V(1) = V(p) since V(1)=0.
        # But we keep the general definition for clarity.
        primes_needed = set([1]) | set(int(p) for p in ps)
        V_pr: dict[int, float] = {1: 0.0}
        for n in sorted(primes_needed):
            if n == 1:
                continue
            V_pr[int(n)] = float(V_of_n(int(n)))

        def A_emp_prime(p: int) -> float:
            # Dirichlet primitive P(p) = mu(1)V(p)+mu(p)V(1)=V(p)
            return float(V_pr[int(p)])

        emp_prime_mags = [abs(float(A_emp_prime(int(p)))) for p in ps]
        emp_prime_mags = [x for x in emp_prime_mags if math.isfinite(x)]
        emp_prime_med_mag = float(np.median(np.asarray(emp_prime_mags, dtype=float))) if emp_prime_mags else float("nan")

        # Dyadic corridor F(m) = V(2^m) for m=1..dyadic_k_max.
        F = [0.0] * (dyadic_k_max + 1)
        for m in range(1, dyadic_k_max + 1):
            n = int(2**m)
            F[m] = float(V_of_n(int(n)))

        ghost_mode = str(args.ghost_mode)
        if ghost_mode == "mobius_primitive":
            inv = mobius_invert_divisor_sum(F)
            g = inv.f  # g[1..]
        elif ghost_mode == "aggregate_as_ghost":
            g = list(F)
        elif ghost_mode == "delta_as_ghost":
            g = [0.0] * (dyadic_k_max + 1)
            for m in range(1, dyadic_k_max + 1):
                g[m] = float(F[m] - F[m - 1])
        else:
            raise RuntimeError("unexpected ghost_mode")

        # Optional normalization: scale g_m <- t^m g_m.
        ghost_t = float(args.ghost_t)
        if int(args.auto_scale_A1_to_emp_prime_median) == 1:
            g1 = float(g[1]) if len(g) > 1 else float("nan")
            if math.isfinite(emp_prime_med_mag) and math.isfinite(g1):
                ghost_t = float(emp_prime_med_mag / max(1e-12, abs(g1)))

        if ghost_t != 1.0:
            for m in range(1, min(len(g), dyadic_k_max + 1)):
                g[m] = float((ghost_t ** int(m)) * float(g[m]))

        # Euler transform coefficients A_k.
        Acoeff = _euler_transform_from_ghost(g, k_max=max_power)

        # Empirical A_emp(n) = Dirichlet primitive of V(n).
        # Evaluate it on the minimal set of n we care about.
        ns: set[int] = set()
        # primes
        for p in ps:
            ns.add(int(p))
        # prime powers up to max_power
        for p in ps:
            pk = 1
            for k in range(1, max_power + 1):
                pk *= int(p)
                ns.add(int(pk))
        # squarefree pairs
        for p, q in itertools.combinations(ps, 2):
            ns.add(int(p * q))
        # mixed p^a q^b
        for p, q in itertools.combinations(ps, 2):
            for a in range(1, max_power + 1):
                for b in range(1, max_power + 1):
                    ns.add(int((p**a) * (q**b)))

        # ensure divisors needed
        all_needed: set[int] = set([1])
        for n in ns:
            for d in divisors(int(n)):
                all_needed.add(int(d))
            all_needed.add(int(n))

        V_map: dict[int, float] = {1: 0.0}
        for n in sorted(all_needed):
            if n == 1:
                continue
            V_map[n] = float(V_of_n(int(n)))

        def A_emp(n: int) -> float:
            acc = 0.0
            for d in divisors(int(n)):
                mu = int(mobius_mu(int(d)))
                if mu == 0:
                    continue
                acc += float(mu) * float(V_map[int(n // d)])
            return float(acc)

        def omega(n: int) -> int:
            f = _prime_factorization(int(n))
            return int(len([p for p, e in f.items() if e > 0]))

        def a_lift(n: int) -> float:
            # multiplicative lift from Acoeff: a_{p^k} = A_k (k>=1)
            f = _prime_factorization(int(n))
            out = 1.0
            for _p, e in f.items():
                e = int(e)
                if e > max_power:
                    return float("nan")
                out *= float(Acoeff[e])
            return float(out)

        # Build per-n rows and class-specific support ratios.
        rows_mode: list[NRow] = []
        C_sf_emp_terms: list[float] = []
        C_pp_emp_terms: list[float] = []
        C_sf_lift_terms: list[float] = []
        C_pp_lift_terms: list[float] = []

        for n in sorted(ns):
            f = _prime_factorization(int(n))
            om = omega(int(n))

            cls = "other"
            if om == 1:
                # prime power
                p = next(iter(f.keys()))
                e = int(f[p])
                if e == 1:
                    cls = "prime"
                else:
                    cls = "primepower"
            elif om == 2 and all(int(e) == 1 for e in f.values()):
                cls = "squarefree_pair"
            elif om == 2:
                cls = "mixed"

            emp = float(A_emp(int(n)))
            lift = float(a_lift(int(n)))

            if (min_abs > 0) and (abs(emp) < min_abs or abs(lift) < min_abs):
                # still record, but log fields become floor-based
                pass

            log_emp = _safe_log_abs(emp, floor=max(min_abs, 1e-300))
            log_lift = _safe_log_abs(lift, floor=max(min_abs, 1e-300))
            log_ratio = float(log_emp - log_lift)

            rows_mode.append(
                NRow(
                    n=int(n),
                    cls=str(cls),
                    omega=int(om),
                    emp_A=float(emp),
                    lift_a=float(lift),
                    log_abs_emp=float(log_emp),
                    log_abs_lift=float(log_lift),
                    log_ratio_abs=float(log_ratio),
                )
            )

        # Support/concentration diagnostics on empirical A_emp.
        for p, q in itertools.combinations(ps, 2):
            ap = float(A_emp(int(p)))
            aq = float(A_emp(int(q)))
            apq = float(A_emp(int(p * q)))
            if min_abs > 0 and (abs(ap) < min_abs or abs(aq) < min_abs or abs(apq) < min_abs):
                continue
            denom = max(max(min_abs, 1e-300), abs(ap) * abs(aq))
            C_sf_emp_terms.append(float(abs(apq) / denom))

            # lifted (should be 1 for this ratio)
            lp = float(a_lift(int(p)))
            lq = float(a_lift(int(q)))
            lpq = float(a_lift(int(p * q)))
            denomL = max(max(min_abs, 1e-300), abs(lp) * abs(lq))
            C_sf_lift_terms.append(float(abs(lpq) / denomL))

        for p in ps:
            ap = float(A_emp(int(p)))
            if min_abs > 0 and abs(ap) < min_abs:
                continue
            for k in range(2, max_power + 1):
                pk = int(p**k)
                apk = float(A_emp(int(pk)))
                if min_abs > 0 and abs(apk) < min_abs:
                    continue
                denom = max(max(min_abs, 1e-300), float(abs(ap) ** int(k)))
                C_pp_emp_terms.append(float(abs(apk) / denom))

                # lifted
                lp = float(a_lift(int(p)))
                lpk = float(a_lift(int(pk)))
                denomL = max(max(min_abs, 1e-300), float(abs(lp) ** int(k)))
                C_pp_lift_terms.append(float(abs(lpk) / denomL))

        C_sf_emp = summarize_median(C_sf_emp_terms)
        C_pp_emp = summarize_median(C_pp_emp_terms)
        C_sf_lift = summarize_median(C_sf_lift_terms)
        C_pp_lift = summarize_median(C_pp_lift_terms)

        summaries.append(
            SummaryRow(
                p_mode=str(p_mode),
                ghost_mode=str(ghost_mode),
                u=float(u),
                gated_out=int(gated_out),
                gen_devI_median=float(gen_dev_med),
                dyadic_k_max=int(dyadic_k_max),
                ghost_t=float(ghost_t),
                g1=float(g[1]) if len(g) > 1 else float("nan"),
                g2=float(g[2]) if len(g) > 2 else float("nan"),
                g3=float(g[3]) if len(g) > 3 else float("nan"),
                A1=float(Acoeff[1]) if len(Acoeff) > 1 else float("nan"),
                A2=float(Acoeff[2]) if len(Acoeff) > 2 else float("nan"),
                A3=float(Acoeff[3]) if len(Acoeff) > 3 else float("nan"),
                C_sf_emp_median=float(C_sf_emp),
                C_pp_emp_median=float(C_pp_emp),
                C_sf_lift_median=float(C_sf_lift),
                C_pp_lift_median=float(C_pp_lift),
                med_abs_logratio_primes=float(med_abs_logratio(rows_mode, "prime")),
                med_abs_logratio_primepowers=float(med_abs_logratio(rows_mode, "primepower")),
                med_abs_logratio_squarefree_pairs=float(med_abs_logratio(rows_mode, "squarefree_pair")),
                med_abs_logratio_mixed=float(med_abs_logratio(rows_mode, "mixed")),
            )
        )

        all_rows.extend(rows_mode)

        print("ghostmap/euler lift")
        print(
            "  p_mode=", p_mode,
            "ghost_mode=", str(ghost_mode),
            "u=", _fmt(u),
            "gate_med_devI=", _fmt(gen_dev_med),
            "gated=" + str(int(gated_out)),
        )
        print("  ghost_t=", _fmt(float(ghost_t)), "(auto=1)" if int(args.auto_scale_A1_to_emp_prime_median) == 1 else "")
        print("  dyadic primitive ghosts g1,g2,g3:", _fmt(float(g[1])), _fmt(float(g[2])), _fmt(float(g[3])))
        print("  Euler A1,A2,A3:", _fmt(float(Acoeff[1])), _fmt(float(Acoeff[2])), _fmt(float(Acoeff[3])))
        print("  C_sf(emp)=", _fmt(C_sf_emp), "C_pp(emp)=", _fmt(C_pp_emp), "| C_sf(lift)=", _fmt(C_sf_lift), "C_pp(lift)=", _fmt(C_pp_lift))
        print(
            "  med |log(emp/lift)| by class:",
            "prime", _fmt(med_abs_logratio(rows_mode, "prime")),
            "pp", _fmt(med_abs_logratio(rows_mode, "primepower")),
            "pq", _fmt(med_abs_logratio(rows_mode, "squarefree_pair")),
            "mix", _fmt(med_abs_logratio(rows_mode, "mixed")),
        )

    out_rows = Path(str(args.out_rows_csv))
    out_rows.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.__dict__ for r in all_rows]).to_csv(str(out_rows), index=False)
    print(f"wrote {out_rows}")

    out_sum = Path(str(args.out_summary_csv))
    out_sum.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.__dict__ for r in summaries]).to_csv(str(out_sum), index=False)
    print(f"wrote {out_sum}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
