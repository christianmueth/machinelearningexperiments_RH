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

from src.arithmetic.mobius import mobius_mu, divisors


def _parse_int_csv(s: str) -> list[int]:
    out: list[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _parse_float_csv(s: str) -> list[float]:
    out: list[float] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
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
class Row:
    u: float
    p_mode: str
    gated_out: int
    gen_devI_median: float

    dyadic_k_max: int
    g1: float
    g2: float
    g3: float

    max_power: int
    A1: float
    A2: float
    A3: float

    # Primary discriminator signals
    A1_abs: float
    invp_A1_near0: int

    # empirical support ratios
    C_sf_emp_median: float
    C_pp_emp_median: float

    # fit-to-emp errors: median |log(emp/lift)| by class
    med_abs_logratio_primes: float
    med_abs_logratio_primepowers: float
    med_abs_logratio_squarefree_pairs: float
    med_abs_logratio_mixed: float


@dataclass(frozen=True)
class SummaryRow:
    p_mode: str
    n_u: int
    n_u_ungated: int
    n_A1_near0: int
    median_A1_abs: float
    median_pq_fit: float
    median_pp_fit: float
    median_mixed_fit: float


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Sweep the ghost-map/Euler-transform lift over a u-window (frozen geometry), using ghost_mode=aggregate_as_ghost only.\n\n"
            "For each u and p_mode in {p,invp}:\n"
            "  - extract g_m = F(m) = V(2^m) where V(n)=log|det(I-S_n)|\n"
            "  - compute Euler-transform coefficients A_k\n"
            "  - build lifted multiplicative model a_n (with a_{p^k}=A_k)\n"
            "  - compare against empirical A_emp(n)=Dirichlet primitive of V(n)\n\n"
            "Outputs A1 and classwise fit errors, especially squarefree pq." 
        )
    )

    ap.add_argument("--ps", default="2,3,5,7,11,13,17")
    ap.add_argument("--us", default="0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24")

    ap.add_argument("--p_modes", default="p,invp")
    ap.add_argument("--min_gen_devI_median", type=float, default=0.8)
    ap.add_argument("--min_abs", type=float, default=1e-12)

    ap.add_argument("--dyadic_k_max", type=int, default=12)
    ap.add_argument("--max_power", type=int, default=3)

    ap.add_argument("--A1_near0_eps", type=float, default=0.2, help="Threshold for declaring |A1| near zero")

    ap.add_argument("--boundary", default="0,3")
    ap.add_argument("--schur_sign", choices=["-", "+"], default="+")
    ap.add_argument("--scattering", choices=["lambda_pm_i", "i_pm_lambda"], default="i_pm_lambda")

    ap.add_argument("--sharp", choices=["transpose", "conj_transpose"], default="conj_transpose")
    ap.add_argument("--X_mode", choices=["raw", "det1"], default="det1")
    ap.add_argument("--X_gamma", type=float, default=1.0)

    ap.add_argument("--out_csv", default="out/u_sweep_ghostlift_rows.csv")
    ap.add_argument("--out_summary_csv", default="out/u_sweep_ghostlift_summary.csv")
    args = ap.parse_args()

    ps = sorted(set(int(p) for p in _parse_int_csv(str(args.ps))))
    if len(ps) < 2:
        raise SystemExit("--ps must contain at least two primes")

    us = _parse_float_csv(str(args.us))
    if not us:
        raise SystemExit("--us must be non-empty")

    p_modes = [m.strip() for m in str(args.p_modes).split(",") if m.strip()]
    if not p_modes:
        raise SystemExit("--p_modes must be non-empty")

    dyadic_k_max = int(args.dyadic_k_max)
    max_power = int(args.max_power)
    if dyadic_k_max < 3:
        raise SystemExit("--dyadic_k_max must be >=3")
    if max_power < 3:
        raise SystemExit("--max_power must be >=3 (we report A1..A3)")

    min_abs = max(0.0, float(args.min_abs))

    boundary_parts = sim._parse_int_list(str(args.boundary))
    if len(boundary_parts) != 2:
        raise SystemExit("--boundary must be i,j")
    boundary = (int(boundary_parts[0]), int(boundary_parts[1]))

    I2 = np.eye(2, dtype=np.complex128)

    def summarize_median(xs: list[float]) -> float:
        arr = np.asarray(xs, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("nan")
        return float(np.median(arr))

    def med_abs_logratio_by_cls(rows: list[tuple[str, float]], cls: str) -> float:
        xs = [abs(v) for c, v in rows if c == cls and math.isfinite(v)]
        return summarize_median(xs)

    out_rows: list[Row] = []

    for u in us:
        u = float(u)
        v = -float(u)

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

            # Gate on generator nontriviality.
            devs = [float(np.linalg.norm(X_p1(int(p)) - I2, ord="fro")) for p in ps]
            devs_arr = np.asarray(devs, dtype=float)
            devs_arr = devs_arr[np.isfinite(devs_arr)]
            gen_dev_med = float(np.median(devs_arr)) if devs_arr.size else float("nan")

            gated_out = 0
            if (not math.isfinite(gen_dev_med)) or gen_dev_med < float(args.min_gen_devI_median):
                gated_out = 1

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

            # dyadic ghost: aggregate_as_ghost
            g = [0.0] * (dyadic_k_max + 1)
            for m in range(1, dyadic_k_max + 1):
                g[m] = float(V_of_n(int(2**m)))

            Acoeff = _euler_transform_from_ghost(g, k_max=max_power)

            # Evaluation set for empirical primitive.
            ns: set[int] = set()
            for p in ps:
                ns.add(int(p))
                pk = 1
                for k in range(2, max_power + 1):
                    pk *= int(p)
                    ns.add(int(pk))
            for p, q in itertools.combinations(ps, 2):
                ns.add(int(p * q))
                for a in range(1, max_power + 1):
                    for b in range(1, max_power + 1):
                        ns.add(int((p**a) * (q**b)))

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

            def a_lift(n: int) -> float:
                f = _prime_factorization(int(n))
                out = 1.0
                for _p, e in f.items():
                    e = int(e)
                    if e > max_power:
                        return float("nan")
                    out *= float(Acoeff[e])
                return float(out)

            # classwise errors and support ratios
            errs: list[tuple[str, float]] = []

            def cls_of_n(n: int) -> str:
                f = _prime_factorization(int(n))
                om = len([1 for e in f.values() if int(e) > 0])
                if om == 1:
                    e = next(iter(f.values()))
                    return "prime" if int(e) == 1 else "primepower"
                if om == 2 and all(int(e) == 1 for e in f.values()):
                    return "squarefree_pair"
                if om == 2:
                    return "mixed"
                return "other"

            for n in ns:
                emp = float(A_emp(int(n)))
                lift = float(a_lift(int(n)))
                log_ratio = float(
                    _safe_log_abs(emp, floor=max(min_abs, 1e-300))
                    - _safe_log_abs(lift, floor=max(min_abs, 1e-300))
                )
                errs.append((cls_of_n(int(n)), float(log_ratio)))

            # empirical support ratios
            C_sf_emp_terms: list[float] = []
            C_pp_emp_terms: list[float] = []
            for p, q in itertools.combinations(ps, 2):
                ap = float(A_emp(int(p)))
                aq = float(A_emp(int(q)))
                apq = float(A_emp(int(p * q)))
                if min_abs > 0 and (abs(ap) < min_abs or abs(aq) < min_abs or abs(apq) < min_abs):
                    continue
                denom = max(max(min_abs, 1e-300), abs(ap) * abs(aq))
                C_sf_emp_terms.append(float(abs(apq) / denom))

            for p in ps:
                ap = float(A_emp(int(p)))
                if min_abs > 0 and abs(ap) < min_abs:
                    continue
                for k in range(2, max_power + 1):
                    apk = float(A_emp(int(p**k)))
                    if min_abs > 0 and abs(apk) < min_abs:
                        continue
                    denom = max(max(min_abs, 1e-300), float(abs(ap) ** int(k)))
                    C_pp_emp_terms.append(float(abs(apk) / denom))

            C_sf_emp = summarize_median(C_sf_emp_terms)
            C_pp_emp = summarize_median(C_pp_emp_terms)

            A1 = float(Acoeff[1])
            A1_abs = float(abs(A1))
            invp_A1_near0 = 1 if (str(p_mode) == "invp" and A1_abs < float(args.A1_near0_eps)) else 0

            row = Row(
                u=float(u),
                p_mode=str(p_mode),
                gated_out=int(gated_out),
                gen_devI_median=float(gen_dev_med),
                dyadic_k_max=int(dyadic_k_max),
                g1=float(g[1]),
                g2=float(g[2]),
                g3=float(g[3]),
                max_power=int(max_power),
                A1=float(Acoeff[1]),
                A2=float(Acoeff[2]),
                A3=float(Acoeff[3]),
                A1_abs=float(A1_abs),
                invp_A1_near0=int(invp_A1_near0),
                C_sf_emp_median=float(C_sf_emp),
                C_pp_emp_median=float(C_pp_emp),
                med_abs_logratio_primes=float(med_abs_logratio_by_cls(errs, "prime")),
                med_abs_logratio_primepowers=float(med_abs_logratio_by_cls(errs, "primepower")),
                med_abs_logratio_squarefree_pairs=float(med_abs_logratio_by_cls(errs, "squarefree_pair")),
                med_abs_logratio_mixed=float(med_abs_logratio_by_cls(errs, "mixed")),
            )
            out_rows.append(row)

            print(
                "u=", _fmt(float(u)),
                "p_mode=", str(p_mode),
                "A1=", _fmt(float(row.A1)),
                "|pq_fit=", _fmt(float(row.med_abs_logratio_squarefree_pairs)),
                "pp_fit=", _fmt(float(row.med_abs_logratio_primepowers)),
                "mixed_fit=", _fmt(float(row.med_abs_logratio_mixed)),
                "gated=" + str(int(gated_out)),
            )

    df = pd.DataFrame([r.__dict__ for r in out_rows])

    # Summaries by mode
    summaries: list[SummaryRow] = []
    for p_mode in sorted(set(df["p_mode"].tolist())):
        sub = df[df["p_mode"] == p_mode]
        n_u = int(sub.shape[0])
        sub_ung = sub[sub["gated_out"] == 0]
        n_u_ung = int(sub_ung.shape[0])
        n_A1_near0 = int(sub_ung["invp_A1_near0"].sum()) if "invp_A1_near0" in sub_ung else 0

        summaries.append(
            SummaryRow(
                p_mode=str(p_mode),
                n_u=int(n_u),
                n_u_ungated=int(n_u_ung),
                n_A1_near0=int(n_A1_near0),
                median_A1_abs=float(np.median(sub_ung["A1_abs"].to_numpy(dtype=float))) if n_u_ung else float("nan"),
                median_pq_fit=float(np.median(sub_ung["med_abs_logratio_squarefree_pairs"].to_numpy(dtype=float)))
                if n_u_ung
                else float("nan"),
                median_pp_fit=float(np.median(sub_ung["med_abs_logratio_primepowers"].to_numpy(dtype=float)))
                if n_u_ung
                else float("nan"),
                median_mixed_fit=float(np.median(sub_ung["med_abs_logratio_mixed"].to_numpy(dtype=float)))
                if n_u_ung
                else float("nan"),
            )
        )

    out_path = Path(str(args.out_csv))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["u", "p_mode"]).to_csv(str(out_path), index=False)
    print(f"wrote {out_path}")

    out_sum = Path(str(args.out_summary_csv))
    out_sum.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([s.__dict__ for s in summaries]).to_csv(str(out_sum), index=False)
    print(f"wrote {out_sum}")

    for s in summaries:
        print("summary", s.p_mode, "ungated", s.n_u_ungated, "/", s.n_u, "A1_near0", s.n_A1_near0, "median|A1|", _fmt(s.median_A1_abs), "median pq_fit", _fmt(s.median_pq_fit))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
