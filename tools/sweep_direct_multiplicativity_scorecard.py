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


def _safe_log_abs(x: float, *, floor: float) -> float:
    return float(math.log(max(float(floor), abs(float(x)))))


@dataclass(frozen=True)
class ScoreRow:
    u: float
    p_mode: str
    gated_out: int
    gen_devI_median: float

    # Level A (omega=2, primes only): m=p, n=q, mn=pq
    levelA_n_pairs: int
    levelA_med_abs_dlog: float
    levelA_med_dx: float
    levelA_med_abs_logR: float

    # New: Möbius-primitive concentration / support checks
    C_sf_median: float
    C_pp_median: float

    # New: pairwise log-additivity summary (signed deltas)
    median_abs_delta_log_pairs: float
    p75_abs_delta_log_pairs: float
    max_abs_delta_log_pairs: float
    median_bias_delta_log_pairs: float
    mad_centered_delta_log_pairs: float
    iqr_centered_delta_log_pairs: float
    max_abs_centered_delta_log_pairs: float
    corr_abs_delta_log_vs_comm_add_scaled: float
    spearman_abs_delta_log_vs_comm_add_scaled: float

    # Aliases (requested names)
    mad_delta_log_pairs: float
    corr_abs_delta_log_vs_comm: float

    # Level B (prime powers)
    levelB_n_terms: int
    levelB_med_abs_pp_dev: float

    # New: prime power log-additivity summary (signed deltas)
    median_abs_delta_log_primepowers: float
    p75_abs_delta_log_primepowers: float
    max_abs_delta_log_primepowers: float
    median_bias_delta_log_primepowers: float
    mad_centered_delta_log_primepowers: float
    iqr_centered_delta_log_primepowers: float
    max_abs_centered_delta_log_primepowers: float

    # Level C (mixed prime powers): m=p^a, n=q^b with p!=q
    levelC_n_pairs: int
    levelC_med_abs_dlog: float
    levelC_med_dx: float
    levelC_med_abs_logR: float

    # New: mixed log-additivity summary (signed deltas)
    median_abs_delta_log_mixed: float
    p75_abs_delta_log_mixed: float
    max_abs_delta_log_mixed: float
    median_bias_delta_log_mixed: float
    mad_centered_delta_log_mixed: float
    iqr_centered_delta_log_mixed: float
    max_abs_centered_delta_log_mixed: float


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Direct multiplicativity scorecard for the frozen composite-n setup:\n"
            "  composition: X-semigroup (X_n = prod_p X_{p,1}^{v_p(n)})\n"
            "  observable:   V(n)=log|det(I-S_n)|\n"
            "  arithmetic:   A(n)=P(n)=Dirichlet primitive of V (Mobius over divisors)\n\n"
            "Reports a minimal scorecard under the same nontriviality gate (median ||X_{p,1}-I||_F).\n\n"
            "Level A (omega-2, primes only): coprime pairs (p,q) with A(pq) vs A(p),A(q)\n"
            "  - median |Δlog| where Δlog=log|A(pq)|-log|A(p)|-log|A(q)|\n"
            "  - median Δ× where Δ×=|A(pq)-A(p)A(q)|/(1+|A(p)A(q)|)\n"
            "  - median |log R| where R=|A(pq)|/(|A(p)||A(q)|)\n\n"
            "Level B (prime powers): median |log|A(p^k)| - k log|A(p)|| for k=2..--max_power."
        )
    )

    ap.add_argument("--ps", default="2,3,5,7,11,13,17", help="Comma-list of primes")
    ap.add_argument("--us", default="0.2", help="Comma-list of u values")

    ap.add_argument("--min_gen_devI_median", type=float, default=0.8)
    ap.add_argument(
        "--min_abs_A",
        type=float,
        default=1e-12,
        help="Minimum |A| threshold to include a term in log-based metrics",
    )

    ap.add_argument("--max_power", type=int, default=3, help="Max prime power exponent for Level B")

    ap.add_argument(
        "--do_levelC",
        type=int,
        default=0,
        help="If 1, also run Level C mixed p^a q^b tests (Euler-style) using a,b<=--max_power",
    )

    ap.add_argument("--boundary", default="0,3")
    ap.add_argument("--schur_sign", choices=["-", "+"], default="+")
    ap.add_argument("--scattering", choices=["lambda_pm_i", "i_pm_lambda"], default="i_pm_lambda")

    ap.add_argument("--sharp", choices=["transpose", "conj_transpose"], default="conj_transpose")
    ap.add_argument("--X_mode", choices=["raw", "det1"], default="det1")
    ap.add_argument("--X_gamma", type=float, default=1.0)

    ap.add_argument("--p_modes", default="p,invp", help="Comma-list of p_modes to compare")

    ap.add_argument("--out_csv", default="out/direct_multiplicativity_scorecard.csv")

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

    boundary_parts = sim._parse_int_list(str(args.boundary))
    if len(boundary_parts) != 2:
        raise SystemExit("--boundary must be i,j")
    boundary = (int(boundary_parts[0]), int(boundary_parts[1]))

    max_power = int(args.max_power)
    if max_power < 2:
        raise SystemExit("--max_power must be >=2")

    min_abs_A = float(args.min_abs_A)
    min_abs_A = max(0.0, min_abs_A)

    rows: list[ScoreRow] = []

    def _summary_signed(deltas: list[float]) -> dict[str, float]:
        arr = np.asarray(deltas, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {
                "n": 0.0,
                "med_abs": float("nan"),
                "p75_abs": float("nan"),
                "max_abs": float("nan"),
                "bias_med": float("nan"),
                "mad_centered": float("nan"),
                "iqr_centered": float("nan"),
                "max_abs_centered": float("nan"),
            }

        abs_arr = np.abs(arr)
        bias = float(np.median(arr))
        centered = arr - bias
        abs_centered = np.abs(centered)
        q75 = float(np.percentile(abs_arr, 75))
        q25_c = float(np.percentile(centered, 25))
        q75_c = float(np.percentile(centered, 75))
        mad = float(np.median(abs_centered))
        iqr = float(q75_c - q25_c)
        return {
            "n": float(arr.size),
            "med_abs": float(np.median(abs_arr)),
            "p75_abs": float(q75),
            "max_abs": float(np.max(abs_arr)),
            "bias_med": float(bias),
            "mad_centered": float(mad),
            "iqr_centered": float(iqr),
            "max_abs_centered": float(np.max(abs_centered)),
        }

    def _corr(xs: list[float], ys: list[float]) -> float:
        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        if x.size < 2:
            return float("nan")
        if float(np.std(x)) <= 0 or float(np.std(y)) <= 0:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    def _spearman_corr(xs: list[float], ys: list[float]) -> float:
        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        if x.size < 2:
            return float("nan")
        # Rank with average method for ties.
        xr = pd.Series(x).rank(method="average").to_numpy(dtype=float)
        yr = pd.Series(y).rank(method="average").to_numpy(dtype=float)
        if float(np.std(xr)) <= 0 or float(np.std(yr)) <= 0:
            return float("nan")
        return float(np.corrcoef(xr, yr)[0, 1])

    for u in us:
        u = float(u)
        v = -float(u)

        for p_mode in p_modes:
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
            devs = [float(np.linalg.norm(X_p1(int(p)) - I2, ord="fro")) for p in ps]
            devs_arr = np.asarray(devs, dtype=float)
            devs_arr = devs_arr[np.isfinite(devs_arr)]
            gen_dev_med = float(np.median(devs_arr)) if devs_arr.size else float("nan")

            gated_out = 0
            if (not math.isfinite(gen_dev_med)) or gen_dev_med < float(args.min_gen_devI_median):
                gated_out = 1
                rows.append(
                    ScoreRow(
                        u=float(u),
                        p_mode=str(p_mode),
                        gated_out=int(gated_out),
                        gen_devI_median=float(gen_dev_med),
                        levelA_n_pairs=0,
                        levelA_med_abs_dlog=float("nan"),
                        levelA_med_dx=float("nan"),
                        levelA_med_abs_logR=float("nan"),
                        C_sf_median=float("nan"),
                        C_pp_median=float("nan"),
                        median_abs_delta_log_pairs=float("nan"),
                        p75_abs_delta_log_pairs=float("nan"),
                        max_abs_delta_log_pairs=float("nan"),
                        median_bias_delta_log_pairs=float("nan"),
                        mad_centered_delta_log_pairs=float("nan"),
                        iqr_centered_delta_log_pairs=float("nan"),
                        max_abs_centered_delta_log_pairs=float("nan"),
                        corr_abs_delta_log_vs_comm_add_scaled=float("nan"),
                        spearman_abs_delta_log_vs_comm_add_scaled=float("nan"),
                        mad_delta_log_pairs=float("nan"),
                        corr_abs_delta_log_vs_comm=float("nan"),
                        levelB_n_terms=0,
                        levelB_med_abs_pp_dev=float("nan"),
                        median_abs_delta_log_primepowers=float("nan"),
                        p75_abs_delta_log_primepowers=float("nan"),
                        max_abs_delta_log_primepowers=float("nan"),
                        median_bias_delta_log_primepowers=float("nan"),
                        mad_centered_delta_log_primepowers=float("nan"),
                        iqr_centered_delta_log_primepowers=float("nan"),
                        max_abs_centered_delta_log_primepowers=float("nan"),
                        levelC_n_pairs=0,
                        levelC_med_abs_dlog=float("nan"),
                        levelC_med_dx=float("nan"),
                        levelC_med_abs_logR=float("nan"),
                        median_abs_delta_log_mixed=float("nan"),
                        p75_abs_delta_log_mixed=float("nan"),
                        max_abs_delta_log_mixed=float("nan"),
                        median_bias_delta_log_mixed=float("nan"),
                        mad_centered_delta_log_mixed=float("nan"),
                        iqr_centered_delta_log_mixed=float("nan"),
                        max_abs_centered_delta_log_mixed=float("nan"),
                    )
                )
                print(
                    "u=", _fmt(float(u)),
                    "p_mode=", str(p_mode),
                    "GATED (gen_devI_median=", _fmt(float(gen_dev_med)), ")",
                )
                continue

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

            # Build the minimal n-set needed for Level A and B.
            needed_ns: set[int] = set([1])
            # primes and their products
            for p in ps:
                needed_ns.add(int(p))
            for p, q in itertools.combinations(ps, 2):
                needed_ns.add(int(p * q))
            # prime powers
            for p in ps:
                pk = int(p)
                for _k in range(2, max_power + 1):
                    pk *= int(p)
                    needed_ns.add(int(pk))

            # optional mixed prime powers for Level C
            if int(args.do_levelC) == 1:
                for p, q in itertools.combinations(ps, 2):
                    for a in range(1, max_power + 1):
                        for b in range(1, max_power + 1):
                            needed_ns.add(int((p**a) * (q**b)))

            # Evaluate V on all divisors needed for Dirichlet inversion.
            all_needed: set[int] = set([1])
            for n in needed_ns:
                all_needed.add(int(n))
                for d in _divisors_from_factorization(_prime_factorization(int(n))):
                    all_needed.add(int(d))

            V_map: dict[int, float] = {1: 0.0}
            for n in sorted(all_needed):
                if n == 1:
                    continue
                V_map[n] = float(V_of_n(int(n)))

            def A_of_n(n: int) -> float:
                # Dirichlet primitive: P(n) = sum_{d|n} mu(d) V(n/d)
                f = _prime_factorization(int(n))
                divs = _divisors_from_factorization(f)
                acc = 0.0
                for d in divs:
                    mu = _mobius_mu(int(d))
                    if mu == 0:
                        continue
                    acc += float(mu) * float(V_map[int(n // d)])
                return float(acc)

            A_map = {int(n): float(A_of_n(int(n))) for n in needed_ns if int(n) != 1}

            # Level A: primes only.
            dlogs_signed: list[float] = []
            abs_dlogs: list[float] = []
            dxs: list[float] = []
            abs_logRs: list[float] = []
            comm_add_scaled: list[float] = []
            C_sf_terms: list[float] = []

            for p, q in itertools.combinations(ps, 2):
                ap = float(A_map.get(int(p), float("nan")))
                aq = float(A_map.get(int(q), float("nan")))
                apq = float(A_map.get(int(p * q), float("nan")))
                if not (math.isfinite(ap) and math.isfinite(aq) and math.isfinite(apq)):
                    continue
                if min_abs_A > 0 and (abs(ap) < min_abs_A or abs(aq) < min_abs_A or abs(apq) < min_abs_A):
                    continue

                dlog = _safe_log_abs(apq, floor=max(min_abs_A, 1e-300)) - (
                    _safe_log_abs(ap, floor=max(min_abs_A, 1e-300)) + _safe_log_abs(aq, floor=max(min_abs_A, 1e-300))
                )
                dlogs_signed.append(float(dlog))
                abs_dlogs.append(float(abs(dlog)))

                denom = 1.0 + abs(ap * aq)
                dxs.append(float(abs(apq - ap * aq) / denom))

                denomR = max(max(min_abs_A, 1e-300), abs(ap) * abs(aq))
                R = float(abs(apq) / denomR)
                abs_logRs.append(float(abs(math.log(max(1e-300, R)))))

                # support/concentration proxy on |A|: M(pq)/(M(p)M(q))
                C_sf_terms.append(float(abs(apq) / denomR))

                # commutator burden (scaled additive commutator)
                Xp = X_p1(int(p))
                Xq = X_p1(int(q))
                dev_denom = float(max(1e-12, np.linalg.norm(Xp - I2, ord="fro") * np.linalg.norm(Xq - I2, ord="fro")))
                add_comm_fro = float(np.linalg.norm((Xp @ Xq - Xq @ Xp).astype(np.complex128), ord="fro"))
                comm_add_scaled.append(float(add_comm_fro / dev_denom))

            def _med_or_nan(xs: list[float]) -> float:
                arr = np.asarray(xs, dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    return float("nan")
                return float(np.median(arr))

            levelA_n_pairs = int(len(abs_dlogs))
            levelA_med_abs_dlog = _med_or_nan(abs_dlogs)
            levelA_med_dx = _med_or_nan(dxs)
            levelA_med_abs_logR = _med_or_nan(abs_logRs)

            # concentration summaries
            C_sf_median = _med_or_nan(C_sf_terms)

            # Level B: prime powers.
            pp_deltas_signed: list[float] = []
            pp_devs: list[float] = []
            C_pp_terms: list[float] = []
            for p in ps:
                ap = float(A_map.get(int(p), float("nan")))
                if not math.isfinite(ap):
                    continue
                if min_abs_A > 0 and abs(ap) < min_abs_A:
                    continue
                log_ap = _safe_log_abs(ap, floor=max(min_abs_A, 1e-300))

                pk = int(p)
                for k in range(2, max_power + 1):
                    pk *= int(p)
                    apk = float(A_map.get(int(pk), float("nan")))
                    if not math.isfinite(apk):
                        continue
                    if min_abs_A > 0 and abs(apk) < min_abs_A:
                        continue
                    log_apk = _safe_log_abs(apk, floor=max(min_abs_A, 1e-300))
                    delta = float(log_apk - float(k) * log_ap)
                    pp_deltas_signed.append(float(delta))
                    pp_devs.append(float(abs(delta)))

                    denom = max(max(min_abs_A, 1e-300), float(abs(ap) ** int(k)))
                    C_pp_terms.append(float(abs(apk) / denom))

            levelB_n_terms = int(len(pp_devs))
            levelB_med_abs_pp_dev = _med_or_nan(pp_devs)

            C_pp_median = _med_or_nan(C_pp_terms)

            # Level C: mixed prime powers (optional).
            levelC_n_pairs = 0
            levelC_med_abs_dlog = float("nan")
            levelC_med_dx = float("nan")
            levelC_med_abs_logR = float("nan")
            dlogs_signed_C: list[float] = []
            if int(args.do_levelC) == 1:
                abs_dlogs_C: list[float] = []
                dxs_C: list[float] = []
                abs_logRs_C: list[float] = []

                for p, q in itertools.combinations(ps, 2):
                    for a in range(1, max_power + 1):
                        for b in range(1, max_power + 1):
                            m = int(p**a)
                            n = int(q**b)
                            mn = int(m * n)
                            am = float(A_map.get(int(m), float("nan")))
                            an = float(A_map.get(int(n), float("nan")))
                            amn = float(A_map.get(int(mn), float("nan")))
                            if not (math.isfinite(am) and math.isfinite(an) and math.isfinite(amn)):
                                continue
                            if min_abs_A > 0 and (
                                abs(am) < min_abs_A or abs(an) < min_abs_A or abs(amn) < min_abs_A
                            ):
                                continue

                            dlog = _safe_log_abs(amn, floor=max(min_abs_A, 1e-300)) - (
                                _safe_log_abs(am, floor=max(min_abs_A, 1e-300))
                                + _safe_log_abs(an, floor=max(min_abs_A, 1e-300))
                            )
                            dlogs_signed_C.append(float(dlog))
                            abs_dlogs_C.append(float(abs(dlog)))

                            denom = 1.0 + abs(am * an)
                            dxs_C.append(float(abs(amn - am * an) / denom))

                            denomR = max(max(min_abs_A, 1e-300), abs(am) * abs(an))
                            R = float(abs(amn) / denomR)
                            abs_logRs_C.append(float(abs(math.log(max(1e-300, R)))))

                levelC_n_pairs = int(len(abs_dlogs_C))
                levelC_med_abs_dlog = _med_or_nan(abs_dlogs_C)
                levelC_med_dx = _med_or_nan(dxs_C)
                levelC_med_abs_logR = _med_or_nan(abs_logRs_C)

            # New: signed delta log summaries
            sumA = _summary_signed(dlogs_signed)
            sumB = _summary_signed(pp_deltas_signed)
            sumC = _summary_signed(dlogs_signed_C) if int(args.do_levelC) == 1 else _summary_signed([])

            corr_abs_dlog_vs_comm = _corr([abs(x) for x in dlogs_signed], comm_add_scaled)
            spearman_abs_dlog_vs_comm = _spearman_corr([abs(x) for x in dlogs_signed], comm_add_scaled)

            rows.append(
                ScoreRow(
                    u=float(u),
                    p_mode=str(p_mode),
                    gated_out=int(gated_out),
                    gen_devI_median=float(gen_dev_med),
                    levelA_n_pairs=int(levelA_n_pairs),
                    levelA_med_abs_dlog=float(levelA_med_abs_dlog),
                    levelA_med_dx=float(levelA_med_dx),
                    levelA_med_abs_logR=float(levelA_med_abs_logR),
                    C_sf_median=float(C_sf_median),
                    C_pp_median=float(C_pp_median),
                    median_abs_delta_log_pairs=float(sumA["med_abs"]),
                    p75_abs_delta_log_pairs=float(sumA["p75_abs"]),
                    max_abs_delta_log_pairs=float(sumA["max_abs"]),
                    median_bias_delta_log_pairs=float(sumA["bias_med"]),
                    mad_centered_delta_log_pairs=float(sumA["mad_centered"]),
                    iqr_centered_delta_log_pairs=float(sumA["iqr_centered"]),
                    max_abs_centered_delta_log_pairs=float(sumA["max_abs_centered"]),
                    corr_abs_delta_log_vs_comm_add_scaled=float(corr_abs_dlog_vs_comm),
                    spearman_abs_delta_log_vs_comm_add_scaled=float(spearman_abs_dlog_vs_comm),
                    mad_delta_log_pairs=float(sumA["mad_centered"]),
                    corr_abs_delta_log_vs_comm=float(corr_abs_dlog_vs_comm),
                    levelB_n_terms=int(levelB_n_terms),
                    levelB_med_abs_pp_dev=float(levelB_med_abs_pp_dev),
                    median_abs_delta_log_primepowers=float(sumB["med_abs"]),
                    p75_abs_delta_log_primepowers=float(sumB["p75_abs"]),
                    max_abs_delta_log_primepowers=float(sumB["max_abs"]),
                    median_bias_delta_log_primepowers=float(sumB["bias_med"]),
                    mad_centered_delta_log_primepowers=float(sumB["mad_centered"]),
                    iqr_centered_delta_log_primepowers=float(sumB["iqr_centered"]),
                    max_abs_centered_delta_log_primepowers=float(sumB["max_abs_centered"]),
                    levelC_n_pairs=int(levelC_n_pairs),
                    levelC_med_abs_dlog=float(levelC_med_abs_dlog),
                    levelC_med_dx=float(levelC_med_dx),
                    levelC_med_abs_logR=float(levelC_med_abs_logR),
                    median_abs_delta_log_mixed=float(sumC["med_abs"]),
                    p75_abs_delta_log_mixed=float(sumC["p75_abs"]),
                    max_abs_delta_log_mixed=float(sumC["max_abs"]),
                    median_bias_delta_log_mixed=float(sumC["bias_med"]),
                    mad_centered_delta_log_mixed=float(sumC["mad_centered"]),
                    iqr_centered_delta_log_mixed=float(sumC["iqr_centered"]),
                    max_abs_centered_delta_log_mixed=float(sumC["max_abs_centered"]),
                )
            )

            print(
                "u=", _fmt(float(u)),
                "p_mode=", str(p_mode),
                "gen_devI_med=", _fmt(float(gen_dev_med)),
                "| C_sf=", _fmt(float(C_sf_median)),
                "C_pp=", _fmt(float(C_pp_median)),
                "| pairs med|dlog|=", _fmt(float(sumA["med_abs"])),
                "p75=", _fmt(float(sumA["p75_abs"])),
                "max=", _fmt(float(sumA["max_abs"])),
                "bias=", _fmt(float(sumA["bias_med"])),
                "MAD=", _fmt(float(sumA["mad_centered"])),
                "corr(|dlog|,comm)=", _fmt(float(corr_abs_dlog_vs_comm)),
                "rho=", _fmt(float(spearman_abs_dlog_vs_comm)),
                "| pp med|d|=", _fmt(float(sumB["med_abs"])),
                "bias=", _fmt(float(sumB["bias_med"])),
                "| mixed med|d|=", _fmt(float(sumC["med_abs"])) if int(args.do_levelC) == 1 else "",
                ("| LevelC n_pairs=" + str(int(levelC_n_pairs)) + " med|dlog|=" + _fmt(float(levelC_med_abs_dlog)) + " med dx=" + _fmt(float(levelC_med_dx)))
                if int(args.do_levelC) == 1
                else "",
            )

    out_path = Path(str(args.out_csv))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.__dict__ for r in rows]).to_csv(str(out_path), index=False)
    print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
