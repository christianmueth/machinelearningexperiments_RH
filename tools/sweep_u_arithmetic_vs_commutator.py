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


@dataclass(frozen=True)
class SweepRow:
    u: float
    p_mode: str

    gated_out: int

    # Composite-n log-additivity (logdet(I-S)).
    composite_logadd_median: float
    composite_logadd_max: float
    composite_n_pairs: int

    # Generator noncommutativity (scaled by ||Xp-I|| ||Xq-I||).
    comm_add_scaled_median: float
    comm_add_scaled_max: float
    comm_grp_scaled_median: float
    comm_grp_scaled_max: float

    # Generator scale diagnostics.
    gen_devI_median: float
    gen_devI_max: float


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Sweep u and compare (A) composite-n arithmetic log-additivity (using logdet(I-S) primitive P(n)) "
            "against (B) scaled generator commutator burden for X_{p,1}.\n\n"
            "Goal: test whether p_mode=p stays best for composite log-additivity across u, while invp stays best for scaled commutators."
        )
    )

    ap.add_argument("--us", default="0.05,0.1,0.15,0.2,0.25,0.3", help="Comma-list of u values")
    ap.add_argument("--p_modes", default="p,invp,p1_over_p", help="Comma-list of p_mode values")

    ap.add_argument(
        "--ns",
        default="2,3,5,6,7,10,11,13,14,15,17,21,22,26,30,33,34,35,39,51,55,65,77,85,91,119,143,187,221",
        help="Comma-list of n to score (must include some coprime products ab inside the set)",
    )
    ap.add_argument("--ps", default="2,3,5,7,11,13,17", help="Comma-list of primes for commutator probe")

    ap.add_argument("--boundary", default="0,3")
    ap.add_argument("--schur_sign", choices=["-", "+"], default="+")
    ap.add_argument("--scattering", choices=["lambda_pm_i", "i_pm_lambda"], default="i_pm_lambda")

    ap.add_argument("--sharp", choices=["transpose", "conj_transpose"], default="conj_transpose")
    ap.add_argument("--X_mode", choices=["raw", "det1"], default="det1")
    ap.add_argument("--X_gamma", type=float, default=1.0)

    ap.add_argument(
        "--min_abs_P",
        type=float,
        default=0.0,
        help="Optional: skip coprime pairs (a,b) if any of |P(a)|,|P(b)|,|P(ab)| below threshold",
    )

    ap.add_argument(
        "--min_gen_devI_median",
        type=float,
        default=0.0,
        help=(
            "Optional nontriviality gate: only score composite log-additivity if median ||X_{p,1}-I||_F across --ps is >= this threshold. "
            "If gate fails, composite score fields are set to NaN and gated_out=1."
        ),
    )

    ap.add_argument("--out_csv", default="out/u_sweep_arithmetic_vs_commutator.csv")
    args = ap.parse_args()

    us = _parse_float_csv(str(args.us))
    if not us:
        raise SystemExit("--us must be non-empty")

    p_modes = [x.strip() for x in str(args.p_modes).split(",") if x.strip()]
    if not p_modes:
        raise SystemExit("--p_modes must be non-empty")

    ns = _parse_int_csv(str(args.ns))
    if not ns:
        raise SystemExit("--ns must be non-empty")
    ns = sorted(set(int(n) for n in ns))

    ps = _parse_int_csv(str(args.ps))
    if len(ps) < 2:
        raise SystemExit("--ps must have >=2 primes")
    ps = sorted(set(int(p) for p in ps))

    boundary_parts = sim._parse_int_list(str(args.boundary))
    if len(boundary_parts) != 2:
        raise SystemExit("--boundary must be i,j")
    boundary = (int(boundary_parts[0]), int(boundary_parts[1]))

    # Precompute which n values are needed for Dirichlet primitive P(n).
    all_needed = set([1])
    for n in ns:
        all_needed.add(int(n))
        for d in _divisors_from_factorization(_prime_factorization(int(n))):
            all_needed.add(int(d))
    all_needed = set(int(x) for x in all_needed)

    def composite_logadd_for(u: float, p_mode: str, *, gate_devI_median: float) -> tuple[float, float, int, int]:
        u = float(u)
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
                    p_mode=str(p_mode),
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

        # Nontriviality gate (median devI over the same prime set used in commutator probe).
        gated_out = 0
        if float(args.min_gen_devI_median) > 0:
            I2 = np.eye(2, dtype=np.complex128)
            devs = []
            for p in ps:
                Xp = X_p1(int(p))
                devs.append(float(np.linalg.norm(Xp - I2, ord="fro")))
            devs_arr = np.asarray(devs, dtype=float)
            devs_arr = devs_arr[np.isfinite(devs_arr)]
            dev_med = float(np.median(devs_arr)) if devs_arr.size else float("nan")
            gate_devI_median = float(dev_med)
            if not math.isfinite(gate_devI_median) or gate_devI_median < float(args.min_gen_devI_median):
                return float("nan"), float("nan"), 0, 1

        def V_of_n(n: int) -> float:
            Xn = build_X_n(int(n))
            A = sim._cayley(Xn)
            Ash = sim._symplectic_partner(A, mode=str(args.sharp))
            B = sim._bulk_B_from_A(A, Ash)
            Lam = sim._schur_complement_Lambda(B, boundary=boundary, sign=str(args.schur_sign))
            S = sim._scattering_from_Lambda(Lam, mode=str(args.scattering))
            I2 = np.eye(2, dtype=np.complex128)
            det = complex(np.linalg.det(I2 - np.asarray(S, dtype=np.complex128)))
            return float(math.log(max(1e-300, abs(det))))

        V_map: dict[int, float] = {1: 0.0}
        for n in sorted(all_needed):
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

        logadd_errs: list[float] = []
        thr = float(args.min_abs_P)

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
                if thr > 0 and (abs(Pa) < thr or abs(Pb) < thr or abs(Pab) < thr):
                    continue
                logadd_errs.append(float(abs(_safe_log_abs(Pab) - (_safe_log_abs(Pa) + _safe_log_abs(Pb)))))

        if not logadd_errs:
            return float("nan"), float("nan"), 0, gated_out
        arr = np.asarray(logadd_errs, dtype=float)
        return float(np.median(arr)), float(np.max(arr)), int(arr.size), gated_out

    def commutator_scaled_for(u: float, p_mode: str) -> tuple[float, float, float, float, float, float]:
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
        for p in ps:
            Xp = X_p1(int(p))
            devs.append(float(np.linalg.norm(Xp - I2, ord="fro")))
        devs_arr = np.asarray(devs, dtype=float)
        devs_arr = devs_arr[np.isfinite(devs_arr)]
        gen_devI_median = float(np.median(devs_arr)) if devs_arr.size else float("nan")
        gen_devI_max = float(np.max(devs_arr)) if devs_arr.size else float("nan")

        add_scaled = []
        grp_scaled = []

        for i, p in enumerate(ps):
            Xp = X_p1(int(p))
            Xp_dev = float(np.linalg.norm(Xp - I2, ord="fro"))
            for q in ps[i + 1 :]:
                Xq = X_p1(int(q))
                Xq_dev = float(np.linalg.norm(Xq - I2, ord="fro"))
                dev_denom = float(max(1e-12, Xp_dev * Xq_dev))

                add_comm = (Xp @ Xq - Xq @ Xp).astype(np.complex128)
                add_scaled.append(float(np.linalg.norm(add_comm, ord="fro") / dev_denom))

                try:
                    Xp_inv = np.linalg.inv(Xp)
                    Xq_inv = np.linalg.inv(Xq)
                    grp = (Xp @ Xq @ Xp_inv @ Xq_inv).astype(np.complex128)
                    grp_scaled.append(float(np.linalg.norm(grp - I2, ord="fro") / dev_denom))
                except np.linalg.LinAlgError:
                    grp_scaled.append(float("nan"))

        add_arr = np.asarray(add_scaled, dtype=float)
        add_arr = add_arr[np.isfinite(add_arr)]
        grp_arr = np.asarray(grp_scaled, dtype=float)
        grp_arr = grp_arr[np.isfinite(grp_arr)]

        add_median = float(np.median(add_arr)) if add_arr.size else float("nan")
        add_max = float(np.max(add_arr)) if add_arr.size else float("nan")
        grp_median = float(np.median(grp_arr)) if grp_arr.size else float("nan")
        grp_max = float(np.max(grp_arr)) if grp_arr.size else float("nan")

        return add_median, add_max, grp_median, grp_max, gen_devI_median, gen_devI_max

    rows: list[SweepRow] = []

    for u in us:
        for pm in p_modes:
            add_med, add_max, grp_med, grp_max, dev_med, dev_max = commutator_scaled_for(float(u), str(pm))
            logadd_med, logadd_max, n_pairs, gated_out = composite_logadd_for(float(u), str(pm), gate_devI_median=float(dev_med))
            rows.append(
                SweepRow(
                    u=float(u),
                    p_mode=str(pm),
                    gated_out=int(gated_out),
                    composite_logadd_median=float(logadd_med),
                    composite_logadd_max=float(logadd_max),
                    composite_n_pairs=int(n_pairs),
                    comm_add_scaled_median=float(add_med),
                    comm_add_scaled_max=float(add_max),
                    comm_grp_scaled_median=float(grp_med),
                    comm_grp_scaled_max=float(grp_max),
                    gen_devI_median=float(dev_med),
                    gen_devI_max=float(dev_max),
                )
            )
            print(
                "u=", _fmt(float(u)),
                "p_mode=", str(pm),
                "| gated=", int(gated_out),
                "logadd_med=", _fmt(float(logadd_med)),
                "comm_add_scaled_med=", _fmt(float(add_med)),
                "comm_grp_scaled_med=", _fmt(float(grp_med)),
                "gen_devI_med=", _fmt(float(dev_med)),
            )

    df = pd.DataFrame([r.__dict__ for r in rows]).sort_values(["u", "p_mode"]).reset_index(drop=True)

    out_path = Path(str(args.out_csv))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out_path), index=False)
    print(f"wrote {out_path}")

    # Simple per-u winners.
    for u in us:
        sub = df[df["u"] == float(u)].copy()
        sub = sub.replace([np.inf, -np.inf], np.nan)
        if sub.empty:
            continue
        # Winner for arithmetic (min composite_logadd_median) among ungated rows.
        a = sub[(sub["gated_out"] == 0)].dropna(subset=["composite_logadd_median"]).sort_values(["composite_logadd_median"])
        if not a.empty:
            best = a.iloc[0]
            print(f"u={_fmt(float(u))} best_arithmetic(p_mode)={best['p_mode']} logadd_med={_fmt(float(best['composite_logadd_median']))}")
        c = sub.dropna(subset=["comm_add_scaled_median"]).sort_values(["comm_add_scaled_median"])
        if not c.empty:
            best = c.iloc[0]
            print(f"u={_fmt(float(u))} best_comm_add(p_mode)={best['p_mode']} add_scaled_med={_fmt(float(best['comm_add_scaled_median']))}")

    # Aggregate win counts (arithmetic winner only), ignoring gated rows.
    win_counts: dict[str, int] = {pm: 0 for pm in p_modes}
    total_count = 0
    for u in us:
        sub = df[(df["u"] == float(u)) & (df["gated_out"] == 0)].dropna(subset=["composite_logadd_median"])
        if sub.empty:
            continue
        best = sub.sort_values(["composite_logadd_median"]).iloc[0]
        win_counts[str(best["p_mode"])] = int(win_counts.get(str(best["p_mode"]), 0) + 1)
        total_count += 1
    if total_count:
        print("arithmetic winner counts (ungated u points):")
        for pm in p_modes:
            print(f"  {pm}: {win_counts.get(pm, 0)}/{total_count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
