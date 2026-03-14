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


def _safe_log_abs(x: float, *, floor: float) -> float:
    return float(math.log(max(float(floor), abs(float(x)))))


@dataclass(frozen=True)
class FitResult:
    u: float
    model: str

    ghost_beta_mode: str
    beta: float
    beta2: float
    beta3: float

    b: float
    c: float

    # training loss pieces
    train_E_sf: float
    train_E_pp: float
    train_E_mix: float
    train_L: float

    # eval over full prime set
    eval_E_sf: float
    eval_E_pp: float
    eval_E_mix: float
    eval_L: float

    # eval over holdout-containing n only
    holdout_E_sf: float
    holdout_E_pp: float
    holdout_E_mix: float
    holdout_L: float

    # eval over "new primes" (eval primes not in train and not in holdout_ps)
    newprime_E_sf: float
    newprime_E_pp: float
    newprime_E_mix: float
    newprime_L: float


@dataclass(frozen=True)
class NRow:
    n: int
    cls: str
    Omega_total: int
    primeset_tag: str

    emp_A: float
    lift_a: float

    log_abs_emp: float
    log_abs_lift: float


def _class_of_n(f: dict[int, int]) -> str:
    if not f:
        return "other"
    if len(f) == 1:
        e = int(next(iter(f.values())))
        return "prime" if e == 1 else "primepower"
    if len(f) == 2 and all(int(e) == 1 for e in f.values()):
        return "squarefree_pair"
    if len(f) == 2:
        return "mixed"
    return "other"


def _median(xs: list[float]) -> float:
    arr = np.asarray(xs, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def _median_abs(xs: list[float]) -> float:
    return _median([abs(float(x)) for x in xs if math.isfinite(float(x))])


def _error_medians(rows: list[NRow], *, b: float, c: float, min_abs: float, holdout_only: bool = False) -> tuple[float, float, float]:
    # Errors are |log|emp| - (log|lift| - c - b*Omega)|
    sf_err: list[float] = []
    pp_err: list[float] = []
    mix_err: list[float] = []

    for r in rows:
        if holdout_only and r.primeset_tag != "holdout":
            continue
        if (min_abs > 0) and (abs(r.emp_A) < min_abs or abs(r.lift_a) < min_abs):
            # keep consistent with the rest of the pipeline: floor-based logs are allowed,
            # but tiny raw magnitudes tend to trivialize the fit; skip them.
            continue

        adj_log_lift = float(r.log_abs_lift) - float(c) - float(b) * float(r.Omega_total)
        err = float(abs(float(r.log_abs_emp) - float(adj_log_lift)))

        if r.cls == "squarefree_pair":
            sf_err.append(err)
        elif r.cls == "primepower":
            pp_err.append(err)
        elif r.cls == "mixed":
            mix_err.append(err)

    return (_median(sf_err), _median(pp_err), _median(mix_err))


def _objective(
    rows: list[NRow],
    *,
    b: float,
    c: float,
    min_abs: float,
    lam_pp: float,
    mu_mix: float,
    holdout_only: bool = False,
) -> float:
    E_sf, E_pp, E_mix = _error_medians(rows, b=b, c=c, min_abs=min_abs, holdout_only=holdout_only)
    # By default we treat missing classes as infinite (should not happen with reasonable sets).
    if not math.isfinite(E_sf):
        return float("inf")
    L = float(E_sf)
    if lam_pp != 0.0:
        if not math.isfinite(E_pp):
            return float("inf")
        L += float(lam_pp) * float(E_pp)
    if mu_mix != 0.0:
        if not math.isfinite(E_mix):
            return float("inf")
        L += float(mu_mix) * float(E_mix)
    return float(L)


def _best_c_candidates(rows: list[NRow], *, b: float, min_abs: float) -> list[float]:
    # Candidate c values near per-class and global medians of residuals.
    # residual r := log_emp - log_lift + b*Omega, and we want to add c to residual (via -c in log_lift)
    # so that r + c is centered around 0.
    by_class: dict[str, list[float]] = {"squarefree_pair": [], "primepower": [], "mixed": [], "all": []}

    for r in rows:
        if (min_abs > 0) and (abs(r.emp_A) < min_abs or abs(r.lift_a) < min_abs):
            continue
        resid = float(r.log_abs_emp - r.log_abs_lift + float(b) * float(r.Omega_total))
        by_class["all"].append(resid)
        if r.cls in by_class:
            by_class[r.cls].append(resid)

    cands: list[float] = [0.0]
    for key in ["squarefree_pair", "primepower", "mixed", "all"]:
        med = _median(by_class[key])
        if math.isfinite(med):
            cands.append(float(-med))

    # de-dup with mild rounding
    out: list[float] = []
    seen: set[float] = set()
    for c in cands:
        cr = float(round(float(c), 6))
        if cr in seen:
            continue
        seen.add(cr)
        out.append(float(c))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Fit minimal log-bias corrections for the ghost/Euler lifted model under the frozen 'p' coordinate.\n\n"
            "Frozen arithmetic: X-semigroup composition, V(n)=log|det(I-S_n)|, empirical A_emp via Dirichlet primitive.\n"
            "Lift: dyadic ghost g_m = V(2^m) (aggregate_as_ghost) -> Euler-transform coefficients A_k -> multiplicative a_n.\n\n"
            "Canonical repo note: the frozen arithmetic backend is the two-tier ghost-depth family `step_m_eq2_and_m_ge3`, which yields the downstream `beta23_plus_c` fit and the frozen beta23 coefficient table. "
            "The generic `c_only`, `bOmega_only`, and `c_plus_bOmega` models remain useful as exploratory ablations, but they are not the canonical backend.\n\n"
            "We fit the smallest correction families on a training prime set (default <=17):\n"
            "  (1) log|a*_n| = log|a_n| - c\n"
            "  (2) log|a*_n| = log|a_n| - c - b*Omega_total(n)\n\n"
            "Objective is a joint closure loss: L = E_sf + lambda*E_pp + mu*E_mix, where each E is a median abs log-error.\n"
            "Then we validate on a larger prime set (default <=23) without refitting and also report holdout-only errors."
        )
    )

    ap.add_argument("--ps_train", default="2,3,5,7,11,13,17")
    ap.add_argument("--ps_eval", default="2,3,5,7,11,13,17,19,23")
    ap.add_argument(
        "--holdout_ps",
        default="",
        help=(
            "Comma-separated holdout primes used only for reporting holdout errors. "
            "If empty, defaults to ps_eval \ ps_train. Example: --holdout_ps 19,23"
        ),
    )
    ap.add_argument("--us", default="0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24")

    ap.add_argument("--min_gen_devI_median", type=float, default=0.8)
    ap.add_argument("--min_abs", type=float, default=1e-12)

    ap.add_argument("--dyadic_k_max", type=int, default=12)
    ap.add_argument("--max_power", type=int, default=3)

    ap.add_argument("--lambda_pp", type=float, default=1.0)
    ap.add_argument("--mu_mix", type=float, default=0.0)

    ap.add_argument("--b_min", type=float, default=-2.0)
    ap.add_argument("--b_max", type=float, default=2.0)
    ap.add_argument("--b_step", type=float, default=0.01)

    ap.add_argument("--boundary", default="0,3")
    ap.add_argument("--schur_sign", choices=["-", "+"], default="+")
    ap.add_argument("--scattering", choices=["lambda_pm_i", "i_pm_lambda"], default="i_pm_lambda")

    ap.add_argument("--sharp", choices=["transpose", "conj_transpose"], default="conj_transpose")
    ap.add_argument("--X_mode", choices=["raw", "det1"], default="det1")
    ap.add_argument("--X_gamma", type=float, default=1.0)

    ap.add_argument("--out_csv", default="out/ghostlift_closure_biasfit_window016_024.csv")

    ap.add_argument(
        "--ghost_beta_mode",
        choices=["none", "step_m_ge2", "linear_m_minus1", "step_m_eq2_and_m_ge3"],
        default="none",
        help=(
            "Optional ghost-depth correction family applied before Euler transform: "
            "g*_m = g_m - beta*h(m), where h(m)=1_{m>=2} for step_m_ge2 or h(m)=(m-1) for linear_m_minus1. "
            "For step_m_eq2_and_m_ge3, we use g*_m = g_m - beta2*1_{m=2} - beta3*1_{m>=3}. "
            "This keeps p/ghost_mode/observable/geometry frozen while testing tower-depth distortion corrections. "
            "The canonical frozen-backend family is step_m_eq2_and_m_ge3; the other choices are exploratory ablations."
        ),
    )
    ap.add_argument("--beta_min", type=float, default=-2.0)
    ap.add_argument("--beta_max", type=float, default=2.0)
    ap.add_argument("--beta_step", type=float, default=0.02)

    ap.add_argument(
        "--beta2_fixed",
        type=float,
        default=float("nan"),
        help=(
            "For step_m_eq2_and_m_ge3: if set, forces beta2 constant across the whole u-window. "
            "If both beta2_fixed and beta3_fixed are set, no beta grid search is performed."
        ),
    )
    ap.add_argument(
        "--beta3_fixed",
        type=float,
        default=float("nan"),
        help=(
            "For step_m_eq2_and_m_ge3: if set, forces beta3 constant across the whole u-window."
        ),
    )

    ap.add_argument(
        "--beta_fixed",
        type=float,
        default=float("nan"),
        help=(
            "If set to a finite number, forces a constant beta across the whole u-window (no per-u refit of beta). "
            "Use with --ghost_beta_mode != none. This is mainly for exploratory ablations rather than the canonical frozen beta23 path. "
            "Example: --beta_fixed 0.82"
        ),
    )
    ap.add_argument(
        "--c_fixed",
        type=float,
        default=float("nan"),
        help=(
            "If set to a finite number, forces a constant c across the whole u-window (no per-u refit of c). "
            "Example: --c_fixed -0.82"
        ),
    )

    args = ap.parse_args()

    ps_train = sorted(set(int(p) for p in _parse_int_csv(str(args.ps_train))))
    ps_eval = sorted(set(int(p) for p in _parse_int_csv(str(args.ps_eval))))
    if not ps_train:
        raise SystemExit("--ps_train must be non-empty")
    if not ps_eval:
        raise SystemExit("--ps_eval must be non-empty")
    if not set(ps_train).issubset(set(ps_eval)):
        raise SystemExit("--ps_train must be a subset of --ps_eval")

    us = [float(u) for u in _parse_float_csv(str(args.us))]
    if not us:
        raise SystemExit("--us must be non-empty")

    dyadic_k_max = int(args.dyadic_k_max)
    if dyadic_k_max < 3:
        raise SystemExit("--dyadic_k_max must be >=3")

    max_power = int(args.max_power)
    if max_power < 1:
        raise SystemExit("--max_power must be >=1")

    min_abs = max(0.0, float(args.min_abs))

    boundary_parts = sim._parse_int_list(str(args.boundary))
    if len(boundary_parts) != 2:
        raise SystemExit("--boundary must be i,j")
    boundary = (int(boundary_parts[0]), int(boundary_parts[1]))

    lam_pp = float(args.lambda_pp)
    mu_mix = float(args.mu_mix)

    b_min = float(args.b_min)
    b_max = float(args.b_max)
    b_step = float(args.b_step)
    if b_step <= 0:
        raise SystemExit("--b_step must be >0")

    b_grid: list[float] = []
    k = 0
    while True:
        b = b_min + k * b_step
        if b > b_max + 1e-12:
            break
        b_grid.append(float(b))
        k += 1

    I2 = np.eye(2, dtype=np.complex128)

    holdout_ps = sorted(set(ps_eval) - set(ps_train))

    holdout_override = sorted(set(int(p) for p in _parse_int_csv(str(args.holdout_ps))))
    if holdout_override:
        holdout_ps = holdout_override

    newprime_ps = sorted(set(ps_eval) - set(ps_train) - set(holdout_ps))

    beta_mode = str(args.ghost_beta_mode)
    beta_min = float(args.beta_min)
    beta_max = float(args.beta_max)
    beta_step = float(args.beta_step)
    if beta_step <= 0:
        raise SystemExit("--beta_step must be >0")

    beta_grid: list[float] = [0.0]
    if beta_mode != "none":
        beta_fixed = float(args.beta_fixed)
        if beta_mode == "step_m_eq2_and_m_ge3":
            # beta_grid is still used as the shared 1D grid for beta2/beta3 (we form a cartesian product below).
            beta_grid = []
            k_beta = 0
            while True:
                beta = beta_min + k_beta * beta_step
                if beta > beta_max + 1e-12:
                    break
                beta_grid.append(float(beta))
                k_beta += 1
        else:
            if math.isfinite(beta_fixed):
                beta_grid = [float(beta_fixed)]
            else:
                beta_grid = []
                k_beta = 0
                while True:
                    beta = beta_min + k_beta * beta_step
                    if beta > beta_max + 1e-12:
                        break
                    beta_grid.append(float(beta))
                    k_beta += 1

    out_rows: list[FitResult] = []

    for u in us:
        v = -float(u)

        # Cache prime generators X_{p,1} for this u.
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
                    p_mode="p",  # frozen
                )
                X_cache[p] = np.asarray(X, dtype=np.complex128)
            return X_cache[p]

        devs = [float(np.linalg.norm(X_p1(int(p)) - I2, ord="fro")) for p in ps_eval]
        devs_arr = np.asarray(devs, dtype=float)
        devs_arr = devs_arr[np.isfinite(devs_arr)]
        gen_dev_med = float(np.median(devs_arr)) if devs_arr.size else float("nan")

        if (not math.isfinite(gen_dev_med)) or gen_dev_med < float(args.min_gen_devI_median):
            # Gate out: still record NaNs so the sweep table is aligned.
            out_rows.append(
                FitResult(
                    u=float(u),
                    model="GATED",
                    b=float("nan"),
                    c=float("nan"),
                    train_E_sf=float("nan"),
                    train_E_pp=float("nan"),
                    train_E_mix=float("nan"),
                    train_L=float("nan"),
                    eval_E_sf=float("nan"),
                    eval_E_pp=float("nan"),
                    eval_E_mix=float("nan"),
                    eval_L=float("nan"),
                    holdout_E_sf=float("nan"),
                    holdout_E_pp=float("nan"),
                    holdout_E_mix=float("nan"),
                    holdout_L=float("nan"),
                )
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

        # Dyadic ghost corridor g_m = V(2^m) (aggregate_as_ghost is frozen).
        g = [0.0] * (dyadic_k_max + 1)
        for m in range(1, dyadic_k_max + 1):
            g[m] = float(V_of_n(int(2**m)))

        def _ghost_h(m: int) -> float:
            if beta_mode == "step_m_ge2":
                return 1.0 if int(m) >= 2 else 0.0
            if beta_mode == "linear_m_minus1":
                return float(int(m) - 1)
            return 0.0

        def _Acoeff_from_betas(beta: float, beta2: float, beta3: float) -> list[float]:
            """Return Acoeff[0..max_power] from corrected g*_m.

            We only need g*_1..g*_max_power because Euler-transform coefficients up to max_power depend only on that prefix.
            This makes two-tier grid search feasible.
            """

            def gstar_j(j: int) -> float:
                j = int(j)
                if beta_mode == "step_m_eq2_and_m_ge3":
                    if j == 2:
                        return float(g[j]) - float(beta2)
                    if j >= 3:
                        return float(g[j]) - float(beta3)
                    return float(g[j])
                # 1-parameter families
                return float(g[j]) - float(beta) * float(_ghost_h(j))

            # Explicit formulas for A1..A3; then fall back to recursion if max_power>3.
            Acoeff = [0.0] * (max_power + 1)
            Acoeff[0] = 1.0
            if max_power >= 1:
                g1 = float(gstar_j(1))
                Acoeff[1] = float(g1)
            if max_power >= 2:
                g1 = float(Acoeff[1])
                g2 = float(gstar_j(2))
                Acoeff[2] = float((g1 * g1 + g2) / 2.0)
            if max_power >= 3:
                g1 = float(Acoeff[1])
                g2 = float(gstar_j(2))
                g3 = float(gstar_j(3))
                Acoeff[3] = float((g1 * Acoeff[2] + g2 * Acoeff[1] + g3) / 3.0)

            if max_power > 3:
                # General recursion using corrected prefix g*_j.
                gstar = [0.0] * (max_power + 1)
                for j in range(1, max_power + 1):
                    gstar[j] = float(gstar_j(j))
                for k_pow in range(1, max_power + 1):
                    if k_pow <= 3:
                        continue
                    acc = 0.0
                    for j in range(1, k_pow + 1):
                        acc += float(gstar[j]) * float(Acoeff[k_pow - j])
                    Acoeff[k_pow] = float(acc) / float(k_pow)

            return Acoeff

        def _a_lift_from_Acoeff(n: int, Acoeff: list[float]) -> float:
            f = _prime_factorization(int(n))
            out = 1.0
            for _p, e in f.items():
                e = int(e)
                if e > max_power:
                    return float("nan")
                out *= float(Acoeff[e])
            return float(out)

        # Enumerate n for evaluation prime set.
        ns_eval: set[int] = set()
        for p in ps_eval:
            ns_eval.add(int(p))
        for p in ps_eval:
            pk = 1
            for k_pow in range(1, max_power + 1):
                pk *= int(p)
                ns_eval.add(int(pk))
        for p, q in itertools.combinations(ps_eval, 2):
            ns_eval.add(int(p * q))
        for p, q in itertools.combinations(ps_eval, 2):
            for a in range(1, max_power + 1):
                for b in range(1, max_power + 1):
                    ns_eval.add(int((p**a) * (q**b)))

        # Ensure divisors needed for A_emp.
        all_needed: set[int] = set([1])
        for n in ns_eval:
            all_needed.add(int(n))
            for d in divisors(int(n)):
                all_needed.add(int(d))

        V_map: dict[int, float] = {1: 0.0}
        for n in sorted(all_needed):
            if n == 1:
                continue
            V_map[int(n)] = float(V_of_n(int(n)))

        def A_emp(n: int) -> float:
            acc = 0.0
            for d in divisors(int(n)):
                mu = int(mobius_mu(int(d)))
                if mu == 0:
                    continue
                acc += float(mu) * float(V_map[int(n // d)])
            return float(acc)

        def Omega_total(n: int) -> int:
            f = _prime_factorization(int(n))
            return int(sum(int(e) for e in f.values()))

        def primeset_tag(n: int) -> str:
            f = _prime_factorization(int(n))
            ps = set(int(p) for p in f.keys())
            if ps.issubset(set(ps_train)):
                return "train"
            if holdout_ps and (len(ps.intersection(set(holdout_ps))) > 0):
                return "holdout"
            if newprime_ps and (len(ps.intersection(set(newprime_ps))) > 0):
                return "newprime"
            return "eval_only"

        # Build per-n rows.
        def build_rows_for_beta(beta: float) -> tuple[list[NRow], list[NRow]]:
            Acoeff = _Acoeff_from_betas(float(beta), float(beta), float(beta))
            rows_eval: list[NRow] = []
            for n in sorted(ns_eval):
                f = _prime_factorization(int(n))
                cls = _class_of_n(f)
                if cls == "other":
                    continue

                emp = float(A_emp(int(n)))
                lift = float(_a_lift_from_Acoeff(int(n), Acoeff))

                log_emp = _safe_log_abs(emp, floor=max(min_abs, 1e-300))
                log_lift = _safe_log_abs(lift, floor=max(min_abs, 1e-300))

                rows_eval.append(
                    NRow(
                        n=int(n),
                        cls=str(cls),
                        Omega_total=int(Omega_total(int(n))),
                        primeset_tag=str(primeset_tag(int(n))),
                        emp_A=float(emp),
                        lift_a=float(lift),
                        log_abs_emp=float(log_emp),
                        log_abs_lift=float(log_lift),
                    )
                )
            rows_train = [r for r in rows_eval if r.primeset_tag == "train"]
            return rows_eval, rows_train

        # Baseline rows: beta=0.
        rows_eval0, rows_train0 = build_rows_for_beta(0.0)

        # Fit models.
        def record(model: str, *, beta: float, rows_eval: list[NRow], rows_train: list[NRow], b: float, c: float) -> None:
            train_E_sf, train_E_pp, train_E_mix = _error_medians(rows_train, b=b, c=c, min_abs=min_abs)
            train_L = _objective(rows_train, b=b, c=c, min_abs=min_abs, lam_pp=lam_pp, mu_mix=mu_mix)

            eval_E_sf, eval_E_pp, eval_E_mix = _error_medians(rows_eval, b=b, c=c, min_abs=min_abs)
            eval_L = _objective(rows_eval, b=b, c=c, min_abs=min_abs, lam_pp=lam_pp, mu_mix=mu_mix)

            hold_E_sf, hold_E_pp, hold_E_mix = _error_medians(rows_eval, b=b, c=c, min_abs=min_abs, holdout_only=True)
            hold_L = _objective(rows_eval, b=b, c=c, min_abs=min_abs, lam_pp=lam_pp, mu_mix=mu_mix, holdout_only=True)

            new_E_sf, new_E_pp, new_E_mix = _error_medians(rows_eval, b=b, c=c, min_abs=min_abs, holdout_only=False)
            # Compute newprime-only by temporarily filtering rows.
            newprime_rows = [r for r in rows_eval if r.primeset_tag == "newprime"]
            newp_E_sf, newp_E_pp, newp_E_mix = _error_medians(newprime_rows, b=b, c=c, min_abs=min_abs)
            newp_L = _objective(newprime_rows, b=b, c=c, min_abs=min_abs, lam_pp=lam_pp, mu_mix=mu_mix)

            out_rows.append(
                FitResult(
                    u=float(u),
                    model=str(model),

                    ghost_beta_mode=str(beta_mode),
                    beta=float(beta),
                    beta2=float(beta),
                    beta3=float(beta),
                    b=float(b),
                    c=float(c),
                    train_E_sf=float(train_E_sf),
                    train_E_pp=float(train_E_pp),
                    train_E_mix=float(train_E_mix),
                    train_L=float(train_L),
                    eval_E_sf=float(eval_E_sf),
                    eval_E_pp=float(eval_E_pp),
                    eval_E_mix=float(eval_E_mix),
                    eval_L=float(eval_L),
                    holdout_E_sf=float(hold_E_sf),
                    holdout_E_pp=float(hold_E_pp),
                    holdout_E_mix=float(hold_E_mix),
                    holdout_L=float(hold_L),

                    newprime_E_sf=float(newp_E_sf),
                    newprime_E_pp=float(newp_E_pp),
                    newprime_E_mix=float(newp_E_mix),
                    newprime_L=float(newp_L),
                )
            )

        # Baseline: no correction.
        record("none", beta=0.0, rows_eval=rows_eval0, rows_train=rows_train0, b=0.0, c=0.0)

        # Constant correction only: pick best among median-based candidates.
        best_c = 0.0
        best_L = float("inf")
        c_fixed = float(args.c_fixed)
        if math.isfinite(c_fixed):
            best_c = float(c_fixed)
        else:
            for c in _best_c_candidates(rows_train0, b=0.0, min_abs=min_abs):
                L = _objective(rows_train0, b=0.0, c=float(c), min_abs=min_abs, lam_pp=lam_pp, mu_mix=mu_mix)
                if L < best_L:
                    best_L = float(L)
                    best_c = float(c)
        record("c_only", beta=0.0, rows_eval=rows_eval0, rows_train=rows_train0, b=0.0, c=float(best_c))

        # b*Omega with c: grid b, then choose best c among candidates.
        best_b = 0.0
        best_c2 = 0.0
        best_L2 = float("inf")
        for b in b_grid:
            cands = _best_c_candidates(rows_train0, b=float(b), min_abs=min_abs)
            for c in cands:
                if math.isfinite(c_fixed) and (abs(float(c) - float(c_fixed)) > 1e-9):
                    continue
                L = _objective(rows_train0, b=float(b), c=float(c), min_abs=min_abs, lam_pp=lam_pp, mu_mix=mu_mix)
                if L < best_L2:
                    best_L2 = float(L)
                    best_b = float(b)
                    best_c2 = float(c)
        record("c_plus_bOmega", beta=0.0, rows_eval=rows_eval0, rows_train=rows_train0, b=float(best_b), c=float(best_c2))

        # Optional: b-only without c (rarely optimal, but useful sanity check).
        best_b3 = 0.0
        best_L3 = float("inf")
        for b in b_grid:
            L = _objective(rows_train0, b=float(b), c=0.0, min_abs=min_abs, lam_pp=lam_pp, mu_mix=mu_mix)
            if L < best_L3:
                best_L3 = float(L)
                best_b3 = float(b)
        record("bOmega_only", beta=0.0, rows_eval=rows_eval0, rows_train=rows_train0, b=float(best_b3), c=0.0)

        # Optional Experiment 2: ghost-depth correction beta (plus optional constant c).
        if beta_mode != "none":
            if beta_mode != "step_m_eq2_and_m_ge3":
                best_beta0 = 0.0
                best_L_beta0 = float("inf")
                best_beta_c = 0.0
                best_c_beta = 0.0
                best_L_beta_c = float("inf")

                for beta in beta_grid:
                    rows_eval_b, rows_train_b = build_rows_for_beta(float(beta))

                    # beta-only (no c)
                    L0 = _objective(rows_train_b, b=0.0, c=0.0, min_abs=min_abs, lam_pp=lam_pp, mu_mix=mu_mix)
                    if L0 < best_L_beta0:
                        best_L_beta0 = float(L0)
                        best_beta0 = float(beta)

                    # beta + best c (b=0)
                    local_best_c = 0.0
                    local_best_L = float("inf")
                    if math.isfinite(c_fixed):
                        local_best_c = float(c_fixed)
                        local_best_L = _objective(rows_train_b, b=0.0, c=float(c_fixed), min_abs=min_abs, lam_pp=lam_pp, mu_mix=mu_mix)
                    else:
                        for c in _best_c_candidates(rows_train_b, b=0.0, min_abs=min_abs):
                            L = _objective(rows_train_b, b=0.0, c=float(c), min_abs=min_abs, lam_pp=lam_pp, mu_mix=mu_mix)
                            if L < local_best_L:
                                local_best_L = float(L)
                                local_best_c = float(c)
                    if local_best_L < best_L_beta_c:
                        best_L_beta_c = float(local_best_L)
                        best_beta_c = float(beta)
                        best_c_beta = float(local_best_c)

                rows_eval_beta0, rows_train_beta0 = build_rows_for_beta(float(best_beta0))
                record("beta_only", beta=float(best_beta0), rows_eval=rows_eval_beta0, rows_train=rows_train_beta0, b=0.0, c=0.0)

                rows_eval_betac, rows_train_betac = build_rows_for_beta(float(best_beta_c))
                record(
                    "beta_plus_c",
                    beta=float(best_beta_c),
                    rows_eval=rows_eval_betac,
                    rows_train=rows_train_betac,
                    b=0.0,
                    c=float(best_c_beta),
                )
            else:
                # Two-tier: beta2 at m=2 and beta3 for m>=3. Fit (beta2,beta3) on training set.
                beta2_fixed = float(args.beta2_fixed)
                beta3_fixed = float(args.beta3_fixed)

                if math.isfinite(beta2_fixed):
                    beta2_vals = [float(beta2_fixed)]
                else:
                    beta2_vals = list(beta_grid)

                if math.isfinite(beta3_fixed):
                    beta3_vals = [float(beta3_fixed)]
                else:
                    beta3_vals = list(beta_grid)

                # Precompute empirical rows once (log_abs_emp, Omega, cls/tag) and just update log_abs_lift per candidate.
                # We reuse rows_eval0/rows_train0 but overwrite the lift-related fields on copies.

                best_beta2 = 0.0
                best_beta3 = 0.0
                best_c_beta = 0.0
                best_L = float("inf")

                # Cache factorization exponents per row: for max_power<=3, lift log depends only on counts of e=1,2,3.
                exp_counts: dict[int, tuple[int, int, int]] = {}
                for r in rows_eval0:
                    f = _prime_factorization(int(r.n))
                    c1 = 0
                    c2 = 0
                    c3 = 0
                    for _p, e in f.items():
                        e = int(e)
                        if e == 1:
                            c1 += 1
                        elif e == 2:
                            c2 += 1
                        elif e == 3:
                            c3 += 1
                    exp_counts[int(r.n)] = (c1, c2, c3)

                def rows_for(beta2: float, beta3: float) -> tuple[list[NRow], list[NRow]]:
                    Acoeff = _Acoeff_from_betas(0.0, float(beta2), float(beta3))
                    # Compute per-exponent logs once.
                    floor = max(min_abs, 1e-300)
                    logA = {
                        1: _safe_log_abs(float(Acoeff[1]), floor=floor),
                        2: _safe_log_abs(float(Acoeff[2]), floor=floor),
                        3: _safe_log_abs(float(Acoeff[3]), floor=floor),
                    }
                    rows_eval: list[NRow] = []
                    for r in rows_eval0:
                        c1, c2, c3 = exp_counts[int(r.n)]
                        log_lift = float(c1) * float(logA[1]) + float(c2) * float(logA[2]) + float(c3) * float(logA[3])
                        # Keep raw lift_a for min_abs filtering: approximate from exp counts.
                        lift_a = float((Acoeff[1] ** c1) * (Acoeff[2] ** c2) * (Acoeff[3] ** c3))
                        rows_eval.append(
                            NRow(
                                n=int(r.n),
                                cls=str(r.cls),
                                Omega_total=int(r.Omega_total),
                                primeset_tag=str(r.primeset_tag),
                                emp_A=float(r.emp_A),
                                lift_a=float(lift_a),
                                log_abs_emp=float(r.log_abs_emp),
                                log_abs_lift=float(log_lift),
                            )
                        )
                    rows_train = [rr for rr in rows_eval if rr.primeset_tag == "train"]
                    return rows_eval, rows_train

                for b2 in beta2_vals:
                    for b3 in beta3_vals:
                        rows_eval_b, rows_train_b = rows_for(float(b2), float(b3))

                        # Fit c (or use fixed) at b=0.
                        if math.isfinite(c_fixed):
                            cand_cs = [float(c_fixed)]
                        else:
                            cand_cs = _best_c_candidates(rows_train_b, b=0.0, min_abs=min_abs)
                        for c in cand_cs:
                            L = _objective(rows_train_b, b=0.0, c=float(c), min_abs=min_abs, lam_pp=lam_pp, mu_mix=mu_mix)
                            if L < best_L:
                                best_L = float(L)
                                best_beta2 = float(b2)
                                best_beta3 = float(b3)
                                best_c_beta = float(c)

                rows_eval_best, rows_train_best = rows_for(float(best_beta2), float(best_beta3))

                # Custom record with beta2/beta3.
                def record_two_tier(model: str, beta2: float, beta3: float, c: float) -> None:
                    train_E_sf, train_E_pp, train_E_mix = _error_medians(rows_train_best, b=0.0, c=c, min_abs=min_abs)
                    train_L = _objective(rows_train_best, b=0.0, c=c, min_abs=min_abs, lam_pp=lam_pp, mu_mix=mu_mix)

                    eval_E_sf, eval_E_pp, eval_E_mix = _error_medians(rows_eval_best, b=0.0, c=c, min_abs=min_abs)
                    eval_L = _objective(rows_eval_best, b=0.0, c=c, min_abs=min_abs, lam_pp=lam_pp, mu_mix=mu_mix)

                    hold_E_sf, hold_E_pp, hold_E_mix = _error_medians(rows_eval_best, b=0.0, c=c, min_abs=min_abs, holdout_only=True)
                    hold_L = _objective(rows_eval_best, b=0.0, c=c, min_abs=min_abs, lam_pp=lam_pp, mu_mix=mu_mix, holdout_only=True)

                    newprime_rows = [r for r in rows_eval_best if r.primeset_tag == "newprime"]
                    newp_E_sf, newp_E_pp, newp_E_mix = _error_medians(newprime_rows, b=0.0, c=c, min_abs=min_abs)
                    newp_L = _objective(newprime_rows, b=0.0, c=c, min_abs=min_abs, lam_pp=lam_pp, mu_mix=mu_mix)

                    out_rows.append(
                        FitResult(
                            u=float(u),
                            model=str(model),
                            ghost_beta_mode=str(beta_mode),
                            beta=float(beta3),
                            beta2=float(beta2),
                            beta3=float(beta3),
                            b=0.0,
                            c=float(c),
                            train_E_sf=float(train_E_sf),
                            train_E_pp=float(train_E_pp),
                            train_E_mix=float(train_E_mix),
                            train_L=float(train_L),
                            eval_E_sf=float(eval_E_sf),
                            eval_E_pp=float(eval_E_pp),
                            eval_E_mix=float(eval_E_mix),
                            eval_L=float(eval_L),
                            holdout_E_sf=float(hold_E_sf),
                            holdout_E_pp=float(hold_E_pp),
                            holdout_E_mix=float(hold_E_mix),
                            holdout_L=float(hold_L),
                            newprime_E_sf=float(newp_E_sf),
                            newprime_E_pp=float(newp_E_pp),
                            newprime_E_mix=float(newp_E_mix),
                            newprime_L=float(newp_L),
                        )
                    )

                record_two_tier("beta23_plus_c", beta2=float(best_beta2), beta3=float(best_beta3), c=float(best_c_beta))

        print(
            f"u={u:.2f} gate_devI_med={gen_dev_med:.3f} "
            f"best(c_only) c={best_c:+.3f} | best(c+bΩ) b={best_b:+.3f} c={best_c2:+.3f}"
        )

    out_path = Path(str(args.out_csv))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.__dict__ for r in out_rows]).to_csv(str(out_path), index=False)
    print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
