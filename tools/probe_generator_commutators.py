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


def _fmt(x: float) -> str:
    if not math.isfinite(float(x)):
        return "nan"
    return f"{float(x):.4g}"


@dataclass(frozen=True)
class PairRow:
    p: int
    q: int
    Xp_fro: float
    Xq_fro: float
    Xp_devI_fro: float
    Xq_devI_fro: float
    add_comm_fro: float
    add_comm_rel: float
    add_comm_scaled: float
    grp_comm_fro: float
    grp_comm_scaled: float
    grp_comm_logdet_abs: float


@dataclass(frozen=True)
class PrimeRow:
    p: int
    X_fro: float
    X_devI_fro: float
    X_logdet_abs: float


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Probe noncommutativity of the prime generators X_{p,1} in GL2.\n\n"
            "For each prime pair (p,q), compute:\n"
            "  additive commutator:   [X_p, X_q]_+ = XpXq - XqXp\n"
            "  group commutator:      [X_p, X_q]   = XpXqXp^{-1}Xq^{-1}\n"
            "and report Frobenius norms and simple relative/logdet diagnostics.\n\n"
            "This is a structural gate: if composite-n log-multiplicativity is emerging, "
            "we expect a preferred p_mode to minimize commutator burden (approximately commuting generators)."
        )
    )

    ap.add_argument("--ps", default="2,3,5,7,11,13,17", help="Comma-list of primes to include")
    ap.add_argument("--u", type=float, default=0.2)

    ap.add_argument("--sharp", choices=["transpose", "conj_transpose"], default="conj_transpose")
    ap.add_argument("--X_mode", choices=["raw", "det1"], default="det1")
    ap.add_argument("--X_gamma", type=float, default=1.0)
    ap.add_argument(
        "--p_mode",
        choices=["p", "logp", "p1_over_p", "p_over_p1", "invp", "p_minus1_over_p"],
        default="p",
    )

    ap.add_argument("--out_csv", default="", help="Optional output CSV")
    ap.add_argument("--out_prime_csv", default="", help="Optional per-prime generator CSV")
    args = ap.parse_args()

    ps = _parse_int_csv(str(args.ps))
    if len(ps) < 2:
        raise SystemExit("--ps must contain at least two primes")
    ps = sorted(set(int(p) for p in ps))

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

    rows: list[PairRow] = []
    prime_rows: list[PrimeRow] = []

    I2 = np.eye(2, dtype=np.complex128)

    # Prime-level stats.
    for p in ps:
        Xp = X_p1(int(p))
        X_fro = float(np.linalg.norm(Xp, ord="fro"))
        X_devI_fro = float(np.linalg.norm(Xp - I2, ord="fro"))
        det = complex(np.linalg.det(Xp))
        X_logdet_abs = float(math.log(max(1e-300, abs(det))))
        prime_rows.append(
            PrimeRow(
                p=int(p),
                X_fro=float(X_fro),
                X_devI_fro=float(X_devI_fro),
                X_logdet_abs=float(X_logdet_abs),
            )
        )

    for i, p in enumerate(ps):
        Xp = X_p1(int(p))
        for q in ps[i + 1 :]:
            Xq = X_p1(int(q))

            Xp_fro = float(np.linalg.norm(Xp, ord="fro"))
            Xq_fro = float(np.linalg.norm(Xq, ord="fro"))
            Xp_devI_fro = float(np.linalg.norm(Xp - I2, ord="fro"))
            Xq_devI_fro = float(np.linalg.norm(Xq - I2, ord="fro"))

            XpXq = (Xp @ Xq).astype(np.complex128)
            XqXp = (Xq @ Xp).astype(np.complex128)

            add_comm = (XpXq - XqXp).astype(np.complex128)
            add_comm_fro = float(np.linalg.norm(add_comm, ord="fro"))

            denom = float(max(1e-12, np.linalg.norm(Xp, ord="fro") * np.linalg.norm(Xq, ord="fro")))
            add_comm_rel = float(add_comm_fro / denom)

            dev_denom = float(max(1e-12, Xp_devI_fro * Xq_devI_fro))
            add_comm_scaled = float(add_comm_fro / dev_denom)

            # Group commutator: Xp Xq Xp^{-1} Xq^{-1}.
            # With X_mode=det1 this should be well-conditioned, but keep it robust.
            try:
                Xp_inv = np.linalg.inv(Xp)
                Xq_inv = np.linalg.inv(Xq)
                grp_comm = (Xp @ Xq @ Xp_inv @ Xq_inv).astype(np.complex128)
                grp_comm_fro = float(np.linalg.norm(grp_comm - np.eye(2, dtype=np.complex128), ord="fro"))
                grp_comm_scaled = float(grp_comm_fro / dev_denom)
                grp_det = complex(np.linalg.det(grp_comm))
                grp_comm_logdet_abs = float(math.log(max(1e-300, abs(grp_det))))
            except np.linalg.LinAlgError:
                grp_comm_fro = float("nan")
                grp_comm_scaled = float("nan")
                grp_comm_logdet_abs = float("nan")

            rows.append(
                PairRow(
                    p=int(p),
                    q=int(q),
                    Xp_fro=float(Xp_fro),
                    Xq_fro=float(Xq_fro),
                    Xp_devI_fro=float(Xp_devI_fro),
                    Xq_devI_fro=float(Xq_devI_fro),
                    add_comm_fro=float(add_comm_fro),
                    add_comm_rel=float(add_comm_rel),
                    add_comm_scaled=float(add_comm_scaled),
                    grp_comm_fro=float(grp_comm_fro),
                    grp_comm_scaled=float(grp_comm_scaled),
                    grp_comm_logdet_abs=float(grp_comm_logdet_abs),
                )
            )

    df = pd.DataFrame([r.__dict__ for r in rows]).sort_values(["p", "q"]).reset_index(drop=True)
    dfp = pd.DataFrame([r.__dict__ for r in prime_rows]).sort_values(["p"]).reset_index(drop=True)

    def summarize(col: str) -> tuple[float, float]:
        a = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return float("nan"), float("nan")
        return float(np.median(a)), float(np.max(a))

    print("generator commutator probe")
    print(f"  u={u:g} p_mode={args.p_mode} X_mode={args.X_mode} gamma={float(args.X_gamma):g} sharp={args.sharp}")
    print(f"  ps={','.join(str(p) for p in ps)}")
    print(f"  n_pairs={int(df.shape[0])}")

    if not dfp.empty:
        a = dfp["X_devI_fro"].to_numpy(dtype=float)
        a = a[np.isfinite(a)]
        if a.size:
            print("  X devI fro median:", _fmt(float(np.median(a))), "max:", _fmt(float(np.max(a))))
        a = dfp["X_fro"].to_numpy(dtype=float)
        a = a[np.isfinite(a)]
        if a.size:
            print("  X fro median:", _fmt(float(np.median(a))), "max:", _fmt(float(np.max(a))))

    med, mx = summarize("add_comm_fro")
    print("  add_comm_fro median:", _fmt(med), "max:", _fmt(mx))
    med, mx = summarize("add_comm_rel")
    print("  add_comm_rel median:", _fmt(med), "max:", _fmt(mx))
    med, mx = summarize("add_comm_scaled")
    print("  add_comm_scaled median:", _fmt(med), "max:", _fmt(mx))
    med, mx = summarize("grp_comm_fro")
    print("  grp_comm_fro median:", _fmt(med), "max:", _fmt(mx))
    med, mx = summarize("grp_comm_scaled")
    print("  grp_comm_scaled median:", _fmt(med), "max:", _fmt(mx))
    med, mx = summarize("grp_comm_logdet_abs")
    print("  grp_comm_logdet_abs median:", _fmt(med), "max:", _fmt(mx))

    if str(args.out_csv).strip():
        pth = Path(str(args.out_csv))
        pth.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(pth), index=False)
        print(f"wrote {pth}")

    if str(args.out_prime_csv).strip():
        pth = Path(str(args.out_prime_csv))
        pth.parent.mkdir(parents=True, exist_ok=True)
        dfp.to_csv(str(pth), index=False)
        print(f"wrote {pth}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
