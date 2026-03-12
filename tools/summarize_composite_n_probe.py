import argparse
import math

import numpy as np
import pandas as pd


def _safe_log_abs(x: float, *, floor: float = 1e-12) -> float:
    return float(math.log(max(float(floor), abs(float(x)))))


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize coprime multiplicativity/log-additivity errors from probe_composite_n_semigroup CSV output.")
    ap.add_argument("--in_csv", required=True)
    ap.add_argument(
        "--min_abs_P",
        type=float,
        default=0.0,
        help=(
            "Optional filter: skip coprime pairs (a,b) if any of |P(a)|, |P(b)|, |P(ab)| is below this threshold. "
            "Useful to reduce log-floor artifacts when P is extremely small."
        ),
    )
    args = ap.parse_args()

    df = pd.read_csv(str(args.in_csv))
    if not {"n", "P"}.issubset(set(df.columns)):
        raise SystemExit("Expected columns: n,P")

    P = {int(r.n): float(r.P) for r in df.itertuples(index=False)}
    ns = sorted(P.keys())

    rel_mult_errs = []
    logadd_errs = []
    n_pairs_total = 0
    n_pairs_kept = 0

    for a in ns:
        for b in ns:
            if a <= 1 or b <= 1:
                continue
            if math.gcd(int(a), int(b)) != 1:
                continue
            ab = int(a * b)
            if ab not in P:
                continue

            n_pairs_total += 1
            Pa = float(P[a])
            Pb = float(P[b])
            Pab = float(P[ab])

            thr = float(args.min_abs_P)
            if thr > 0 and (abs(Pa) < thr or abs(Pb) < thr or abs(Pab) < thr):
                continue

            n_pairs_kept += 1
            denom = max(1e-12, abs(Pa) * abs(Pb))
            rel_mult_errs.append(float(abs(Pab - Pa * Pb) / denom))
            logadd_errs.append(float(abs(_safe_log_abs(Pab) - (_safe_log_abs(Pa) + _safe_log_abs(Pb)))))

    def fmt(x: float) -> str:
        if not math.isfinite(float(x)):
            return "nan"
        return f"{float(x):.4g}"

    print("composite-n probe summary")
    print(f"  in_csv: {args.in_csv}")
    print(f"  ns: {','.join(str(n) for n in ns)}")
    if float(args.min_abs_P) > 0:
        print(f"  min_abs_P: {float(args.min_abs_P):g} (kept {n_pairs_kept}/{n_pairs_total} coprime pairs)")
    if rel_mult_errs:
        a = np.asarray(rel_mult_errs, dtype=float)
        print("  multiplicativity relerr median:", fmt(float(np.median(a))), "max:", fmt(float(np.max(a))), "n:", int(a.size))
    else:
        print("  multiplicativity relerr: (no coprime pairs)")

    if logadd_errs:
        a = np.asarray(logadd_errs, dtype=float)
        print("  log|P| additivity abs err median:", fmt(float(np.median(a))), "max:", fmt(float(np.max(a))), "n:", int(a.size))
    else:
        print("  log|P| additivity abs err: (no coprime pairs)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
