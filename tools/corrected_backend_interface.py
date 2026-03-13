from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_COEFF_CSV = Path("out/corrected_factor_injection_beta23_window_coefficients.csv")


@dataclass(frozen=True)
class CorrectedFactors:
    u: float
    beta2: float
    beta3: float
    c: float
    A1_star: float
    A2_star: float
    A3_star: float


def load_corrected_factors(*, coeff_csv: Path | str = DEFAULT_COEFF_CSV, u: float, tol: float = 1e-12) -> CorrectedFactors:
    path = Path(str(coeff_csv))
    if not path.exists():
        raise SystemExit(f"missing coefficient csv: {path}")
    df = pd.read_csv(path)
    mask = np.isclose(df["u"].astype(float), float(u), atol=float(tol))
    df = df.loc[mask].copy()
    if df.empty:
        raise SystemExit(f"no coefficient row at u={u} in {path}")
    row = df.iloc[0]
    return CorrectedFactors(
        u=float(row["u"]),
        beta2=float(row.get("beta2", 0.0)),
        beta3=float(row.get("beta3", 0.0)),
        c=float(row.get("c", 0.0)),
        A1_star=float(row["A1_star"]),
        A2_star=float(row["A2_star"]),
        A3_star=float(row["A3_star"]),
    )


def build_prime_local_backend_feed(*, primes: list[int], factors: CorrectedFactors) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    base = asdict(factors)
    for p in primes:
        rows.append({"p": int(p), **base})
    return pd.DataFrame(rows)
