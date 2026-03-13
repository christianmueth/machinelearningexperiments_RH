from __future__ import annotations

import argparse
import cmath
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _anchor_r(y: float) -> complex:
    return complex(0.5, float(y))


def _anchor_rs(y: float) -> complex:
    return complex(0.5, -float(y))


def _anchor_matrix(y: float) -> np.ndarray:
    r = _anchor_r(float(y))
    rs = _anchor_rs(float(y))
    return np.asarray([[1.0 + 0.0j, r], [-(1.0 / rs), 1.0 + 0.0j]], dtype=np.complex128)


def _anchor_det_numeric(y: float) -> complex:
    return complex(np.linalg.det(_anchor_matrix(float(y))))


def _anchor_det_exact(y: float) -> complex:
    r = _anchor_r(float(y))
    rs = _anchor_rs(float(y))
    return complex(1.0 + r / rs)


def _anchor_det_closed_form(y: float) -> complex:
    return complex(2.0 / (1.0 - 2j * float(y)))


def _log_anchor_det(y: float) -> complex:
    return cmath.log(_anchor_det_closed_form(float(y)))


def _even_log_coeff(k: int) -> float:
    if int(k) < 1:
        raise ValueError("k must be >= 1")
    return float(((-1) ** int(k)) * (2 ** (2 * int(k))) / float(2 * int(k)))


def _odd_log_coeff(k: int) -> complex:
    if int(k) < 0:
        raise ValueError("k must be >= 0")
    n = 2 * int(k) + 1
    return complex((2j) ** n / float(n))


def _anchored_real_polynomial(x: float, t: float, even_coeffs: dict[int, float]) -> float:
    h = complex(float(x), float(t))
    hit = complex(0.0, float(t))
    total = 0.0
    for power, coeff in even_coeffs.items():
        total += float(coeff) * float(np.real((h ** int(power)) - (hit ** int(power))))
    return float(total)


def _quartic_anchored_component(x: float, t: float) -> float:
    x = float(x)
    t = float(t)
    return float((x**4) - 6.0 * (x**2) * (t**2))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Derive the centered-even completion curvature from the dyadic anchor matrix, "
            "expand log det M(y) around y=0, and compare the induced anchored quartic term with the empirical completion coefficient."
        )
    )
    ap.add_argument("--empirical_a4", type=float, default=-0.2)
    ap.add_argument("--max_even_order", type=int, default=8)
    ap.add_argument("--sample_t", type=float, default=28.0)
    ap.add_argument("--sigma_min", type=float, default=0.45)
    ap.add_argument("--sigma_max", type=float, default=0.55)
    ap.add_argument("--sigma_steps", type=int, default=21)
    ap.add_argument("--out_prefix", default="out/anchor_completion_derivation")
    args = ap.parse_args()

    max_even_order = int(args.max_even_order)
    if max_even_order < 2 or (max_even_order % 2) != 0:
        raise ValueError("max_even_order must be an even integer >= 2")

    coeff_rows: list[dict[str, float | int | str]] = []
    even_coeffs: dict[int, float] = {}
    for power in range(2, max_even_order + 1, 2):
        k = power // 2
        c = _even_log_coeff(k)
        even_coeffs[int(power)] = float(c)
        coeff_rows.append(
            {
                "term_type": "even",
                "power": int(power),
                "coefficient": float(c),
                "interpretation": f"c_{power} in log(det M(y))",
            }
        )
    for power in range(1, max_even_order + 1, 2):
        k = (power - 1) // 2
        c = _odd_log_coeff(k)
        coeff_rows.append(
            {
                "term_type": "odd",
                "power": int(power),
                "coefficient_re": float(np.real(c)),
                "coefficient_im": float(np.imag(c)),
                "interpretation": f"odd coefficient at y^{power}",
            }
        )

    y_samples = np.asarray([-0.1, -0.05, 0.0, 0.05, 0.1], dtype=float)
    det_rows: list[dict[str, float]] = []
    for y in y_samples.tolist():
        det_num = _anchor_det_numeric(float(y))
        det_exact = _anchor_det_exact(float(y))
        det_closed = _anchor_det_closed_form(float(y))
        log_det = _log_anchor_det(float(y))
        det_rows.append(
            {
                "y": float(y),
                "det_numeric_re": float(np.real(det_num)),
                "det_numeric_im": float(np.imag(det_num)),
                "det_exact_re": float(np.real(det_exact)),
                "det_exact_im": float(np.imag(det_exact)),
                "det_closed_re": float(np.real(det_closed)),
                "det_closed_im": float(np.imag(det_closed)),
                "logdet_re": float(np.real(log_det)),
                "logdet_im": float(np.imag(log_det)),
            }
        )

    derived_c2 = float(even_coeffs.get(2, 0.0))
    derived_c4 = float(even_coeffs.get(4, 0.0))
    empirical_a4 = float(args.empirical_a4)
    inverse_anchor_a4 = -derived_c4
    scale_direct = float((abs(empirical_a4) / abs(derived_c4)) ** 0.25) if abs(derived_c4) > 0 else float("nan")
    sign_match_direct = bool(math.copysign(1.0, empirical_a4) == math.copysign(1.0, derived_c4)) if derived_c4 != 0.0 else False
    sign_match_inverse = bool(math.copysign(1.0, empirical_a4) == math.copysign(1.0, inverse_anchor_a4)) if inverse_anchor_a4 != 0.0 else False

    summary_row = {
        "anchor_det_formula": "2/(1-2iy)",
        "logdet_formula": "log(2)-log(1-2iy)",
        "derived_c2": float(derived_c2),
        "derived_c4": float(derived_c4),
        "inverse_anchor_c4": float(inverse_anchor_a4),
        "empirical_a4": float(empirical_a4),
        "quartic_sign_matches_direct": bool(sign_match_direct),
        "quartic_sign_matches_inverse": bool(sign_match_inverse),
        "implied_y_scale_if_inverse": float(scale_direct),
    }

    sigma_values = np.linspace(float(args.sigma_min), float(args.sigma_max), int(args.sigma_steps), dtype=float)
    profile_rows: list[dict[str, float | str]] = []
    for sigma in sigma_values.tolist():
        x = float(sigma - 0.5)
        quartic_shape = _quartic_anchored_component(x=x, t=float(args.sample_t))
        derived_anchor_logw = _anchored_real_polynomial(x=x, t=float(args.sample_t), even_coeffs={2: derived_c2, 4: derived_c4})
        inverse_anchor_logw = -derived_anchor_logw
        empirical_logw = float(empirical_a4) * quartic_shape
        profile_rows.extend(
            [
                {
                    "profile": "derived_anchor",
                    "sigma": float(sigma),
                    "x": float(x),
                    "t": float(args.sample_t),
                    "quartic_shape": float(quartic_shape),
                    "log_weight": float(derived_anchor_logw),
                },
                {
                    "profile": "inverse_anchor",
                    "sigma": float(sigma),
                    "x": float(x),
                    "t": float(args.sample_t),
                    "quartic_shape": float(quartic_shape),
                    "log_weight": float(inverse_anchor_logw),
                },
                {
                    "profile": "empirical_quartic_only",
                    "sigma": float(sigma),
                    "x": float(x),
                    "t": float(args.sample_t),
                    "quartic_shape": float(quartic_shape),
                    "log_weight": float(empirical_logw),
                },
            ]
        )

    out_prefix = Path(str(args.out_prefix))
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    coeff_path = Path(str(out_prefix) + "_coefficients.csv")
    det_path = Path(str(out_prefix) + "_determinant_samples.csv")
    summary_path = Path(str(out_prefix) + "_summary.csv")
    profile_path = Path(str(out_prefix) + "_profiles.csv")
    pd.DataFrame(coeff_rows).to_csv(coeff_path, index=False)
    pd.DataFrame(det_rows).to_csv(det_path, index=False)
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False)
    pd.DataFrame(profile_rows).to_csv(profile_path, index=False)

    print(f"wrote {coeff_path}")
    print(f"wrote {det_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {profile_path}")
    print(f"derived c2={derived_c2:.12g}, c4={derived_c4:.12g}, inverse-anchor c4={inverse_anchor_a4:.12g}")
    print(f"empirical a4={empirical_a4:.12g}, implied y-scale if inverse={(scale_direct):.12g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())