"""Parse FE discrimination run CSVs and print p_mode rankings.

This is a lightweight utility for comparing three regimes:
- k_max=1, prime_power_mode=direct (legacy 'disc_fixed_*' outputs)
- k_max=3, prime_power_mode=direct (disc_direct_*_k3)
- k_max=3, prime_power_mode=x_power (disc_semigroup_*_k3)

It summarizes mean rel_l2(Fu) over u in {-0.2, +0.2}.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_fe"


@dataclass(frozen=True)
class RunMeta:
    completion: str
    k_max: int
    prime_power_mode: str
    p_mode: str
    filename: str


def _parse_p_mode_from_name(name: str) -> str | None:
    # p_mode tokens can include underscores (e.g. p1_over_p), so we parse
    # between "pmode_" and the next "_b" boundary marker.
    m = re.search(r"pmode_(.+?)_b", name)
    return m.group(1) if m else None


def _parse_kmax_from_name(name: str) -> int | None:
    m = re.search(r"_k(\d+)\.csv$", name)
    return int(m.group(1)) if m else None


def _parse_completion_from_name(name: str) -> str | None:
    if "_gl2_" in name or name.startswith("disc_fixed_gl2"):
        return "gl2"
    if "_zeta_" in name or name.startswith("disc_fixed_zeta"):
        return "zeta"
    return None


def _infer_meta_from_filename(path: Path) -> RunMeta:
    name = path.name
    completion = _parse_completion_from_name(name)
    if completion is None:
        raise ValueError(f"Cannot infer completion from filename: {name}")

    # Special-case the legacy smoke baseline file (has no pmode_ token).
    if name == "smoke_disc_fixed_zeta_p_k1_direct.csv":
        return RunMeta(
            completion="zeta",
            k_max=1,
            prime_power_mode="direct",
            p_mode="p",
            filename=name,
        )

    p_mode = _parse_p_mode_from_name(name)
    if p_mode is None:
        raise ValueError(f"Cannot infer p_mode from filename: {name}")

    if name.startswith("disc_fixed_") and name.endswith("_b05.csv"):
        return RunMeta(
            completion=completion,
            k_max=1,
            prime_power_mode="direct",
            p_mode=p_mode,
            filename=name,
        )

    if name.startswith("disc_direct_"):
        k_max = _parse_kmax_from_name(name)
        if k_max is None:
            raise ValueError(f"Cannot infer k_max from filename: {name}")
        return RunMeta(
            completion=completion,
            k_max=k_max,
            prime_power_mode="direct",
            p_mode=p_mode,
            filename=name,
        )

    if name.startswith("disc_semigroup_"):
        k_max = _parse_kmax_from_name(name)
        if k_max is None:
            raise ValueError(f"Cannot infer k_max from filename: {name}")
        return RunMeta(
            completion=completion,
            k_max=k_max,
            prime_power_mode="x_power",
            p_mode=p_mode,
            filename=name,
        )

    if name.startswith("disc_bulkpower_"):
        k_max = _parse_kmax_from_name(name)
        if k_max is None:
            raise ValueError(f"Cannot infer k_max from filename: {name}")
        return RunMeta(
            completion=completion,
            k_max=k_max,
            prime_power_mode="bulk_power",
            p_mode=p_mode,
            filename=name,
        )

    raise ValueError(f"Unrecognized run filename pattern: {name}")


def _mean_fu_rel_l2(df: pd.DataFrame) -> float:
    if "label" not in df.columns:
        raise ValueError("CSV missing required column 'label'")
    if "u" not in df.columns:
        raise ValueError("CSV missing required column 'u'")
    if "rel_l2" not in df.columns:
        raise ValueError("CSV missing required column 'rel_l2'")

    dff = df[(df["label"].astype(str) == "Fu") & (df["u"].isin([-0.2, 0.2]))]
    if dff.empty:
        raise ValueError("No Fu rows found at u=±0.2")
    return float(dff["rel_l2"].mean())


def main() -> None:
    patterns = [
        "disc_fixed_zeta_pmode_*_b05.csv",
        "disc_fixed_gl2_pmode_*_b05.csv",
        "smoke_disc_fixed_zeta_p_k1_direct.csv",
        "disc_direct_zeta_pmode_*_b05_k*.csv",
        "disc_direct_gl2_pmode_*_b05_k*.csv",
        "disc_semigroup_zeta_pmode_*_b05_k*.csv",
        "disc_semigroup_gl2_pmode_*_b05_k*.csv",
        "disc_bulkpower_zeta_pmode_*_b05_k*.csv",
        "disc_bulkpower_gl2_pmode_*_b05_k*.csv",
    ]

    paths: list[Path] = []
    for pat in patterns:
        paths.extend(sorted(RUNS.glob(pat)))

    if not paths:
        raise SystemExit(f"No matching CSVs found under: {RUNS}")

    rows: list[dict[str, object]] = []
    for path in paths:
        meta = _infer_meta_from_filename(path)
        df = pd.read_csv(path)
        score = _mean_fu_rel_l2(df)
        rows.append(
            {
                "completion": meta.completion,
                "k_max": meta.k_max,
                "prime_power_mode": meta.prime_power_mode,
                "p_mode": meta.p_mode,
                "mean_rel_l2_Fu_u02": score,
                "file": meta.filename,
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["completion", "k_max", "prime_power_mode", "mean_rel_l2_Fu_u02"])

    # Print per-completion rankings
    for completion in ["zeta", "gl2"]:
        sub = out[out["completion"] == completion].copy()
        if sub.empty:
            continue
        print(f"\n=== {completion} ===")
        print(sub[["k_max", "prime_power_mode", "p_mode", "mean_rel_l2_Fu_u02"]].to_string(index=False))

        print("\nRankings:")
        for (kmax, ppm), grp in sub.groupby(["k_max", "prime_power_mode"], dropna=False):
            grp = grp.sort_values("mean_rel_l2_Fu_u02")
            parts = [f"{pm}({val:.6g})" for pm, val in zip(grp["p_mode"], grp["mean_rel_l2_Fu_u02"]) ]
            print(f"  k_max={kmax} | prime_power_mode={ppm}: " + " < ".join(parts))

    # Completion-stability: compare rank order between zeta and gl2 for each regime.
    # We compute Kendall tau on the ranks of p_mode among the intersection set.
    print("\n=== Completion Stability (Kendall tau) ===")
    for (kmax, ppm), grp in out.groupby(["k_max", "prime_power_mode"], dropna=False):
        g_z = grp[grp["completion"] == "zeta"].copy()
        g_g = grp[grp["completion"] == "gl2"].copy()
        if g_z.empty or g_g.empty:
            continue
        common = sorted(set(g_z["p_mode"]).intersection(set(g_g["p_mode"])))
        if len(common) < 2:
            continue
        rz = {pm: i for i, pm in enumerate(g_z.sort_values("mean_rel_l2_Fu_u02")["p_mode"].tolist())}
        rg = {pm: i for i, pm in enumerate(g_g.sort_values("mean_rel_l2_Fu_u02")["p_mode"].tolist())}
        a = np.array([rz[pm] for pm in common], dtype=int)
        b = np.array([rg[pm] for pm in common], dtype=int)

        n = int(len(common))
        conc = 0
        disc = 0
        for i in range(n):
            for j in range(i + 1, n):
                s1 = int(a[i] - a[j])
                s2 = int(b[i] - b[j])
                if s1 == 0 or s2 == 0:
                    continue
                if (s1 > 0) == (s2 > 0):
                    conc += 1
                else:
                    disc += 1
        denom = conc + disc
        tau = float((conc - disc) / denom) if denom > 0 else float("nan")
        print(f"k_max={int(kmax)} | prime_power_mode={ppm}: tau={tau:.3f} over {len(common)} p_modes")

    # Save machine-readable summary
    summary_path = RUNS / "summary_rankings_u02.csv"
    out.to_csv(summary_path, index=False)
    print(f"\nWrote {os.path.relpath(summary_path, ROOT)}")


if __name__ == "__main__":
    main()
