# Stage Gate Summary — v5 Scan/Confirm (Phase‑3B legal)

Date: 2026‑01‑29

This document summarizes the preregistered scan (`N_null=1024`) and confirm (`N_null=16384`) results for the v5 fork while keeping the Phase‑3B measurement spine frozen (DN‑map/logdet aggregation, label‑scramble null, α=0.01).

## Where the outputs live

- Scan root: `out_phase3B_preregistered_uniqueness_64_127_v5`
  - `preregistration.json` (frozen protocol)
  - `preregistration_digest.txt` (digest: `da3663ff6d51c570`)
  - `raw.csv`, `kpi_by_seed.csv`, `kpi_fraction.csv`, `uniqueness.csv`
- Confirm root: `out_phase3B_confirm_preregistered_uniqueness_rejecting_seeds_N16384_v5scan`
  - `raw.csv`

## Preregistration snapshot

From `out_phase3B_preregistered_uniqueness_64_127_v5/preregistration.json`:

- Mode: `dnmap_only` (no closure channel)
- Null family: `label_scramble`
- α: `0.01`
- Seeds: 64–127 inclusive (64 seeds)
- Anchors: 7 anchors (2, 9, 14, 44, 46, 51, 60)
- Windows: (2.0, 5.0) primary; (0.6, 7.5) secondary
- `N_null` (scan): 1024
- `dnmap_null_batch`: 8 (compute only)
- Backends:
  - `legacy`
  - `geom_v4_theta0p125_autotune` (v4, eps auto‑tuned to target ΔTp≈0.02)
  - `geom_v5_theta0p125_autotune` (v5, eps auto‑tuned to target ΔTp≈0.02; harmonic modulation γ=1.0 strength=1.0)

## Scan results (N_null=1024)

From `out_phase3B_preregistered_uniqueness_64_127_v5/kpi_fraction.csv`:

- `legacy`: 0/64 rejecting seeds in both windows.
- `geom_v5_theta0p125_autotune`: 0/64 rejecting seeds in both windows.
- `geom_v4_theta0p125_autotune`:
  - Secondary window (0.6, 7.5): 0/64 rejecting seeds.
  - Primary window (2.0, 5.0): 1/64 rejecting seeds (1.5625%).

Seed responsible (primary window):
- backend: `geom_v4_theta0p125_autotune`
- seed: 110
- scan best p (over anchors): 0.007805

## Confirm results (N_null=16384)

Confirm was run for backend `geom_v4_theta0p125_autotune` and seed 110 across the 7 preregistered anchors.

Summary from `out_phase3B_confirm_preregistered_uniqueness_rejecting_seeds_N16384_v5scan/raw.csv`:

- Primary window (2.0, 5.0):
  - 1/7 anchors rejected at α=0.01
  - The rejecting anchor was 46 with p≈0.009155
  - Other anchors were near-threshold but non‑rejecting (≈0.0104–0.0117)
- Secondary window (0.6, 7.5):
  - 0/7 anchors rejected

Interpretation: the scan hit for v4 exists but is not robust under confirm across anchors. v5 did not produce any scan hits under this preregistered configuration.

## Null spread diagnostics (derived-only)

Confirm rows include derived null-spread fields (`dn_null_std`, `dn_null_cv`, `dn_snr`). For seed 110, primary window values were consistent across anchors, with `dn_null_cv` around ~0.40–0.41.

These are diagnostics only; they do not affect p-values or the frozen measurement definition.

## Status

- Scan: complete
- Confirm: complete (only for v4 seed 110, per scan selection)
- Next action if desired: repeat the same pipeline with altered *admissible* Tp-backend knobs (still Phase‑3B legal) to see if v5 can produce scan hits that survive confirm across anchors.
