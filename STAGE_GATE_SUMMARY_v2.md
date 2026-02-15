# Stage Gate Summary (v2): Preregistered Uniqueness Scan + Confirm

Date: 2026-01-29  
Workspace: `machinelearningexperiments_RH`  

This note summarizes the **v2** rerun (the rerun that pins the v4 deformation magnitude and logs a geometry-policy digest), consisting of:
- A preregistered scan over seeds 64–127 with N_null=1024.
- A preregistered confirm run at N_null=16384 for scan-selected rejecting seeds only.

## What ran (frozen measurement spine)
**Frozen / invariant across all runs** (Phase‑3B legal):
- DN-map proxy via Schur complement + logdet aggregation over fixed energy grids per window.
- Label-scramble null; empirical p-values; decision threshold α=0.01.
- Windows:
  - Primary: (2.0, 5.0)
  - Secondary: (0.6, 7.5)
- Anchors: [2, 9, 14, 44, 46, 51, 60]
- Stride: dnmap_stride=2

**Backends in scan (exactly 3):**
- `legacy`
- `geom_v4_theta0p125` (v4, fixed eps band; tp_delta median ≈ 0.0149)
- `geom_v4_theta0p125_autotune` (v4, eps auto-tuned to target delta 0.02)

Scan outputs: [out_phase3B_preregistered_uniqueness_64_127_v2](out_phase3B_preregistered_uniqueness_64_127_v2)  
Confirm outputs: [out_phase3B_confirm_preregistered_uniqueness_rejecting_seeds_N16384_v2](out_phase3B_confirm_preregistered_uniqueness_rejecting_seeds_N16384_v2)

## Completion + invariance checks
**Completion**
- Scan jobs complete: 1344 / 1344 (64 seeds × 7 anchors × 3 backends)
- Confirm jobs complete: 7 / 7 (selected seeds × 7 anchors)

**Digest invariance**
- Scan digest invariance summary: [out_phase3B_preregistered_uniqueness_64_127_v2/digest_invariance_summary.csv](out_phase3B_preregistered_uniqueness_64_127_v2/digest_invariance_summary.csv)
  - `meas_params_digest=1`, `meas_boundary_digest=1`, `meas_E_digest=1` for all backends/windows
  - `geom_policy_digest=1` for all backends/windows
- Confirm digest invariance summary: [out_phase3B_confirm_preregistered_uniqueness_rejecting_seeds_N16384_v2/digest_invariance_summary.csv](out_phase3B_confirm_preregistered_uniqueness_rejecting_seeds_N16384_v2/digest_invariance_summary.csv)
  - All digests match (=1) for both windows

## Scan results (N_null=1024)
Source: [out_phase3B_preregistered_uniqueness_64_127_v2/kpi_fraction.csv](out_phase3B_preregistered_uniqueness_64_127_v2/kpi_fraction.csv)

**Seed-level reject frequency (α=0.01; any anchor rejects)**
- Primary window (2.0, 5.0):
  - `legacy`: 0/64 seeds reject
  - `geom_v4_theta0p125`: 0/64 seeds reject
  - `geom_v4_theta0p125_autotune`: 1/64 seeds reject (seed 110)
- Secondary window (0.6, 7.5):
  - `legacy`: 0/64 seeds reject
  - `geom_v4_theta0p125`: 1/64 seeds reject
  - `geom_v4_theta0p125_autotune`: 0/64 seeds reject

**Scan-selected rejecting seed(s), primary (2.0, 5.0)**
- `geom_v4_theta0p125_autotune`: seed 110

## Confirm results (N_null=16384; selected seeds only)
Confirm plan: [out_phase3B_confirm_preregistered_uniqueness_rejecting_seeds_N16384_v2/confirm_plan.json](out_phase3B_confirm_preregistered_uniqueness_rejecting_seeds_N16384_v2/confirm_plan.json)

- Confirmed backend: `geom_v4_theta0p125_autotune`
- Selected seeds: [110]
- Anchors: [2, 9, 14, 44, 46, 51, 60]

Aggregate results: [out_phase3B_confirm_preregistered_uniqueness_rejecting_seeds_N16384_v2/aggregate.csv](out_phase3B_confirm_preregistered_uniqueness_rejecting_seeds_N16384_v2/aggregate.csv)

- Primary (2.0, 5.0):
  - `reject_rate = 1/7 ≈ 0.1429`
  - `best_p ≈ 0.009155` (best across the 7 anchors)
- Secondary (0.6, 7.5):
  - `reject_rate = 0/7 = 0`
  - `best_p ≈ 0.013183`

Interpretation (strict, reviewer-safe): under this preregistered design, the **seed-level scan hit does not translate into a strong across-anchors confirm effect** (reject rate is not high; “best p” stays near threshold).

## Uniqueness KPI snapshot (scan; medians)
These are descriptive (not used for the α=0.01 decision), but help quantify the “uniqueness / injectivity” behavior.

Computed from: [out_phase3B_preregistered_uniqueness_64_127_v2/uniqueness.csv](out_phase3B_preregistered_uniqueness_64_127_v2/uniqueness.csv)

Median values across all (seed, anchor) jobs, by backend:
- `legacy`:
  - `tp_delta_mean ≈ 0.0000`
  - primary (2.0,5.0): `dn_identity_rank ≈ 568.5`, `dn_collision_rate ≈ 0.5542`
- `geom_v4_theta0p125`:
  - `tp_delta_mean ≈ 0.0149`
  - primary (2.0,5.0): `dn_identity_rank ≈ 599.0`, `dn_collision_rate ≈ 0.5840`
- `geom_v4_theta0p125_autotune`:
  - `tp_delta_mean ≈ 0.01994`
  - primary (2.0,5.0): `dn_identity_rank ≈ 605.0`, `dn_collision_rate ≈ 0.5898`

(These are just medians; the full distribution is in `uniqueness.csv`.)

## Bottom line
- v2 scan+confirm completed cleanly and is audit-safe (all digests match; prereg confirm uses scan preregistration + selected seeds).
- Primary (2.0, 5.0): one prereg scan rejecting seed for v4-autotune (seed 110), but confirm shows only 1/7 anchors reject at N_null=16384.
- Secondary (0.6, 7.5): one scan rejecting seed for v4-fixed; confirm was not triggered there (by design).

## Reproduce (commands)
Scan (already done):
- `python preregistered_uniqueness_seedblock_64_127.py --seed_start 64 --seed_end_exclusive 128 --anchors 2,9,14,44,46,51,60 --N_null 1024 --dnmap_stride 2 --windows "2.0,5.0;0.6,7.5" --out_root out_phase3B_preregistered_uniqueness_64_127_v2 --resume`

Confirm (already done):
- `python confirm_preregistered_uniqueness_rejecting_seeds_N16384.py --scan_root out_phase3B_preregistered_uniqueness_64_127_v2 --out_root out_phase3B_confirm_preregistered_uniqueness_rejecting_seeds_N16384_v2 --backend_label geom_v4_theta0p125_autotune --N_null 16384 --anchors 2,9,14,44,46,51,60 --dnmap_stride 2 --windows "2.0,5.0;0.6,7.5" --collision_tol 0.0 --resume`
