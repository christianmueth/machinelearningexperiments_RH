# Results Packet (for handoff) — 2026‑01‑30

This is a compact, copy/paste‑friendly packet you can give another GPT.

## Ground rules / invariants

- Frozen measurement spine: DN‑map proxy via Schur complement + logdet aggregation.
- Energy windows fixed: primary (2.0, 5.0) and secondary (0.6, 7.5).
- Null family fixed: label‑scramble; empirical lower‑tail p‑values with add‑one smoothing.
- Decision threshold fixed: reject at α=0.01.
- Audit policy: digest invariances must remain constant within a run (measurement + geometry policy + FE policy).

## Phase‑3C observable (what changed vs Phase‑3B)

- New FE/involution channel computed on the *same* DN energy grid using the canonical involution **E ↦ −E**.
- Clear separation of concepts:
  - **Analytic completion (null‑free):** `fe_completion_mode=analytic_harmonic` with
    - \(G(E)=\beta\,\log(1/4+(\alpha_E E)^2)\)
    - parameters used here: `fe_alpha_E=1.0`, `fe_beta=1.0`.
  - **Null calibration (null‑dependent):** treated as stabilizer only; in the runs below we used `fe_calibration_mode=none`.
- FE aggregation: `fe_agg=mean`.
- Cross‑window combine rule (FE): take the max across windows (conservative AND across windows).

## Completed prereg scans (seedblock 64–127)

### A) FE AND rigid gate (combined)

- Scan root: `out_phase3C_fe_scan64_127_clean`
  - prereg: `preregistration.json`, `preregistration_digest.txt`
  - raw: `raw.csv`
  - KPI: `kpi_fraction.csv`, `kpi_by_seed.csv`
  - audit: `digest_invariance_summary.csv`

Gate semantics:
- `p_rigid = min(p_dnmap(window1), p_dnmap(window2))`
- `p_family = max(p_fe, p_rigid)` (AND gate)

Headline KPI (from `kpi_fraction.csv`):
- 0/64 rejecting seeds for both backends and both windows.
- Best p over all seeds/anchors:
  - `legacy`: ≈ 0.2117073
  - `geom_v9_phasetransport_autotune`: ≈ 0.3970732

Audit:
- `digest_invariance_summary.csv` reports all `nuniq_* = 1` (measurement + geometry policy + FE policy).

### B) FE‑only gate

- Scan root: `out_phase3C_fe_scan64_127_feonly_clean`
  - prereg: `preregistration.json`, `preregistration_digest.txt`
  - raw: `raw.csv`
  - KPI: `kpi_fraction.csv`, `kpi_by_seed.csv`
  - audit: `digest_invariance_summary.csv`

Gate semantics:
- `p_family = p_fe` (FE only)

Headline KPI (from `kpi_fraction.csv`):
- 0/64 rejecting seeds for both backends and both windows.
- Best p over all seeds/anchors (near‑misses only):
  - `legacy`: ≈ 0.01463415
  - `geom_v9_phasetransport_autotune`: ≈ 0.02634146

Audit:
- `digest_invariance_summary.csv` reports all `nuniq_* = 1` (measurement + geometry policy + FE policy).

## Confirm runs (N_null = 16384) for near‑miss seeds

All confirms below are FE‑only gate, analytic completion on, calibration off (`none`), 5 anchors each, both windows.

### Confirm 1: legacy seed 76

- Confirm root: `out_phase3C_fe_confirm_seed76_N16384`
  - KPI by window: `kpi_by_window.csv`
  - raw: `raw.csv`
  - audit: `digest_invariance_summary.csv`

Result:
- No rejects.
- Combined FE p‑value increased vs scan:
  - scan best p ≈ 0.014634 → confirm best p ≈ 0.01934696.

### Confirm 2: legacy seed 85

- Confirm root: `out_phase3C_fe_confirm_seed85_legacy_N16384`
  - KPI by window: `kpi_by_window.csv`
  - raw: `raw.csv`
  - audit: `digest_invariance_summary.csv`

Result:
- No rejects.
- Best combined FE p ≈ 0.03753433.

### Confirm 3: geom_v9 seed 124

- Confirm root: `out_phase3C_fe_confirm_seed124_geomv9_N16384`
  - KPI by window: `kpi_by_window.csv`
  - raw: `raw.csv`
  - audit: `digest_invariance_summary.csv`

Result:
- No rejects.
- Best combined FE p ≈ 0.02807446.

## Bottom‑line interpretation (honest + audit‑clean)

- Under the frozen DN‑map measurement spine, the Phase‑3C FE/involution observable (E↦−E) with analytic harmonic completion and no null calibration produces **no α=0.01 rejections** on the preregistered seedblock 64–127.
- FE‑only scanning shows near‑misses around ~0.015–0.03, but confirm at `N_null=16384` pushes the top near‑miss upward (seed 76: ~0.0146 → ~0.0193), consistent with statistical fluctuation rather than a stable effect.
- All runs pass digest invariance checks (measurement + geometry policy + FE policy), supporting the interpretation as a real negative result for this observable configuration.

## Suggested next step (if continuing Phase‑3C)

- Decide whether to preregister a *calibration* variant (e.g., `center_null_mean` or `zscore_null`) strictly as calibration/stabilization, and rerun a scan+confirm pair, clearly labeling it as null‑dependent.
- Alternatively, keep calibration off and explore preregistered sweeps over the analytic completion parameters (`fe_alpha_E`, `fe_beta`) while keeping the rest frozen, then confirm only scan‑selected near‑misses.

---

## Phase‑3D operator/channel diagnostics (2026‑01‑31 update)

Goal shift (post Phase‑3C): stop p‑value hunting and instead certify what the *operator itself* is doing.

### What Phase‑3D measures

- Snapshot DN proxy \(\Lambda(E)\) (Schur complement) at fixed energies per window.
- Form Cayley transform \(S(E;\eta)=(\Lambda-i\eta I)(\Lambda+i\eta I)^{-1}\).
- Extract “channels” by clustering eigenphases of \(S(E)\) into \(M=8\) groups and forming the corresponding spectral subspaces (QR‑orthonormalized eigenvector blocks).

KPIs:
- **\(\delta_\Lambda\)**: Hermitian defect of the *raw* (unsymmetrized) Schur complement output.
- **Anchor stability**: same \(S(E)\), different deterministic clustering initializations (“anchors”), principal angles after best-permutation matching.
- **Energy transport (diagnostic)**: principal angles for matched channel subspaces across energies, plus overlap \(\frac1d\mathrm{Tr}(PQ)\).
- **Same‑E control (new)**: run the channel extraction twice at the same \(E\) with different clustering RNG salts; any nonzero “transport” here is extraction noise.

### Scale run executed (8 seeds × 7 anchors)

Outputs:
- `out_phase3D_channel_diag_scale8_fullanchors_legacy/`
- `out_phase3D_channel_diag_scale8_fullanchors_geomv9/`

New postpass files per output root:
- `kpi_energy_transport_pairs.csv` (per-adjacent-step per-channel rows: angle, overlap, gap proxies, ΔE)
- `kpi_extraction_noise.csv` (same‑E control)

Additional Phase‑3D artifacts (2026‑01‑31):
- `kpi_energy_transport_pairs_tracked.csv` / `kpi_energy_transport_tracked.csv`: continuity‑tracked transport (overlap‑assignment of eigenvectors to persist channel dims)
- `transport_angle_vs_gap_center.png`, `transport_overlap_vs_gap_center.png`, `transport_angle_per_dE_vs_gap_center.png`: “angle vs gap” views
- `transport_angle_hist_compare.png`, `transport_overlap_hist_compare.png`: clustering vs tracked distribution comparisons

### Headline findings (honest interpretation)

- **\(\delta_\Lambda\)** is tiny (median ~1e‑18, max ~1e‑16), confirming the Schur complement pipeline is producing an almost‑Hermitian DN map *before* symmetrization.
  - Consequence: Cayley‑unitarity is largely “free”; it is not evidence of a nontrivial mechanism.

- **Same‑E control is ~0**:
  - `sameE_angle_median_median` ~ 0
  - `sameE_overlap_median_median` ~ 1
  This rules out a baseline offset/normalization bug in the angle metric: the big cross‑energy angles are not dominated by extraction noise.

- **Energy transport is *not* canonically stable in typical cases**:
  - Adjacent‑energy median max‑angle is ~0.65 rad (≈37°) with median overlap ~0.62–0.65.
  - This is “moderate rotation”, not “stable invariant subspace transport” in the usual operator‑theory sense.

- **Gap conditioning matters**:
  - Using a centroid‑gap proxy on eigenphases, “large‑gap” cases are rare (legacy: ~2.3% of adjacent channel pairs have gap ≥ 0.4 rad).
  - In those rare large‑gap cases (legacy), the transport angle drops (median ~0.24 rad) and overlap rises (~0.94), consistent with adiabatic stability when spectral separation exists.

- **Continuity‑tracked transport did not fix the typical rotation**:
  - We implemented a continuity‑based assignment step (carry channel dims forward by choosing next‑energy eigenvectors that maximize overlap with previous channel subspaces).
  - Summary medians remained essentially unchanged (tracked median angle ~0.66 rad; overlaps ~0.62–0.63), indicating the moderate rotation is not just a per‑snapshot clustering artifact.

### Important reporting fix

- The Phase‑3D postpass now separates:
  - `overall_pass_basic`: unitarity/commutator/anchor‑stability style gates
  - `overall_pass_with_transport`: includes energy‑transport gates
  - `transport_is_gate` (default false): whether transport gates are included in `overall_pass`

Interpretation policy:
- Treat energy transport as **diagnostic until calibrated** (ΔE‑normalized + gap‑stratified), because without eigengaps “channels” are not canonically transportable.

---

## Phase‑3E connection + monodromy pack (2026‑01‑31)

Motivation: Phase‑3D indicates fine spectral projectors are ill‑posed in a pervasive small‑gap regime. Phase‑3E pivots to **coarse blocks + a canonical discrete connection**, and reports **holonomy/monodromy invariants** around a 2‑parameter loop.

Implementation:
- Script: `phase3e_connection_monodromy_pack.py`
- Coarse blocks: split eigenphases into `blocks=4` equal‑count blocks (rank fixed by construction)
- Loop: rectangle in `(E, η)` using stored `lambda_snap`:
  - `(E0, η1) → (E1, η1) → (E1, η2) → (E0, η2) → (E0, η1)` with `η2 = 2·η1`
- Connection: per‑step complex Procrustes (closest unitary) on block subspaces
- Reported holonomy invariants per block: `||I−M||_F`, det‑phase, eigenphase stats

Outputs (per root):
- `phase3e_monodromy_rows.csv`
- `phase3e_monodromy_summary.json`

Headline (scale8 fullanchors):
- Legacy: median `||I−M||_F` ~ 2.45, mean|eigphase| ~ 1.19 rad
- geom_v9: median `||I−M||_F` ~ 2.37, mean|eigphase| ~ 1.13 rad

Interpretation: holonomy is strongly nontrivial and stable to compute (valid_fraction=1.0 on 112 artifacts), suggesting the right object to track in dense spectra is **connection + monodromy**, not “small principal angles”.

### Phase-3E.1 loop sanity suite (correction)

We added a loop sanity suite and upgraded the discrete connection to a more gauge-stable, block-coordinate form.

- Script: phase3e_monodromy_suite.py
- Outputs per root:
  - phase3e_monodromy_suite_rows.csv
  - phase3e_monodromy_suite_summary.json
  - plots: phase3e_suite_strength_vs_area.png, phase3e_suite_reverse_consistency_hist.png, phase3e_suite_phi_cdf_by_anchor.png

Key result: with the gauge-stable connection (and reusing bases at repeated loop points), the (E, eta) rectangle holonomy collapses to ~identity (machine-precision ||I-M||_F), and the degenerate / reverse / double-loop controls behave as expected.

Interpretation update: the earlier nontrivial ||I-M||_F values from phase3e_connection_monodromy_pack.py are likely dominated by gauge/basis ambiguity from the original n x n Procrustes construction and/or block-boundary relabeling effects. For meaningful monodromy, the second parameter in the loop should deform Lambda itself (geometry / boundary / split), not just the Cayley eta parameter (which largely preserves eigenvectors at fixed E).
