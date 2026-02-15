# GPT Checkpoint — 2026-02-01 (Phase-3F continuation)

## What changed (code)

### 1) Composition/locality suite generalized to handle event2 with a different E-rectangle
- File: phase3f_elambda_composition_suite.py
- Change: lassos are now based at a shared basepoint `(i_base, j_base)` and include a connector in **lambda** then **energy** to each rectangle’s basepoint, so `event2` can differ in `(loop_e0, loop_e1)`.
- New CLI arg: `--i_base` (default = `event1 e0`).
- New output columns:
  - `event1_e0,event1_e1,event2_e0,event2_e1`
  - `i_base`
  - `flip_overlap_M1_M2` = `|<v_-1(M1), v_-1(M2)>|` where `v_-1(M)` is the normalized eigenvector for the eigenvalue closest to `-1`.

### 2) Multi-π candidate finder can now search “true multi-lambda walls”
- File: phase3f_elambda_find_multipi_pairs.py
- New CLI flag: `--fixed_e 1` counts distinct **lambda-cells** per fixed `(loop_e0, loop_e1)`; `--fixed_e 0` (default) counts distinct full cells `(loop_e0,loop_e1,lam0,lam1)`.

## New results (nontrivial Phase-3F composition + locality + cancellation)

### Source run (dim=2 blocks=2)
- Run folder: out_phase3E_elambda_scale8_legacy_geomv9_alpha1p00_blocks2
- Grid: lambdas `0,0.25,0.5,0.75,1.0` (so cells are coarse, but dim=2 makes flip-mode nontrivial).

### Multi-cell π candidates exist (but across different E-rectangles, not across different lambda cells)
- Output: out_phase3E_elambda_scale8_legacy_geomv9_alpha1p00_blocks2/phase3f_multipi_pairs_tol0p25.csv
- Fixed-E scan is empty:
  - out_phase3E_elambda_scale8_legacy_geomv9_alpha1p00_blocks2/phase3f_multipi_pairs_fixedE_tol0p25.csv (header only)
  - i.e. no (pair,block) shows π in ≥2 distinct lambda-cells for the same `(loop_e0,loop_e1)` at tol=0.25.

### Composition test A (seed=80, anchor=2)
- Output folder: out_phase3F_blocks2_comp_twoEevents_seed80_anchor2_b0
- Pair: `(seed,anchor,wlo,whi)=(80,2,2.0,5.0)`
- Event1: `(loop_e0,loop_e1,lam0,lam1,block,dim)=(0,2,0,0.25,0,2)`
- Event2: `(1,2,0,0.25,0,2)`

Observed (for both blocks 0 and 1):
- Single-event holonomies are π-quantized:
  - `M1_abs_det_phase ≈ π`, `M2_abs_det_phase ≈ π`
  - `num_eigs_near_minus1 = 1`, `min_dist_to_minus1 ~ 2.7e-13` (mechanism fingerprint)
- Cancellation:
  - `M1_twice_abs_det_phase ≈ 0` and `||I - M1^2||_F ~ 5e-13`
- Locality control (adjacent lambda cell, non-enclosing):
  - `M1_ctrl_right_abs_det_phase ~ 1e-13` (≈ 0)
- Composition + order:
  - `M12_abs_det_phase ≈ 0` and equals `M21_abs_det_phase` (order_detphase_diff=0)
  - `||M12 - M1@M2||_F ~ 1e-16`, `||M21 - M2@M1||_F ~ 1e-16` (numerically exact composition)
- Flip-mode identity across the two events:
  - `flip_overlap_M1_M2 ≈ 0.99999996` (block0) and `≈ 0.99999708` (block1)

Interpretation: for this same system instance (same pair), **two distinct wall events** (different E-rectangles but same lambda cell) each contribute a Z₂ flip, and composing them yields trivial det-phase (π+π → 0). The flip-mode vectors align across the two events.

### Composition test B (seed=96, anchor=2)
- Output folder: out_phase3F_blocks2_comp_twoEevents_seed96_anchor2_b0
- Pair: `(96,2,0.6,7.5)`
- Event1: `(0,1,0,0.25,0,2)`
- Event2: `(0,2,0,0.25,0,2)`

Observed (both blocks 0 and 1): same qualitative behavior
- `M1_abs_det_phase ≈ π`, `M2_abs_det_phase ≈ π`
- `M12_abs_det_phase ≈ 0`, order difference 0, and `M12≈M1@M2` to ~1e-16
- `flip_overlap_M1_M2 ≈ 0.99998` (block0) and `≈ 0.99993` (block1)

## Flip-mode stability within a single event cell (dim=2)
From out_phase3E_elambda_scale8_legacy_geomv9_alpha1p00_blocks2/phase3f_event_cells_top20.csv:
- Example: `E0_2_lam0_0.25_b0_d2` has `flip_overlap_n=14`, `flip_overlap_min≈0.733`, `median≈0.867` (nontrivial; mode identity is measurable but not perfect across anchors).
- Many other cells are near-1.0.

## Immediate next steps
1) If we want “two distinct lambda walls” (same `(loop_e0,loop_e1)` but different lambda cells), we likely need:
   - finer lambda grids / local refinement around hotspots, or larger lambda ranges (current grid is coarse), or different alpha.
2) Use the generalized lasso composition suite to test more multi-cell candidates (the multipi list has 42 entries at tol=0.25 in the blocks=2 run).
3) If desired: extend the event registry to compute cross-event flip-mode overlaps when a pair is π in multiple event cells (so we can summarize `flip_overlap_M1_M2` distribution without running composition loops explicitly).

## Addendum: refined-λ attempt (still no fixed-E multi-λ primes)

To explicitly search for “two distinct lambda walls for the same fixed E-rectangle”, we ran a dense refinement on `[0,0.25]` (step 0.0125) over a targeted subset of 21 multipi pairs:

- Artifact subset roots:
  - `out_phase3D_channel_diag_scale8_fullanchors_legacy__subset_multipi_blocks2`
  - `out_phase3D_channel_diag_scale8_fullanchors_geomv9__subset_multipi_blocks2`
- Refined Phase-3E run:
  - `out_phase3E_elambda_blocks2_subset_multipi_localrefine0p00_0p25`
  - `n_pairs=21`, `blocks=2`, `lambdas=[0,0.0125,...,0.25,0.5]`

Result: the fixed-E scan remains empty at `tol_pi=0.25`:

- `out_phase3E_elambda_blocks2_subset_multipi_localrefine0p00_0p25/phase3f_multipi_pairs_fixedE_tol0p25.csv` (header only)

Interpretation: for these pairs (and this λ range), π events behave like **a single narrow λ wall-band per fixed E-rectangle**, but the same underlying wall can be detected by multiple E-rectangles (hence multi-cell π across E). This supports the picture of a codim-1 wall sheet that is localized in λ but extended in E.

### Addendum (continued): widened to `[0,0.5]` still empty

We then widened the dense refinement to `[0,0.5]` (step 0.0125), using a faster “fwd-rectangle only” suite mode:

- Refined Phase-3E run:
  - `out_phase3E_elambda_blocks2_subset_multipi_localrefine0p00_0p50`
  - `suite_mode=fwd_rect`, `n_rows=26460`, `valid_fraction=1.0`

Result: fixed-E multi-λ candidates remain empty, even at a looser tolerance:

- `tol_pi=0.25`: `out_phase3E_elambda_blocks2_subset_multipi_localrefine0p00_0p50/phase3f_multipi_pairs.csv` (header only)
- `tol_pi=0.4`: `out_phase3E_elambda_blocks2_subset_multipi_localrefine0p00_0p50/phase3f_multipi_pairs_tol0p4_fixedE.csv` (header only)

Interpretation: within this targeted multipi subset and for λ up to 0.5, **each fixed E-rectangle sees at most one π λ-cell** per (pair,block). Multi-cell π behavior continues to arise from varying the E-rectangle (probing the same wall sheet in different E locations), not from multiple distinct λ-bands at the same E.
