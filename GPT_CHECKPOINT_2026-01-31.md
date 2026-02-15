# GPT checkpoint — 2026-01-31

## Executive summary

We now have strong, falsifiable evidence that the det-phase $\approx\pi$ events behave like thin “walls” in $\lambda$ (for fixed E-loop) under the legacy↔geomv9 Λ-deformation:

- After fixing gauge/basis-reuse artifacts, $(E,\eta)$ holonomy became trivial ($\approx I$), so $\eta$ is not a meaningful second parameter.
- Under true Λ-deformation (legacy↔geomv9), holonomy det-phase quantizes near 0/π with robust loop sanity checks.
- π-events are *always* concentrated in a conditioning-stress subset (small `step_smin_min`); above-threshold π-rate collapses to 0.
- Within-backend pseudo-geometry controls (anchor-offset pairing within same backend) produce π=0 across suites.

Interpretation: π is deformation-specific and appears only when the discrete transport hits a conditioning-stress locus.

## New results since last checkpoint

### 1) Deeper micro-refine for (0,2)

We refined around the previously localized (0,2) wall near $\lambda\approx 0.41$.

Run:
- [out_phase3E_elambda_scale8_legacy_geomv9_autorefine4_v6_lp02_fg/phase3e_elambda_suite_summary.json](out_phase3E_elambda_scale8_legacy_geomv9_autorefine4_v6_lp02_fg/phase3e_elambda_suite_summary.json)
- [out_phase3E_elambda_scale8_legacy_geomv9_autorefine4_v6_lp02_fg/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv](out_phase3E_elambda_scale8_legacy_geomv9_autorefine4_v6_lp02_fg/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv)
- [out_phase3E_elambda_scale8_legacy_geomv9_autorefine4_v6_lp02_fg/phase3e_elambda_wall_cells_top20.csv](out_phase3E_elambda_scale8_legacy_geomv9_autorefine4_v6_lp02_fg/phase3e_elambda_wall_cells_top20.csv)

Key readout:
- Overall π fraction: `0.0056818`
- π micro-intervals further localized to:
  - $[0.4065398437, 0.4069345982]$ with `pi_rate=0.015625`
  - $[0.4189544643, 0.4193451531]$ with `pi_rate=0.015625`
- Conditioning gate remains sharp: π-rate above `step_smin_min > 0.05` is `0.0`

### 2) Rectangle/path surgery for (1,2) edge π

Slice-swap on the autorefine3 grid showed (1,2) had π only on coarse edge intervals (not near the mid-band λ≈0.41 wall).
We refined those edge intervals to test whether they also localize.

Low-λ dense run $\lambda\in[0,0.26]$:
- [out_phase3E_elambda_scale8_legacy_geomv9_rectsurg_v1_lp12_band0p00_0p26/phase3e_elambda_suite_summary.json](out_phase3E_elambda_scale8_legacy_geomv9_rectsurg_v1_lp12_band0p00_0p26/phase3e_elambda_suite_summary.json)
- [out_phase3E_elambda_scale8_legacy_geomv9_rectsurg_v1_lp12_band0p00_0p26/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv](out_phase3E_elambda_scale8_legacy_geomv9_rectsurg_v1_lp12_band0p00_0p26/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv)
- [out_phase3E_elambda_scale8_legacy_geomv9_rectsurg_v1_lp12_band0p00_0p26/phase3e_elambda_wall_cells_top20.csv](out_phase3E_elambda_scale8_legacy_geomv9_rectsurg_v1_lp12_band0p00_0p26/phase3e_elambda_wall_cells_top20.csv)

Key readout:
- Overall π fraction: `0.0004381`
- π localizes to a narrow window near $\lambda\approx 0.156$: $[0.155, 0.1575]$ with `pi_rate=0.015625`
- Conditioning gate: π-rate above `step_smin_min > 0.01` is `0.0`

High-λ dense run $\lambda\in[0.49,1.0]$:
- [out_phase3E_elambda_scale8_legacy_geomv9_rectsurg_v1_lp12_band0p49_1p00/phase3e_elambda_suite_summary.json](out_phase3E_elambda_scale8_legacy_geomv9_rectsurg_v1_lp12_band0p49_1p00/phase3e_elambda_suite_summary.json)
- [out_phase3E_elambda_scale8_legacy_geomv9_rectsurg_v1_lp12_band0p49_1p00/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv](out_phase3E_elambda_scale8_legacy_geomv9_rectsurg_v1_lp12_band0p49_1p00/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv)
- [out_phase3E_elambda_scale8_legacy_geomv9_rectsurg_v1_lp12_band0p49_1p00/phase3e_elambda_wall_cells_top20.csv](out_phase3E_elambda_scale8_legacy_geomv9_rectsurg_v1_lp12_band0p49_1p00/phase3e_elambda_wall_cells_top20.csv)

Key readout:
- Overall π fraction: `0.0004507`
- π localizes to two narrow windows:
  - $[0.5, 0.505]$ with `pi_rate=0.015625`
  - $[0.93, 0.935]$ with `pi_rate=0.015625`
- Conditioning gate remains strong: π-rate above `step_smin_min > 0.05` is `0.0`

## Current best narrative

- π-events behave like wall-crossings in $\lambda$ under cross-backend Λ-deformation.
- The wall location depends on the E-loop (e.g., (0,1) and (0,2) share the ~0.41 wall; (1,2) has different walls).
- The mechanism signature remains consistent: π rows show an odd −1 eigenvalue (holonomy has exactly one eigenvalue near −1).
- Conditioning appears to be a *necessary* trigger (π absent above thresholds).

## Questions for GPT

1) Conceptual model: is “thin λ walls + E-dependent location + Z2 det-phase” most consistent with a codim-1 degeneracy surface in (E,λ), where the loop crosses it?
2) Since π always lives at small `step_smin_min`, what is the best way to argue this is a *physical* near-degeneracy (gap closure/avoided crossing) rather than numeric instability?
3) What’s the cleanest next discriminating experiment?
   - Deeper micro-refine each (1,2) window (e.g. [0.155,0.1575], [0.5,0.505], [0.93,0.935]) to see if each is a single wall.
   - “Path surgery” in λ: shift rectangles to avoid/include a wall and confirm π toggles only when crossing.
   - Build a direct “wall proxy curve”: λ-midpoint vs `step_smin_min` and `holonomy_min_dist_to_minus1`, stratified by det-phase class.
