# RESULTS PACKET — 2026-01-31

## What changed today

Added a fast postprocessor to quantify:
- (A) π-event concentration vs conditioning, and
- (B) wall localization in $(E,\lambda)$ cells,

without rerunning any expensive suites.

Postprocessor:
- [phase3e_elambda_wall_postprocess.py](phase3e_elambda_wall_postprocess.py)

## Direction A — π rate vs conditioning threshold

Cross-backend run analyzed:
- [out_phase3E_elambda_scale8_legacy_geomv9_gidkpi_hotspots/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv](out_phase3E_elambda_scale8_legacy_geomv9_gidkpi_hotspots/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv)

Key readout (from that CSV):
- For `step_smin_min <= 0.05`: coverage 0.1068 of rows; π-rate below = 0.195; π-rate above = 0.0
- For `step_smin_min <= 0.10`: coverage 0.2070 of rows; π-rate below = 0.1006; π-rate above = 0.0
- For `step_smin_min <= 0.01`: coverage 0.0286 of rows; π-rate below = 0.4545; π-rate above = 0.0080

Interpretation: for this run, π-events are concentrated in a small conditioning-stress subset; outside that subset they are absent.

Within-backend controls (anchor offset 3) analyzed:
- [out_phase3E_elambda_within_legacy_anchoroff3_full/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv](out_phase3E_elambda_within_legacy_anchoroff3_full/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv)
- [out_phase3E_elambda_within_geomv9_anchoroff3_full/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv](out_phase3E_elambda_within_geomv9_anchoroff3_full/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv)

Result: π-rate is 0.0 for all thresholds in both within-backend controls.

## Direction B — wall localization in $(E,\lambda)$

Cross-backend wall tables:
- Per-loop collapsed stats: [out_phase3E_elambda_scale8_legacy_geomv9_gidkpi_hotspots/phase3e_elambda_loop_wall_stats.csv](out_phase3E_elambda_scale8_legacy_geomv9_gidkpi_hotspots/phase3e_elambda_loop_wall_stats.csv)
- Per-cell wall map: [out_phase3E_elambda_scale8_legacy_geomv9_gidkpi_hotspots/phase3e_elambda_wall_cells.csv](out_phase3E_elambda_scale8_legacy_geomv9_gidkpi_hotspots/phase3e_elambda_wall_cells.csv)
- Top-20 hot cells: [out_phase3E_elambda_scale8_legacy_geomv9_gidkpi_hotspots/phase3e_elambda_wall_cells_top20.csv](out_phase3E_elambda_scale8_legacy_geomv9_gidkpi_hotspots/phase3e_elambda_wall_cells_top20.csv)

Top hot cells (highest `pi_rate`) currently are:
- $(E_0,E_1)=(0,2)$, $\lambda\in[0.25,0.50]$: `pi_rate = 0.046875`, `min_step_smin_min ≈ 0.002674`
- $(E_0,E_1)=(0,1)$, $\lambda\in[0.25,0.50]$: `pi_rate = 0.046875`, `min_step_smin_min ≈ 0.006387`

## How to run

To postprocess any existing E–λ run:

```
python phase3e_elambda_wall_postprocess.py --out_root <OUT_DIR>
```

Example:

```
python phase3e_elambda_wall_postprocess.py --out_root out_phase3E_elambda_scale8_legacy_geomv9_gidkpi_hotspots
```

## Next minimal-runtime follow-ups

- Micro-refine one hot interval: rerun E–λ with a dense local `--lambdas` list over the hottest λ segment (e.g. 21 points from 0.25 to 0.50) to see whether π-events cluster into a tighter band.
- Blocks invariance: rerun cross-backend E–λ with `--blocks 2` and `--blocks 8` (equalcount scheme) and compare (i) overall π fraction and (ii) the `holonomy_num_eigs_near_minus1` signature inside π rows.

## Update — micro-refinement (local λ grid)

Dense-λ run (focused on the coarse-hot interval $\lambda\in[0.25,0.50]$):
- [out_phase3E_elambda_scale8_legacy_geomv9_localrefine_0p25_0p50/phase3e_elambda_suite_summary.json](out_phase3E_elambda_scale8_legacy_geomv9_localrefine_0p25_0p50/phase3e_elambda_suite_summary.json)
- [out_phase3E_elambda_scale8_legacy_geomv9_localrefine_0p25_0p50/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv](out_phase3E_elambda_scale8_legacy_geomv9_localrefine_0p25_0p50/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv)
- [out_phase3E_elambda_scale8_legacy_geomv9_localrefine_0p25_0p50/phase3e_elambda_wall_cells_top20.csv](out_phase3E_elambda_scale8_legacy_geomv9_localrefine_0p25_0p50/phase3e_elambda_wall_cells_top20.csv)

Key readout (from the thresholds CSV):
- For `step_smin_min <= 0.05`: coverage 0.1196 of rows; π-rate below = 0.03693; π-rate above = 0.0
- Overall π fraction (threshold 1.0): 0.004416

Wall localization note (from `wall_cells_top20.csv`): the refined band shows π-hot sub-intervals around $\lambda\approx 0.40$–$0.45$ at resolution 0.0125 (e.g. [0.4000,0.4125], [0.4125,0.4250], [0.4375,0.4500]).

## Update — blocks invariance sweep (Direction C)

Blocks=2 run:
- [out_phase3E_elambda_scale8_legacy_geomv9_alpha1p00_blocks2/phase3e_elambda_suite_summary.json](out_phase3E_elambda_scale8_legacy_geomv9_alpha1p00_blocks2/phase3e_elambda_suite_summary.json)
- [out_phase3E_elambda_scale8_legacy_geomv9_alpha1p00_blocks2/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv](out_phase3E_elambda_scale8_legacy_geomv9_alpha1p00_blocks2/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv)

Key readout (blocks=2):
- Overall π fraction: 0.03125
- For `step_smin_min <= 0.05`: coverage 0.09375; π-rate below = 0.3333; π-rate above = 0.0

Blocks=8 run:
- [out_phase3E_elambda_scale8_legacy_geomv9_alpha1p00_blocks8/phase3e_elambda_suite_summary.json](out_phase3E_elambda_scale8_legacy_geomv9_alpha1p00_blocks8/phase3e_elambda_suite_summary.json)
- [out_phase3E_elambda_scale8_legacy_geomv9_alpha1p00_blocks8/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv](out_phase3E_elambda_scale8_legacy_geomv9_alpha1p00_blocks8/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv)

Key readout (blocks=8):
- Overall π fraction: 0.020833
- For `step_smin_min <= 0.05`: coverage 0.1068; π-rate below = 0.1951; π-rate above = 0.0

Interpretation: π persists under charting changes (blocks 2→8), and remains concentrated in conditioning-stress subsets (above-threshold π-rate collapses to 0.0).

## Update — deformation amplitude α (Direction D)

Completed α=0.25 run (λ_eff = 0.25·λ):
- [out_phase3E_elambda_scale8_legacy_geomv9_alpha0p25_blocks4/phase3e_elambda_suite_summary.json](out_phase3E_elambda_scale8_legacy_geomv9_alpha0p25_blocks4/phase3e_elambda_suite_summary.json)
- [out_phase3E_elambda_scale8_legacy_geomv9_alpha0p25_blocks4/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv](out_phase3E_elambda_scale8_legacy_geomv9_alpha0p25_blocks4/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv)

Key readout (α=0.25):
- Overall π fraction: 0.005208
- For `step_smin_min <= 0.01`: coverage 0.02604; π-rate below = 0.2; π-rate above = 0.0
- For `step_smin_min <= 0.05`: coverage 0.09635; π-rate below = 0.05405; π-rate above = 0.0

Interpretation: decreasing α strongly suppresses π-events while keeping them confined to conditioning-stress regions.

Additional completed α runs (postprocessed):

- α=0.50: [out_phase3E_elambda_scale8_legacy_geomv9_alpha0p50_blocks4/phase3e_elambda_suite_summary.json](out_phase3E_elambda_scale8_legacy_geomv9_alpha0p50_blocks4/phase3e_elambda_suite_summary.json), [out_phase3E_elambda_scale8_legacy_geomv9_alpha0p50_blocks4/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv](out_phase3E_elambda_scale8_legacy_geomv9_alpha0p50_blocks4/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv)
	- Overall π fraction: 0.013021
	- For `step_smin_min <= 0.05`: coverage 0.1016; π-rate below = 0.1282; π-rate above = 0.0

- α=0.75: [out_phase3E_elambda_scale8_legacy_geomv9_alpha0p75_blocks4/phase3e_elambda_suite_summary.json](out_phase3E_elambda_scale8_legacy_geomv9_alpha0p75_blocks4/phase3e_elambda_suite_summary.json), [out_phase3E_elambda_scale8_legacy_geomv9_alpha0p75_blocks4/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv](out_phase3E_elambda_scale8_legacy_geomv9_alpha0p75_blocks4/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv)
	- Overall π fraction: 0.015625
	- For `step_smin_min <= 0.05`: coverage 0.1120; π-rate below = 0.1395; π-rate above = 0.0

- α=1.00 (blocks=4 baseline): [out_phase3E_elambda_scale8_legacy_geomv9_alpha1p00_blocks4/phase3e_elambda_suite_summary.json](out_phase3E_elambda_scale8_legacy_geomv9_alpha1p00_blocks4/phase3e_elambda_suite_summary.json), [out_phase3E_elambda_scale8_legacy_geomv9_alpha1p00_blocks4/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv](out_phase3E_elambda_scale8_legacy_geomv9_alpha1p00_blocks4/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv)
	- Overall π fraction: 0.020833
	- For `step_smin_min <= 0.05`: coverage 0.1068; π-rate below = 0.1951; π-rate above = 0.0

Sweep summary (overall π fraction):
- α=0.25: 0.005208
- α=0.50: 0.013021
- α=0.75: 0.015625
- α=1.00: 0.020833

## Update — α sweep compact CSV artifact

Generated a compact α “phase diagram” table (blocks=4) combining overall π fraction + π-rate/coverage at thresholds:
- [out_phase3E_elambda_alpha_sweep_summary_blocks4.csv](out_phase3E_elambda_alpha_sweep_summary_blocks4.csv)

## Update — slice swap on autorefine3 λ grid

Goal: test whether the localized wall near $\lambda\approx 0.41$ is shared across E-pairs.

Slice swap runs (same refined `lambdas` as the autorefine3 (0:1) run):
- (0,2): [out_phase3E_elambda_scale8_legacy_geomv9_autorefine3_v5_lp02_fg/phase3e_elambda_suite_summary.json](out_phase3E_elambda_scale8_legacy_geomv9_autorefine3_v5_lp02_fg/phase3e_elambda_suite_summary.json), [out_phase3E_elambda_scale8_legacy_geomv9_autorefine3_v5_lp02_fg/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv](out_phase3E_elambda_scale8_legacy_geomv9_autorefine3_v5_lp02_fg/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv), [out_phase3E_elambda_scale8_legacy_geomv9_autorefine3_v5_lp02_fg/phase3e_elambda_wall_cells_top20.csv](out_phase3E_elambda_scale8_legacy_geomv9_autorefine3_v5_lp02_fg/phase3e_elambda_wall_cells_top20.csv)
	- Overall π fraction: 0.005952
	- The same two micro-intervals are π-hot: $[0.405366,0.406924]$ and $[0.417801,0.419327]$ with `pi_rate=0.015625`.
	- Conditioning gate remains sharp: π-rate above `step_smin_min > 0.05` is 0.0.
- (1,2): [out_phase3E_elambda_scale8_legacy_geomv9_autorefine3_v5_lp12_fg/phase3e_elambda_suite_summary.json](out_phase3E_elambda_scale8_legacy_geomv9_autorefine3_v5_lp12_fg/phase3e_elambda_suite_summary.json), [out_phase3E_elambda_scale8_legacy_geomv9_autorefine3_v5_lp12_fg/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv](out_phase3E_elambda_scale8_legacy_geomv9_autorefine3_v5_lp12_fg/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv), [out_phase3E_elambda_scale8_legacy_geomv9_autorefine3_v5_lp12_fg/phase3e_elambda_wall_cells_top20.csv](out_phase3E_elambda_scale8_legacy_geomv9_autorefine3_v5_lp12_fg/phase3e_elambda_wall_cells_top20.csv)
	- Overall π fraction: 0.002232
	- No π events in the refined mid-band intervals near $\lambda\approx 0.41$; π appears only in coarse edge intervals (e.g. $[0,0.25]$, $[0.5,0.75]$, $[0.75,1.0]$).
	- Conditioning gate remains sharp: π-rate above `step_smin_min > 0.05` is 0.0.

Interpretation: the wall location is E-dependent (shared by (0,1) and (0,2), but not by (1,2) on this λ grid), while the “π only under conditioning stress” rule continues to hold.

## Update — deeper micro-refine for (0,2) wall

Tighter micro-refine around the two (0,2) π-hot micro-intervals inside $\lambda\in[0.39,0.43]$:
- [out_phase3E_elambda_scale8_legacy_geomv9_autorefine4_v6_lp02_fg/phase3e_elambda_suite_summary.json](out_phase3E_elambda_scale8_legacy_geomv9_autorefine4_v6_lp02_fg/phase3e_elambda_suite_summary.json)
- [out_phase3E_elambda_scale8_legacy_geomv9_autorefine4_v6_lp02_fg/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv](out_phase3E_elambda_scale8_legacy_geomv9_autorefine4_v6_lp02_fg/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv)
- [out_phase3E_elambda_scale8_legacy_geomv9_autorefine4_v6_lp02_fg/phase3e_elambda_wall_cells_top20.csv](out_phase3E_elambda_scale8_legacy_geomv9_autorefine4_v6_lp02_fg/phase3e_elambda_wall_cells_top20.csv)

Key readout:
- Overall π fraction (threshold 1.0): 0.0056818
- The two previously-identified π micro-intervals further localize to:
	- $[0.4065398437, 0.4069345982]$ with `pi_rate=0.015625`
	- $[0.4189544643, 0.4193451531]$ with `pi_rate=0.015625`
- Conditioning gate remains sharp: π-rate above `step_smin_min > 0.05` is 0.0

Interpretation: on (0,2), the wall-crossing continues to behave like a thin λ-locus (collapsing with refinement rather than “smearing out”).

## Update — rectangle surgery for (1,2) edge π events

Goal: for (1,2), refine the coarse edge intervals where π appeared (in the slice-swap run) and see whether they localize into narrow λ windows.

Low-λ edge refinement (dense on $\lambda\in[0,0.26]$):
- [out_phase3E_elambda_scale8_legacy_geomv9_rectsurg_v1_lp12_band0p00_0p26/phase3e_elambda_suite_summary.json](out_phase3E_elambda_scale8_legacy_geomv9_rectsurg_v1_lp12_band0p00_0p26/phase3e_elambda_suite_summary.json)
- [out_phase3E_elambda_scale8_legacy_geomv9_rectsurg_v1_lp12_band0p00_0p26/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv](out_phase3E_elambda_scale8_legacy_geomv9_rectsurg_v1_lp12_band0p00_0p26/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv)
- [out_phase3E_elambda_scale8_legacy_geomv9_rectsurg_v1_lp12_band0p00_0p26/phase3e_elambda_wall_cells_top20.csv](out_phase3E_elambda_scale8_legacy_geomv9_rectsurg_v1_lp12_band0p00_0p26/phase3e_elambda_wall_cells_top20.csv)

Key readout:
- Overall π fraction: 0.0004381
- π localizes to a narrow window near $\lambda\approx 0.156$: $[0.155,0.1575]$ with `pi_rate=0.015625`
- Conditioning gate: π-rate above `step_smin_min > 0.01` is 0.0

High-λ edge refinement (dense on $\lambda\in[0.49,1.0]$):
- [out_phase3E_elambda_scale8_legacy_geomv9_rectsurg_v1_lp12_band0p49_1p00/phase3e_elambda_suite_summary.json](out_phase3E_elambda_scale8_legacy_geomv9_rectsurg_v1_lp12_band0p49_1p00/phase3e_elambda_suite_summary.json)
- [out_phase3E_elambda_scale8_legacy_geomv9_rectsurg_v1_lp12_band0p49_1p00/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv](out_phase3E_elambda_scale8_legacy_geomv9_rectsurg_v1_lp12_band0p49_1p00/phase3e_elambda_pi_rate_thresholds_step_smin_min.csv)
- [out_phase3E_elambda_scale8_legacy_geomv9_rectsurg_v1_lp12_band0p49_1p00/phase3e_elambda_wall_cells_top20.csv](out_phase3E_elambda_scale8_legacy_geomv9_rectsurg_v1_lp12_band0p49_1p00/phase3e_elambda_wall_cells_top20.csv)

Key readout:
- Overall π fraction: 0.0004507
- π localizes to two narrow windows:
	- $[0.5,0.505]$ with `pi_rate=0.015625`
	- $[0.93,0.935]$ with `pi_rate=0.015625`
- Conditioning gate remains strong: π-rate above `step_smin_min > 0.05` is 0.0

Interpretation: (1,2) also exhibits wall-like localization under refinement, but at different λ locations than the (0,*) walls.
