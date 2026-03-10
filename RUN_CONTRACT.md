# Run Contract (Experiment E)

This document freezes the **minimum required artifacts** for an Experiment E run directory under `runs/`, so that later mathematical/prose claims are not undermined by “the harness changed”.

The Experiment E driver is `experiments/exp_E_doc_validation_track.py`.

## 1) Always-required run artifacts

Every run directory must contain:

- `config.json` — resolved configuration used for the run.
- `doc_meta.json` — doc-aligned metadata snapshot used by the driver.
- `E_summary.md` — human-readable summary of what executed.

The driver enforces presence by default (`run_contract_enforce: true`).

## 2) Convergence-sweep-only contract (`do_convergence_sweep_only: true`)

Required:

- `E_Qlambda_convergence_sweep.csv`
  - Must include columns: `N`, `basis`, `block`, `boundary_b0`, `err_phi_norm_p95`, `err_phi_norm_top5_mean`, `err_phi_norm_median`, `gap_phi_min`.
- `E_Qlambda_convergence_fit.csv`
  - Fit summary table for convergence curves.

If `convergence_do_winding: true`, also required:

- `E_Qlambda_convergence_winding.csv` (or `convergence_wind_out_csv` if overridden)
  - Must include columns: `source`, `N`, `basis`, `block`, `n_edge`, `wind_track`, `wind_pointwise`, `failures`, `eta`.
  - **Policy:** Any divisor-clean / argument-principle claim must reference the **tracked** channel (`wind_track*`).
    `wind_pointwise*` is permitted only as a selector-stability diagnostic.

## 3) Boundary autopick contract (`boundary_autopick: true`)

Required:

- `E_boundary_autopick_policy.md`
  - Textual policy statement (candidate set, constraints, objective, guard, winding check channel).
- `E_boundary_autopick_candidates.csv`
  - One row per candidate `delta` per `N`.
  - Must include booleans: `metric_ok`, `gap_ok`, `wind_ok_stage1`, `top5_guard_ok`.
- `E_boundary_autopick_choice.csv`
  - Must include columns: `N`, `chosen_delta`.
- `E_boundary_autopick_acceptance.csv`
  - Per-`N` acceptance and rejection counts.
  - Must include both:
    - Independent fails: `n_fail_metric`, `n_fail_gap`, `n_fail_wind_stage1`, `n_fail_top5_guard`
    - Ordered pipeline counts: `pipeline_order`, `n_after_*`, `n_rej_*`

## 4) Optional modules (required only if enabled)

- Dip atlas (`do_dip_atlas: true`):
  - `E_dip_atlas_points.csv`
  - `E_dip_atlas_lines.csv`
  - `E_dip_atlas_suggested_rect.json`

- Pole-line scan (`do_pole_line_scan: true`):
  - `E_pole_line_scan.csv`

- Argument principle / divisor diagnostics (non-convergence runs):
  - If `do_arg_c_emp: true`: `E_arg_principle_c_emp.csv`
  - If `do_arg_lambda_branch: true`: `E_divisor_arg_principle_Q_lambda.csv`

## 5) Cross-run comparison artifact

The script `tools/compare_expE_convergence_runs.py` writes:

- `runs/_compare_latest_baseline_vs_autopick.csv`
  - Per-`N` worst-channel curve plus acceptance counts.

This file is not produced by a single run directory; it is an aggregate report.
