# Frozen Oracle Sanity Sheet

This note records the final minimal validation pass for the frozen anchored RH/scattering pipeline before treating it as a structured oracle for AI/ML work. The goal is not to claim a completed theorem. The goal is to document what has in fact been numerically stabilized well enough to justify using the architecture as a frozen inductive prior.

## Frozen operating point

- Frozen backend coefficients: `out/corrected_factor_injection_beta23_window_coefficients.csv`
- Canonical frontend: `A3_exact_frontend`
- Canonical completion: `completion_even_mode=anchored_real`
- Default quartic anchor: `completion_even_a4=-0.2`
- Canonical analytic operating point: `u=0.24`

## Validation status

### Test A. Euler-style closure validation

Interpretation: pass on the stronger frozen `beta23_plus_c` artifact; partial on the fresh generic `c_plus_bOmega` sweep.

Fresh generic closure rerun:

- Artifact: `out/final_validation_closure.csv`
- Best generic corrected model by total loss: `c_plus_bOmega`
- Median eval comparison versus `none`:
  - `eval_E_sf`: `0.161254` vs `0.954558` for the stronger frozen run, but on the fresh rerun the corrected model improves squarefree and mixed terms while not beating baseline on prime-power error.
  - holdout behavior is similar: better squarefree, better mixed, better total loss, but not better prime-power.
- Conclusion: the fresh minimal generic fit shows coherent improvement but does not by itself satisfy the strictest all-metrics Euler pass condition.

Frozen stronger closure artifact:

- Artifact: `out/ghostlift_closure_beta23_mumix050_fit17_eval29_hold19_23.csv`
- Best corrected model: `beta23_plus_c`
- Median eval metrics:
  - `beta23_plus_c`: `eval_E_sf=0.165631`, `eval_E_pp=0.846297`, `eval_E_mix=1.054788`, `eval_L=1.561769`
  - `none`: `eval_E_sf=0.954558`, `eval_E_pp=1.132957`, `eval_E_mix=1.775693`, `eval_L=2.972947`
- Median holdout metrics:
  - `beta23_plus_c`: `holdout_E_sf=0.233999`, `holdout_E_pp=1.158028`, `holdout_E_mix=1.530673`, `holdout_L=2.143855`
  - `none`: `holdout_E_sf=1.043889`, `holdout_E_pp=2.065819`, `holdout_E_mix=2.499357`, `holdout_L=4.359386`
- Conclusion: on the frozen `beta23_plus_c` comparison, the corrected model beats the uncorrected baseline on all three requested closure components and on total loss, on both eval and holdout.

What this validates:

- the frozen arithmetic correction materially improves Euler-style closure diagnostics,
- the improvement is not limited to one metric family,
- and the corrected arithmetic layer is strong enough to serve as a frozen prior for downstream oracle features.

What this does not validate:

- exact equality with the classical Euler product.

### Test B. Mangoldt validation

Interpretation: pass.

Artifacts:

- `out/final_validation_mangoldt_sixby6.csv`
- `out/final_validation_mangoldt_corrected2x2.csv`
- `out/final_validation_mangoldt_a3exact.csv`

At `sigma=2.0`, using the repo’s Mangoldt/log-derivative probe, the best current frontend (`A3_exact_frontend`) beats both the six-by-six baseline and the corrected 2x2 injection on the two user-requested metrics.

Key worst-case relative RMSE figures:

- sixby6 baseline:
  - per-prime `log p` ladder: about `0.000276`
  - global Mangoldt template: about `0.134459`
- corrected 2x2 injection:
  - per-prime `log p` ladder: about `0.000247`
  - global Mangoldt template: about `0.160181`
- `A3_exact_frontend`:
  - per-prime `log p` ladder: about `0.000184`
  - global Mangoldt template: about `0.037430`

Conclusion:

- the accepted frontend wins on both the `log p` ladder metric and the global Mangoldt-template metric,
- so the current frontend is the right frozen realization to carry into oracle work.

What this does not validate:

- an exact Mangoldt identity or exact logarithmic-derivative identification theorem.

### Test C. Critical-line local centering validation

Interpretation: pass.

Artifacts:

- `out/final_validation_anchored_fe_summary.csv`
- `out/final_validation_anchored_sigma_scan_summary.csv`
- `out/final_validation_anchored_zero_tracking.csv`

Anchored completed-object local scan near the persistent `t≈28` cluster:

- `sigma_at_min_abs_det = 0.5`
- `critical_line_preference_gap = 0.0`
- `abs_det_at_critical = min_abs_det`

Tracked across `u=0.21, 0.22, 0.23, 0.24`:

- `sigma_at_transverse_min = 0.5` on every row,
- `critical_line_preferred = True` on every row,
- the candidate cluster stays in the same narrow `t≈27.95..28.1` band.

Conclusion:

- the anchored completion recenters the local transverse minimum onto the critical line,
- and it does so without destroying the existing zero-candidate cluster identity.

What this does not validate:

- a theorem that all zeros of the completed object lie on the critical line.

### Test D. Oracle stability validation

Interpretation: pass.

Artifact:

- `out/final_validation_oracle_feature_table.csv`

Compared nearby operating points `u=0.22, 0.23, 0.24`:

- movement from `u=0.24` to `u=0.23`:
  - median absolute feature difference about `0.0113`
  - median relative feature difference about `0.0619`
- movement from `u=0.24` to `u=0.22`:
  - median absolute feature difference about `0.0331`
  - median relative feature difference about `0.1720`

Persistent identities across the compared rows:

- `tracking.sigma_at_transverse_min = 0.5`
- `global.best_zero_t` remains in the same `t≈27.85..28.0` cluster
- FE, zero-cluster, closure, and rigidity features stay finite and move smoothly rather than flipping discontinuously.

Conclusion:

- the frozen oracle feature map is locally stable under small changes in `u`,
- so it is suitable as a versioned structured feature source for AI/ML experiments.

What this does not validate:

- that the oracle is uniquely canonical in any theorem-level sense,
- or that feature smoothness alone implies mathematical correctness of the global object.

## Bottom line

What we can now say cleanly:

- the frozen corrected arithmetic layer improves Euler-style closure on the strongest frozen comparison artifact,
- the accepted frontend wins the requested Mangoldt diagnostics against the main baselines,
- the anchored completion centers the persistent local zero cluster on `sigma=0.5`,
- and the exported oracle features are stable enough to use as a frozen inductive-bias source.

This is enough to justify the repo’s next phase:

- use the anchored RH/scattering stack as a frozen structured oracle for AI/ML experiments,
- while keeping the remaining analytic identification statements explicitly open.

Those remaining open statements are still:

- exact classical Euler-product equality,
- exact Mangoldt/log-derivative identity,
- equality with the Riemann zeta function or a classical scattering determinant,
- and a proof that all zeros lie on the critical line.