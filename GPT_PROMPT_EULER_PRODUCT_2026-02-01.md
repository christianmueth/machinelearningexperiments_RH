# Prompt to send to GPT (Euler-product / primes-surrogate formalization)

You are helping formalize a “Route‑B primes surrogate” narrative for a computational pipeline that finds stable π-wall generators in a deformation family and tests their nonabelian algebra.

## What we have (current state)

We have a pipeline:
1) Phase‑3E: scan an (E, λ) grid (and sometimes µ) for loop holonomies; detect π-events (det-phase near π) and dump holonomies.
2) Phase‑3F: cluster π-cells into stable wall generators; build a primitive dictionary (primitive generator classes, with a v_-1 eigenvector signature to avoid involution collapse); run nonabelian tests (commutators and word tests).
3) Refinement/persistence: refine λ-grids around hotspots and re-run; candidates are ranked by persistence + nonabelianity.

Important constraint: we treat “true 2D” in (λ,µ) as a hard gate. µ must not collapse (rank‑2 direction) in a structurally meaningful witness.

## Flagship vs mismatch cases

Flagship: seed80/a14
Mismatch: seed96/a14

Wide‑µ validation summaries include a set of µ-direction “lenses” comparing the µ direction (B2−B1) vs the λ direction (B1−A) at snapshot energies. The KPI is median |cos| between those directions; small = good (non-colinear), near 1 = collapse.

For the validate runs (n_keys=1 each):

- seed80/a14 validate:
  - lambda_snap: |cos|≈0.133, ratio≈11.11, rank≈2
  - S_snap:     |cos|≈0.119, ratio≈11.63, rank≈2
  - trace_powers: |cos|=1, ratio≈1, rank=1
  - logdet_IminusS: |cos|=1, ratio≈1, rank=1
  - series_partial: |cos|≈0.493, ratio≈0.00235, rank≈2
  - resid_partial:  |cos|≈0.918, ratio≈0.579, rank≈2

- seed96/a14 validate:
  - lambda_snap: |cos|≈0.150, ratio≈10.48, rank≈2
  - S_snap:     |cos|≈0.996, ratio≈1.001, rank≈2  (strong collapse)
  - trace_powers: |cos|=1, ratio≈1, rank=1
  - logdet_IminusS: |cos|=1, ratio≈1, rank=1
  - series_partial: |cos|≈0.776, ratio≈0.598, rank≈2
  - resid_partial:  |cos|≈0.985, ratio≈0.560, rank≈2

Interpretation so far: seed80/a14 passes the conservative µ gate (lambda_snap + S_snap). seed96/a14 fails because S_snap is nearly colinear even though lambda_snap looks 2D.

## What we want from you

We want a mathematically defensible “Euler product / zeta-like” analogue that uses:
- primitive generator classes (prime-like objects),
- word algebra / noncommutativity (nonabelian structure),
- refinement persistence (stability under zoom),
- but does NOT overclaim literal integer primes.

### Questions

1) Given a collection of primitive generators with associated holonomy matrices (unitary-ish), what are 1–2 clean, defensible definitions of a zeta-like object Z that factorizes over primitive objects (Euler-product style), e.g.
   - commutative: Z(s)=∏_p (1-w(p,s))^{-1}
   - or a trace/logdet formulation: log Z(s)=Σ_{k≥1} (1/k) Σ_p w(p,s)^k
   - or a matrix-valued / noncommutative analogue (if appropriate)

2) What should we use for the primitive weights w(p,s) so that:
   - they are stable under refinement,
   - they respect µ‑2D gating (avoid collapse artifacts),
   - they connect to observables we already compute (det-phase, traces, spectral gaps, ||I−M||, etc.)?

3) What are the minimal “sanity checks” to claim an Euler-product‑like expansion here?
   Examples: convergence diagnostics; independence from gauge choices; stability under lasso/basepoint choices; robustness to π-tolerance.

4) How should we phrase the narrative so it’s precise?
   We want wording like: “primitive cycle factorization / prime-like generators” rather than “primes”.

## Reference artifacts you can assume exist

- Checkpoints:
  - GPT_CHECKPOINT_2026-01-31.md
  - GPT_CHECKPOINT_2026-02-01.md

- Current smoke root outputs:
  - out_phase3E_elambda_2D_mu_B2pilotv2_smoke/prime_persistence_table.csv
  - out_phase3E_elambda_2D_mu_B2pilotv2_smoke/refined_ranking.csv

- Validate runs:
  - out_phase3E_elambda_2D_mu_B2pilotv2_smoke/validate_mugrid_prime_seed80_a14_w2_5_mu0_1_rs3/phase3e_elambda_suite_summary.json
  - out_phase3E_elambda_2D_mu_B2pilotv2_smoke/validate_mugrid_prime_seed96_a14_w0.6_7.5_mu0_1_rs3/phase3e_elambda_suite_summary.json

Please propose concrete definitions + a short experimental checklist we can run next to support them.
