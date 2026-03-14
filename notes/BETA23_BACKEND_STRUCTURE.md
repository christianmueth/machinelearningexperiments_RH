# Beta23 Backend Structure

This note explains what is structurally special about the repo's canonical `beta23` arithmetic backend and why it is treated as the frozen path rather than just one more fit variant.

## Short version

`beta23` is the accepted two-tier dyadic-depth correction of the ghost/Euler lift.

It is not just a stronger version of the generic `c_plus_bOmega` bias family.

The generic family applies a post-lift logarithmic correction of the form

`log|a*_n| = log|a_n| - c - b*Omega_total(n)`

which is a coarse global penalty based only on total prime-factor multiplicity.

By contrast, `beta23` changes the dyadic ghost tower before the Euler transform:

- `g*_2 = g_2 - beta2`
- `g*_m = g_m - beta3` for `m >= 3`

and only then rebuilds the corrected Euler coefficients `A1_star`, `A2_star`, `A3_star`.

So `beta23` is structurally upstream of the packet realization layer, while the generic `bOmega` path is structurally downstream and phenomenological.

## Canonical distinction

There are two different correction philosophies in the repo.

### 1. Generic closure-bias family

Implemented in [tools/fit_ghostlift_closure_bias.py](tools/fit_ghostlift_closure_bias.py).

Representative models:

- `c_only`
- `bOmega_only`
- `c_plus_bOmega`

These models:

- leave the underlying dyadic ghost tower unchanged,
- leave the Euler-transform coefficients unchanged,
- and correct only the final log-magnitude relation by a constant and or an `Omega_total(n)` term.

This is useful as a diagnostic family, but it is not the frozen canonical backend.

### 2. Two-tier ghost-depth family

Also implemented in [tools/fit_ghostlift_closure_bias.py](tools/fit_ghostlift_closure_bias.py), under:

- `ghost_beta_mode=step_m_eq2_and_m_ge3`
- model name `beta23_plus_c`

This family:

- modifies the ghost tower itself,
- distinguishes the `m=2` layer from the deeper `m>=3` layers,
- then recomputes the Euler coefficients from the corrected tower,
- and finally carries only a residual constant `c` after that structural correction.

That is why the downstream frozen coefficient table stores

- `beta2`
- `beta3`
- `c`
- `A1_star`
- `A2_star`
- `A3_star`

rather than a generic `bOmega` parameter.

## Why beta23 is treated as special

The repo treats `beta23` as special for three reasons.

### A. It is structurally aligned with the architecture

The architecture is dyadic-seed first, then ghost tower, then Euler lift, then corrected local coefficients, then packet realization.

`beta23` corrects the tower at the ghost level, which respects that layering.

The generic `bOmega` path does not. It is a useful closure surrogate, but it acts only after multiplicative data have already been assembled.

### B. It is the frozen backend actually consumed downstream

The canonical coefficient table is [out/corrected_factor_injection_beta23_window_coefficients.csv](out/corrected_factor_injection_beta23_window_coefficients.csv).

The narrow loader in [tools/corrected_backend_interface.py](tools/corrected_backend_interface.py) reads exactly that table and exposes:

- `u`
- `beta2`
- `beta3`
- `c`
- `A1_star`
- `A2_star`
- `A3_star`

Downstream factor-injection and frontend probes already default to the `beta23_plus_c` fit artifact:

- [tools/probe_corrected_factor_injection.py](tools/probe_corrected_factor_injection.py)
- [tools/probe_corrected_factor_companion3.py](tools/probe_corrected_factor_companion3.py)
- [tools/probe_corrected_factor_damped_unitary3.py](tools/probe_corrected_factor_damped_unitary3.py)
- [tools/probe_corrected_factor_normal3_search.py](tools/probe_corrected_factor_normal3_search.py)
- [tools/probe_corrected_factor_normal4_search.py](tools/probe_corrected_factor_normal4_search.py)
- [tools/probe_corrected_factor_cayley3_search.py](tools/probe_corrected_factor_cayley3_search.py)
- [tools/probe_exact_A3_chart_constrained.py](tools/probe_exact_A3_chart_constrained.py)
- [tools/probe_exact_Aj_superstructure.py](tools/probe_exact_Aj_superstructure.py)

So the frozen architecture already assumes `beta23` at the arithmetic interface.

### C. It is the variant that satisfies the strongest closure validation

The final validation note [notes/FROZEN_ORACLE_SANITY_SHEET.md](notes/FROZEN_ORACLE_SANITY_SHEET.md) records that the stronger frozen artifact

- [out/ghostlift_closure_beta23_mumix050_fit17_eval29_hold19_23.csv](out/ghostlift_closure_beta23_mumix050_fit17_eval29_hold19_23.csv)

passes the requested Euler-style closure comparison against baseline on:

- squarefree closure error,
- prime-power closure error,
- mixed closure error,
- and total loss,

on both eval and holdout.

The fresh generic `c_plus_bOmega` rerun did not satisfy the same strict standard on prime-power error.

That is evidence that the two-tier dyadic correction is not just prettier conceptually; it is also the stronger validated frozen closure path in this repo.

## Operational rule for the repo

The canonical frozen arithmetic path is:

`dyadic tower -> two-tier ghost correction (beta2, beta3) -> corrected Euler coefficients -> A3 frontend -> anchored completion`

The generic `c_only`, `bOmega_only`, and `c_plus_bOmega` families should be treated as exploratory diagnostics only.

They may remain in fitting tools for comparison and ablation, but they are not part of the repo's canonical frozen architecture.