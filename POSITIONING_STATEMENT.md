# Field-facing positioning statement (Phase‑3B Stage Gate)


## Key numbers (auto-generated)

Generated from [out_stage_gate_figure_table.csv](out_stage_gate_figure_table.csv) via `update_key_numbers_blocks.py`.

<!-- BEGIN KEY_NUMBERS -->
**Preregistered generalization (seeds 32–63, N_null=4096)**

| window | backend | n_seeds | seeds w/ any reject | frac | best_p_overall | median_best_p |
|---:|---|---:|---:|---:|---:|---:|
| (0.6, 7.5) | geom_theta0p125 | 32 | 0 | 0.00000 | 0.0107396 | 0.239688 |
| (0.6, 7.5) | legacy | 32 | 0 | 0.00000 | 0.123749 | 0.636563 |
| (2.0, 5.0) | geom_theta0p125 | 32 | 5 | 0.15625 | 0.000244081 | 0.252014 |
| (2.0, 5.0) | legacy | 32 | 0 | 0.00000 | 0.208201 | 0.660117 |

**High-null confirm (N_null=16384) on preregistered rejecting seeds**
- Seeds: `36,42,45,54,58`
- Window: (2.0, 5.0)

| backend | rows | reject_rate | best_p | median_p | worst_p |
|---|---:|---:|---:|---:|---:|
| geom_theta0p125 | 35 | 1.000 | 0.000244126 | 0.00396704 | 0.0075679 |
| legacy | 35 | 0.000 | 0.276594 | 0.436802 | 0.728044 |
<!-- END KEY_NUMBERS -->

This is paste-ready and intentionally conservative: it states what is established under the frozen Phase‑3B measurement spine, and what it is **not** (i.e., not a proof of RH).

## What this program is

A Hilbert–Pólya‑style computational harness designed to be **falsifiable**, not merely “plausible.” It couples:

- **Self-adjoint spectral proxy:** a deterministic symmetric operator family whose boundary-completion observable is a DN‑map proxy computed by Schur complement, aggregated by log‑determinant over energy windows.
- **Multiplicativity/Hecke-shaped constraints:** closure residual channels (pq / pqr / prime‑power) that operationalize Euler/Hecke consistency as rigidity constraints.
- **Adversarial nulls:** label‑scramble null families that preserve the measurement spine while destroying prime labeling structure.

## What’s new vs most RH work

Most RH programs emphasize either (i) analytic equivalences, (ii) spectral plausibility arguments, or (iii) automorphic trace formula machinery. This harness is different in one key way: it produces an *end‑to‑end mechanism candidate* with a fixed measurement observable and an explicit adversarial null family, so “structure” is tested by survival under scrambles rather than by heuristic plausibility.

## What can be claimed now (precise)

Under a **frozen DN‑map/logdet measurement spine**, a fixed label‑scramble null family, and preregistered evaluation, there exists a Tp‑only geometry deformation that induces **strong spectral rigidity** detectable by the DN‑map observable, while the legacy backend does not.

- Preregistered generalization (seeds 32–63, $N_{null}=4096$): geometry has a nonzero fraction of rejecting seeds; legacy has zero.
- High-null confirm on the preregistered rejecting seeds ($N_{null}=16384$): geometry rejects 35/35 anchor-runs; legacy rejects 0/35.

These statements are backed by:

- Stage-gate KPI table: [out_stage_gate_figure_table.csv](out_stage_gate_figure_table.csv)
- Preregistered seedblock outputs: [out_phase3B_preregistered_seedblock_32_63/kpi_fraction.csv](out_phase3B_preregistered_seedblock_32_63/kpi_fraction.csv)
- N=16384 confirm outputs: [out_phase3B_confirm_preregistered_rejecting_seeds_N16384/aggregate.csv](out_phase3B_confirm_preregistered_rejecting_seeds_N16384/aggregate.csv), [out_phase3B_confirm_preregistered_rejecting_seeds_N16384/aggregate_by_seed.csv](out_phase3B_confirm_preregistered_rejecting_seeds_N16384/aggregate_by_seed.csv)

## Interpretation (what the DN-map result means)

The clean “RH-adjacent” interpretation is:

- There exists a Tp-only, self-adjoint geometry deformation such that a determinant-like completion observable (DN-map/Schur/logdet) exhibits **rigidity under adversarial label-scramble nulls**, while an undeformed baseline does not.
- The effect is **window-localized** (strong in (2.0, 5.0); absent in (0.6, 7.5) at tested null levels), which is consistent with the completion observable having a sensitive spectral band.
- The effect **generalizes but is not uniform across seeds**, which is consistent with “seed” acting as a gauge/branch choice within a larger operator class than the true target object.

## Where this sits in the RH landscape

This work is best described as a **mechanism and falsification harness**:

- Compared to many equivalence-based approaches (Li/Weil/Nyman–Beurling), it emphasizes an explicit candidate completion observable and adversarial falsification, rather than deriving new equivalences.
- Compared to Hilbert–Pólya discussions, it provides an operator-style spectral proxy together with a concrete, repeatable test for whether structure survives null families.
- Compared to de Branges–style programs, it is more constructive/experimental, and correspondingly less advanced on analytic closure.

## What remains (now sharply defined)

The remaining proof-aligned gap is no longer “find any mechanism.” It is three formal upgrades:

- **Admissible class definition:** specify the canonical ratio chart, branch/unwrap rules, gauge equivalences, and boundedness. Canonical definition: [ADMISSIBLE_GEOMETRY_DEFINITION.md](ADMISSIBLE_GEOMETRY_DEFINITION.md).
- **Rigidity ⇒ multiplicativity:** promote DN-map rigidity into Hecke/Euler-type identities (coprime and prime-power consistency) within the admissible class.
- **Determinant identity:** prove analyticity/entirety after normalization, exact involution symmetry, and zeros ↔ self-adjoint spectrum for the completed determinant.

## The safe non-oversell line

This is **not a proof of RH**. It is evidence for a *mechanism candidate*: a self‑adjoint completion observable whose rigidity is measurably controlled by a geometry-derived Tp construction, surviving preregistration and heavy adversarial nulls. The remaining work is to formalize admissibility/gauge, upgrade closure residuals to identities, and connect the completed determinant to an RH‑equivalent analytic object.
