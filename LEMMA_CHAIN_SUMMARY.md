# Lemma chain summary (one page)

This is a compact summary of the proof-aligned pipeline as a lemma chain. It is intentionally conservative: it describes the intended implication structure without claiming the analytic steps are already complete.

## Setup

- Input: a prime-indexed family $\{T_p\}$ constructed from a self-dual geometry (warp chart + branch convention) and satisfying self-adjointness.
- Completion observable: a DN-map/Schur-complement log-determinant completion $\Xi_T$ built from a self-adjoint bulk/boundary completion.
- Null family: adversarial label-scramble permutations that preserve the measurement spine.

## Lemma A (Admissible class)

Define the admissible deformation class (canonical chart, branch/unwrap convention, gauge equivalence, bounded deformation) and prove the model is well-defined modulo gauge.

Canonical reference: [ADMISSIBLE_GEOMETRY_DEFINITION.md](ADMISSIBLE_GEOMETRY_DEFINITION.md)

## Lemma B (Rigidity ⇒ multiplicativity)

Show that DN-map rigidity of $\Xi_T$ against the adversarial label-scramble null family forces Hecke/Euler-type multiplicativity constraints for the induced arithmetic data (coprime multiplicativity + prime-power recurrences) within the admissible class.

Empirical analogue: closure residual channels should vanish only on DN-rigid admissible slices.

## Lemma C (Completed determinant identity)

Show that the normalized completion $\Xi_T$ is the correct “completed object”:

- analytic/entire after normalization (no spurious poles),
- exact involution symmetry (functional-equation analogue),
- zeros correspond to the spectrum of a self-adjoint operator.

This splits into two explicit proof obligations already tracked elsewhere:

- **Finite→limit control (convergence obligations):** safe-rectangle operator consistency and divisor control on rectangles, upgraded from finite $N$.
	- Safe-rectangle lemma: [notes/lemma_safe_rectangle_convergence.md](notes/lemma_safe_rectangle_convergence.md)
	- What convergence *does* and *does not* prove: [notes/closure_from_convergence.md](notes/closure_from_convergence.md)
	- H2 proof blueprint for the current finite sections: [notes/h2_completion_blueprint.md](notes/h2_completion_blueprint.md)
	- Geometric H2 plan + spec (RH target): [notes/h2_geometric_growing_boundary_plan.md](notes/h2_geometric_growing_boundary_plan.md) and [notes/geometric_discretization_spec_option_B.md](notes/geometric_discretization_spec_option_B.md)

- **Model-side analytic structure (non-convergence obligations):** existence/meromorphy + involution/FE + strip growth for the limiting scattering object, and identification of the extracted channel with the modular scattering coefficient.
	- Task A (scattering operator presentation): [notes/operator_construction_and_meromorphy.md](notes/operator_construction_and_meromorphy.md)
	- Task B (Eisenstein/scattering identification): [notes/eisenstein_intertwiner_identification.md](notes/eisenstein_intertwiner_identification.md)
	- Hinge closure: [notes/proposition_identification_hinge.md](notes/proposition_identification_hinge.md)

## Theorem (RH corollary)

If Lemma A, Lemma B, and Lemma C hold for the construction, then the nontrivial zeros lie on the critical line $\Re(s)=1/2$ (RH).

## Pointers

- Field-facing summary: [POSITIONING_STATEMENT.md](POSITIONING_STATEMENT.md)
- Full roadmap: [THEOREM_ROADMAP.md](THEOREM_ROADMAP.md)
- Stage-gate KPIs: [out_stage_gate_figure_table.csv](out_stage_gate_figure_table.csv)
