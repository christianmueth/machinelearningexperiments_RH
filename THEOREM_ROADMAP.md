# Formal theorem roadmap (Phase‑3B → proof-aligned)


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

This is written in “Definitions → Lemmas → Theorems” shape. The harness corresponds to finite-dimensional surrogates of the objects in the statements.

## 0. Candidate theorem (target statement)

This is the **target statement** the program is converging toward. It is phrased conservatively: it names the objects and the logical gaps without claiming RH is already proved.

**Target theorem (implication form).**
Let $\{T_p\}_{p\ \mathrm{prime}}$ be a prime-indexed family in a Hilbert-space model such that:

- (Self-adjointness) each $T_p$ is self-adjoint (or yields a self-adjoint bulk operator in the completion construction).
- (Involution symmetry) the geometry is equivariant under the involution exchanging the two self-dual branches, with critical locus $r=r_s$.
- (Admissible geometry) $\{T_p\}$ lies in a canonical admissible deformation class (chart/branch/gauge/boundedness) as defined in [ADMISSIBLE_GEOMETRY_DEFINITION.md](ADMISSIBLE_GEOMETRY_DEFINITION.md).
- (Frozen completion observable) a DN-map/completion determinant observable $\Xi_T(E)$ is defined from $\{T_p\}$ via Schur-complement/DN completion and a normalization choice.

If $\Xi_T$ exhibits rigidity against the adversarial label-scramble null family on the critical locus in the admissible limit, then the induced arithmetic structure satisfies Hecke-shaped multiplicativity constraints (coprime and prime-power recurrences) in that admissible class.

Notes:

- Phase‑3B establishes the **rigidity premise** for a specific Tp-only geometry deformation under a frozen DN-map/logdet spine, but does not (yet) upgrade rigidity → Hecke identities → completed determinant analyticity.
- The remaining proof work is exactly: (A) admissible class formalization, (B) rigidity ⇒ multiplicativity, (C) determinant identity (entirety + involution symmetry + zeros ↔ spectrum).

## 1. Lemma chain (make the arrows explicit)

The intended logical structure is:

**Lemma A (Admissible class / gauge fixing).** Define the admissible geometry class (chart + branch/unwrap + gauge equivalence + boundedness) and show the construction of $\{T_p\}$ is well-defined modulo the gauge equivalence.

**Lemma B (Rigidity ⇒ multiplicativity).** In the admissible limit, rigidity of the completed DN-map observable $\Xi_T$ forces Hecke/Euler-type multiplicativity constraints (coprime and prime-power) for the induced arithmetic data.

**Lemma C (DN/logdet = completed $\xi$-object).** The normalized DN/logdet completion defines an analytic object with exact involution symmetry and a zeros ↔ self-adjoint spectrum correspondence.

**Theorem (RH corollary).** Lemma A + Lemma B + Lemma C imply the nontrivial zeros lie on the critical line $\Re(s)=1/2$ (RH).

## D. Definitions (frozen spine)

**D0 (Measurement spine / DN map).**
Let $H(E)$ be a symmetric/self-adjoint family (finite-dimensional surrogate in code). With a boundary/interior split, write the block form

$$
H(E)=\begin{pmatrix}A(E) & B(E)\\ C(E) & D(E)\end{pmatrix}.
$$

Assuming $A(E)$ invertible, define the Schur complement

$$
S(E)=D(E)-C(E)A(E)^{-1}B(E).
$$

Define the completion observable (surrogate) as a log-determinant aggregate over an energy window $[w_{lo},w_{hi}]$:

$$
\mathcal{D}(w_{lo},w_{hi})=\frac{1}{|\mathcal{E}|}\sum_{E\in\mathcal{E}}\log\det(S(E)+\varepsilon I),
$$

with a fixed deterministic energy grid $\mathcal{E}$ and regularization $\varepsilon>0$.

Implementation anchor: [machinelearning_rh_colab_cells.py](machinelearning_rh_colab_cells.py).

**D1 (Null family).**
A label-scramble null is a permutation of prime labels in the Tp generator family that preserves the measurement spine but destroys Euler labeling.

**D2 (Decision rule).**
Lower-tail empirical p-values from the null distribution with add-one smoothing; reject if $p\le \alpha$.

## G. Geometry (self-dual constraint + chart)

You requested the critical locus be the equality $r=r_s$ and the chart treat equality as a distinguished symmetry point (not a pole).

**G0 (Self-dual normalization).**
Impose the symmetry/gauge condition

$$
1=r+r_s.
$$

Derivation (not an extra assumption): from $r+r_s=1$,

$$
\frac{1}{r}+\frac{1}{r_s}=\frac{r+r_s}{rr_s}=\frac{1}{rr_s}.
$$

$$
\frac{1}{rr_s}=\frac{1}{r}+\frac{1}{r_s}.
$$

**G1 (Warp chart).**
Use a self-dual Möbius-type warp chart that treats $r=r_s$ as a distinguished locus:

$$
W(r,r_s)=\frac{r-r_s}{r+r_s}.
$$

This makes $r=r_s$ map to $W=0$, i.e. a symmetry locus rather than a pole.

**G2 (Horn depth + chirality).**
Define

$$
\log W(p)=u(p)+i\,\theta(p)
$$

where $u=\Re\log W$ and $\theta=\Im\log W$ depend on a branch convention (the branch convention is part of the admissible class).

## H. Generator family and Hecke-shaped closure

**H0 (Prime generators).**
Define a prime-indexed family $\{T_p\}$ where geometry enters only through $(u(p),\theta(p))$.

**H1 (Closure constraints).**
Define coprime closure (pq), triangle/two-path closure (pqr), and prime-power closure (tower/pow). In the proof-aligned target, these should become exact identities (or converge to them under an admissible limit).

## T. Theorems to target (roadmap)

**Theorem A (Admissible class / gauge fixing).**
Define an admissible set of charts/branches/gauges such that the construction of $(u(p),\theta(p))$ is canonical up to an explicitly described equivalence, and such that the rigidity phenomenon is stable within this class.

Computational analogue: “seed-conditional” behavior should be reinterpretable as gauge/branch selection, not hidden measurement drift.

**Theorem B (Exact multiplicativity / Hecke algebra).**
Within the admissible class, show closure residuals vanish (or the exact algebraic relations hold):

- coprime multiplicativity,
- prime-power recurrences,
- higher-order associativity/two-path consistency.

This is the formal bridge from closure channels to Euler/Hecke structure.

**Theorem C (Completed determinant identity).**
Show the completed determinant built from the DN-map observable is an analytic object with:

- analyticity/entire-ness after normalization,
- a functional symmetry corresponding to the involution (RH-style $s\mapsto 1-s$ analogue),
- a spectral-zero correspondence (zeros correspond to the spectrum of a self-adjoint operator).

This is the hardest analytic step, but it is now sharply defined because the candidate completion observable is explicit.

## Implementation anchors (outputs)

- Combined KPI “figure table”: [out_stage_gate_figure_table.csv](out_stage_gate_figure_table.csv)
- Preregistered generalization outputs: [out_phase3B_preregistered_seedblock_32_63/](out_phase3B_preregistered_seedblock_32_63/)
- N=16384 confirm outputs: [out_phase3B_confirm_preregistered_rejecting_seeds_N16384/](out_phase3B_confirm_preregistered_rejecting_seeds_N16384/)
