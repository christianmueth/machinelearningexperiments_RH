# Admissible geometry definition (canonical)

This file is **foundational**. It contains definitions only (no empirical claims, no references to runs or p-values).

## A. Self-dual coordinates and critical locus

Let $r$ and $r_s$ be complex coordinates equipped with an involution exchanging branches.

**A1 (Self-dual normalization).** Fix the gauge by

$$
1 = r + r_s.
$$

Derivation: multiplying by $1/(rr_s)$ gives the harmonic identity

$$
\frac{1}{r r_s} = \frac{1}{r} + \frac{1}{r_s}.
$$

**A2 (Critical locus).** The distinguished (critical) locus is the fixed-point set

$$
r = r_s.
$$

## B. Canonical warp chart and branch convention

**B1 (Canonical warp).** Fix the canonical warp chart (the “ratio” chart)

$$
W(r,r_s) = \frac{r - r_s}{r + r_s}.
$$

This treats $r=r_s$ as the distinguished symmetry locus $W=0$ (not a pole).

**B2 (Depth and chirality).** Define

$$
\log W(p) = u(p) + i\,\theta(p).
$$

The admissible class must specify:

- A branch convention for $\log$ (e.g. principal branch + an unwrapping rule).
- How unwrapping is applied consistently across primes.

## C. Generator construction (Tp family)

Let $\{T_p\}$ be a prime-indexed family built from the warp-derived coordinates $(u(p),\theta(p))$.

**C1 (Self-adjointness / spectral reality constraint).** Each $T_p$ is constructed so that the derived spectral observable is compatible with self-adjointness (finite-dimensional surrogate: symmetric/Hermitian matrices; target theory: self-adjoint operators on a Hilbert space).

**C2 (Non-decomposable coupling allowed, bounded).** The admissible class allows a controlled coupling term that prevents trivial decompositions.

## D. Bounded deformation class

To exclude trivial overfitting, the admissible class includes a uniform bound relative to a baseline (e.g. a “legacy” family):

$$
\sup_p \frac{\|T_p - T_p^{(\mathrm{base})}\|_F}{\|T_p^{(\mathrm{base})}\|_F + \varepsilon} \le \delta,
$$

for a small fixed $\delta$ and regularizer $\varepsilon>0$.

(Exact norms/metrics can be swapped, but the point is a uniform smallness/boundedness requirement.)

## E. Gauge equivalence (what we mod out by)

Two Tp families are considered equivalent if they differ only by operations that should not change the underlying arithmetic/spectral content:

- Prime relabeling that preserves the experimental protocol’s “true labeling” reference.
- Branch/unwrap conventions within the admissible rules.
- Conjugation by a fixed unitary/orthogonal transformation.

- Prime relabeling as an abstract reindexing symmetry (when the underlying arithmetic labeling is treated as structure).
- Branch/unwrap conventions within the admissible rules.
- Conjugation by a fixed unitary/orthogonal transformation.
