# Proposition: Identification Hinge (quotient rigidity)

## Purpose

This is the single complex-analytic proposition that turns “operator-level match + one normalization” into full identification.

It is intentionally classical and short: the work in the project is *verifying the hypotheses* using the safe-rectangle convergence lemma and the operator/representation-theoretic identification.

Numerical experiments are used solely to witness the hypotheses of the analytic statements (invertibility margins, spectral gaps, and divisor cleanliness on safe rectangles); no step of the analytic proof relies on numerical computation.

---

## Setup

Let $\lambda(s)$ be the limiting scalar channel (Swap± eigenchannel) extracted from the limiting scattering/Cayley object.

Let
\[
\phi_{mod}(s) := \frac{\xi(2s-1)}{\xi(2s)}.
\]

Define the quotient
\[
Q(s) := \frac{\lambda(s)}{\phi_{mod}(s)}.
\]

---

## Functional equation / involution constraint (model-side)

Assume the model scattering satisfies a duality involution of the form
\[
S(1-s) = D_b\,\big(S(s)^*\big)^{-1}\,D_b^{-1}
\]
for a fixed unitary involution $D_b$ induced by the boundary Swap/duality symmetry.

Assume the scalar channel $\lambda(s)$ is extracted in a symmetry-compatible way so that the induced scalar constraint is
\[
\lambda(1-s)\;\overline{\lambda(s)} = 1\quad\text{on }\Re(s)=\tfrac12,
\]
(or an equivalent formulation; this is the one the harness treats as “unitarity-type” on the critical line).

Since $|\phi_{mod}(1/2+it)|=1$ on $\Re(s)=1/2$, the same constraint transfers to $Q$:
\[
Q(1-s)\;\overline{Q(s)} = 1\quad\text{on }\Re(s)=\tfrac12.
\]
In particular,
\[
|Q(\tfrac12+it)|=1\quad\forall t\in\mathbb{R}.
\]

---

## Proposition (rigidity of the quotient)

**Proposition (Identification hinge).**
Suppose $Q$ satisfies:

1) (**Meromorphic continuation**) $Q$ is meromorphic on $\mathbb{C}$.

2) (**Divisor control on safe rectangles**) There exists a family of safe rectangles $\{R_k\}$ exhausting the plane (or at least an open connected domain to which analytic continuation applies) such that:
   - $Q$ is holomorphic and nonvanishing on each $R_k$,
   - and the winding of $Q(\partial R_k)$ is zero.

   (Equivalently: the divisor of $Q$ is empty in each $R_k$.)

3) (**Functional equation / involution**) On the critical line,
\[
Q(1-s)\;\overline{Q(s)} = 1,\quad \Re(s)=\tfrac12.
\]
In particular $|Q(1/2+it)|=1$.

4) (**Growth bound**) $Q$ has controlled growth in vertical strips (e.g. finite order as an entire function after clearing poles, or bounded by $\exp(C|t|^\alpha)$ in strips). Any standard growth hypothesis sufficient for a Phragmén–Lindelöf-type argument is acceptable.

5) (**Normalization**) There exists a basepoint $s_0$ in the domain of holomorphy such that
\[
Q(s_0)=1.
\]

Then
\[
Q(s)\equiv 1\quad\text{on }\mathbb{C},
\]
and therefore
\[
\lambda(s)\equiv \phi_{mod}(s).
\]

---

## Proof sketch (classical)

This is the standard “bounded on a symmetry line + analytic continuation + growth control ⇒ constant” argument.

One clean route that matches the harness logic is:

1) On each safe rectangle $R_k$ where $Q$ is holomorphic and nonvanishing with winding 0, define an analytic branch of $\log Q$ on $R_k$.

2) Use the involution constraint on the critical line to show $\Re(\log Q(1/2+it))=0$.

3) Apply a maximum principle / Phragmén–Lindelöf argument in strips (using the growth bound) to conclude $\Re(\log Q)$ is constant, hence $\log Q$ is constant, hence $Q$ is constant.

4) The normalization $Q(s_0)=1$ forces that constant to be 1.

The proposition does not depend on number theory. The number theory enters only in identifying the target $\phi_{mod}$ and verifying the model-side involution and growth hypotheses.

---

## Notes on verification responsibility

- Hypothesis (2) is exactly what the winding artifacts certify on chosen safe rectangles for finite $N$, and what the Safe-Rectangle Convergence Lemma upgrades to the limiting $Q$.
- Hypothesis (3) is the operator-algebra consequence of DN-duality + Cayley map + symmetry-compatible channel extraction.
- Hypothesis (4) is the main analytic requirement that must come from the operator/representation-theoretic construction (e.g. trace-class/Fredholm determinant formulation, or standard bounds for intertwiners).

See also: `notes/hypotheses_to_artifacts.md` for the audit mapping from hypotheses to run artifacts.
