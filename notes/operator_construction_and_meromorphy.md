# Operator construction and meromorphy (Task A)

## Scope and intent

This note supplies the missing operator-theoretic backbone assumed by:

- [notes/lemma_safe_rectangle_convergence.md](notes/lemma_safe_rectangle_convergence.md)
- [notes/proposition_identification_hinge.md](notes/proposition_identification_hinge.md)

It does **not** add new experimental claims. It defines a limiting boundary operator family $\Lambda(s)$ and the associated scattering/Cayley transform $S_\eta(s)$, and records the standard analytic facts needed:

1) meromorphic dependence of $\Lambda(s)$ and $S_\eta(s)$,
2) the duality / functional-equation (FE) involution at the operator level,
3) the symmetry decomposition (Swap±) at the operator level.

The concrete “model-to-modular identification” is deferred to Task B.

This note is intentionally **citation-forward**: it records a standard route from a self-adjoint Laplacian on a cusp surface to an operator-valued Weyl/DN function with meromorphic continuation, and then to a scattering operator with the expected involution/unitarity constraints.

---

## 1) Geometric/operator model to target

### 1.1 Base operator

Let $X$ be a complete Riemannian surface with one cusp (ultimately: $X=\mathrm{SL}(2,\mathbb{Z})\backslash \mathbb{H}$) and let $\Delta_X$ be the nonnegative Laplace–Beltrami operator on $L^2(X)$.

Use the standard spectral parameterization
\[
(\Delta_X - s(1-s))u = 0.
\]

On $\Re(s)>1$ the resolvent
\[
R(s) := (\Delta_X - s(1-s))^{-1}
\]
exists as a bounded operator on $L^2(X)$ (up to the discrete spectrum).

**Standard reference points (no details needed here).**
For geometrically finite hyperbolic surfaces (including finite-area one-cusp surfaces), the resolvent and scattering objects admit meromorphic continuation in $s$ by an analytic Fredholm argument. This is classical in the spectral/scattering literature on cusps.

### 1.2 Cutting along a boundary and boundary Hilbert space

Fix a horocycle cut at height $y=Y$ (or any admissible cross-section) so that
\[
X = X_{\mathrm{int}} \cup \big([Y,\infty)\times (\mathbb{R}/\mathbb{Z})\big)
\]
with interface boundary $\partial X_Y \cong \mathbb{R}/\mathbb{Z}$.

Define the boundary Hilbert space
\[
\mathcal{H}_\partial := L^2(\partial X_Y),
\]
and the usual Sobolev boundary trace spaces $H^{1/2}(\partial X_Y)$ (Dirichlet data) and $H^{-1/2}(\partial X_Y)$ (Neumann data).

Under the cusp cross-section identification $\partial X_Y\cong \mathbb{R}/\mathbb{Z}$ (with coordinate $x$), the associated scattering operator is diagonal in boundary Fourier modes and in particular preserves the constant-mode subspace $\mathrm{ran}(P_0)$ (the channel used in Task B).

**Two-channel / Swap structure.** In the project’s “two-channel symmetric” setup, the boundary space is doubled (two copies or two oriented components) and equipped with a fixed involution (“Swap”) exchanging channels. Abstractly,
\[
\mathcal{H}_\partial^{(2)} := \mathcal{H}_\partial \oplus \mathcal{H}_\partial,
\]
with Swap operator $J(f,g)=(g,f)$ and projections
\[
\Pi_\pm := \tfrac12(I\pm J).
\]

This is the infinite-dimensional analog of the finite Swap± block decomposition used in the code.

---

## 2) Limiting DN map and its meromorphy

### 2.1 Poisson operator and boundary-to-solution map

For $s$ in a domain where the boundary value problem is well-posed (e.g. $\Re(s)>1$ away from the point spectrum), define the Poisson operator
\[
\mathcal{P}(s): H^{1/2}(\partial X_Y) \to H^1_{\mathrm{loc}}(X)
\]
by: $u=\mathcal{P}(s)f$ is the solution to $(\Delta_X-s(1-s))u=0$ with Dirichlet trace $u|_{\partial X_Y}=f$ and with the appropriate outgoing/cusp asymptotic condition.

Equivalently (and often more cleanly for meromorphy), $\mathcal{P}(s)$ is obtained from a cut-off resolvent/parametrix construction plus an analytic Fredholm step.

### 2.2 Dirichlet-to-Neumann (DN) / Weyl operator

Define the limiting DN map (often also called a boundary **Weyl function** in the boundary-triple literature)
\[
\Lambda(s): H^{1/2}(\partial X_Y) \to H^{-1/2}(\partial X_Y)
\]
by
\[
\Lambda(s)f := \partial_\nu (\mathcal{P}(s)f)\big|_{\partial X_Y},
\]
where $\partial_\nu$ is the outward normal derivative on $\partial X_Y$.

This is the operator-level target of the finite Schur complement DN maps $\Lambda_N(s)$.

**Boundary-triple viewpoint (optional, for clean analytic structure).**
One can package the boundary traces as abstract boundary maps $\Gamma_0,\Gamma_1$ (Dirichlet/Neumann) on a suitable maximal domain for the differential operator, obtaining a (quasi-)boundary triple. In that language, the DN/Weyl map is the Weyl function
\[
M(s) := \Gamma_1(\Gamma_0|_{\ker(\Delta_X-s(1-s))})^{-1},
\]
which coincides with $\Lambda(s)$ on the corresponding trace spaces.

### 2.3 Meromorphic continuation

**Claim (standard).** The family $s\mapsto \Lambda(s)$ extends meromorphically to $\mathbb{C}$ as an operator-valued meromorphic function between the boundary Sobolev spaces.

**Justification route (what to cite/use).** This is a standard consequence of:

- meromorphic continuation of the resolvent $R(s)$ on geometrically finite hyperbolic surfaces (analytic Fredholm framework), and
- the fact that boundary trace and normal-derivative maps are continuous between the resolvent’s mapping spaces.

Equivalently, one may formulate the DN map via boundary triples and Weyl functions: in that language, the Weyl function is known to be operator-valued meromorphic.

This is the only meromorphy fact needed to feed the “hinge proposition”; it does not require explicit formulas.

**Remark (what is and is not being claimed).**
No attempt is made here to prove global resolvent continuation from scratch. The role of this note is to reduce the project’s analytic needs to a standard black box: “Weyl/DN functions for self-adjoint elliptic operators on cusp manifolds are operator-valued meromorphic functions of the spectral parameter.”

---

## 3) Cayley/scattering operator and involution

### 3.1 Cayley transform at impedance \(\eta\)

Fix $\eta>0$. On any domain where $(\Lambda(s)+i\eta I)$ is invertible as a boundary operator, define
\[
S_\eta(s) := (\Lambda(s)-i\eta I)(\Lambda(s)+i\eta I)^{-1}.
\]

**Meromorphy.** Since $\Lambda(s)$ is meromorphic, $S_\eta(s)$ is meromorphic wherever the inverse exists; its poles correspond to zeros of $\det(\Lambda(s)+i\eta I)$ (interpreted appropriately in the operator setting).

In particular, the only singularities introduced by the Cayley map are the “Cayley bottlenecks” where $(\Lambda(s)+i\eta I)$ fails to be invertible.

### 3.2 Critical-line unitarity (scattering interpretation)

On the critical line $\Re(s)=1/2$ (away from poles), $\Lambda(1/2+it)$ is self-adjoint (in the appropriate boundary pairing), hence $S_\eta(1/2+it)$ is unitary. This is the operator-theoretic content behind the harness’s “unitarity-type” tests.

### 3.3 Duality / FE involution

Let $D_b$ denote the operator-level boundary involution induced by the geometric duality (the infinite-dimensional analogue of the code’s boundary duality/Swap conjugation).

**Target identity.** The operator FE statement needed later is
\[
\Lambda(1-s) = D_b\,\Lambda(s)^*\,D_b^{-1}.
\]

From the Cayley transform, this implies
\[
S_\eta(1-s) = D_b\,\big(S_\eta(s)^*\big)^{-1}\,D_b^{-1}.
\]

On $\Re(s)=1/2$ this reduces to a unitarity relation.

**Where this should come from.** In the boundary triple/Weyl function framework, adjoint symmetry of the underlying self-adjoint operator implies the above relation. In the cusp setting, it is also compatible with the standard scattering functional equation.

Concretely, one route is:
- use Green’s identity to relate Dirichlet/Neumann pairings under $s\mapsto 1-s$;
- identify the boundary involution $D_b$ induced by the geometric symmetry (or by the two-channel construction);
- transfer the adjointness relation to the Weyl/DN map.

---

## 4) Swap± symmetry and channel extraction at the operator level

Assume the two-channel boundary space is equipped with a unitary involution $J$ (Swap) commuting with $\Lambda(s)$ and hence with $S_\eta(s)$:
\[
J\Lambda(s)=\Lambda(s)J,\qquad JS_\eta(s)=S_\eta(s)J.
\]

Then the decomposition
\[
\mathcal{H}_\partial^{(2)} = \mathrm{ran}(\Pi_+)\oplus \mathrm{ran}(\Pi_-)
\]
reduces the scattering operator into $\pm$ blocks.

The scalar channel $\lambda(s)$ is then defined as the unique (or isolated) eigenvalue in the relevant block singled out by the cusp/scattering channel, with isolation controlled by a spectral gap hypothesis on safe rectangles (the operator version of the harness gap condition).

On safe rectangles where the channel is isolated, $\lambda(s)$ is meromorphic and can be tracked via Riesz projections.

---

## 5) Interface with the finite-N objects

The finite harness constructs $\Lambda_N(s)$ via a Schur complement on a truncated/discretized bulk operator. The intended analytic statement to prove (later, not here) is:

- for each safe rectangle $R$, $\Lambda_N(s)\to\Lambda(s)$ uniformly on $R$ in operator norm (or a norm sufficient to control the Cayley transform),
- the Swap± decomposition is consistent under truncation,
- and the isolated channel eigenvalue converges uniformly on $R$.

These are exactly the hypotheses assumed in [notes/lemma_safe_rectangle_convergence.md](notes/lemma_safe_rectangle_convergence.md).

---

## 6) What remains after this note

This note only constructs/records $\Lambda(s)$ and $S_\eta(s)$ abstractly and their meromorphy + involution.

The remaining conceptual bridge is Task B: identify the scalar scattering channel $\lambda(s)$ with the Eisenstein intertwining/scattering coefficient for $\mathrm{SL}(2,\mathbb{Z})\backslash\mathbb{H}$, including normalization.

See also: `notes/hypotheses_to_artifacts.md` for the single-page audit table used in the other notes.

---

## References (standard sources to cite)

The project only needs these results as black boxes; the intent here is a **hybrid-minimal** citation stack covering (i) DN/Weyl functions and (ii) cusp-surface resolvent/scattering continuation.

- DN/Weyl functions via boundary triples (operator-valued meromorphy):

  - V. A. Derkach and M. M. Malamud, boundary triple / Weyl function references (a survey-style source is sufficient).
  - J. Behrndt, S. Hassi, H. de Snoo, survey/monograph sources on boundary triples and Weyl functions.

- Scattering matrices expressed via Weyl functions (extension theory / boundary triples):

  - J. Behrndt, M. M. Malamud, H. Neidhardt, *Scattering matrices and Weyl functions*, Proc. London Math. Soc. (2008). arXiv:math-ph/0604013. DOI: 10.1112/plms/pdn016.
  - J. Behrndt, H. Neidhardt, E. R. Racec, P. N. Racec, U. Wulf, *On Eisenbud's and Wigner's R-matrix: A general approach*, J. Differential Equations 244 (2008), 2545–2577. DOI: 10.1016/j.jde.2008.02.004.

- Meromorphic continuation of the resolvent/scattering on cusp/geometrically finite hyperbolic surfaces:

  - R. Mazzeo and R. B. Melrose, meromorphic continuation of the resolvent on (asymptotically) hyperbolic / conformally compact settings.
  - C. Guillopé and M. Zworski, resolvent/scattering continuation for geometrically finite hyperbolic surfaces.
