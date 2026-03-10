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

Let $\Gamma=\mathrm{PSL}(2,\mathbb{Z})$ and let
\[
X := \Gamma\backslash \mathbb{H}
\]
be the modular surface equipped with the hyperbolic metric $ds^2=y^{-2}(dx^2+dy^2)$. (All statements below also apply to any finite-area hyperbolic surface with cusps; the one-cusp case is what the pipeline ultimately uses.)

Let $\Delta_X$ denote the nonnegative Laplace–Beltrami operator acting on $L^2(X)$ with its standard self-adjoint realization.

Use the standard spectral parameterization
\[
(\Delta_X - s(1-s))u = 0.
\]

Equivalently, set the spectral parameter
\(
\lambda := s(1-s)
\)
and write $(\Delta_X-\lambda)u=0$. The involution $s\mapsto 1-s$ leaves $\lambda$ invariant; the scattering/FE statements below are the operator-level avatar of this symmetry.

On $\Re(s)>1$ (equivalently $\Re(\lambda)<0$) the resolvent
\[
R(s) := (\Delta_X - s(1-s))^{-1}
\]
exists as a bounded operator on $L^2(X)$ (up to the discrete spectrum).

**Standard black box (to cite, but not reprove here).**
For finite-area hyperbolic surfaces with cusps, the resolvent and the associated scattering objects admit meromorphic continuation in $s$ to $\mathbb{C}$ (analytic Fredholm framework). This is classical; see the references list at the end of this note.

### 1.2 Cutting along a boundary and boundary Hilbert space

Fix $Y>1$ and define the truncated surface
\[
X_Y := \{[z]\in X : \Im z \le Y\},\qquad \partial X_Y \cong \mathbb{R}/\mathbb{Z}
\]
(a horocycle cross-section at height $Y$).

Then $X$ decomposes as
\[
X = X_{\mathrm{int}} \cup \big([Y,\infty)\times (\mathbb{R}/\mathbb{Z})\big)
\]
with interface boundary $\partial X_Y \cong \mathbb{R}/\mathbb{Z}$.

Define the boundary Hilbert space
\[
\mathcal{H}_\partial := L^2(\partial X_Y),
\]
and the usual Sobolev boundary trace spaces $H^{1/2}(\partial X_Y)$ (Dirichlet data) and $H^{-1/2}(\partial X_Y)$ (Neumann data).

**Trace maps.** Let $\gamma_0:H^1(X_Y)\to H^{1/2}(\partial X_Y)$ denote the (Dirichlet) trace and let $\gamma_1:H^1(X_Y)\to H^{-1/2}(\partial X_Y)$ denote the (outward) conormal derivative trace; they satisfy the Green identity for the Laplacian on $X_Y$.

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

This section fixes a concrete definition of the limiting DN/Weyl operator family in a way that is compatible with (quasi-)boundary triples and standard cusp scattering.

### 2.1 Poisson operator and boundary-to-solution map

For $s$ in a domain where the boundary value problem is well-posed (e.g. $\Re(s)>1$ away from the point spectrum), define the Poisson operator
\[
\mathcal{P}(s): H^{1/2}(\partial X_Y) \to H^1_{\mathrm{loc}}(X)
\]
by: $u=\mathcal{P}(s)f$ is the solution to $(\Delta_X-s(1-s))u=0$ with Dirichlet trace $u|_{\partial X_Y}=f$ and with the appropriate outgoing/cusp asymptotic condition.

Equivalently (and often more cleanly for meromorphy), $\mathcal{P}(s)$ is obtained from a cut-off resolvent/parametrix construction plus an analytic Fredholm step.

**Alternative (boundary triple) formulation.** One may equivalently construct $\mathcal{P}(s)$ as the boundary-to-solution map associated to a self-adjoint extension pair on $X_Y$ (Dirichlet vs Neumann boundary conditions) and the corresponding Weyl function; see §2.2.

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
One can package the boundary traces as abstract boundary maps $\Gamma_0,\Gamma_1$ (Dirichlet/Neumann) on a suitable maximal domain for the differential operator, obtaining a (quasi-)boundary triple. In that language, the DN/Weyl map is (up to a fixed declared linear fractional transform) the Weyl function
\[
M(s) := \Gamma_1(\Gamma_0|_{\ker(\Delta_X-s(1-s))})^{-1},
\]
which coincides with $\Lambda(s)$ on the corresponding trace spaces.

For a one-cusp surface, the scattering matrix is scalar on the constant Fourier mode, and the Task B identification targets that scalar channel. Task A’s role is strictly prior: it supplies the existence/meromorphy/adjointness/FE symmetry of the operator families.

### 2.3 Meromorphic continuation

**Theorem A (standard; meromorphy of the DN/Weyl operator).**
For a finite-area hyperbolic cusp surface $X$ and any fixed cut height $Y>1$, the family
\(
s\mapsto \Lambda(s):H^{1/2}(\partial X_Y)\to H^{-1/2}(\partial X_Y)
\)
extends meromorphically to $\mathbb{C}$ as an operator-valued meromorphic function (with poles corresponding to the discrete spectrum/resonances appropriate to the chosen formulation).

**Justification route (what to cite/use).** This is a standard consequence of:

- meromorphic continuation of the resolvent $R(s)$ on geometrically finite hyperbolic surfaces (analytic Fredholm framework), and
- the fact that boundary trace and normal-derivative maps are continuous between the resolvent’s mapping spaces.

Equivalently, one may formulate the DN map via boundary triples and Weyl functions: in that language, the Weyl function is known to be operator-valued meromorphic.

This is the only meromorphy fact needed to feed the hinge proposition; it does not require explicit formulas.

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

Task A fixes a single presentation for the “model-side involution” used by the hinge proposition: it is most cleanly stated for the **geometric cusp scattering operator**, and then transferred to any equivalent scattering normalization (including a Cayley transform of a Weyl/DN object).

#### 3.3.1 Geometric scattering operator (definition)

Let $E(z,s)$ denote the (vector-valued) Eisenstein series for $X$ associated to the cusp(s). For $\Re(s)>1$, its constant term in the cusp has the form
\[
E(z,s) \sim y^s + \mathcal{S}(s)\,y^{1-s} \quad (y\to\infty),
\]
where $\mathcal{S}(s)$ is the (finite-dimensional) **scattering matrix** acting on cusp boundary data (in the one-cusp modular case this is a scalar function).

Equivalently: $\mathcal{S}(s)$ is characterized as the operator mapping incoming to outgoing coefficients in the Fourier-mode expansion of generalized eigenfunctions of $\Delta_X$.

#### 3.3.2 Theorem B (meromorphy + functional equation + unitarity)

**Theorem B (standard cusp scattering properties).**
For a finite-area hyperbolic surface with cusps (in particular $X=\mathrm{PSL}(2,\mathbb{Z})\backslash\mathbb{H}$), the scattering matrix $\mathcal{S}(s)$ has the following properties:

1) (**Meromorphic continuation**) $\mathcal{S}(s)$ extends meromorphically to $\mathbb{C}$.

2) (**Functional equation**) One has
\[
\mathcal{S}(s)\,\mathcal{S}(1-s)=I
\]
as a meromorphic identity.

3) (**Adjoint symmetry / unitarity on the critical line**) One has
\[
\mathcal{S}(\overline{s}) = \mathcal{S}(s)^*,
\]
and in particular for $s=\tfrac12+it$ away from poles,
\(
\mathcal{S}(\tfrac12+it)
\)
is unitary.

These statements are classical in the spectral theory of Eisenstein series and scattering on finite-area surfaces; see the references in §References.

#### 3.3.3 Transfer to the hinge’s involution constraint

The hinge proposition [notes/proposition_identification_hinge.md](notes/proposition_identification_hinge.md) asks for an involution constraint of the form
\(
\lambda(1-s)\,\overline{\lambda(s)}=1
\)
on $\Re(s)=1/2$ for the extracted scalar channel.

Once Task B identifies the project’s limiting scalar channel $\lambda(s)$ with a scalar channel of $\mathcal{S}(s)$ (up to an $s$-independent scalar/unitary normalization fixed at a basepoint), Theorem B immediately supplies the hinge’s involution hypothesis:

- functional equation gives $\lambda(1-s)=\lambda(s)^{-1}$,
- unitarity on the line gives $\overline{\lambda(s)}=\lambda(s)^{-1}$ for $s=1/2+it$,

so $\lambda(1-s)\overline{\lambda(s)}=1$ on $\Re(s)=1/2$.

**Remark (why we do not force an FE directly at the DN-map level).**
The $s\mapsto 1-s$ functional equation is a statement about *incoming/outgoing* asymptotics for generalized eigenfunctions at the cusp; it is native to the scattering operator. In contrast, the DN/Weyl family $\Lambda(s)$ is most naturally a meromorphic family in $\lambda=s(1-s)$ with adjoint symmetry; attempting to write an FE at the DN level generally requires choosing a specific boundary normalization and relating it back to cusp incoming/outgoing data. For this reason, Task A treats the FE at the scattering level and uses extension-theory/boundary-triple formulas only to relate normalizations.

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

## 5) Growth control in strips (for the hinge)

The hinge proposition [notes/proposition_identification_hinge.md](notes/proposition_identification_hinge.md) accepts any standard growth hypothesis sufficient for a Phragmén–Lindelöf argument applied to $\log Q$ in vertical strips.

Task A’s deliverable is therefore not an optimal bound, but a clean, citable *finite-order / strip-growth* statement for the limiting scattering object (and hence for any scalar eigenchannel extracted from it on divisor-free regions).

### 5.1 Standard routes to a strip-growth bound

Any one of the following standard inputs is sufficient (choose whichever matches the eventual write-up/citation preference):

1) (**Finite order via Selberg zeta / scattering determinant.**) For finite-area hyperbolic cusp surfaces, the determinant of the scattering matrix can be expressed in terms of the Selberg zeta function (or closely related zeta data), implying the scattering determinant is a meromorphic function of finite order. Finite order implies a growth bound of the type allowed by the hinge proposition.

2) (**Polynomial/exponential bounds for Eisenstein/scattering in strips.**) Classical estimates for Eisenstein series and scattering coefficients in vertical strips (away from poles) imply polynomial (or at worst subexponential) bounds in $|t|$. These bounds can be propagated to the boundary scattering operator and its scalar channels.

3) (**Abstract operator-theoretic bounds.**) If $S_\eta(s)$ admits a Fredholm determinant representation $\det(I+K(s))$ with $K(s)$ trace class and $\|K(\sigma+it)\|_{\mathfrak{S}_1}$ controlled in strips, then $\log|\det(I+K(s))|$ satisfies a strip-growth bound. This is the extension-theory/boundary-triple analogue of (1).

### 5.2 Theorem C (finite order / strip growth for scattering determinant)

**Theorem C (standard; scattering determinant has finite order).**
For a finite-area hyperbolic surface with cusps, the scattering determinant
\(
\det \mathcal{S}(s)
\)
is a meromorphic function of **finite order** in $s$ (indeed it admits an explicit factorization in terms of Selberg zeta data).

Consequently, after clearing poles, $\det\mathcal{S}(s)$ satisfies a growth bound in vertical strips of the type required by the hinge proposition: for each strip $\sigma_1\le \Re(s)\le \sigma_2$ there exist constants $A,B>0$ such that
\[
\log\big|\det\mathcal{S}(\sigma+it)\big|\;\le\; A\,(1+|t|)^B
\]
for all $\sigma\in[\sigma_1,\sigma_2]$ away from the poles.

This (finite-order) statement is the most direct way to discharge the hinge’s growth hypothesis in a citation-forward way.

### 5.3 How this supplies the hinge’s growth hypothesis for the scalar channel

On divisor-free regions (safe rectangles) where the scalar channel $\lambda(s)$ is isolated and varies meromorphically, any of the following is sufficient for the hinge:

- a finite-order bound for $\lambda(s)$ (or for a finite product of scalar channels), or
- a strip growth bound $|\lambda(\sigma+it)|\le \exp(C|t|^\alpha)$.

Once Task B identifies $\lambda(s)$ with the modular cusp channel (or more generally a scalar channel of $\mathcal{S}(s)$) up to a fixed $s$-independent normalization, Theorem C (finite order of $\det\mathcal{S}(s)$) provides the required strip growth for $\lambda$ (and hence for $Q$ in the hinge) after clearing poles.

**Alternative acceptable route (if preferred).**
Instead of scattering determinants, one can cite classical vertical-strip bounds for Eisenstein series and the scattering coefficient itself (e.g. from Iwaniec/Hejhal). The determinant/finite-order route is usually the cleanest for a rigidity hinge.

---

## 6) Interface with the finite-N objects

The finite harness constructs $\Lambda_N(s)$ via a Schur complement on a truncated/discretized bulk operator. The intended analytic statement to prove (as part of H2 / Task A) is:

- for each safe rectangle $R$, $\Lambda_N(s)\to\Lambda(s)$ uniformly on $R$ in operator norm (or a norm sufficient to control the Cayley transform),
- the Swap± decomposition is consistent under truncation,
- and the isolated channel eigenvalue converges uniformly on $R$.

These are exactly the hypotheses assumed in [notes/lemma_safe_rectangle_convergence.md](notes/lemma_safe_rectangle_convergence.md).

For a clean, model-agnostic mechanism showing “bulk truncation convergence \(\Rightarrow\) DN/Weyl convergence”, see:

- [notes/lemma_truncation_to_dnmap_convergence.md](notes/lemma_truncation_to_dnmap_convergence.md)

For the fully instantiated statement of H2 in terms of the exact finite objects Experiment E constructs (bulk matrix, boundary index set, Schur complement DN map, Cayley/relative scattering, Swap± reduction, and the scalar channel selection rule), see:

- [notes/h2_operator_consistency_instantiated.md](notes/h2_operator_consistency_instantiated.md)

---

## 7) What remains after this note

This note only constructs/records $\Lambda(s)$ and $S_\eta(s)$ abstractly and their meromorphy + involution.

The remaining conceptual bridge is Task B: identify the scalar scattering channel $\lambda(s)$ with the Eisenstein intertwining/scattering coefficient for $\mathrm{SL}(2,\mathbb{Z})\backslash\mathbb{H}$, including normalization.

See also: `notes/hypotheses_to_artifacts.md` for the single-page audit table used in the other notes.

---

## 8) Referee checklist (Task A)

This is the minimal set of analytic statements that must be pinned down (by explicit model specification + standard citations) for Task A to be considered “paper-complete”.

### A1 (Geometric operator model)

- Specify the surface/manifold, cusp structure, and the exact operator (typically the nonnegative Laplacian \(\Delta_X\) on \(L^2(X)\)).
- Fix the spectral parameter convention \((\Delta_X-s(1-s))u=0\), and define the cut \(\partial X_Y\) (horocycle or equivalent).

### A2 (Boundary space + trace maps)

- Define \(\mathcal{H}_\partial\) and the trace maps \(\Gamma_0\) (Dirichlet) and \(\Gamma_1\) (Neumann/flux), including the Green identity and the adjoint pairing convention.

### A3 (Meromorphy of the resolvent and Weyl/DN map)

- Cite meromorphic continuation of the resolvent on the chosen cusp geometry (analytic Fredholm framework).
- Deduce operator-valued meromorphy of \(\Lambda(s)\) (Weyl function) and hence of \(S_\eta(s)\).

### A4 (Operator-level involution / functional equation)

- Prove (or cite) the adjoint/duality identity \(\Lambda(1-s)=D_b\,\Lambda(s)^*\,D_b^{-1}\) and the induced Cayley/scattering involution.

### A5 (Growth control in strips)

- Supply a strip growth bound sufficient for the hinge proposition’s Phragmén–Lindelöf step (or name the alternative rigidity route being used).

### A6 (Finite \(N\) truncations \(\Rightarrow\) H2 consistency)

- Define the finite objects \(\Lambda_N(s)\) as Schur complements of a truncation scheme consistent with the limiting model.
- Prove uniform convergence on safe rectangles in a norm strong enough to pass through the Cayley transform.

For the abstract Schur-complement step used in A6, see [notes/lemma_truncation_to_dnmap_convergence.md](notes/lemma_truncation_to_dnmap_convergence.md).

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

- Modular surface / finite-area cusp scattering functional equation and growth control (choose one standard spectral source):

  - D. Hejhal, *The Selberg Trace Formula for PSL(2,\mathbb{R})*, Vol. II (scattering theory for finite-area surfaces; functional equation/unitarity; scattering determinant).
  - H. Iwaniec, *Spectral Methods of Automorphic Forms* (Eisenstein series, constant terms, scattering coefficients; standard strip bounds).
  - D. Borthwick, *Spectral Theory of Infinite-Area Hyperbolic Surfaces* (contains general resolvent/scattering analytic Fredholm technology; many statements also cover the finite-area cusp case).
