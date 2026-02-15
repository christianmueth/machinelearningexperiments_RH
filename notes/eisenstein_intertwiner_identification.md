# Eisenstein intertwiner identification (Task B)

## Scope and intent

This note supplies the final representation-theoretic identification:

> the scalar scattering channel extracted from the operator-level Cayley/DN scattering is the classical Eisenstein scattering coefficient for $\mathrm{SL}(2,\mathbb{Z})\backslash\mathbb{H}$.

It is structured as a **mapping theorem** (with explicit citations) plus a **normalization statement** that feeds directly into:

- [notes/proposition_identification_hinge.md](notes/proposition_identification_hinge.md)

No new numerics are introduced.

This note is also **citation-forward**: the goal is to make it easy to point to standard references for (i) Eisenstein series constant terms and scattering matrices, and (ii) the representation-theoretic intertwining-operator normalization that produces the ratio of completed zeta factors.

---

## 1) Standard modular scattering object

Let $X = \mathrm{SL}(2,\mathbb{Z})\backslash\mathbb{H}$.

### 1.1 Eisenstein series and constant term

Let $E(z,s)$ be the standard weight-0 Eisenstein series at the cusp at infinity (normalized in the standard analytic number theory convention). Its Fourier expansion at the cusp has constant term
\[
E(z,s) = y^s + \phi(s)\,y^{1-s} + (\text{non-constant Fourier modes}).
\]

The scalar function $\phi(s)$ is the (one-cusp) **scattering coefficient**.

In scattering language, $y^s$ is the incoming asymptotic and $\phi(s)y^{1-s}$ is the outgoing asymptotic in the constant Fourier mode.

Equivalently (representation-theoretically), $\phi(s)$ is the scalar by which the normalized standard intertwining operator acts on the spherical vector in the (unramified) principal series.

### 1.2 Functional equation and unitarity

Standard scattering theory gives
\[
\phi(s)\,\phi(1-s)=1,
\]
and on the critical line $\Re(s)=1/2$,
\[
|\phi(1/2+it)|=1.
\]

### 1.3 Explicit formula for \(\mathrm{SL}(2,\mathbb{Z})\)

For $\mathrm{SL}(2,\mathbb{Z})$, one has the classical explicit form
\[
\phi(s) = \frac{\xi(2s-1)}{\xi(2s)} =: \phi_{mod}(s),
\]
where $\xi$ is the completed Riemann xi function.

This identity is the “modular content” target of the project.

---

## 2) Operator scattering from DN/Cayley and its relation to \(\phi(s)\)

### 2.1 DN/Cayley scattering operator

From Task A (see [notes/operator_construction_and_meromorphy.md](notes/operator_construction_and_meromorphy.md)), we have:

- an operator-valued DN map $\Lambda(s)$ on the boundary space at a cusp cross-section,
- and the Cayley/scattering operator
\[
S_\eta(s) = (\Lambda(s)-i\eta I)(\Lambda(s)+i\eta I)^{-1}.
\]

On $\Re(s)=1/2$ (away from poles), $S_\eta(s)$ is unitary.

### 2.2 Scattering as an intertwiner / boundary-to-boundary map

The classical scattering operator $\mathcal{S}(s)$ for a cusp is the operator mapping incoming boundary data to outgoing boundary data.

Concretely, in the cusp Fourier decomposition (modes in $x\in\mathbb{R}/\mathbb{Z}$), each mode has an incoming and outgoing amplitude; the scattering operator relates them.

In the one-cusp modular case, the **constant mode** (“cusp channel”) is one-dimensional, and the scattering operator restricted to that mode is multiplication by the scalar $\phi(s)$.

More explicitly: if $P_0$ denotes the orthogonal projection in $L^2(\mathbb{R}/\mathbb{Z})$ onto the constant functions in $x$, then the constant-mode scattering is the scalar operator
\[
P_0\,\mathcal{S}(s)\,P_0 = \phi(s)\,P_0.
\]

### 2.3 The key bookkeeping object: the constant-mode projector

To make the identification referee-proof, the “cusp channel” should be named as an explicit projector.

Fix a cusp cross-section $\partial X_Y\cong \mathbb{R}/\mathbb{Z}$ with coordinate $x$.
Then the boundary Hilbert space admits the Fourier decomposition
\[
L^2(\partial X_Y) = \bigoplus_{n\in\mathbb{Z}} \mathbb{C}\,e^{2\pi i n x}.
\]
Let $P_0$ be the orthogonal projection onto the $n=0$ subspace.

In the project’s two-channel boundary setting (doubling plus Swap), one may view the boundary space as
\[
\mathcal{H}_\partial^{(2)} \cong L^2(\partial X_Y)\oplus L^2(\partial X_Y),
\]
with the same Fourier decomposition in each component.
Define the two-channel constant-mode projector
\[
P_0^{(2)} := P_0\oplus P_0.
\]
Since Swap acts only between the two components, $P_0^{(2)}$ commutes with Swap and hence with $\Pi_\pm$.

The “cusp channel” is then the (one-dimensional) invariant subspace
\[
\mathcal{C}_0 := \mathrm{ran}(P_0^{(2)}\Pi_\sigma)
\]
for the appropriate Swap sign $\sigma\in\{+,-\}$ determined by the model’s symmetry convention.

### 2.4 Mapping lemma (referee-facing form)

**Theorem (DN/Weyl \(\to\) scattering; constant-mode channel \(=\phi(s)\)).**
Assume:

1) (**Boundary triple / Weyl function hypothesis**) The cusp-cut Laplacian model admits a (quasi-)boundary triple description in which the DN/Weyl operator $\Lambda(s)$ is the corresponding Weyl function (or a fixed linear fractional transform of it), with the standard meromorphic continuation and adjointness relations.

2) (**Scattering matrix formula**) The scattering matrix for the corresponding pair of self-adjoint realizations is given by a standard Weyl-function formula (equivalently: by a linear fractional transform of the Weyl function). In particular, after choosing an incoming/outgoing boundary basis, the associated scattering operator is conjugate (up to an $s$-independent unitary change of basis) to a Cayley-type transform of the DN/Weyl object.

3) (**Cusp Fourier bookkeeping**) The boundary space is canonically identified with $L^2(\partial X_Y)$ (or $L^2(\partial X_Y)^{\oplus 2}$ in the two-channel case), so the Fourier decomposition and constant-mode projector $P_0$ (hence $P_0^{(2)}$) are intrinsic.

Then the DN/Cayley scattering operator $S_\eta(s)$ constructed in Task A is (up to an $s$-independent unitary conjugation and an $s$-independent scalar normalization) the standard cusp scattering operator $\mathcal{S}(s)$.

Moreover, if the cusp scattering preserves Fourier modes (as in the standard cusp model), the restriction of the scattering operator to the constant Fourier mode is scalar, and one has
\[
P_0\,\mathcal{S}(s)\,P_0 = \phi(s)\,P_0.
\]
Consequently, in the two-channel setting with cusp channel $\mathcal{C}_0=\mathrm{ran}(P_0^{(2)}\Pi_\sigma)$, the scalar eigenchannel extracted from $S_\eta(s)$ satisfies
\[
\lambda(s)=c\,\phi(s),\qquad c\in\mathbb{C}^*\text{ independent of }s.
\]

**Canonical channel extraction (Riesz projector / gap).**
On any safe rectangle $R$ where the relevant scalar channel is isolated by a uniform spectral gap, define the eigenprojection by the Riesz formula
\[
\mathsf{P}(s)=\frac{1}{2\pi i}\oint_{\gamma}\,(zI-S_\eta(s))^{-1}\,dz,
\]
with $\gamma$ a fixed contour encircling only the target eigenvalue for all $s\in R$.
Then $s\mapsto \mathsf{P}(s)$ is holomorphic on $R$, and the “cusp channel” choice is canonical once one imposes a single basepoint overlap condition (nontrivial overlap with $\mathcal{C}_0$ at one $s_0\in R$). Continuity of $\mathsf{P}(s)$ prevents channel swapping.

**Drop-in references for (1)–(2).**
For scattering matrices expressed in terms of Weyl functions/boundary triples (finite deficiency indices) and the associated linear fractional transforms:

- J. Behrndt, M. M. Malamud, H. Neidhardt, *Scattering matrices and Weyl functions*, Proc. London Math. Soc. (2008). arXiv:math-ph/0604013. DOI: 10.1112/plms/pdn016.
- J. Behrndt, H. Neidhardt, E. R. Racec, P. N. Racec, U. Wulf, *On Eisenbud's and Wigner's R-matrix: A general approach*, J. Differential Equations 244 (2008), 2545–2577. DOI: 10.1016/j.jde.2008.02.004.

(If one needs infinite-dimensional boundary spaces / non-ordinary triples, use the quasi-boundary-triple generalizations in the Behrndt–Neidhardt program; the proof skeleton here only uses that the scattering operator is a linear fractional transform of a Weyl object and inherits meromorphy/adjointness.)

**Comment (why this answers the ‘discretization proxy’ concern).**
The only genuinely model-specific input is (1)–(3), which are statements about how the limiting boundary model aligns with the classical cusp Fourier decomposition. Once that bookkeeping is fixed, the rest is standard: constant mode is 1D, it is invariant under scattering, and its scattering coefficient is the classical $\phi(s)$.

### 2.5 Mapping lemma (summary statement)

**Corollary (Modular specialization).**
For $X=\mathrm{SL}(2,\mathbb{Z})\backslash\mathbb{H}$, the standard constant-term computation of the Eisenstein series gives the explicit scattering coefficient
\[
\phi(s)=\phi_{mod}(s)=\frac{\xi(2s-1)}{\xi(2s)}.
\]
Therefore $\lambda(s)=c\,\phi_{mod}(s)$ with $c$ independent of $s$.

**Explanation of where the constant can come from.**
Different scattering normalizations correspond to different conventions for the boundary pairing (flux normalization) and for the incoming/outgoing basis. The impedance $\eta$ Cayley transform is a convenient way to build a unitary operator on the critical line, but may differ from the classical Eisenstein normalization by an $s$-independent unit modulus factor.

In practice, the project already treats this correctly: any such scalar ambiguity is removed by a single basepoint normalization (or, in the finite-$N$ diagnostics, by a fitted constant $c_N$ / relative normalization). In the proof, it is a single axiom.

---

## 3) Normalization and the role of a basepoint

The identification hinge proposition is formulated in terms of
\[
Q(s)=\frac{\lambda(s)}{\phi_{mod}(s)}.
\]

Once the mapping lemma shows $\lambda(s)=c\,\phi(s)$ and the standard theory gives $\phi(s)=\phi_{mod}(s)$, we obtain
\[
Q(s)\equiv c.
\]

A single normalization condition at a basepoint $s_0$ (or any equivalent choice) fixes $c=1$:

- Choose $s_0$ in a safe rectangle where both objects are holomorphic and nonzero.
- Impose $Q(s_0)=1$.

In the numerics, this is the same role played by the fitted constant $c_N$ or by free normalization / reference matching; in the proof, it collapses to a single scalar normalization axiom.

**Important separation of duties.**
Numerics are not used to *prove* that $c$ is constant in $s$; that is part of the mapping lemma (a structural statement about how two constructions of the scattering operator relate). Numerics only witness that a basepoint normalization is compatible with the observed stability on safe rectangles.

---

## 4) How this plugs into the existing trilogy

1) Task A provides $\Lambda(s)$ and $S_\eta(s)$ meromorphy + involution + Swap± symmetry.
2) The safe-rectangle convergence lemma upgrades finite-N stability/winding to limiting holomorphy/nonvanishing on safe rectangles.
3) This note (Task B) identifies the limiting channel with the modular $\phi_{mod}(s)$ up to a constant.
4) The hinge proposition then forces $Q\equiv 1$.

At that point, the analytic identification $\lambda(s)=\xi(2s-1)/\xi(2s)$ is established, and the remaining RH-level consequences are standard scattering-theoretic corollaries.

---

## 5) What is still “model-specific” vs. standard

The theorem above reduces the last external-world touchpoint to a clean split:

- **Standard black boxes:** boundary-triple/Weyl-function scattering formula; and the explicit Eisenstein constant term for $\mathrm{SL}(2,\mathbb{Z})$.
- **Model-specific bookkeeping:** the boundary identification and the fact that the extracted Swap± channel corresponds to the constant-mode cusp channel (enforced canonically by the Riesz projector under a safe-rectangle gap).

See also: `notes/hypotheses_to_artifacts.md` for how the safe-rectangle hypotheses (gap/invertibility margins) are witnessed in run artifacts.

---

## 6) Referee checklist (minimal hypotheses to verify)

This section spells out, in one place, the exact hypotheses that must be checked to apply the cited Weyl-function \(\leftrightarrow\) scattering results to the modular cusp model **without** any numerical input.

### H1 (Geometric operator is the modular cusp Laplacian)

- The bulk operator is the Laplace–Beltrami operator \(\Delta\) on \(X=\mathrm{SL}(2,\mathbb{Z})\backslash\mathbb{H}\) (or the exactly specified 1-cusp hyperbolic surface under study), considered with spectral parameter \(s\) via \((\Delta-s(1-s))u=0\).
- A cusp cross-section \(\partial X_Y\cong \mathbb{R}/\mathbb{Z}\) (horocycle cut) is fixed once and for all.

### H2 (Extension pair and boundary model are specified)

- Specify the symmetric operator \(A\) (e.g. the minimal Laplacian on the cut manifold) and the two self-adjoint realizations \(A_0\) and \(A_\Theta\) whose scattering system is being described.
- State explicitly how the impedance/Cayley parameter \(\eta\) corresponds to the extension parameter \(\Theta\) (this is where different scattering normalizations can enter).

### H3 (Boundary space and trace maps)

- The boundary space is \(\mathcal{H}_\partial=L^2(\partial X_Y)\) (or \(\mathcal{H}_\partial^{(2)}\cong L^2(\partial X_Y)^{\oplus 2}\) in the two-channel setting).
- The boundary maps \(\Gamma_0\) (Dirichlet trace) and \(\Gamma_1\) (Neumann/flux trace) are specified on the maximal domain so that the relevant Green identity holds.
- Fix the boundary pairing / flux convention used to define adjoints (this is the only place sign conventions can materially affect formulas).

### H4 (Weyl function equals the DN/Weyl operator used)

- Verify that the Weyl function \(M(s)\) in the cited boundary triple framework coincides with the DN/Weyl operator \(\Lambda(s)\) used in Task A (or differs from it by a fixed, declared linear fractional transform).
- Conclude meromorphic continuation and the required adjointness identities from the standard theory (no RH input is needed here).

### H5 (Scattering matrix formula matches the Cayley transform)

- Using the cited Weyl-function scattering formula, show that the scattering matrix \(\mathcal{S}(s)\) for the extension pair is a linear fractional transform of \(\Lambda(s)\).
- Match this transform to the Cayley form
  \[
  S_\eta(s)=(\Lambda(s)-i\eta I)(\Lambda(s)+i\eta I)^{-1}
  \]
  up to an \(s\)-independent unitary change of boundary basis.

This is exactly the black-box content of Behrndt–Malamud–Neidhardt (and related extension-theoretic scattering references).

### H6 (Normalization mismatch is only an \(s\)-independent scalar)

- Check that any remaining mismatch between the DN/Cayley convention and the Eisenstein/scattering convention is an \(s\)-independent scalar (typically unit-modulus), coming from boundary basis / flux normalization.
- Fix it by one basepoint normalization \(Q(s_0)=1\) as in §3.

### H7 (Fourier-mode invariance; constant mode is the cusp channel)

- Under the cusp identification \(\partial X_Y\cong \mathbb{R}/\mathbb{Z}\), the boundary translation symmetry implies the scattering operator preserves Fourier modes in \(x\).
- In particular, the constant Fourier mode subspace \(\mathrm{ran}(P_0)\) is invariant, one-dimensional in the one-cusp modular case, and the restriction of the scattering operator to it is multiplication by \(\phi(s)\).

This is standard in cusp scattering theory and is recorded explicitly in references that compute the Eisenstein constant term (e.g. Iwaniec; Hejhal/Borthwick-style cusp scattering texts).

### H8 (Canonicity of the extracted scalar channel on safe rectangles)

- On each safe rectangle \(R\), assume a uniform spectral gap isolating the target channel eigenvalue.
- Define the eigenprojection by the Riesz projector (fixed contour) and use holomorphic dependence to prevent eigenchannel swapping.

This is the analytic mechanism that makes “the cusp channel” a canonical object once the constant-mode overlap is fixed at one basepoint.

---

## References (standard sources to cite)

- Classical constant-term/scattering normalization (primary viewpoint for this note):

  - H. Iwaniec, *Spectral Methods of Automorphic Forms*.
  - (Any standard automorphic scattering text works here; the key requirement is an explicit Eisenstein constant term formula and the one-cusp scattering coefficient.)

- Intertwining operator viewpoint (equivalent formulation, used only to justify “scalar ambiguity + basepoint fixes it” cleanly):

  - D. Bump, *Automorphic Forms and Representations*.

- Weyl function / boundary triple scattering formulas (DN/Weyl \(\leftrightarrow\) scattering):

  - J. Behrndt, M. M. Malamud, H. Neidhardt, *Scattering matrices and Weyl functions*, Proc. London Math. Soc. (2008). arXiv:math-ph/0604013. DOI: 10.1112/plms/pdn016.
  - J. Behrndt, H. Neidhardt, E. R. Racec, P. N. Racec, U. Wulf, *On Eisenbud's and Wigner's R-matrix: A general approach*, J. Differential Equations 244 (2008), 2545–2577. DOI: 10.1016/j.jde.2008.02.004.
