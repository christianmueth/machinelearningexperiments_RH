# Mandelbrot Local Connectivity (MLC): one-hinge lemma + closure, and the RH-pipeline analogy

## Scope

This note records (i) the **clean logical skeleton** behind the standard “MLC via puzzle-piece shrinking” program, and (ii) how that skeleton matches the RH/Experiment-E pipeline structure in this repository.

This note is **not** a claim that MLC is proven globally. The whole point of the reduction is that the global conjecture remains open because the required **uniform admissibility engine** (and the corresponding shrinking/closure mechanism on that admissible class) is not known in full generality.

---

## Part I. MLC in “one-hinge lemma + closure” form

### Object

Fix a parameter $c$ in the Mandelbrot set $\mathcal{M}$. Let
\[
P_n(c) \ni c
\]
be the nested **parapuzzle pieces** (parameter-space puzzle neighborhoods) defined via the Yoccoz puzzle / parapuzzle machinery.

The pieces satisfy (by construction) $P_{n+1}(c) \subseteq P_n(c)$ and are connected neighborhoods of $c$.

### Conjecture (local form)

MLC at $c$ is equivalent to
\[
\operatorname{diam}(P_n(c)) \to 0 \quad (n\to\infty).
\]
Once the diameter shrinks to $0$, the impression of the nested neighborhoods is a point, which yields local connectivity at $c$ (this is the “formal closure step”; see below).

### Closure step (formal)

If $(P_n)$ is a nested sequence of connected neighborhoods of $c$ with $\operatorname{diam}(P_n)\to 0$, then
\[
\bigcap_{n\ge 1} \overline{P_n} = \{c\},
\]
and the local connectivity conclusion at $c$ follows from standard topological facts (no dynamical content here).

### One-hinge lemma (the only substantive content)

All the dynamical difficulty is pushed into: **admissibility $\Rightarrow$ controlled refinement with a ghost-eliminating closure mechanism**.

- Let $R$ denote the parapuzzle refinement operator (“go one depth deeper”).
- Let $A$ denote an admissible class of puzzle data (bounded geometry / a priori bounds).
- Let $d$ be a geometry-controlling metric on the appropriate moduli space (typically Teichmüller distance, or a modulus-based distance that controls distortion/shape).

**Hinge lemma (template: contraction form).** If there exists $\kappa<1$ such that for all $x,y\in A$,
\[
 d(Rx, Ry) \le \kappa\, d(x,y),
\]
then refinement is a strict contraction on $A$, and one deduces exponential shrinking of geometry:
\[
\operatorname{diam}(P_n(c)) \le C\,\kappa^n \to 0.
\]

**Hinge lemma (template: rigidity/uniqueness closure form).** Even without a literal $\kappa<1$, one can still aim for an MLC-style conclusion if admissibility produces:

- non-expansive (or uniformly well-conditioned) refinement on $A$,
- a canonicalization mechanism that prevents mode swapping / fragmentation,
- and a rigidity/uniqueness principle that forces a single limit object once invariants hold.

Interpretation:
- The “closure step” (nested pieces + diameters shrink $\Rightarrow$ MLC) is **formal**.
- The “one-hinge lemma” is where all real work sits: proving that your refinement map is controlled on an admissible class, and that a suitable contraction or rigidity/closure mechanism forces shrinking.

### Why global MLC remains hard (in this skeleton)

To apply an MLC-style hinge uniformly over all $c\in\mathcal{M}$ you need a uniform admissible class $A$ (a priori bounds, complex bounds, control of renormalizations, etc.) that is stable under refinement and strong enough to force shrinking via either (i) a contraction argument or (ii) an alternative rigidity/uniqueness closure mechanism.

This is precisely the missing global engine.

---

## Part II. Exact structural match to the RH/Experiment-E pipeline

The RH pipeline in this repository has the same two-level structure:

1) **Admissibility / stability engine** (what numerics can witness)
2) **Rigidity closure** (purely analytic: once admissible hypotheses hold, the quotient is forced to be constant)

### RH pipeline as a refinement system

- **State space.** Operator/scattering data extracted from truncation level $N$:
  - DN/Weyl map $\Lambda_N(s)$,
  - Cayley/scattering $S_{\eta,N}(s)$,
  - symmetry-reduced scalar channel $\lambda_N(s)$,
  - quotient $Q_N(s)=\lambda_N(s)/\phi_{mod}(s)$.

- **Refinement.** Increase truncation depth $N$ (plus analytic continuation in $s$ on chosen rectangles).

- **Admissibility class (the “a priori bounds” analogue).** The safe-rectangle + gap + margin conditions:
  - uniform invertibility margin for $(\Lambda_N(s)+ i\eta I)$ on the rectangle boundary,
  - spectral gap isolating the cusp channel so Riesz projectors vary continuously (no swapping),
  - divisor cleanliness / winding diagnostics on rectangle boundaries.

This is the RH analogue of “bounded geometry / a priori bounds”: it prevents degeneracy under refinement and makes the extracted channel canonical.

### Closure principle (the RH analogue of “nested pieces shrink”)

Once Tasks A/B supply the analytic obligations (meromorphy, growth/functional equation, correct identification of the channel), and once admissibility ensures divisor control on safe rectangles, the complex-analytic hinge forces
\[
Q(s) \equiv 1.
\]
That closure step is the RH analogue of “nested connected neighborhoods + diameters shrink $\Rightarrow$ local connectivity”: it is formal once the hypotheses are secured.

### What corresponds to the MLC contraction constant $\kappa<1$?

In RH you do **not** literally need a strict contraction of a metric, because on the critical line unitarity/non-expansiveness is the natural symmetry.

The closest honest structural match is:

- **Non-expansive backbone** (unitarity on $\Re(s)=1/2$ away from poles), plus
- **Admissibility-gated Lipschitz control** (invertibility margins keeping Cayley stable), plus
- **Canonical channel tracking** (gap + Riesz projector preventing swapping), plus
- **Rigidity/closure** (the quotient-hinge proposition forcing canonicity once invariants and divisor-control hold).

What you need instead is the bundle of strict controls that prevent “ghost” poles/zeros and prevent channel ambiguity:

- **Invertibility margins** (keep Cayley analytic; no spurious poles).
- **Riesz projector continuity / spectral gap** (channel is canonical; no swapping).
- **Normal-family type bounds** (subsequence limits exist and are unique on rectangles).

This is exactly the “MLC-style admissibility first, then closure” doctrine in operator language.

---

## Part III. How this answers the two gap questions

### Gap A: “Is the pipeline internally consistent and numerically stable?”

MLC lens: are you operating inside an admissible class where refinement cannot degenerate?

Translated to this repo’s language:

- Safe rectangles are actually safe: boundary margins stay bounded away from $0$.
- The isolated cusp channel stays isolated: a gap exists and the Riesz projector is stable in $s$ and $N$.
- Winding/argument diagnostics stabilize once divisor-clean boundaries are enforced.

If these hold, the pipeline is internally stable in the only sense that matters: the computed objects are coherent witnesses for a limiting analytic object, not artifacts of degeneracy.

### Gap B: “Is it computing the *modular* DN/scattering object (or a controlled surrogate)?”

MLC lens: are you sure your refinement/state space is the correct one, not a stable look-alike?

Here the admissibility engine does *not* suffice. You need external identification:

- The mapping theorem hypotheses for the actual intended operator model (cusp-cut modular Laplacian or an exactly controlled surrogate).
- Normalization pinned down (to rule out an $s$-dependent scalar factor that preserves symmetry but changes divisor data).
- Observable match: the extracted channel is truly the constant Fourier mode / correct cusp channel of the intended scattering operator.

This is exactly why Task B carries a referee-style hypothesis checklist: stability is necessary, but identification is a separate obligation.

---

## Part IV. Minimal calibration tests (upgrade confidence without pretending it’s proof)

These are “falsification-first” checks of the *operator-theoretic semantics* of the DN→Cayley→channel extraction machinery.

1) **1D Sturm–Liouville toy:** DN is explicit; compare the Cayley transform to a known reflection/transmission coefficient.
2) **Finite graph with ports:** DN via Schur complement is exact; scattering can be computed independently by linear algebra.
3) **Separable cylinder/cusp model:** Fourier modes decouple; verify constant-mode projection + Riesz extraction returns the analytically known coefficient.

Passing these tests supports Gap A strongly and reduces the chance that “modular-looking behavior” is a code artifact.

---

## Pointers inside this repo

- RH scaffold overview: [notes/proof_roadmap.md](proof_roadmap.md)
- Admissibility-as-witnesses principle: [notes/methodology_numerics_as_witnesses.md](methodology_numerics_as_witnesses.md)
- Safe rectangles and convergence: [notes/lemma_safe_rectangle_convergence.md](lemma_safe_rectangle_convergence.md)
- Channel canonicity (no swapping): [notes/lemma_selector_admissibility.md](lemma_selector_admissibility.md)
- Operator/scattering backbone: [notes/operator_construction_and_meromorphy.md](operator_construction_and_meromorphy.md)
- Modular identification obligations: [notes/eisenstein_intertwiner_identification.md](eisenstein_intertwiner_identification.md)
- Rigidity hinge for $Q(s)$: [notes/proposition_identification_hinge.md](proposition_identification_hinge.md)
