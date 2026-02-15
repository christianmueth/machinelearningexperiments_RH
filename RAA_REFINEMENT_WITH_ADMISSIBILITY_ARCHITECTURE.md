# Refinement-with-Admissibility Architecture (RAA)

## Scope

This note records an **abstraction** suggested by the repository’s core structure:

- the code/logic here is potentially interesting for AI systems **not because it “proves RH”**, but because it instantiates a *general-purpose pattern* for building **stable refinement systems** with explicit **admissibility** and explicit **ghost diagnostics**.

This is written in “research pitch” form while staying grounded in what this repo actually implements (auditability, admissibility gating, and failure-mode instrumentation).

---

## 1) Unifying core: refinement systems and ghost elimination

Across three domains:

- **Mandelbrot (MLC):** parapuzzle refinement should shrink; otherwise one gets persistent branching/non-shrinking geometry.
- **RH/scattering pipeline (this repo):** analytic continuation/refinement should not introduce spurious divisor structure; otherwise one gets “ghost” zeros/poles and channel ambiguity.
- **Recursive AI reasoning:** multi-step refinement should collapse ambiguity; otherwise one gets hallucination cascades and branch explosion.

A shared template:

- a refinement map $R$ on a state space,
- an admissible subset $A$ where refinement behaves well,
- a metric/geometry-control quantity $d$,

and an engine that enforces (at least)

- **non-expansiveness** globally (no blow-ups), and
- **admissibility-gated stability** on $A$ (well-conditioned/Lipschitz dependence plus a *canonicalization mechanism* that prevents mode swapping and eliminates ghost branches).

In many serious systems you do **not** need (and should not overclaim) a literal contraction constant $\kappa<1$ in refinement depth. A cleaner and often more defensible closure pattern is:

- **Non-expansive backbone** (bounded dynamics; often isometric/unitary on a symmetry locus), plus
- **Admissibility-gated Lipschitz control** (explicit conditioning margins), plus
- **Canonical channel tracking** (gap/projector-style anti-swapping), plus
- **Rigidity/closure** (once invariants hold on admissible regions, a uniqueness principle forces a canonical outcome).

**Why MLC matters:** it’s the cleanest theorem template where admissibility engineering turns “refinement” into a controlled dynamical system, and where the last step becomes formal once the right closure mechanism (contraction or rigidity/uniqueness) is in place.

Pointers:
- MLC hinge skeleton and its RH mapping: [notes/mlc_hinge_lemma_and_connection.md](notes/mlc_hinge_lemma_and_connection.md)
- RH proof-chain scaffold: [notes/proof_roadmap.md](notes/proof_roadmap.md)

### κ and “curvature”: what we do and do not claim

In contraction-style refinements one often summarizes stability by a constant $\kappa$ in an inequality
\[
d(Rx,Ry)\le \kappa\,d(x,y).
\]

- $\kappa<1$ is **strict contraction** (hyperbolic/shrinking behavior).
- $\kappa=1$ is **non-expansive** (isometric/neutral behavior).
- $\kappa>1$ is **expanding** (unstable behavior).

In the RH/scattering backbone, the natural symmetry locus is the critical line $\Re(s)=1/2$, where (away from poles) the scattering/Cayley object is unitary. That is structurally a **$\kappa=1$ backbone**: it preserves norm rather than shrinking distances.

Separately, the pipeline uses “curvature-like” diagnostics (phase/log-derivative bending, Cayley Möbius geometry, conditioning margins $\delta$ away from bottlenecks) to detect where analytic continuation becomes ill-conditioned. These are **not** a contraction constant for refinement depth; they are conditioning/rigidity diagnostics.

So the honest RAA claim this repo supports is not “we have $\kappa<1$,” but:

- a non-expansive backbone on a symmetry locus,
- admissibility-gated conditioning margins (Lipschitz control; avoid near-singular regimes),
- canonical mode tracking (anti-swapping),
- and a rigidity/closure principle that forces uniqueness once invariants hold.

### Curvature in RH/scattering vs curvature in MLC/renormalization

The word “curvature” shows up in both stories, but it refers to different geometry.

**RH/scattering (this repo) — curvature-like phenomena are analytic/conditioning diagnostics.** Typical examples:

- **Spectral/phase bending in $s$:** quantities like $\frac{d}{ds}\log\lambda(s)$ or $\frac{d}{ds}\arg\lambda(s)$ detect proximity to poles/zeros and changes in divisor geometry.
- **Cayley/Möbius geometry:** the linear fractional transform mapping self-adjoint data to a unitary scattering object is nonlinear geometry of representation; on $\Re(s)=1/2$ it is isometric/unitary (neutral, $\kappa=1$).
- **Admissibility margin $\delta$:** invertibility margins bound conditioning away from Cayley bottlenecks (Lipschitz control), but do not by themselves imply metric contraction under refinement depth.

**MLC/renormalization — curvature is dynamical/hyperbolic when it exists.** In the idealized contraction picture:

- Teichmüller/hyperbolic geometry of renormalization yields $d_T(Rf,Rg)\le \kappa\,d_T(f,g)$ with $\kappa<1$ on an admissible class.
- A priori bounds (modulus lower bounds) prevent degeneration and enable hyperbolicity/contraction arguments.

**Bottom line:** RH uses a neutral backbone + admissibility + rigidity closure; MLC (when available) uses admissibility + contraction (or another shrinking/uniqueness closure) to force piece shrinking.

---

## 1.5) Optional: where a genuine $\kappa<1$ could live for $N\mapsto N'$ refinement

Unitarity ($\kappa=1$) is about the scattering object on the symmetry locus at a fixed discretization level. A strict contraction $\kappa<1$ would have to concern a different axis: the refinement map taking a coarse discretization to a finer one.

Below are plausible places where a “refinement contraction” could emerge **without** contradicting the unitary backbone.

1) **Contraction after quotienting gauge/basis symmetries**

Raw operator norms can fail to contract if refinements differ by harmless unitary conjugacy (basis/gauge). A gauge-quotiented distance is
\[
d_{\mathrm{gauge}}(\Lambda_N,\Lambda_{N'}) := \inf_{U\in\mathcal{U}}\;\sup_{s\in R}\;\|\Lambda_N(s)-U^*\Lambda_{N'}(s)U\|.
\]
More surgically (and closer to what the proof actually uses), one can measure only the **channel projector** distance
\[
d_P(N,N') := \sup_{s\in R}\;\|P_N(s)-P_{N'}(s)\|,
\]
where $P_N(s)$ is the Riesz projector onto the isolated channel.

2) **Resolution-aware (RG-style) metrics via coarse-graining**

Comparing $N$ to $2N$ directly can hide contraction because the ambient space grows. If $\Pi_N$ is a coarse-graining/restriction map from the fine boundary space to the coarse one, define
\[
d_{\mathrm{RG}}(N,2N):=\sup_{s\in R}\;\|\Lambda_N(s)-\Pi_N\,\Lambda_{2N}(s)\,\Pi_N^*\|.
\]
If refinement mainly adds high-frequency detail, this kind of metric can contract.

3) **Metrics on spectral measures rather than operators**

Even when operators do not converge in norm, the spectral data used by the pipeline may converge rapidly in a weak topology. One can compare smoothed phase/derivative statistics (e.g. via bounded-Lipschitz or Wasserstein distances on induced measures).

4) **Projective/Hilbert metrics on positive cones (if a positive transfer object exists)**

Strict contractions are most naturally proved for strongly positive maps acting on cones. The scattering operator is unitary (not positive), but a derived positive object (kernel/density/transfer operator) might admit a projective metric where refinement is contractive.

5) **Rigidity closure as a replacement for contraction (what this repo already uses)**

Even if no natural $\kappa<1$ exists for refinement depth in any reasonable metric, the RH pipeline has a different closure mechanism:

- admissibility (margins/gaps/divisor cleanliness),
- canonical channel tracking (projectors under a gap),
- analytic rigidity (quotient forced constant).

In many settings this is stronger than contraction, because it forces uniqueness without requiring geometric shrinkage.

### Concrete “next step” proxy tests (no proof claims)

If you want to look for an empirical contraction signal across refinement, two low-friction proxies are:

- **Projector contraction proxy:**
\[
\Delta_P(N):=\sup_{s\in\partial R}\;\|P_{2N}(s)-P_N(s)\|.
\]
- **Coarse-grained DN proxy:**
\[
\Delta_\Lambda(N):=\sup_{s\in\partial R}\;\|\Lambda_N(s)-\Pi_N\,\Lambda_{2N}(s)\,\Pi_N^*\|.
\]

If either exhibits a pattern like $\Delta(2N)\approx \kappa\,\Delta(N)$ with $\kappa<1$ on safe rectangles, that would define a meaningful “refinement contraction” notion for the pipeline (again: this would be a stability diagnostic, not a proof ingredient).

---

## 2) What this repo already implements (that many AI stacks don’t)

### A) Methodology firewall (assumed vs witnessed)

The repo explicitly separates:

- **proof logic** (theorem-level dependencies), from
- **numeric witnesses** (used only to validate hypotheses, detect pathologies, and falsify).

Pointers:
- Firewall statement: [notes/methodology_numerics_as_witnesses.md](notes/methodology_numerics_as_witnesses.md)
- “Hypotheses ↔ artifacts” audit table: [notes/hypotheses_to_artifacts.md](notes/hypotheses_to_artifacts.md)
- Frozen artifact contract: [RUN_CONTRACT.md](RUN_CONTRACT.md)

### B) Admissibility is first-class (a priori bounds analogue)

The safe-rectangle discipline is “a priori bounds in operator form”:

- exclude Cayley bottlenecks (near-singular denominators),
- enforce invertibility margins,
- enforce spectral gap + projector continuity to prevent channel swapping,
- track winding/divisor proxies as failure detectors.

Pointer:
- Safe-rectangle lemma (theorem-shaped): [notes/lemma_safe_rectangle_convergence.md](notes/lemma_safe_rectangle_convergence.md)

### C) Ghost-branch diagnostics are instrumented

The harness turns “ghosts” into measurable failure modes:

- winding instability (divisor events),
- gap collapse / swapping risk,
- blowups near Cayley dips,
- non-normal-family behavior across refinement depth.

This is the same pattern as MLC’s “non-shrinking pieces / infinite renormalization” showing up as observable pathology.

---

## 3) The AGI-relevant abstraction: “MLC for reasoning” (structural, not metaphorical)

### 3.1 Reasoning puzzle pieces

Let $P_n$ denote the set of possible internal belief/plan states after $n$ refinement steps (tool calls, self-consistency passes, constraint injections, search iterations).

A stability target is an “MLC-like” collapse:
\[
\operatorname{diam}(P_n) \to 0,
\]
meaning ambiguity shrinks and a canonical belief/plan emerges.

Failure corresponds to persistent branching: hallucination modes that survive refinement.

### 3.2 Three-part engine (directly analogous to this repo)

1) **Non-expansive backbone:** updates are norm-controlled (no blow-ups).
2) **Admissibility filter:** reject states/updates in degenerate regimes; preserve $R(A)\subseteq A$.
3) **Canonical channel extraction / collapse rule:** enforce mode stability (prevent plan/goal drift).

In the RH pipeline the “collapse rule” is implemented via **spectral gap + Riesz projector** to prevent channel swapping; in reasoning it would be an explicit mechanism that forces the agent to stay in a stable “thought channel” when refinement depth increases.

---

## 4) Concrete translation sketch (operators \(\to\) reasoning)

This section is intentionally modular: it’s a recipe for building an RAA-style system, not a claim that today’s code already does it for AI.

- **DN/Weyl operator analogue:** a “boundary response operator” mapping boundary conditions (prompt constraints, retrieved facts, environment observations) to responses (logits, plans, actions).
  - Possible instantiations: Jacobians/Fisher blocks, implicit-layer response operators, constrained decoding operators, tool-augmented response maps.

- **Cayley transform analogue:** a stability-normalized update map.
  - In AI terms: proximal/trust-region/implicit updates that bound sensitivity and prevent near-singular steps.

- **Safe rectangles analogue:** safe operating regimes.
  - Criteria: avoid near-singular updates, enforce margins, detect blowup regions, restrict recursion to stable domains.

- **Riesz projector analogue:** canonical thought-channel extraction.
  - Goal: isolate the dominant reasoning/plan mode and prevent swaps (plan drift/goal drift).

---

## 5) Diagnostics (unit tests for recursive reasoning stability)

Borrowed directly from the RH harness pattern:

- **Gap stability:** principal reasoning mode remains separated (failure = mode swapping / plan drift).
- **Invertibility margin:** updates aren’t near-singular (failure = bottlenecks / blow-ups).
- **Winding/branch persistence proxies:** refinement doesn’t reintroduce contradictory branches (failure = ghost branches).
- **Normal-family behavior across depth:** boundedness + convergence of state trajectories (failure = explosion/oscillation).

---

## 6) Calibration tasks (to show it’s not just an analogy)

Mirror the repo’s falsification-first discipline:

1) **Toy deterministic planning** with known convergent refinement: verify non-expansion/Lipschitz control and a canonicalization/closure mechanism in a chosen metric.
2) **Synthetic multi-branch reasoning with planted contradiction:** verify admissibility eliminates the planted ghost branch.
3) **Tool-augmented reasoning loop:** verify refinement stabilizes rather than oscillates.

---

## 7) Honest constraint (the non-negotiable deliverable)

To claim RAA is more than a compelling analogy, one must provide:

- a meaningful semantic metric $d$ on internal belief/plan states,
- a proof (or tight empirical proxy with explicit assumptions) that refinement is non-expansive globally,
- and that admissibility yields **well-conditioned refinement** (Lipschitz control with explicit margins) plus a **canonicalization/closure mechanism** that prevents persistent branching (ghost modes).

MLC is the canonical warning: without the right metric/geometry control, one can “shrink the wrong thing” and still have ghosts.

---

## 8) Where this connects back to the RH work (without overclaim)

This repo already demonstrates the engineering discipline RAA needs:

- explicit admissibility and failure detectors,
- a frozen contract to prevent demo drift,
- and a clean separation between witnessed hypotheses and analytic obligations.

What remains for RH specifically is tracked in: [REMAINING_OBLIGATIONS.md](REMAINING_OBLIGATIONS.md)
