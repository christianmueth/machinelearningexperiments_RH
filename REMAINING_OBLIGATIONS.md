# Remaining obligations (proof delta)

This page is the one-screen checklist of what is still missing for a full RH proof *in the repository’s framework*, and which items are currently:

- **Numeric witness (per run)** — logged evidence that a hypothesis holds on chosen safe rectangles (not a proof).
- **Analytic theorem (needed)** — must be discharged by standard operator/representation theory (no numerics).
- **Formally complete** — already a stated proposition/lemma in the notes (still requires hypotheses).

It is intentionally link-forward: each obligation points to the note where it is stated/used.

---

## 0) The “hinge” structure (what closes the argument)

- **Hinge proposition (formal closure):** [notes/proposition_identification_hinge.md](notes/proposition_identification_hinge.md)
  - Status: **Formally complete** as a proposition.
  - Remaining work is verifying its hypotheses (Items 1–4 below).

---

## 1) Task A: limiting operator/DN/scattering object (analytic)

Primary note: [notes/operator_construction_and_meromorphy.md](notes/operator_construction_and_meromorphy.md)

- **A1. Specify the limiting geometric operator model precisely** (surface/cusp cut, domains, and spectral parameter convention).
  - Status: **Write-up now explicit** (still needs final citation lock-in).
- **A2. Define the limiting DN/Weyl family \(\Lambda(s)\)** on a fixed boundary Hilbert space and prove meromorphic continuation.
  - Status: **Write-up now explicit** (meromorphy is a cited standard theorem; final citation choice still required).
- **A3. Prove the operator functional equation / adjoint symmetry** \(\Lambda(1-s)=D_b\Lambda(s)^*D_b^{-1}\) and the induced involution for \(S_\eta(s)\).
  - Status: **Citable in chosen presentation** (proved at the geometric scattering level; transfers to the scalar channel once Task B identifies the channel normalization).
- **A4. Prove a growth bound sufficient for a Phragmén–Lindelöf step** (or an equivalent rigidity route).
  - Status: **Citable in chosen presentation** (finite-order/strip growth via scattering determinant / Selberg zeta; transfers to the scalar channel once Task B identifies the channel normalization).

---

## 2) Finite-\(N\) \(\to\) limiting consistency and safe-rectangle convergence (mixed)

Primary note: [notes/lemma_safe_rectangle_convergence.md](notes/lemma_safe_rectangle_convergence.md)

This lemma is the “admissibility engine” for the pipeline; it turns safe-rectangle hypotheses into limiting holomorphy/nonvanishing statements on those rectangles.

- **C1/C2/C3 safe-rectangle convergence lemma** (statement + proof skeleton).
  - Status: **Formally complete** as a lemma.

Its hypotheses split as follows (see audit mapping): [notes/hypotheses_to_artifacts.md](notes/hypotheses_to_artifacts.md)

- **H1 (invertibility margin for \(\Lambda_N+i\eta I\) on a chosen rectangle)**
  - Status: **Numeric witness (per run)** via run artifacts.
- **H3 (spectral gap isolating the target channel; no swapping)**
  - Status: **Numeric witness (per run)** via run artifacts.
- **H4 (no modular divisor on \(\partial R\))**
  - Status: **Formally complete (simple complex analysis)** once rectangle boundaries are chosen to avoid the discrete divisor set.
  - Note: [notes/lemma_modular_divisor_free_boundary.md](notes/lemma_modular_divisor_free_boundary.md)
- **Additional “divisor-clean boundary for \(Q_N\)” / winding=0 on \(\partial R\)** (the operational version of divisor control)
  - Status: **Numeric witness (per run)** via winding/argument-principle artifacts.
- **H2 (operator consistency: \(\Lambda_N\to\Lambda\) uniformly on \(R\))**
  - Status: **Analytic theorem (needed)**.
  - Mechanism note: [notes/lemma_truncation_to_dnmap_convergence.md](notes/lemma_truncation_to_dnmap_convergence.md)
  - Instantiated statement (matching Experiment E objects): [notes/h2_operator_consistency_instantiated.md](notes/h2_operator_consistency_instantiated.md)
  - Completion blueprint (what remains to actually prove for `build_A`/finite sections): [notes/h2_completion_blueprint.md](notes/h2_completion_blueprint.md)
  - Clarification: fully discharging H2 means proving the rectangle-uniform DN/Weyl convergence theorem (finite→limit). It is a key bridge, but RH still also requires Task A/B (limiting meromorphy/FE/growth + channel identification).
  - Geometric Option-B spec (treat current Hecke construction as a Galerkin scheme): [notes/geometric_discretization_spec_option_B.md](notes/geometric_discretization_spec_option_B.md)
  - **Core missing bridge theorem (packages B1–B4 as one statement):** [notes/bridge_theorem_option_B_minimal.md](notes/bridge_theorem_option_B_minimal.md)

Run protocol freezing (what is logged, when, and with which field names): [RUN_CONTRACT.md](RUN_CONTRACT.md)

---

## 3) Task B: identify the extracted channel with modular scattering (analytic, with a clear checklist)

Primary note: [notes/eisenstein_intertwiner_identification.md](notes/eisenstein_intertwiner_identification.md)

- **Mapping theorem “DN/Weyl \(\to\) scattering; constant Fourier mode equals \(\phi(s)\)”**
  - Status: **Analytic theorem (needed)** (the note is citation-forward, but the model-specific hypotheses must be verified in the exact operator setup).

Referee checklist (explicit): see §6 (H1–H8) in [notes/eisenstein_intertwiner_identification.md](notes/eisenstein_intertwiner_identification.md)

A useful status summary of H1–H8:

- **H1–H5 (operator model, boundary triple/Weyl function match, and scattering formula):** **Analytic theorem (needed)**.
- **H6 (normalization mismatch is only an \(s\)-independent scalar):** **Analytic theorem (needed)** (then fixed by one basepoint normalization).
- **H7 (Fourier-mode invariance; constant mode is the cusp channel):** **Analytic theorem (needed)** (standard in cusp scattering, but must be pinned to the exact boundary identification).
- **H8 (canonicity via gap + Riesz projector on safe rectangles):** **Mixed**
  - analytic mechanism is standard;
  - the *gap premise* is currently a **Numeric witness (per run)**.

---

## 4) Quotient rigidity closure: what must feed the hinge

Primary note: [notes/proposition_identification_hinge.md](notes/proposition_identification_hinge.md)

To apply the hinge proposition to \(Q(s)=\lambda(s)/\phi_{mod}(s)\), you need:

- **(i) Meromorphic continuation of \(Q\)**
  - Source obligation: Task A + Task B.
  - Status: **Analytic theorem (needed)**.
- **(ii) Divisor control on an exhaustion by safe rectangles (winding=0; no zeros/poles)**
  - Source obligation: safe-rectangle lemma + safe-rectangle witnesses.
  - Status: **Mixed**
    - “Upgrade finite-\(N\) winding stability \(\to\) limiting divisor control” is **formal** once H1–H3/H2 hold;
    - the premises (margin/gap/winding=0) are currently **Numeric witnesses (per run)**.
- **(iii) Functional equation / involution constraint on the critical line**
  - Source obligation: Task A + channel extraction compatibility.
  - Status: **Analytic theorem (needed)**.
- **(iv) Growth bound in strips (or equivalent rigidity argument)**
  - Source obligation: Task A/B.
  - Status: **Analytic theorem (needed)**.
- **(v) One basepoint normalization \(Q(s_0)=1\)**
  - Status: **Axiomatically easy** once the identification is fixed (and already mirrored operationally by how the numerics normalize).

---

## 5) What this implies about “code completeness”

### A recurring point of confusion: “we already did analytic continuation numerically”

The harness *does* evaluate complex-parameter objects off the real axis (and runs branch tracking, winding/argument diagnostics, etc.). That is best understood as **numerical exploration of finite-\(N\) surrogates** and **witnessing** that certain hypotheses look stable on chosen safe rectangles.

However, the missing proof steps are not just “more samples.” They are theorem-grade upgrades of the form:

- (**Existence + meromorphy for the limiting object**) define the limiting DN/Weyl family \(\Lambda(s)\) on a fixed Hilbert boundary space and prove its meromorphic continuation + adjoint/functional-equation symmetry (Task A).
- (**Uniform finite-to-limit control**) prove \(\Lambda_N\to\Lambda\) on safe rectangles in a norm strong enough to pass through the Cayley transform and the channel extraction (H2 / Task A).
- (**Semantic identification**) prove the extracted scalar channel is the classical modular cusp scattering coefficient \(\phi(s)=\xi(2s-1)/\xi(2s)\) (Task B).
- (**Growth/finite-order bounds**) supply whatever strip-growth control is used by the hinge proposition (Task A/B).

Numerics can still be extremely valuable for falsification and for mapping admissible regions (where margins/gaps/winding behave), but they do not by themselves certify these global/uniform analytic statements.

- The harness is already structured as a **witness generator** with a frozen artifact contract: [RUN_CONTRACT.md](RUN_CONTRACT.md)
- The proof delta is dominated by **analytic identification** (Tasks A/B) and **hypothesis discharge**, not by new algorithms.

Practical impact on code if the proof advances:

- **Likely no impact:** proving Task A/B as written (same DN/Cayley/scattering conventions, same channel definition) mostly adds citations/theorems and turns currently “assumed” hypotheses into proven ones.
- **Potential impact:** if the analytic identification forces a different normalization, a different boundary pairing, or a different canonical channel criterion, then the extraction/normalization logic could need adjustment (but that would be driven by the math, not by “better numerics”).
