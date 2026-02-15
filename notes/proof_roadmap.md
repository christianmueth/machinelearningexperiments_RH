# Proof roadmap (Experiment E scaffold)

## Scope

This note is a **one-page navigation map** tying together:

- the frozen experimental protocol and artifacts (`RUN_CONTRACT.md`),
- what the numerics are allowed to claim (hypothesis witnesses, not proofs), and
- the analytic/representation-theoretic steps that turn those hypotheses into the desired identification.

It does **not** add new hypotheses beyond what is already spelled out in the other notes; it only orders them.

Numerical experiments are used solely to witness the hypotheses of the analytic statements (invertibility margins, spectral gaps, and divisor cleanliness on safe rectangles); no step of the analytic proof relies on numerical computation.

See also: `notes/methodology_numerics_as_witnesses.md` for a short reader-facing preface.

Structural analogue: Mandelbrot Local Connectivity (MLC) in the same “one-hinge lemma + closure” style, mapped onto this pipeline: `notes/mlc_hinge_lemma_and_connection.md`.

---

## What is being identified

The pipeline constructs, for each truncation level $N$ and spectral parameter $s$, a boundary DN map $\Lambda_N(s)$ and a Cayley/scattering object.
After symmetry reduction (Swap±) and branch-tracking, it extracts a scalar channel $\lambda_N(s)$.

The target comparison function is
\[
\phi_{mod}(s) = \frac{\xi(2s-1)}{\xi(2s)},
\]
and the central quotient is
\[
Q_N(s) := \frac{\lambda_N(s)}{\phi_{mod}(s)}.
\]

The proof skeleton aims to justify (in the $N\to\infty$ limit) that
\[
\lambda(s) \equiv \phi_{mod}(s)
\]
by showing $Q(s)\equiv 1$.

---

## Contract → audit table → lemmas (the “numerics are witnesses” principle)

1) **Protocol freeze**

- The driver produces a fixed set of artifacts for each run under `runs/`.
- The required minimum is frozen in `RUN_CONTRACT.md` so later claims cannot shift with harness changes.

2) **Audit mapping (numerics → hypotheses)**

- `notes/hypotheses_to_artifacts.md` is the bridge document.
- It states which hypotheses are *analytic obligations* and which are *witnessed in a run* by named CSV/JSON fields.

3) **Local safe-rectangle convergence (finite → limiting channel)**

- `notes/lemma_safe_rectangle_convergence.md` packages the computational regime into a lemma statement.
- It isolates four hypotheses (H1–H4) and shows how they imply:
  - stability of the Cayley/scattering object on a chosen rectangle $R$,
  - convergence of the symmetry-reduced channel eigenvalue $\lambda_N(s)\to\lambda(s)$ on $R$,
  - and eventual stability of the winding/argument-principle diagnostics for $Q_N$.

4) **Selector admissibility (boundary autopick is not “cheating”)**

- `notes/lemma_selector_admissibility.md` treats the boundary autopick as a deterministic finite selector under explicit constraints.
- It explains why “scan a small window, enforce winding=0 and gap≥γ, optimize a robust tail metric (plus guard)” is a legitimate *admissible choice rule* and not post-hoc tuning.

At this point, the numerics serve one job: **provide auditable evidence that H1/H3/H4 and the selector constraints are satisfiable and hold for the chosen run rectangles**.

---

## Two analytic tasks (non-numeric obligations)

Everything below is *not* proven by the harness; it is the remaining math.

### Task A: Define the limiting operator and its meromorphy

- `notes/operator_construction_and_meromorphy.md` is the operator-theory backbone.
- It must supply:
  - a clean definition of the limiting DN/Weyl family $\Lambda(s)$,
  - its meromorphic continuation (as appropriate for the geometric/scattering setting),
  - and the operator consistency statement H2 used in the safe-rectangle lemma.

### Task B: Identify the extracted scalar channel with the classical intertwiner

- `notes/eisenstein_intertwiner_identification.md` is the representation-theoretic mapping.
- It must justify:
  - which symmetry-reduced scalar channel corresponds to the classical Eisenstein scattering coefficient / intertwining operator,
  - the normalization convention (including the basepoint constant),
  - and that the functional-equation/involution constraint transfers to the scalar channel $\lambda(s)$.

These two tasks are the “source” of the analytic hypotheses that numerics cannot supply: meromorphy, growth control, and the correct conceptual identification of the channel.

---

## Hinge proposition (turn hypotheses into equality)

- `notes/proposition_identification_hinge.md` is the purely complex-analytic step.
- Once Tasks A/B provide:
  - meromorphy + growth control for $Q(s)$,
  - the involution/functional equation constraint on the critical line,
  - and a normalization $Q(s_0)=1$,
- and once the safe-rectangle lemma + winding witnesses ensure **divisor control** (no zeros/poles on safe rectangles),

then the hinge proposition forces $Q(s)\equiv 1$, hence $\lambda(s)\equiv \phi_{mod}(s)$.

---

## Suggested reading order

1) `RUN_CONTRACT.md`
2) `notes/hypotheses_to_artifacts.md`
3) `notes/lemma_safe_rectangle_convergence.md`
4) `notes/lemma_selector_admissibility.md`
5) `notes/operator_construction_and_meromorphy.md`
6) `notes/eisenstein_intertwiner_identification.md`
7) `notes/proposition_identification_hinge.md`
