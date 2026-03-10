# Proposition: Identification Hinge (quotient rigidity)

## Purpose

This is the single complex-analytic proposition that turns “operator-level match + one normalization” into full identification.

It is intentionally classical and short: the work in the project is *verifying the hypotheses* using the safe-rectangle convergence lemma and the operator/representation-theoretic identification.

Numerical experiments are used solely to witness the hypotheses of the analytic statements (invertibility margins, spectral gaps, and divisor cleanliness on safe rectangles); no step of the analytic proof relies on numerical computation.

---

## Setup

Let $\lambda(s)$ be the limiting scalar channel (Swap± eigenchannel) extracted from the limiting scattering/Cayley object.

This proposition is intentionally **target-agnostic**: it applies to any explicitly specified reference scalar $\phi_{ref}(s)$ that you aim to identify $\lambda(s)$ with.

Two references appear in the two-tier plan:

- **Tier‑1 (uncompleted / finite prime set):**
   \[
   \phi_P(s) := \prod_{p\in P}\frac{1-p^{-2s}}{1-p^{-(2s-1)}}.
   \]
- **Tier‑2 (completed modular target):**
   \[
   \phi_{mod}(s) := \frac{\xi(2s-1)}{\xi(2s)}.
   \]

Define the quotient
\[
Q_{ref}(s) := \frac{\lambda(s)}{\phi_{ref}(s)}.
\]

In Tier‑1 you set $\phi_{ref}=\phi_P$ and prove identification in the **uncompleted normalization**.
In Tier‑2 you set $\phi_{ref}=\phi_{mod}$, but only *after* you have proved the required normalization/dictionary lemma (see below).

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

Since $|\phi_{mod}(1/2+it)|=1$ on $\Re(s)=1/2$, the same constraint transfers to $Q_{mod}$ when $\phi_{ref}=\phi_{mod}$:
\[
Q_{mod}(1-s)\;\overline{Q_{mod}(s)} = 1\quad\text{on }\Re(s)=\tfrac12.
\]
In particular,
\[
|Q_{mod}(\tfrac12+it)|=1\quad\forall t\in\mathbb{R}.
\]

For Tier‑1 ($\phi_{ref}=\phi_P$), the corresponding “unitarity on the critical line” statement for $Q_P$ depends on the *uncompleted* normalization implemented by the pipeline (and is the correct Tier‑1 object to feed into this hinge).

---

## Proposition (rigidity of the quotient)

**Proposition (Identification hinge).**
Suppose $Q_{ref}$ satisfies:

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

### Tier‑2 remaining obligation (dictionary/normalization map; not “more sweeps”)

Empirically, Experiment‑E exhibits a stable analytic channel but rejects “multiply the reference by $g_\infty$” as a valid identification move for the computed $\lambda$ (it produces $Q\approx 1/g_\infty$ instead of $Q\approx 1$). This pins the real Tier‑2 burden:

**Normalization-map lemma (Tier‑2).** Construct (and justify) an operator-level normalization map $\mathcal{N}$ acting on the scattering/channel extraction so that the Tier‑2 scalar is obtained by
\[
\lambda_{mod}(s) := (\mathcal{N}\lambda)(s) = c\,g_\infty(s)\,\lambda(s)
\]
(or an equivalent conjugation/renormalization statement), with $c$ fixed by a basepoint.

Only after this lemma is proved/implemented should one apply the hinge proposition with $\phi_{ref}=\phi_{mod}$.

---

## Tier‑2 normalization-map lemma: numerical witnesses (Step‑1/2/3)

This section records *witnesses* for the Tier‑2 normalization-map lemma (and for rejecting common foils).
All numbers below are taken from the convergence-only diagnostics produced by Experiment‑E.

The consolidated table for the current run set is:

- `runs/_summary_20260216_expE.csv`

where the key columns are:

- `logabs_corr`: correlation of $\log|Q|$ with $\log|g_\infty|$ (detects the $Q\sim 1/g_\infty$ signature)
- `abs_absQ_minus1_median`: median $||Q|-1|$ (magnitude-only)
- `abs_argQ_median`: median $|\arg(Q)|$ (phase diagnostic; distinguishes full complex vs magnitude-only fixes)
- `wind_maxabs`: max absolute boundary winding (divisor cleanliness witness)

New (Task‑1 / simulation‑first) columns:

- `fp_*`: the same magnitude/phase witnesses but computed on a dense critical-line scan ($\sigma=1/2$, $t\in[t_{min},t_{max}]$). These are produced when `do_critical_line_fingerprint: true`.

### Step‑1: ablation — “not a branch-selection artifact”

Freeze *selection* and toggle only the operator-side normalization.

Example (selection frozen to uncompleted Euler-partial, target completed):

- `20260216_005828__expE_doc_validation_track` (no normalization): `logabs_corr≈-0.965`, `abs_argQ_median≈1.64`, `abs_absQ_minus1_median≈2.51`.
- `20260216_010206__expE_doc_validation_track` (apply $g_\infty$ on observable): `logabs_corr≈+0.499`, `abs_argQ_median≈0.039`, `abs_absQ_minus1_median≈0.047`.

Since `phi_select_mode` is held fixed, the removal of the $Q\sim 1/g_\infty$ signature cannot be explained by branch switching driven by the reference.

Important nuance: in this frozen-selection regime the winding witness becomes nontrivial for larger $N$ (e.g. `wind_maxabs≈9`), indicating that “pointwise agreement” and “single analytic branch on $\partial R$” can decouple when the selector is not the divisor-clean modular selector.

### Step‑2: stress test across $\eta$ (robustness)

With modular selection and completed target, the same operator-side normalization improves both magnitude and phase and remains divisor-clean across $\eta\in\{1,10\}$:

- `20260216_012525__expE_doc_validation_track` (eta=1, norm=none): `logabs_corr≈-0.794`, `abs_argQ_median≈1.70`.
- `20260216_015000__expE_doc_validation_track` (eta=1, norm=ginf_multiply): `logabs_corr≈+0.688`, `abs_argQ_median≈0.258`.
- `20260216_021345__expE_doc_validation_track` (eta=10, norm=none): `logabs_corr≈-0.722`, `abs_argQ_median≈1.73`.
- `20260216_022235__expE_doc_validation_track` (eta=10, norm=ginf_multiply): `logabs_corr≈+0.642`, `abs_argQ_median≈0.249`.

In all these modular-selection runs, `wind_maxabs` stays at numerical zero-scale (≈1e‑15), supporting divisor cleanliness on the chosen safe rectangle.

### Step‑2b: stress test across rectangles (robustness beyond the main safe rectangle)

We additionally tested a narrow rectangle around $t\approx 3.5$:

- $R_{narrow}=(0.55,1.06)\times(3.3855,3.6815)$.

Across $\eta\in\{1,3,10\}$, the same pattern holds: without the operator-side normalization the phase diagnostic stays large (order 1–2 radians) and magnitude agreement is poor (median $||Q|-1|\sim 2$), while `ginf_multiply` reduces both magnitude and phase errors.

- eta=1:
   - `20260216_035203__expE_doc_validation_track` (norm=none): `abs_absQ_minus1_median≈2.04`, `abs_argQ_median≈1.72`, `wind_maxabs≈5e‑16`.
   - `20260216_035519__expE_doc_validation_track` (norm=ginf_multiply): `abs_absQ_minus1_median≈0.140`, `abs_argQ_median≈0.286`, `wind_maxabs≈5e‑16`.
- eta=3:
   - `20260216_034840__expE_doc_validation_track` (norm=none): `abs_absQ_minus1_median≈2.18`, `abs_argQ_median≈1.79`.
   - `20260216_034541__expE_doc_validation_track` (norm=ginf_multiply): `abs_absQ_minus1_median≈0.242`, `abs_argQ_median≈0.319`.

   Nuance: for eta=3, `wind_maxabs≈1` is driven by the smallest truncation (`N=48`) only; the winding drops back to numerical zero for `N≥64` in the same run’s winding table.

- eta=10:
   - `20260216_035743__expE_doc_validation_track` (norm=none): `abs_absQ_minus1_median≈2.14`, `abs_argQ_median≈1.67`, `wind_maxabs≈7e‑16`.
   - `20260216_035957__expE_doc_validation_track` (norm=ginf_multiply): `abs_absQ_minus1_median≈0.200`, `abs_argQ_median≈0.236`, `wind_maxabs≈7e‑16`.

In this narrow rectangle, `logabs_corr` is not as discriminative as on the main wide safe rectangle (the correlation can be positive even when magnitude/phase agreement fails), so the phase + magnitude diagnostics are the primary witnesses.

### Step‑3: foils (specificity)

The “full complex $g_\infty(s)$” normalization is not replaceable by smooth magnitude-only or constant factors:

- **Magnitude-only foil fails phase**:
   - `20260216_022919__expE_doc_validation_track` (norm=`ginf_abs_multiply`) gives good magnitude stats but `abs_argQ_median≈1.71` (phase mismatch persists).
- **Constant-in-$s$ foil fails**:
   - `20260216_023447__expE_doc_validation_track` (norm=`ginf_const_multiply`, sref=(1.10,3.50)) keeps `logabs_corr≈-0.747` and `abs_argQ_median≈1.68`.
- **Inverse factor recreates the signature**:
   - `20260216_023917__expE_doc_validation_track` (norm=`ginf_divide`) yields extreme mismatch (`logabs_corr≈-0.949`, magnitude error explodes).

These foils are designed so that the only successful move is the targeted complex archimedean completion factor.

### Task‑1 addendum: critical-line fingerprinting + subfactor comparison

To support the “simulation-first” Tier‑2 contract (cheap, dense, falsifiable critical-line fingerprints), Experiment‑E now emits two additional artifacts (when enabled):

- `E_Qlambda_critical_line_fingerprint_points.csv` — per‑$t$ points on $s=\tfrac12+it$ (includes gap + selection relerr + magnitude/phase witnesses).
- `E_Qlambda_critical_line_fingerprint_summary.csv` — per‑$N$ summary stats over the critical-line grid.

Additionally, it can emit a single-run “candidate table” comparing the key archimedean factors on the same point set:

- `E_Qlambda_critical_line_candidates_summary.csv` — per‑$N$, per‑candidate summaries for candidates in `{none, gamma_only, pi_only, ginf_multiply, ginf_abs_multiply}`.

Example run (eta=3, modular selector, completed target, candidate table enabled):

- `20260216_172033__expE_doc_validation_track`

This is intended to fingerprint the *minimal* archimedean correction needed: on the critical line, the $\pi^{1/2-s}$ factor has unit modulus, so magnitude-only witnesses are primarily sensitive to the $\Gamma(s-1/2)/\Gamma(s)$ piece, while phase witnesses distinguish full complex $g_\infty$ from magnitude-only foils.

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
