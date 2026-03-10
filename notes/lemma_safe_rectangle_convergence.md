# Lemma: Safe-Rectangle Convergence (finite DN/Cayley → limiting scattering channel)

## Purpose

This lemma packages the exact regime that the harness already enforces (dip-safe rectangle selection + invertibility margin + Swap± gap health + winding stability) into a theorem-shaped statement.

Numerical experiments are used solely to witness the hypotheses of the analytic statements (invertibility margins, spectral gaps, and divisor cleanliness on safe rectangles); no step of the analytic proof relies on numerical computation.

It is deliberately **local-in-$s$** (one rectangle $R$) and **local-in-$N$** (eventually in $N$). No global continuation is attempted here.

---

## Minimal definitions

### 1) Finite boundary/DN operators

For each truncation level $N$ and spectral parameter $s \in \mathbb{C}$, let

- $\Lambda_N(s)$ denote the **finite Dirichlet-to-Neumann (DN) map** on the boundary space $\mathcal{H}_\partial^{(N)}$ produced by the bulk-to-boundary Schur complement used in the code.

(Concretely, it is the boundary Schur complement after eliminating the interior degrees of freedom.)

### 2) Cayley/scattering map at impedance $\eta$

Fix $\eta>0$. Define the Cayley transform of a boundary operator by

\[
\mathcal{C}_\eta(\Lambda) := (\Lambda - i\eta I)\,(\Lambda + i\eta I)^{-1}.
\]

Define the finite scattering matrix

\[
S_{N,\eta}(s) := \mathcal{C}_\eta(\Lambda_N(s)).
\]

**Two equivalent normalizations used in the harness.**
The analysis can be carried out using either:

- the raw Cayley scattering $S_{N,\eta}(s)$ defined above, or
- the free-normalized relative scattering $S_{rel,N}(s)=S_{model,N}(s)\,S_{free,N}(s)^{-1}$.

The harness’s “incoming basis” channel matrix is a fixed (invertible) conjugation of the chosen scattering normalization by the Cayley denominator $(\Lambda_N(s)+i\eta I)$ (so it represents the same analytic object away from Cayley bottlenecks). This lemma treats the resulting channel matrix abstractly as $M_N(s)$.

### 3) Swap± reduction and channel eigenvalue

Let $\Pi_{\pm}$ denote the fixed “Swap±” symmetry projection (the $\pm$ eigenspaces of the Swap involution on the two-channel boundary space).

Let $B_{N,\pm}(s)$ denote the projected block used to define the scalar channel, i.e.

\[
B_{N,\pm}(s) := \Pi_{\pm}\,M_N(s)\,\Pi_{\pm}\big|_{\mathrm{ran}(\Pi_{\pm})},
\]

where $M_N(s)$ is the channel matrix built from the chosen scattering normalization (incoming-basis or rel-basis, per the harness).

Let $\lambda_N(s)$ denote the **distinguished channel eigenvalue** of $B_{N,\pm}(s)$ tracked by the branch-continuation rule used in the winding check.

### 4) Reference target and quotient (Tier‑1 vs Tier‑2)

This lemma is meant to support the **two-tier completion plan**:

- **Tier‑1 (unconditional for the computed object):** fix a finite prime set $P$ and use an **uncompleted / truncated** reference target.
- **Tier‑2 (modular identification):** upgrade the reference target to the fully completed modular scattering coefficient via explicit dictionary lemmas.

Accordingly, we define a *parameterized* reference target $\phi_{ref}(s)$ and form the quotient against it.

**Tier‑2 (completed modular target).**
\[
\phi_{mod}(s) := \frac{\xi(2s-1)}{\xi(2s)}.
\]

**Tier‑1 (finite-prime Euler-product target).** For a fixed finite prime set $P$,
\[
\phi_{P}(s) := \prod_{p\in P}\frac{1-p^{-2s}}{1-p^{-(2s-1)}}.
\]
(Optionally one may also consider a partially completed variant $g_\infty(s)\,\phi_P(s)$, but Tier‑1 does not assume any such dictionary.)

**Empirical fork (record once; do not revisit).** For the **operator/scattering normalization implemented by Experiment‑E**, the archimedean completion factor is empirically rejected as a target multiplier.

On the verified safe rectangle $R=[0.8,1.4]\times i[0.5,6.5]$ (and using the branch-control split `phi_select_mode` vs `phi_target_mode`), replacing the Tier‑1 reference $\phi_P$ by $g_\infty\phi_P$:

- preserves divisor-cleanliness on $\partial R$ (winding stays machine-zero with 0 failures), but
- *destroys* interior value agreement to the reference (errors jump by an order of magnitude and `phi_wins_frac` collapses to 0).

Witness runs/artifacts (same rectangle, $N\in\{48,64,96,128\}$):

- Uncompleted reference ($\phi_{ref}=\phi_P$): [runs/20260215_222622__expE_doc_validation_track/E_Qlambda_convergence_sweep.csv](runs/20260215_222622__expE_doc_validation_track/E_Qlambda_convergence_sweep.csv) and [runs/20260215_222622__expE_doc_validation_track/E_Qlambda_convergence_magphase.csv](runs/20260215_222622__expE_doc_validation_track/E_Qlambda_convergence_magphase.csv)
- Partially completed reference ($\phi_{ref}=g_\infty\phi_P$): [runs/20260215_223000__expE_doc_validation_track/E_Qlambda_convergence_sweep.csv](runs/20260215_223000__expE_doc_validation_track/E_Qlambda_convergence_sweep.csv) and [runs/20260215_223000__expE_doc_validation_track/E_Qlambda_convergence_magphase.csv](runs/20260215_223000__expE_doc_validation_track/E_Qlambda_convergence_magphase.csv)

Moreover, the mag/phase diagnostic shows the mismatch is exactly archimedean: in the completed-reference run, the quotient satisfies $Q(s)\approx 1/g_\infty(s)$ (equivalently $Q(s)\,g_\infty(s)\approx 1$), rather than $Q(s)\approx 1$.

Therefore: **the computed Tier‑1 observable is in an uncompleted normalization**. Any appearance of $g_\infty$ belongs to Tier‑2 dictionary/normalization lemmas (operator-level change), not to Tier‑1 target multiplication.

**Positive Tier‑2 clue (scalar proxy for an operator-level normalization).**
If one keeps the *completed* reference $\phi_{ref}=g_\infty\phi_P$ fixed but applies the candidate scalar normalization map
\[
\lambda_N^{(norm)}(s):=g_\infty(s)\,\lambda_N(s)
\]
before forming the quotient $Q_N=\lambda_N^{(norm)}/\phi_{ref}$, then the completed-reference quotient statistics collapse back to the uncompleted-reference behavior (as expected from cancellation): the huge $|Q|$ blow-up and strong $\log|Q|$–$\log|g_\infty|$ anticorrelation seen in the completed-target run disappear.

Witness run (same rectangle and $N$ set):

- Completed reference with scalar normalization ($\phi_{ref}=g_\infty\phi_P$ and `lambda_norm_mode: ginf_multiply`): [runs/20260215_234741__expE_doc_validation_track/E_summary.md](runs/20260215_234741__expE_doc_validation_track/E_summary.md), [runs/20260215_234741__expE_doc_validation_track/E_Qlambda_convergence_magphase.csv](runs/20260215_234741__expE_doc_validation_track/E_Qlambda_convergence_magphase.csv)

This does not by itself identify the modular target, but it does empirically validate a very specific Tier‑2 dictionary candidate: **the archimedean factor enters on the observable/operator side (normalization of the channel scalar), not as a post-hoc multiplier of the reference target alone**.

**Reference target.** In a given run/lemma instance, choose
\[
\phi_{ref}(s)\in\{\phi_{mod}(s),\;\phi_P(s)\}\quad\text{(or another explicitly specified reference)}.
\]

Define the scalar quotient
\[
Q_N(s) := \frac{\lambda_N(s)}{\phi_{ref}(s)}.
\]

---

## Tier‑2 normalization dictionary (stub; not used in Tier‑1)

Tier‑2 is the place where one states (and proves) **operator/scattering normalization** lemmas. Empirically, “multiplying the target by $g_\infty$” is *not* equivalent to the normalization implemented by the current pipeline.

Concrete knobs that can change the normalization at the operator level (examples; to be made precise in Tier‑2):

- boundary pairing / inner product on $\mathcal{H}_\partial$ (cusp measure / weighting)
- incoming/outgoing convention and the exact Cayley impedance choice
- free-normalization choice (what is divided out at the matrix level)
- constant-term normalization of the Eisenstein/incoming basis (where the archimedean factor can enter)


---

## Hypotheses (match harness quantities)

Let $R \subset \mathbb{C}$ be a closed axis-aligned rectangle, with boundary $\partial R$.

### H1 (invertibility margin: Cayley denominator stays away from singularity)

There exist $\delta>0$ and $N_0$ such that for all $N\ge N_0$,

\[
\inf_{s\in R}\; s_{\min}(\Lambda_N(s)+ i\eta I)\;\ge\;\delta.
\]

**Harness correspondence.** This is exactly the “dip-safe rectangle” condition:
- empirically chosen using dip-atlas / pole-line diagnostics;
- directly audited by the convergence winding artifact (failures) and any optional line scans.

### H2 (uniform operator consistency on the safe rectangle)

There exists a limiting boundary operator family $\Lambda(s)$ on a limiting boundary Hilbert space $\mathcal{H}_\partial$ such that

\[
\sup_{s\in R}\;\|\Lambda_N(s)-\Lambda(s)\|\;\to\;0\quad\text{as }N\to\infty.
\]

(Here “$\|\cdot\|$” is an operator norm in a fixed identification of boundary spaces; the exact topology is part of the later operator construction.)

Mechanism note (bulk truncation convergence \(\Rightarrow\) DN/Weyl convergence):
- [notes/lemma_truncation_to_dnmap_convergence.md](notes/lemma_truncation_to_dnmap_convergence.md)

### H3 (channel isolation: spectral gap lower bound)

There exists $\gamma>0$ and $N_0$ such that for all $N\ge N_0$ and all $s\in R$, the distinguished eigenvalue $\lambda_N(s)$ of $B_{N,\pm}(s)$ is isolated by a gap of size at least $\gamma$ from the rest of the spectrum of $B_{N,\pm}(s)$.

**Harness correspondence.** This is the role of the logged gap statistics (e.g. `gap_phi_min` and related “gap” fields in the convergence sweep), and the per-candidate `gap_ok` filtering in autopick.

### H4 (no reference-target divisor on the boundary)

The function $\phi_{ref}(s)$ has no zeros or poles on $\partial R$.

(Equivalently: $\inf_{s\in \partial R}|\phi_{ref}(s)|>0$.)

Notes:
- For **Tier‑2** ($\phi_{ref}=\phi_{mod}$), existence of divisor-free boundaries can be discharged analytically:
  - [notes/lemma_modular_divisor_free_boundary.md](notes/lemma_modular_divisor_free_boundary.md)
- For **Tier‑1** ($\phi_{ref}=\phi_P$), $\phi_P$ is a finite product of elementary factors, so divisor-free rectangles are obtained by simply avoiding its explicit pole/zero loci.

---

## Conclusions

### C1 (uniform convergence of Cayley/scattering on the rectangle)

Define the limiting scattering object
\[
S_{\eta}(s) := \mathcal{C}_\eta(\Lambda(s))
\]
on $R$.

Then under H1–H2,
\[
\sup_{s\in R}\;\|S_{N,\eta}(s)-S_{\eta}(s)\| \to 0.
\]

### C2 (convergence of the channel spectral projector and eigenvalue)

Under H1–H3, the Riesz projector onto the isolated channel eigenspace converges uniformly on $R$, and in particular
\[
\sup_{s\in R}\;|\lambda_N(s)-\lambda(s)|\to 0
\]
for a limiting meromorphic (in fact holomorphic on $R$) function $\lambda(s)$.

### C3 (eventual winding stability for the quotient)

Under H1–H4 and C2, the winding number of $Q_N$ along $\partial R$ is eventually constant in $N$.

In particular, if the harness observes machine-zero winding for all sufficiently large $N$ on $\partial R$, then $Q_N$ is eventually holomorphic and nonvanishing on $R$ (i.e. no divisor events occur in the limit on that safe rectangle).

---

## Proof skeleton (standard, intentionally short)

### Step 1: resolvent identity for the Cayley denominator

Write
\[
(\Lambda+i\eta I)^{-1}-(\Lambda_N+i\eta I)^{-1}
= (\Lambda+i\eta I)^{-1}(\Lambda_N-\Lambda)(\Lambda_N+i\eta I)^{-1}.
\]

H1 implies a uniform bound on the resolvent norms on $R$:
\[
\|(\Lambda_N+i\eta I)^{-1}\|\le \delta^{-1},\qquad \|(\Lambda+i\eta I)^{-1}\|\le \delta^{-1}
\]
(after $N$ is large enough).

Combine with H2 to get uniform resolvent convergence on $R$.

### Step 2: Cayley map is Lipschitz under the margin

Use the algebraic identity
\[
\mathcal{C}_\eta(\Lambda)-\mathcal{C}_\eta(\Lambda_N)
= (\Lambda-i\eta I)(\Lambda+i\eta I)^{-1}-(\Lambda_N-i\eta I)(\Lambda_N+i\eta I)^{-1}
\]
plus resolvent convergence from Step 1 to conclude C1.

### Step 3: isolated eigenvalue continuity via Riesz projectors

Fix a contour $\Gamma$ enclosing $\lambda_N(s)$ and separating it from the rest of the spectrum by margin $\gamma$ uniformly on $R$ (H3). Define the Riesz projector
\[
P_N(s)=\frac{1}{2\pi i}\int_{\Gamma} (zI-B_{N,\pm}(s))^{-1}\,dz.
\]

Uniform convergence of $B_{N,\pm}(s)$ (a consequence of C1 and the fixed finite-rank projections) implies uniform convergence of the resolvents on $\Gamma$, hence $P_N(s)\to P(s)$ and the corresponding eigenvalue converges uniformly, yielding C2.

### Step 4: winding stability

C2 gives uniform convergence of $\lambda_N$ on $\partial R$. H4 gives a positive lower bound for $|\phi_{ref}|$ on $\partial R$, hence uniform convergence of $Q_N$ on $\partial R$.

If additionally $Q_N$ is nonvanishing on $\partial R$ for all large $N$, then the argument principle implies the winding of $Q_N(\partial R)$ stabilizes (and equals the divisor count inside $R$).

---

## Where each hypothesis is audited in artifacts

- H1: “dip-safe” selection + failure-free winding evaluation on $\partial R$;
  - `E_Qlambda_convergence_winding.csv` (or the configured `convergence_wind_out_csv`) logs `failures` and `n_edge`.
- H3: gap health
  - `E_Qlambda_convergence_sweep.csv` logs gap statistics per channel (e.g. `gap_phi_min`).
- Winding stability conclusion
  - `E_Qlambda_convergence_winding.csv` logs `winding` per $N$.

See also: `notes/hypotheses_to_artifacts.md` for a compact audit table.
