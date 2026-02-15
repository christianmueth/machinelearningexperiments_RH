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

### 4) Modular target and quotient

Let
\[
\phi_{mod}(s) := \frac{\xi(2s-1)}{\xi(2s)}.
\]

Define the scalar quotient
\[
Q_N(s) := \frac{\lambda_N(s)}{\phi_{mod}(s)}.
\]

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

### H3 (channel isolation: spectral gap lower bound)

There exists $\gamma>0$ and $N_0$ such that for all $N\ge N_0$ and all $s\in R$, the distinguished eigenvalue $\lambda_N(s)$ of $B_{N,\pm}(s)$ is isolated by a gap of size at least $\gamma$ from the rest of the spectrum of $B_{N,\pm}(s)$.

**Harness correspondence.** This is the role of the logged gap statistics (e.g. `gap_phi_min` and related “gap” fields in the convergence sweep), and the per-candidate `gap_ok` filtering in autopick.

### H4 (no modular divisor on the boundary)

The function $\phi_{mod}(s)$ has no zeros or poles on $\partial R$.

(Equivalently: $\inf_{s\in \partial R}|\phi_{mod}(s)|>0$.)

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

C2 gives uniform convergence of $\lambda_N$ on $\partial R$. H4 gives a positive lower bound for $|\phi_{mod}|$ on $\partial R$, hence uniform convergence of $Q_N$ on $\partial R$.

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
