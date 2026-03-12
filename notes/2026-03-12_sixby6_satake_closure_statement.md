# Six-by-six → Satake closure (theorem-shaped statement)

**Informal statement (internal)**

Fix the six-by-six local construction (boundary choice + Schur sign + Cayley/scattering conventions), and consider the mixed-sign SL(2) deformation family

$$X_{p,k}(u) = U(u)\,D_{p,k}\,L(-u),$$

with determinant-1 diagonal normalization $D_{p,k}=\mathrm{diag}(p^{k/2},p^{-k/2})$.

For each prime $p$ (with $k=1$), the intrinsic six-by-six pipeline produces a $2\times 2$ unitary scattering matrix $S_p(u)$ with eigenvalues $(\alpha_p(u),\beta_p(u))$.

Define the **exact Satake table** at deformation parameter $u$ by

$$p \mapsto (\alpha_p(u),\beta_p(u)).$$

Build a **global** Fredholm determinant from the prime packets via

$$D_u(s)=\det\bigl(I-K_u(s)\bigr),\qquad K_u(s)=\mathrm{diag}\bigl(e^{-s\log p}\,S_p(u)\bigr)_{p\in\mathcal P}.$$

Then, for the tested prime sets and $s$-grids, the global objects computed in two ways agree up to machine precision:

1) **Intrinsic**: compute $S_p(u)$ from the six-by-six construction and then assemble $K_u(s)$.

2) **Injected**: inject the extracted exact Satake table $(\alpha_p(u),\beta_p(u))$ directly (local_model='satake') and assemble $K_u(s)$.

Concretely, the following agree to ~1e-16 relative/absolute error (depending on the reported metric):

- $D_u(s)$,
- $\log D_u(s)$ (branch-consistent on the sampled grid),
- $\Phi_u(s) = -\partial_s \log D_u(s)=\mathrm{Tr}((I-K_u)^{-1}K_u')$.

**Interpretation**

This is an **exact local-to-global closure check**: the global determinant in this architecture is fully explained by the extracted local Satake packets, and this closure remains stable across the tested deformation family $u$.
