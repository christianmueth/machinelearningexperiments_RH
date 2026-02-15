# Lemma: Selector Admissibility (boundary autopick is not parameter tuning)

## Purpose

This lemma formalizes the boundary “autopick” as a **deterministic finite selector** constrained by structural conditions (divisor-clean winding and spectral isolation) and guarded against extreme-tail regressions.

It is designed to neutralize the objection “you tuned the discretization until it matched” while remaining honest: the selector does not change the limiting analytic object; it chooses among a tiny fixed local candidate set.

---

## Definitions (match policy + CSV artifacts)

Fix a truncation level $N$ and a nominal boundary size $b^\star(N)$ (the baseline discretization).

### Candidate set (local, finite)

For a fixed window $w\in\mathbb{N}$ (default $w=2$ in the harness), define
\[
\mathcal{B}_N := \{\,b^\star(N)+\Delta : \Delta\in\{-w,\ldots,w\}\,\}.
\]

Equivalently, in terms of the stored `delta` values, candidates are $\Delta\in\{-w,\ldots,w\}$.

### Candidate evaluation functional

For each candidate boundary $b\in\mathcal{B}_N$ and each safe rectangle $R$ (fixed for the sweep), the harness evaluates a robustness statistic on a finite boundary grid of $s$-values on $R$:

- worst-channel p95 error (primary),
- worst-channel top5 mean error (secondary),
- worst-channel median error (tertiary),

as implemented by the “worst-channel, lexicographic” mode.

These are exactly the stored per-candidate columns (`err_p95`, `err_top5`, `err_med`) and (optionally) the argmax channel identity fields.

### Structural constraints (booleans in candidates table)

A candidate is called **admissible** if it satisfies the following boolean predicates (as logged per candidate):

- `metric_ok`: the evaluation completed (no NaNs / no hard failures).
- `gap_ok`: the observed channel isolation is acceptable (gap ≥ γ).
- `wind_ok_stage1`: winding check on $\partial R$ succeeds at stage-1 resolution and returns 0 (within tolerance).
- `top5_guard_ok`: (optional) passes the “no top5 regression” guard relative to the baseline candidate $\Delta=0$.

These appear in `E_boundary_autopick_candidates.csv`.

The per-$N$ acceptance summary is logged in `E_boundary_autopick_acceptance.csv`, including both independent fail counts (`n_fail_*`) and ordered pipeline counts (`n_rej_*`, `n_after_*`).

### Objective (deterministic ranking)

Among admissible candidates, the selector chooses the unique minimizer under lexicographic rank
\[
(\mathrm{p95},\;\mathrm{top5},\;\mathrm{median}).
\]

Tie-breaking is deterministic (stable sort by this tuple plus a fixed iteration order over $\Delta$).

### Two-stage winding confirmation

After selecting a short list of top candidates (e.g. top-$k$ by the objective), the harness recomputes winding at higher boundary resolution and chooses the first candidate that passes (stage 2). This is recorded in the candidate table by `wind_ok_stage2` and related fields.

---

## Lemma statement (admissibility and non-cheating)

**Lemma (Selector admissibility on safe rectangles).**
Fix a safe rectangle $R$ and impedance $\eta$.
Assume the hypotheses of the Safe-Rectangle Convergence Lemma hold on $R$ (invertibility margin and channel isolation, plus the existence of a limiting operator object).

Define $b_0(N)$ to be the boundary choice produced by the autopick policy described above.

Then:

1) (**Deterministic, finite, local**) For each $N$, the selection map $N\mapsto b_0(N)$ is deterministic and satisfies
\[
|b_0(N)-b^\star(N)|\le w.
\]

2) (**Structural admissibility**) Every selected candidate satisfies the structural constraints:
- winding = 0 on $\partial R$ (at stage 2 when enabled),
- gap ≥ γ,
- and (when enabled) worst-channel top5 does not exceed the baseline top5 at $\Delta=0$ up to the configured slack.

3) (**Asymptotic irrelevance of selection**) On $R$, the selector does not alter the limiting analytic object: it only chooses among discretizations whose approximation error is uniformly controlled on $R$.
More precisely, if each candidate family $\{\Lambda_{N,b}(s)\}_{N}$ converges to the same limiting $\Lambda(s)$ uniformly on $R$, then the selected family $\Lambda_{N,b_0(N)}(s)$ also converges to $\Lambda(s)$ uniformly on $R$.

4) (**Non-vacuity, auditability**) The acceptance artifacts provide an auditable record that:
- the candidate set is fixed and tiny (e.g. 5 points),
- constraints are structural,
- and the guard’s activation pattern can be reported (typically active mainly at small $N$).

---

## Proof sketch (theorem-compatible)

1) Determinism and locality are immediate from the fixed candidate window and lexicographic ordering.

2) Structural admissibility is immediate from the filter-then-confirm pipeline: only candidates satisfying the constraints can be selected.

3) Asymptotic irrelevance: if for each fixed $\Delta$ in the finite set the corresponding discretization converges to the same limit on $R$, then selecting any sequence $\Delta(N)$ with $\Delta(N)\in\{-w,\ldots,w\}$ preserves convergence to that same limit. (This is a compactness/finite-choice argument, not an optimization argument.)

4) The acceptance tables do not prove convergence; they support the methodological claim that selection is constrained and auditable.

---

## Direct mapping to run artifacts

- Policy text: `E_boundary_autopick_policy.md`
- Candidate-level booleans and objective statistics: `E_boundary_autopick_candidates.csv`
- Selected deltas: `E_boundary_autopick_choice.csv`
- Acceptance rates (independent + ordered pipeline): `E_boundary_autopick_acceptance.csv`

See also: `notes/hypotheses_to_artifacts.md` for a single-table summary.
