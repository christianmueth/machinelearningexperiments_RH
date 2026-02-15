# Hypotheses ↔ artifacts (audit table)

This table is the “numerics → hypotheses” bridge: the harness does not prove the analytic statements, but it **records auditable witnesses** that the hypotheses are satisfiable and (in a given run) hold on the chosen safe rectangle(s).

| Item | Mathematical statement (paper form) | Witnessed/verified by (artifact + fields) |
|---|---|---|
| H1 (invertibility margin) | $\inf_{s\in R} s_{\min}(\Lambda_N(s)+i\eta I)\ge\delta$ for all $N\ge N_0$ on a chosen safe rectangle $R$ | Dip/pole diagnostics: `E_pole_line_scan.csv` (when enabled); safe-rectangle selection via `E_dip_atlas_*` (when enabled). Practical audit: `E_Qlambda_convergence_winding.csv` (`failures`, `n_edge`, `eta`, `cayley_eps`) should show stable evaluation on $\partial R$. |
| H2 (operator consistency) | $\sup_{s\in R}\|\Lambda_N(s)-\Lambda(s)\|\to 0$ | Operator theory + truncation argument (not a logged numeric field). See `notes/operator_construction_and_meromorphy.md` for defining $\Lambda(s)$, and `notes/lemma_safe_rectangle_convergence.md` for how H2 is used. |
| H3 (channel isolation) | Channel eigenvalue is isolated by gap $\ge\gamma$ on $R$ | Convergence sweep: `E_Qlambda_convergence_sweep.csv` (`gap_phi_min` and related gap fields per channel); autopick filter: `E_boundary_autopick_candidates.csv` (`gap_ok`, `gap_min` if present). |
| H4 (divisor-clean boundary) | $\phi_{mod}$ has no zeros/poles on $\partial R$ and the quotient is nonvanishing on $\partial R$ (winding=0) | Winding logs: `E_Qlambda_convergence_winding.csv` (`winding`, `failures`, `n_edge`) for convergence-only runs; and/or divisor diagnostics: `E_divisor_arg_principle_*.csv` (non-convergence runs) with recorded `n_edge`/rectangle bounds. |
| Selector locality | $b_0(N)\in\{b^\star(N)-w,\dots,b^\star(N)+w\}$ | Policy: `E_boundary_autopick_policy.md` (candidate window); candidates: `E_boundary_autopick_candidates.csv` (`delta`, `b_star`, `deltas`). |
| Selector constraints | winding=0 (two-stage), gap $\ge\gamma$, evaluation success | Candidates: `E_boundary_autopick_candidates.csv` (`metric_ok`, `gap_ok`, `wind_ok_stage1`, `wind_ok_stage2`). |
| Guard non-vacuity | “No top-5 regression” guard is active mainly at small $N$ | Acceptance summary: `E_boundary_autopick_acceptance.csv` (`n_fail_top5_guard`, plus ordered pipeline `n_rej_top5_guard`, `n_after_*`). |
| Worst-channel stability curve | worst-channel p95/top5 vs $N$ (baseline→autopick) | Aggregate report: `runs/_compare_latest_baseline_vs_autopick.csv` (worst-channel p95/top5 deltas + acceptance `ap_*` counts). |

Notes:
- The ordered-pipeline counts (`n_rej_*`, `n_after_*`) are disjoint and “reader-friendly”; independent fails (`n_fail_*`) can overlap.
- The only non-numeric hinge is H2 (operator consistency), which is an operator-theory/truncation proof obligation.
