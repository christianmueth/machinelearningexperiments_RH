Strict s01 oracle-first proof summary

Benchmark artifact
- Benchmark: out/ai/checkpoints/gsm8k_phi3mini_structuredsteps_scale_v1_branchranker/world_model_stress/third_batch/s01/s01.jsonl
- Benchmark summary: out/ai/checkpoints/gsm8k_phi3mini_structuredsteps_scale_v1_branchranker/world_model_stress/third_batch/s01/s01_summary.json
- Suite report: out/ai/checkpoints/gsm8k_phi3mini_structuredsteps_scale_v1_branchranker/world_model_stress/third_batch/s01/s01_suite.json

Benchmark composition
- 24 selected groups from a 34-group strict source pool
- All 24 selected groups are mixed
- Mean max ambiguity score: 0.8683732413419912
- Mean gold-minus-best-wrong surface score: -0.24958333333333327
- Ambiguity tags are concentrated in arithmetic_slip plus dependency_conflict or dependency_omission, with structurally_plausible_wrong present in 23 of 24 groups

Head-to-head challenger-only result
- Surface chooser: 0.0, or 0 of 24
- Best text-side baseline: 0.25, or 6 of 24
- Oracle path: 0.2916666666666667, or 7 of 24
- Controller path: 0.16666666666666666, or 4 of 24
- Branch-ranker path: 0.08333333333333333, or 2 of 24

Primary claim
- This strict slice now cleanly defeats shallow surface selection.
- On that slice, the frozen oracle is the strongest structural evaluator and beats the strongest text-side baseline.
- The strict proof object for Track B should therefore be oracle-first, with controller and branch-ranker results reported as secondary diagnostics.

Branch-ranker regression audit
- Audit artifact: out/ai/checkpoints/gsm8k_phi3mini_structuredsteps_scale_v1_branchranker/world_model_stress/third_batch/s01/s01_branch_regressions_vs_oracle.json
- Branch ranker has 5 regressions versus oracle and 0 wins versus oracle on this slice.
- All 5 regressions occur on zero-oracle-margin mixed groups with a 3-candidate branch set.
- 4 of the 5 regressions are already controller regressions: controller bonuses move the top choice away from the oracle-selected gold candidate, and the branch ranker then preserves the wrong branch preference.
- 1 of the 5 regressions is a true branch-ranker-only flip: the controller still selects the gold candidate, but the branch ranker promotes a wrong challenger over it.
- No evidence from this slice suggests structural deduplication or signature consensus is the main issue. The failure pattern is score preference inside a small low-margin branch, not duplicate-state collapse.

Interpretation for next work
- The benchmark is now strong enough to support the proof claim that oracle beats surface-style selection on a strict surface-hard mixed regime.
- The main improvement target is not broader learned scoring. It is translator or state-extraction quality, plus controller behavior in uncertainty regions.
- Branch-ranker work should remain secondary and tied to Track A unless a local change demonstrably helps the larger mainline benchmark offline.