Strict oracle-first proof progress

Context
- Date: 2026-03-15
- Mainline stack remains frozen: beta23 oracle core, current translator path, safe controller policy, current 5-feature branch ranker checkpoint.
- Project is now split into Track A mainline engineering benchmark and Track B strict oracle-first proof benchmark.

Most important strict artifact
- Benchmark summary: out/ai/checkpoints/gsm8k_phi3mini_structuredsteps_scale_v1_branchranker/world_model_stress/third_batch/s01/s01_summary.json
- Suite report: out/ai/checkpoints/gsm8k_phi3mini_structuredsteps_scale_v1_branchranker/world_model_stress/third_batch/s01/s01_suite.json

Strict s01 result
- source_groups: 34
- selected_groups: 24
- selected_group_types: all mixed
- mean_max_ambiguity_score: 0.8683732413419912
- mean_gold_minus_best_wrong_surface_score: -0.24958333333333327

Challenger-only head-to-head on strict s01
- surface chooser: 0.0
- best text-side baseline: 0.25
- oracle path: 0.2916666666666667
- controller path: 0.16666666666666666
- branch-ranker path: 0.08333333333333333

Interpretation
- The strict slice is now doing the intended job: shallow surface selection is fully defeated.
- Oracle is the strongest structural path on this proof regime.
- Controller and branch ranker underperform oracle on the same slice, so they should be reported as secondary diagnostic paths here rather than the main proof claim.
- This supports the architectural emphasis: generator -> translator/state extractor -> frozen structural verifier/oracle -> narrow controller or local resolver.

Operational note
- Windows path depth remains a practical constraint for strict runs.
- Short artifact prefixes are currently the safest way to complete deep strict benchmark runs on this machine.

Next work
1. Audit the 24 strict mixed-only groups to classify why branch ranker regresses relative to oracle.
2. Produce an oracle-first proof summary anchored on the strict s01 slice.
3. Prioritize translator and state-extraction quality improvements before reopening broad learned-scoring changes.