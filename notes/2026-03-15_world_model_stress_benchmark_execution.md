World-model stress benchmark phase

Frozen stack for this phase
- beta23 oracle core
- current translator path
- safe controller policy
- current 5-feature branch ranker checkpoint at 0.500
- no live-path feature search beyond optional offline diagnostics

Architecture emphasis for this phase
- Main architectural story: generator -> translator/state extractor -> frozen structural verifier/oracle -> narrow controller or local resolver.
- Do not treat the branch ranker as the universal proof object.
- Treat learned refinement as a narrow local tool that is only primary on the larger mainline benchmark where it has already earned lift.

Primary goal
- Build one proof benchmark that isolates structural ambiguity cases where shallow surface plausibility is not enough and run a single direct comparison across surface chooser, rerankers, oracle path, controller path, and branch-ranker path.

Current interpretation
- The RH-inspired direction is aligned with process-verifier and controller-style work: the strongest signals in this stack are structural state-coherence signals, not generic semantic reward-model signals.
- On the strict surface-hard regime, the oracle path is currently the strongest structural candidate and should be the lead proof object.
- On the larger structured benchmark, the branch ranker remains a valid mainline engineering component because it still delivers measurable low-margin lift.

Deliverables
- A fixed-size proof benchmark JSONL built from the frozen stack.
- A suite report on that same benchmark with all baselines on the same slice.
- A summary that states whether the oracle path beats the strongest text-side baseline on challenger-only evaluation, with controller and branch-ranker paths reported as secondary challengers.
- A second independently generated benchmark batch if the first result is promising.

Project split
1. Track A: Mainline system
	- Use the larger structured benchmark as the engineering benchmark.
	- Keep the live stack frozen unless a local refinement clearly improves that benchmark offline.
	- The branch ranker remains allowed here because it has already shown lift under low-margin conditions.
2. Track B: Proof benchmark
	- Use the strict surface-hard mixed or mixed+dependency regime.
	- Lead with oracle vs text-style selection.
	- Treat controller as secondary and branch ranker as optional challenger, not the central claim.

Stop conditions
- Stop benchmark polishing if the proof benchmark is large enough to evaluate and the suite gives a clean head-to-head result.
- Do not reopen architecture unless the proof benchmark repeatedly shows the frozen stack losing on the intended regime.
- Do not promote new branch-ranker features unless they beat the 0.500 checkpoint on the main benchmark offline.

Immediate sequence
1. Keep the current live stack frozen on the mainline benchmark.
2. Use the strict surface-hard proof slice to report oracle-first head-to-head results against surface and text-side baselines.
3. Use controller and branch-ranker results on the strict slice as diagnostic secondary comparisons, not as the main proof claim.
4. Put translator and state-extraction quality ahead of generic branch-ranker feature search.
5. If the oracle still loses on a clean strict slice, improve the benchmark or translator before reopening broad learned-scoring work.