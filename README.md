# machinelearningexperiments_RH

This repository is the working proof-and-architecture package for a dyadic/Euler determinant program aimed at building a completed global analytic object from a structured spectral backend. In the current state of the project, the repository serves three roles at once:

- a proof-navigation and obligation-tracking workspace for the RH/scattering program,
- a reproducible numerical witness package for the frozen anchored pipeline,
- and a machine-learning inductive-bias testbed built around that pipeline as a frozen structured oracle.

At a high level, the architecture described in [RH Architecture.pdf](RH%20Architecture.pdf) is:

`dyadic anchor -> dyadic tower -> ghost coordinates -> Euler/multiplicative lift -> corrected local coefficients -> prime-local packet realization -> determinant assembly -> anchored centered-even completion -> completed analytic object -> FE/zero/rigidity probes`

The companion note [Machine learning wRH.pdf](Machine%20learning%20wRH.pdf) reframes that same stack as a reasoning prior rather than only a proof engine: a latent structured world model with a global consistency score, suitable for reranking or constraining language-guided reasoning while keeping the mathematical core frozen.

Inductive-bias themes represented in the repo:

- spectral stability bias
- determinant-level global invariants
- primitive-vs-composite factor structure
- cross-resolution/operator-consistency checks
- operator-based diagnostics as reasoning supervision

## What this repository is actually trying to build

The central object is not a raw Dirichlet series assembled term-by-term. The architecture instead starts from a dyadic spectral anchor, passes through an additive ghost representation and an Euler-style multiplicative lift, realizes the resulting local arithmetic data through small spectral packet models, and then assembles a determinant-based completed object whose behavior is studied through functional-equation, zero-tracking, rigidity, and stability probes.

In the language of the PDF architecture note, the repo is organized around four interacting layers:

1. Dyadic generator layer.
	A local anchor matrix supplies the primitive normalization and the dyadic `2^k` tower grammar from which the rest of the arithmetic structure is grown.
2. Arithmetic lift layer.
	Ghost coordinates and Euler-style transforms convert the dyadic seed into multiplicative local data, followed by correction and calibration of the low-order coefficients used downstream.
3. Spectral realization layer.
	Prime-local packet models realize the corrected local data as concrete low-dimensional spectral objects.
4. Global analytic layer.
	The packet family is assembled into a determinant-like global object, then completed with an anchored centered-even correction so that FE-like and transverse geometric diagnostics can be measured on a stable completed object.

## Current status in one page

What is already established in the repository, at the level of architecture plus numerical witness evidence:

- a coherent dyadic-to-ghost-to-Euler pipeline exists and has been frozen into reproducible artifacts,
- corrected local coefficient tables can be extracted and reused without regenerating the backend,
- prime-local spectral packet realizations exist for those corrected coefficients,
- the assembled determinant object exhibits FE-like symmetry diagnostics, stable zero-candidate structure, and rigidity/stability observables,
- the architecture cleanly separates structural/operator statements from the remaining analytic identification obligations.

What is not yet claimed as proved, and is treated honestly as open in the project notes:

- exact equality with the classical Euler product,
- exact Mangoldt/log-derivative identity,
- equality with the Riemann zeta function or modular scattering determinant,
- a theorem that all zeros of the constructed global object lie on the critical line.

That distinction matters: the repo is strongest today as a structural architecture plus witness framework, not as a completed theorem package.

## Final validation before AI/ML use

The repo now has a short final validation note for the frozen anchored pipeline: [notes/FROZEN_ORACLE_SANITY_SHEET.md](notes/FROZEN_ORACLE_SANITY_SHEET.md).

That note documents four minimal pre-ML checks:

- Euler-style closure improvement on the strongest frozen corrected artifact,
- Mangoldt/log-derivative improvement of the accepted `A3_exact_frontend` versus the main baselines,
- critical-line recentering of the persistent `t≈28` zero-candidate cluster under the anchored completion,
- and local stability of exported oracle features across nearby operating points.

The practical conclusion is narrow but useful: the current anchored architecture is validated well enough to serve as a frozen structured oracle or inductive bias for AI/ML experiments. The note also restates the project’s open boundaries explicitly: no theorem is claimed here for exact Euler-product identity, exact Mangoldt identity, equality with zeta/classical scattering, or universal critical-line location of zeros.

## PDF-guided architecture summary

The two root PDFs are useful as high-level summaries of what the codebase is doing and why.

- [RH Architecture.pdf](RH%20Architecture.pdf) describes the full mathematical stack from the original dyadic anchor through determinant assembly and completed-object probes. It emphasizes that the key conceptual move is the additive ghost representation followed by the Euler lift, and that the completed object is a hybrid arithmetic-spectral determinant rather than a conventional directly-written Dirichlet series.
- [Machine learning wRH.pdf](Machine%20learning%20wRH.pdf) interprets the same stack as a structured latent prior for AI systems: local packets become latent realizers, the determinant/completion layer becomes a global coherence score, and the frozen anchored pipeline becomes a scientific oracle that should guide reasoning rather than be overwritten by gradient updates.

## Start here

## Start here

- Proof navigation map: [notes/proof_roadmap.md](notes/proof_roadmap.md)
- Numerics-as-witnesses methodology firewall: [notes/methodology_numerics_as_witnesses.md](notes/methodology_numerics_as_witnesses.md)
- Frozen run protocol / contract: [RUN_CONTRACT.md](RUN_CONTRACT.md)
- Remaining obligations (proof delta: what’s still analytic vs witnessed): [REMAINING_OBLIGATIONS.md](REMAINING_OBLIGATIONS.md)

## Proof chain (notes)

- Convergence + safe-rectangle hypotheses (finite \(N\) \(\to\) limiting channel): [notes/lemma_safe_rectangle_convergence.md](notes/lemma_safe_rectangle_convergence.md)
- Modular divisor-free rectangle boundaries (H4 discharge): [notes/lemma_modular_divisor_free_boundary.md](notes/lemma_modular_divisor_free_boundary.md)
- Selector admissibility (channel isolation / no swapping): [notes/lemma_selector_admissibility.md](notes/lemma_selector_admissibility.md)
- Task A (operator backbone: DN/Weyl \(\Lambda(s)\), Cayley/scattering \(S_\eta(s)\), meromorphy/involution): [notes/operator_construction_and_meromorphy.md](notes/operator_construction_and_meromorphy.md)
- Truncation \(\to\) DN/Weyl convergence mechanism (H2): [notes/lemma_truncation_to_dnmap_convergence.md](notes/lemma_truncation_to_dnmap_convergence.md)
- H2 specialized to the concrete Experiment E objects: [notes/h2_operator_consistency_instantiated.md](notes/h2_operator_consistency_instantiated.md)
- Task B (mapping theorem + canonical cusp-channel extraction; modular \(\phi_{mod}(s)\)): [notes/eisenstein_intertwiner_identification.md](notes/eisenstein_intertwiner_identification.md)
- Rigidity hinge for \(Q(s)=\lambda(s)/\phi_{mod}(s)\): [notes/proposition_identification_hinge.md](notes/proposition_identification_hinge.md)
- One-page audit table (hypotheses \(\leftrightarrow\) artifacts): [notes/hypotheses_to_artifacts.md](notes/hypotheses_to_artifacts.md)

## Structural analogue (MLC blueprint)

- Mandelbrot Local Connectivity (MLC) in the same “one-hinge lemma + closure” style, and its connection to the DN/Cayley/scattering pipeline: [notes/mlc_hinge_lemma_and_connection.md](notes/mlc_hinge_lemma_and_connection.md)

## AI/AGI abstraction (RAA)

- Refinement-with-Admissibility Architecture (RAA): stable refinement systems + ghost diagnostics, grounded in the repo’s methodology firewall and admissibility engine: [RAA_REFINEMENT_WITH_ADMISSIBILITY_ARCHITECTURE.md](RAA_REFINEMENT_WITH_ADMISSIBILITY_ARCHITECTURE.md)
- Frozen-oracle reasoning-model spec: a concrete ML design that treats the current RH/scattering stack as a fixed structured consistency prior for language-guided reasoning: [notes/FROZEN_ORACLE_REASONING_MODEL_SPEC.md](notes/FROZEN_ORACLE_REASONING_MODEL_SPEC.md)
- Practical AI workspace for datasets, benchmarks, and reranker experiments around the frozen oracle: [ai/README.md](ai/README.md)

## Evidence packets (human-readable summaries)

- Results packet: [RESULTS_PACKET_2026-01-29.md](RESULTS_PACKET_2026-01-29.md)
- Lemma-chain summary: [LEMMA_CHAIN_SUMMARY.md](LEMMA_CHAIN_SUMMARY.md)
- Theorem/proof roadmap overview: [THEOREM_ROADMAP.md](THEOREM_ROADMAP.md)

## Reproducibility (code + configs)

- Experiment configs (YAML): [configs/](configs/)
- Experiment drivers: [experiments/](experiments/)
- Core library (DN + scattering utilities): [src/](src/)
- Helper tooling (aggregation/reporting): [tools/](tools/)

## Current architecture

The active RH assembly pipeline in this repository is organized as a frozen backend plus a canonical frontend and completed-object analysis layer. This is the operational version of the broader architecture summarized in [RH Architecture.pdf](RH%20Architecture.pdf).

1. Frozen backend coefficients.
	The arithmetic side is treated as fixed input once corrected coefficients have been extracted. The canonical table is `out/corrected_factor_injection_beta23_window_coefficients.csv`, and `tools/corrected_backend_interface.py` is the narrow loader that exposes `u`, `beta2`, `beta3`, `c`, and the corrected local coefficients `A1_star`, `A2_star`, `A3_star`.

2. Canonical frontend realization.
	The accepted frontend is the `A3_exact_frontend`, implemented in `tools/probe_frontend_realization.py`. For each selected `u`, it fits a normal `3x3` packet host whose eigenvalues match the corrected low-order coefficient targets while staying inside the spectral-radius constraint. This frontend is the default realization used by the downstream completed-object probes.

3. Global determinant assembly.
	`tools/probe_completed_global_object.py` builds the global determinant object from the frozen coefficient row and the canonical packet family. It evaluates determinant tracks over `t`, extracts critical-line zero candidates, and measures functional-equation consistency across the working `u` window.

4. Anchored completion layer.
	The current completion law is a centered-even correction carried as an anchored real log-weight rather than a raw multiplicative quartic. The implementation lives in `tools/probe_completed_global_object.py` and supports coefficients through orders `2, 4, 6, 8` with `completion_even_mode=anchored_real`. The practical default remains an anchored quartic term with `completion_even_a4=-0.2`, with higher inverse-anchor families explored separately.

5. Analytic characterization and search probes.
	`tools/characterize_completed_object.py` is the main characterization driver: FE surface summaries, transverse `sigma` scans, zero tracking, rigidity, and stability checks. Supporting probes include:
	`tools/densify_local_zero_scan.py` for dense local `(sigma, t)` scans around the persistent `t≈28` cluster.
	`tools/investigate_completion_layer.py` and `tools/search_selfdual_completion_templates.py` for completion-family comparisons.
	`tools/derive_anchor_completion_weight.py` for deriving the even completion curvature from the dyadic anchor.
	`tools/search_anchor_lambda_family.py` for higher-order inverse-anchor `lambda` searches.

6. Operating conventions.
	The backend is intentionally not regenerated during frontend/completion experiments. The default analytic operating point is `u=0.24`, the frontend default is `A3_exact_frontend`, and completion experiments are evaluated against the persistent zero cluster near `t≈27.85..28.05`.

In short: the repo’s current numerical architecture is `corrected coefficient table -> constrained A3 frontend packets -> global determinant assembly -> anchored even completion -> FE/zero diagnostics`.

## ML interpretation of the same architecture

The machine-learning direction in this repo does not treat the RH/scattering stack as a trainable neural network. Following [Machine learning wRH.pdf](Machine%20learning%20wRH.pdf), it treats the anchored pipeline as a frozen structured oracle with three conceptual layers:

1. Language layer.
	A conventional encoder or LLM handles problem text and candidate reasoning traces.
2. Structured latent layer.
	The dyadic backend, corrected coefficients, local packets, and completion settings define a latent world model with explicit factorization and spectral structure.
3. Global consistency layer.
	Oracle summaries derived from determinant assembly, FE stability, zero structure, and related probes act as a coherence signal for reranking or controlling reasoning.

The implementation rule is deliberate: learn around the oracle, not through it. The PDFs argue for the following guardrails, which the repo now follows:

- keep the anchored backend frozen,
- never backpropagate into the mathematical oracle,
- keep oracle outputs interpretable and versioned,
- compare every learned component against a no-oracle baseline,
- use the oracle as structured supervision for consistency, not as a raw text target.

That design is documented concretely in [notes/FROZEN_ORACLE_REASONING_MODEL_SPEC.md](notes/FROZEN_ORACLE_REASONING_MODEL_SPEC.md).

## Suggested reading order

If you are new to the repository, the most efficient sequence is:

1. Read this README for the system-level map.
2. Read [RH Architecture.pdf](RH%20Architecture.pdf) for the end-to-end mathematical architecture.
3. Read [notes/proof_roadmap.md](notes/proof_roadmap.md) and [REMAINING_OBLIGATIONS.md](REMAINING_OBLIGATIONS.md) to separate established structure from open analytic hinges.
4. Read [Machine learning wRH.pdf](Machine%20learning%20wRH.pdf) and [notes/FROZEN_ORACLE_REASONING_MODEL_SPEC.md](notes/FROZEN_ORACLE_REASONING_MODEL_SPEC.md) if you care about the inductive-bias / reasoning-model direction.
5. Use the `tools/` drivers and `out/` artifacts to inspect the frozen anchored operating point directly.

## Outputs

Local run outputs typically land under directories like `out_*/` and `runs/`.
These are treated as generated artifacts and are not intended to be versioned in git (see [.gitignore](.gitignore)).
