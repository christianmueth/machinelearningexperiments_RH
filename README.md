# machinelearningexperiments_RH

This repository contains the RH/Experiment-E proof package notes, the frozen run protocol, and the code/configs used to generate the numerical *witness* artifacts.

Inductive bias research:

spectral stability bias	- uncommon

determinant-level invariants	- very uncommon

cross-resolution operator tests	- rare

operator-based diagnostics	- emerging research area

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

## Evidence packets (human-readable summaries)

- Results packets: [RESULTS_PACKET_2026-01-29.md](RESULTS_PACKET_2026-01-29.md), [RESULTS_PACKET_2026-01-30.md](RESULTS_PACKET_2026-01-30.md), [RESULTS_PACKET_2026-01-31.md](RESULTS_PACKET_2026-01-31.md)
- Lemma-chain summary: [LEMMA_CHAIN_SUMMARY.md](LEMMA_CHAIN_SUMMARY.md)
- Theorem/proof roadmap overview: [THEOREM_ROADMAP.md](THEOREM_ROADMAP.md)

## Reproducibility (code + configs)

- Experiment configs (YAML): [configs/](configs/)
- Experiment drivers: [experiments/](experiments/)
- Core library (DN + scattering utilities): [src/](src/)
- Helper tooling (aggregation/reporting): [tools/](tools/)

## Current architecture

The active RH assembly pipeline in this repository is organized as a frozen backend plus a canonical frontend and completed-object analysis layer.

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

## Outputs

Local run outputs typically land under directories like `out_*/` and `runs/`.
These are treated as generated artifacts and are not intended to be versioned in git (see [.gitignore](.gitignore)).
