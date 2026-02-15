# machinelearningexperiments_RH

This repository contains the RH/Experiment-E proof package notes, the frozen run protocol, and the code/configs used to generate the numerical *witness* artifacts.

## Start here

- Proof navigation map: [notes/proof_roadmap.md](notes/proof_roadmap.md)
- Numerics-as-witnesses methodology firewall: [notes/methodology_numerics_as_witnesses.md](notes/methodology_numerics_as_witnesses.md)
- Frozen run protocol / contract: [RUN_CONTRACT.md](RUN_CONTRACT.md)
- Remaining obligations (proof delta: what’s still analytic vs witnessed): [REMAINING_OBLIGATIONS.md](REMAINING_OBLIGATIONS.md)

## Proof chain (notes)

- Convergence + safe-rectangle hypotheses (finite \(N\) \(\to\) limiting channel): [notes/lemma_safe_rectangle_convergence.md](notes/lemma_safe_rectangle_convergence.md)
- Selector admissibility (channel isolation / no swapping): [notes/lemma_selector_admissibility.md](notes/lemma_selector_admissibility.md)
- Task A (operator backbone: DN/Weyl \(\Lambda(s)\), Cayley/scattering \(S_\eta(s)\), meromorphy/involution): [notes/operator_construction_and_meromorphy.md](notes/operator_construction_and_meromorphy.md)
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

## Outputs

Local run outputs typically land under directories like `out_*/` and `runs/`.
These are treated as generated artifacts and are not intended to be versioned in git (see [.gitignore](.gitignore)).
