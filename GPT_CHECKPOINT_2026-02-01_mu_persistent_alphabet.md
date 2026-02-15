# GPT collaborator checkpoint — 2026-02-01 (µ-stability with persistent generator alphabet)

## Executive summary
We now have a working “µ-stability zeta diagnostic” pipeline that can be run on validate_mugrid runs and produces paste-ready comparison tables.

Key update at this checkpoint: we implemented stronger controls for “same primes across µ” by enforcing a *persistent generator alphabet* across µ slices.

- A strict ID-intersection control (`alphabet_strategy=persistent_intersection`) turned out to be *too strict* on the current data: the intersection of generator IDs across µ={0.75,1.0} is empty for both seed80 and seed96 in the current validate_mugrid outputs. This indicates generator IDs are not stable labels across µ (renumbering / clustering changes), so “same gen_id across µ” is not a valid persistence criterion.
- We therefore added an invariant-based matching control (`alphabet_strategy=match_invariants`) that matches generators across µ by simple matrix invariants, then compares the resulting matched alphabet across µ.

This gives us an apples-to-apples stability probe that is stronger than fixed-k “first generators” while not relying on unstable IDs.

## What we have now
### Tools added/updated
- [tools/phase3f_mu_zeta_stability.py](tools/phase3f_mu_zeta_stability.py)
  - Adds `--alphabet_strategy {slice,persistent_intersection,match_invariants}`
  - Adds matching knobs for invariants:
    - `--match_ref_mu` (default: smallest included µ)
    - `--match_max_cost` (standardized feature distance threshold)
  - Writes additional bookkeeping columns to the summaries (strategy, match parameters, avail/used generator counts).
  - Safety: if no rows are produced (e.g., intersection empty + skip), it now writes empty CSVs instead of crashing.

### Outputs created (in the smoke out_dir)
Folder: `out_phase3E_elambda_2D_mu_B2pilotv2_smoke/`

1) Strict ID-intersection attempt (expected to be empty, but now handled gracefully)
- `mu_zeta_seed80_validate_persistk4_mu075_1_{scalar,word}_summary.csv`
- `mu_zeta_seed96_validate_persistk4_mu075_1_{scalar,word}_summary.csv`
- `mu_zeta_validate_persistk4_compare.md` (+ `_long.csv`)
- `mu_zeta_validate_persistk4_cases.txt`

2) Invariant-matched “persistent alphabet” comparison (works; uses 3 matched generators)
- `mu_zeta_seed80_validate_matchk4_mu075_1_{scalar,word}_summary.csv`
- `mu_zeta_seed96_validate_matchk4_mu075_1_{scalar,word}_summary.csv`
- `mu_zeta_validate_matchk4_compare.md` (+ `_long.csv`)
- `mu_zeta_validate_matchk4_cases.txt`

Notable detail: in these runs, the matched persistent alphabet size is 3 for both seeds (so k=4 request degrades to 3 under `use_all`). This is recorded via `persistent_intersection_size` and `n_gens_used`.

3) Robustness sweep over matching threshold (`match_max_cost`)
- `mu_zeta_validate_matchk_costs_compare.md` (+ `_long.csv`)
- `mu_zeta_validate_matchk_costs_cases.txt`

This table is the key “falsification” artifact: it shows whether conclusions are stable as we loosen/tighten the matching criterion.

## Interpretation (conservative)
- The zeta-like µ diagnostics are now reproducible and controllable.
- Generator IDs are not stable across µ slices, so “persistence across µ” must be defined structurally (via invariants or higher-level primitive-class tracking), not by raw IDs.
- Invariant-matching provides a workable structural notion of “same alphabet across µ”, but we should treat it as a *diagnostic* rather than a proof: it is greedy and uses a low-dimensional feature embedding.

## Open questions / request for collaborator direction
1) Better matching criterion?
   - Should we implement a more principled assignment (Hungarian) for cross-µ matching? (Currently: greedy minimal-cost matching.)
   - Should the feature vector include additional invariants (e.g., spectrum moments, trace of powers) to reduce collisions?

2) What should be the “blessed” stability claim?
   Options:
   - A) Fixed-k per-µ (controls generator-count but not identity)
   - B) Invariant-matched persistent alphabet (controls identity up to matching)
   - C) Track persistence at primitive-class level (harder but closer to the narrative)

3) How to package the result mathematically?
   - Right now we have: nonabelian witnesses + a zeta-like Euler surrogate + word-zeta cancellation diagnostics.
   - What would count as “structure identified” in your view: e.g., approximate representation of a known group, or an intrinsic phase transition signature across µ?

## Minimal reproduction commands
(Windows paths; run from repo root)

- Invariant-matched µ-stability for a validate_mugrid run:
  - `C:/Users/chris_xht57xv/machinelearningexperiments_RH/.venv/Scripts/python.exe tools/phase3f_mu_zeta_stability.py --run_dir out_phase3E_elambda_2D_mu_B2pilotv2_smoke/validate_mugrid_prime_seed80_a14_w2_5_mu0_1_rs3 --out_dir out_phase3E_elambda_2D_mu_B2pilotv2_smoke --mu_values 0.75,1.0 --alphabet_strategy match_invariants --match_ref_mu 0.75 --match_max_cost 1.25 --restrict_k_gens 4 --insufficient_gens_mode use_all --out_prefix mu_zeta_seed80_validate_matchk4_mu075_1`

- Compare cases listed in a case file:
  - `C:/Users/chris_xht57xv/machinelearningexperiments_RH/.venv/Scripts/python.exe tools/phase3f_mu_zeta_stability_compare.py --out_dir out_phase3E_elambda_2D_mu_B2pilotv2_smoke --case_file out_phase3E_elambda_2D_mu_B2pilotv2_smoke/mu_zeta_validate_matchk4_cases.txt --out_prefix mu_zeta_validate_matchk4_compare`

## Suggested “decision point”
If the cost-sweep comparison stays qualitatively consistent (same seed80 vs seed96 separation and similar µ-trends), then we likely have a robust checkpoint worth sending for new theoretical direction. If it is highly sensitive to `match_max_cost`, we should refine the matching definition before escalating.
