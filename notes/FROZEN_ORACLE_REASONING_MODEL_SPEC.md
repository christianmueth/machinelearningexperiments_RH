# Frozen-Oracle Reasoning Model Spec

## Purpose

This note translates the current RH/scattering architecture into a concrete ML system design without modifying the scientific stack itself.

The rule is strict:

- the dyadic/ghost backend stays frozen,
- the corrected coefficient table stays frozen,
- the A3 frontend stays frozen,
- the anchored completion law stays frozen,
- the completed determinant object stays frozen.

The learned system is allowed to learn only how to:

- map language or structured problems into oracle queries,
- read back oracle summaries,
- and use those summaries to improve reasoning-time selection.

That keeps the repo's scientific object intact and makes the inductive-bias claim testable.

This spec is anchored-pipeline only. Historical inductive-bias memos are background material, not implementation inputs. The implementation target here is only the latest canonical stack built around:

- `A3_exact_frontend`,
- anchored completion with `completion_even_mode=anchored_real`,
- the canonical quartic default `completion_even_a4=-0.2`,
- and the corresponding anchored output artifacts under `out/`.

## High-level architecture

The proposed reasoning system is:

$$
\boxed{
\text{language encoder}
\to
\text{latent translator}
\to
\text{frozen structured oracle}
\to
\text{reasoning controller / reranker}
}
$$

Expanded pipeline:

1. A language model encodes the problem and produces candidate reasoning traces.
2. A learned translator converts each trace into a small oracle query.
3. The frozen oracle evaluates structured consistency features.
4. A reranker or controller scores the trace using both text and oracle features.
5. The top-ranked trace is selected, or generation continues using the score as guidance.

## The frozen oracle

The oracle is the repo's existing fixed stack:

1. Frozen backend coefficients.
   Source: `out/corrected_factor_injection_beta23_window_coefficients.csv`.
   Accessor: `tools/corrected_backend_interface.py`.

2. Canonical frontend realization.
   Default family: `A3_exact_frontend`.
   Driver: `tools/probe_frontend_realization.py`.

3. Completed global object.
   Driver: `tools/probe_completed_global_object.py`.

4. Anchored completion and characterization.
   Main evaluator: `tools/characterize_completed_object.py`.
   Supporting probes:
   - `tools/densify_local_zero_scan.py`
   - `tools/investigate_completion_layer.py`
   - `tools/search_selfdual_completion_templates.py`
   - `tools/derive_anchor_completion_weight.py`
   - `tools/search_anchor_lambda_family.py`

Operationally, the oracle should be treated as a deterministic scientific prior with versioned feature outputs.

The canonical anchored artifact set for the first ML wrapper is:

- `out/completed_global_object_anchored_default_summary.csv`
- `out/completed_global_object_anchored_default_stability.csv`
- `out/analytic_characterization_a3default_quartic_a4m02_anchored_fe_summary.csv`
- `out/analytic_characterization_a3default_quartic_a4m02_anchored_sigma_scan_summary.csv`
- `out/analytic_characterization_a3default_quartic_a4m02_anchored_rigidity.csv`
- `out/analytic_characterization_a3default_quartic_a4m02_anchored_stability.csv`
- `out/analytic_characterization_a3default_quartic_a4m02_anchored_zero_tracking.csv`

## ML module breakdown

### Module A: Language Encoder

Use a standard transformer or frozen LLM.

Responsibilities:

- encode the problem statement,
- encode candidate reasoning traces,
- optionally generate multiple candidate chains,
- produce a fixed trace embedding for reranking.

This module should not learn the spectral or arithmetic structure directly.

Recommended first version:

- frozen encoder or lightly fine-tuned small transformer,
- output one embedding per candidate trace,
- no dependency on raw oracle internals.

### Module B: Latent Translator

This is the main learned bridge between language and the oracle.

Responsibilities:

- choose which oracle operating point to inspect,
- choose which summary family to request,
- optionally choose which local windows or perturbation probes to query.

Recommended initial outputs:

- `u_choice` from a small discrete set such as `{0.16, ..., 0.24}`,
- `sigma_family` selector such as `critical_only`, `fe_surface`, or `sigma_scan`,
- `feature_mask` over summary groups,
- `cluster_window_id` for which local zero window to inspect.

Important restriction:

- this module may select queries,
- it may not rewrite coefficients,
- it may not mutate the oracle,
- it may not backprop through the oracle.

### Module C: Frozen Structured Oracle

The oracle accepts a compact query and returns a curated feature vector.

Recommended query schema:

```json
{
  "u": 0.24,
  "feature_families": ["closure", "spectral", "global", "stability"],
  "sigma_mode": "critical_only",
  "cluster_window": "canonical_t28",
  "include_perturbation_features": true
}
```

Recommended output schema:

```json
{
  "u": 0.24,
  "closure.squarefree": 0.0,
  "closure.primepower": 0.0,
  "closure.mixed": 0.0,
  "spectral.radius": 0.91,
  "spectral.packet_rigidity": 0.07,
  "spectral.err1": 0.003,
  "spectral.err2": 0.018,
  "spectral.err3": 0.021,
  "global.fe_rel": 0.012,
  "global.fe_log_rel": 0.004,
  "global.zero_t": 27.95,
  "global.zero_depth": 0.16,
  "global.critical_gap": 0.0,
  "completion.a2": 0.0,
  "completion.a4": -0.2,
  "completion.a6": 0.0,
  "completion.a8": 0.0,
  "stability.window_spread": 0.03,
  "stability.phase_sensitivity": 0.14,
  "stability.radius_sensitivity": 0.02
}
```

The exact feature names can differ, but the feature families should be stable and interpretable.

### Module D: Reasoning Controller / Reranker

This is the first learned decision module that should deliver measurable payoff.

Inputs:

- trace embedding from Module A,
- oracle feature vector from Module C,
- optionally the translator state from Module B.

Outputs:

- scalar score for a candidate trace,
- optional binary accept/reject decision,
- optional next-step priority for closed-loop generation.

Recommended first implementation:

- concatenate text embedding and oracle features,
- pass through a small MLP,
- train with listwise or pairwise ranking loss.

## File boundaries for a first implementation

The cleanest first build is a separate ML layer that wraps existing tools rather than rewriting them.

Suggested new files:

- `src/ml_oracle/oracle_schema.py`
  Defines typed query and response dataclasses.

- `src/ml_oracle/frozen_oracle_client.py`
  Thin wrapper over existing probe outputs and summary loaders.

- `src/ml_oracle/feature_registry.py`
  Maps raw probe summaries into a stable feature vector.

- `src/ml_oracle/translator.py`
  Learned or heuristic translator from trace embeddings to oracle queries.

- `src/ml_oracle/reranker.py`
  MLP or shallow transformer head that scores candidate traces.

- `src/ml_oracle/datasets.py`
  Candidate-trace dataset and collation logic.

- `src/ml_oracle/train_reranker.py`
  First training loop.

- `src/ml_oracle/eval_reranker.py`
  Offline benchmark evaluation against a no-oracle baseline.

- `tools/export_oracle_feature_table.py`
  Utility to precompute oracle summaries for the allowed query grid.

This keeps the current scientific pipeline in `tools/` and places the learned ML layer under `src/ml_oracle/`.

## Oracle feature families

The first system should consume only curated summaries, not raw determinant tracks.

### Closure features

- squarefree closure error,
- prime-power closure error,
- mixed closure error,
- local factor fit residuals.

Interpretation:

- does the candidate trace align with primitive-to-composite structure,
- or is it forcing an incoherent factorization.

### Spectral features

- packet rigidity,
- spectral radius,
- frontend coefficient fit errors,
- eigenvalue geometry summaries.

Interpretation:

- does the trace correspond to a compact, stable local realization.

### Global analytic features

- FE defect summaries,
- best zero-candidate depth,
- best zero-candidate location,
- transverse critical-line gap,
- completion curvature summaries.

Interpretation:

- does the candidate trace assemble into a globally coherent explanation.

### Stability features

- sensitivity across neighboring `u`,
- local window persistence,
- perturbation sensitivity under radius and phase shifts.

Interpretation:

- is the trace relying on brittle structure or stable structure.

## Recommended benchmark path

Do not begin with open-domain chat.

Start with tasks where:

- multiple candidate traces can be generated,
- correctness is easy to score,
- global consistency matters.

Recommended first benchmarks:

1. symbolic equation solving,
2. arithmetic or GSM8K-style reranking,
3. program execution trace verification,
4. graph path or combinatorial reasoning,
5. theorem-step verification on short formal chains.

The key requirement is candidate diversity plus objective correctness labels.

## Minimal experiment

### Goal

Test whether frozen-oracle features improve selection of correct reasoning traces.

### Baseline

- language-model trace embedding,
- reranker trained only on text features.

### Proposed system

- same text embedding,
- same reranker size,
- plus oracle feature vector.

### Success criterion

- higher top-1 selection accuracy among candidate traces,
- especially on multi-step problems where local plausibility is insufficient.

## First training loop

The first version should be offline reranking, not closed-loop generation.

### Data format

Each training item should contain:

- a problem statement,
- `K` candidate reasoning traces,
- one or more correctness labels,
- precomputed oracle queries and feature vectors for each candidate.

Suggested record format:

```json
{
  "problem_id": "ex-001",
  "prompt": "Solve the equation ...",
  "candidates": [
    {
      "text": "candidate reasoning trace 1",
      "label": 1,
      "oracle_query": {"u": 0.24, "feature_families": ["global", "spectral"]},
      "oracle_features": [0.12, 0.03, 0.91, 0.0, 27.95]
    },
    {
      "text": "candidate reasoning trace 2",
      "label": 0,
      "oracle_query": {"u": 0.18, "feature_families": ["global", "spectral"]},
      "oracle_features": [0.23, 0.11, 0.96, 0.04, 28.07]
    }
  ]
}
```

### Training loop sketch

```python
for batch in dataloader:
    text_emb = language_encoder(batch.candidate_texts)
    oracle_feat = batch.oracle_features
    joint = concatenate(text_emb, oracle_feat)
    scores = reranker(joint)
    loss = ranking_loss(scores, batch.labels, group_ids=batch.problem_ids)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Recommended losses

- pairwise hinge loss,
- listwise softmax cross-entropy,
- or margin ranking loss.

The first pass should avoid complicated reinforcement objectives.

## Versioned query policy

To keep the experiment interpretable, begin with a finite query grid.

Example grid:

- `u` in `{0.16, 0.18, 0.21, 0.24}`,
- `feature_families` in a small fixed set,
- one canonical local cluster window,
- one canonical completion mode.

This has two advantages:

- oracle calls can be precomputed,
- improvements are easier to attribute.

Only after reranking works should the translator become a learned selector.

## Suggested development phases

### Phase ML-1: Oracle wrapper

Build a stable Python wrapper around the current frozen architecture.

Deliverables:

- query dataclass,
- response dataclass,
- feature registry,
- precomputation script for the allowed query grid.

### Phase ML-2: Candidate reranker

Train a small reranker using:

- text embeddings,
- frozen oracle features,
- correctness labels over candidate traces.

Deliverable:

- comparison against text-only baseline.

### Phase ML-3: Learned translator

Replace manual query selection with a small learned selector over:

- `u`,
- feature subsets,
- local windows.

Still keep the oracle frozen.

### Phase ML-4: Closed-loop controller

Use reranker scores during generation:

- propose next-step candidates,
- query oracle features,
- keep coherent branches,
- prune inconsistent branches.

This is the first real reasoning-controller version.

## Guardrails

To preserve the scientific architecture, the following constraints should be treated as non-negotiable:

1. Never backpropagate into the frozen oracle.
2. Never let the learned system rewrite backend coefficients.
3. Never use raw oracle outputs as plain-text supervision targets.
4. Keep oracle feature names versioned and documented.
5. Always compare to a no-oracle baseline.
6. Keep the first experiments on structured benchmarks with objective labels.

## Why this is a serious inductive-bias test

This design tests a specific claim:

$$
\text{reasoning quality improves when candidate explanations are filtered by a fixed local-to-global consistency prior.}
$$

The prior here is unusual because it combines:

- additive latent structure,
- multiplicative lift,
- packetized local realization,
- and determinant-style global coherence.

That is a concrete, falsifiable ML hypothesis even if the RH program itself remains mathematically separate.

## Practical first target

The best first paper-scale experiment is:

- baseline LLM candidate reranker,
- versus the same reranker augmented with frozen-oracle features,
- evaluated on a structured reasoning benchmark with multiple candidate traces.

If the oracle-augmented system wins on trace selection, that is already evidence that the architecture supplies a nontrivial reasoning inductive bias.

## Smoke-test path in this repo

The current scaffold includes a small anchored-only sample dataset at:

- `configs/anchored_oracle_reranker_smoke.jsonl`

and the first export/train/eval commands are:

```bash
python tools/export_oracle_feature_table.py --out_csv out/anchored_oracle_feature_table.csv
python -m src.ml_oracle.train_reranker --dataset configs/anchored_oracle_reranker_smoke.jsonl --out_model out/ml_oracle_smoke_reranker.npz
python -m src.ml_oracle.eval_reranker --dataset configs/anchored_oracle_reranker_smoke.jsonl --model out/ml_oracle_smoke_reranker.npz
```

Those commands are intended only as a reproducible smoke test for the anchored A3 oracle wrapper and reranker path. They are not a scientific benchmark.

The repo now also includes a fixed tiny split for explicit mode comparison:

- `configs/anchored_oracle_reranker_smoke_train.jsonl`
- `configs/anchored_oracle_reranker_smoke_eval.jsonl`

and a comparison runner:

```bash
python -m src.ml_oracle.compare_modes \
  --train_dataset configs/anchored_oracle_reranker_smoke_train.jsonl \
  --eval_dataset configs/anchored_oracle_reranker_smoke_eval.jsonl \
  --out_csv out/ml_oracle_mode_comparison.csv
```

That runner trains and evaluates three matched rerankers on the same split:

- `text`
- `oracle`
- `text+oracle`

This is the first direct utility comparison for the anchored frozen-oracle path.

The repo also now supports anchored feature-family ablations through:

```bash
python -m src.ml_oracle.compare_ablations \
  --train_dataset configs/anchored_oracle_reranker_smoke_train.jsonl \
  --eval_dataset configs/anchored_oracle_reranker_smoke_eval.jsonl \
  --out_csv out/ml_oracle_ablation_comparison.csv
```

The current ablation families are:

- `closure`
- `packet`
- `zero`
- `fe_stability`

and the ablation runner evaluates each family alone, selected cumulative combinations, and matched `text+oracle` combinations on the same fixed split.

## Next benchmark package

The next serious step is now documented in:

- `notes/ANCHORED_ORACLE_BENCHMARK_TEMPLATE.md`
- `configs/anchored_oracle_benchmark_manifest.csv`

This package defines a 50-group benchmark template built around near-miss candidate traces where text plausibility and deeper consistency come apart. That is the intended regime for the anchored frozen oracle.