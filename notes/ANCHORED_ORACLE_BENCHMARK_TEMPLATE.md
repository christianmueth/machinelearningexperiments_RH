# Anchored Oracle Benchmark Template

## Purpose

This note defines the next benchmark target for the frozen anchored oracle ML path.

The goal is not to make the oracle look good on easy text tasks. The goal is to construct a grouped-candidate benchmark where:

- text-only reranking is not saturated,
- wrong candidates are locally plausible,
- and the distinction between candidates is primarily about deeper consistency rather than surface wording.

That is the regime where the anchored oracle has a legitimate chance to add value.

## Benchmark principle

Each example should be a grouped-candidate item with:

1. one problem prompt,
2. three to five candidate reasoning traces,
3. exactly one preferred candidate,
4. at least one wrong candidate that is linguistically plausible,
5. a clear latent consistency failure mode for each negative.

The negatives should not be obviously wrong from wording alone.

## What this benchmark is testing

The benchmark is specifically testing the claim:

$$
\text{text candidates} + \text{anchored frozen-oracle features} \to \text{better selection of globally coherent traces.}
$$

It is not testing open-domain QA, style quality, or memorized textbook correctness.

## Candidate design rules

Each grouped example should contain these candidate types when possible:

1. Gold coherent trace.
   Correct and globally consistent.

2. Local-plausibility failure.
   The trace looks good line by line but breaks a later constraint or fails to assemble globally.

3. Shortcut failure.
   The trace jumps to a convenient local pattern without preserving the full constraint set.

4. Instability failure.
   The trace relies on a brittle branch that would fail under small perturbation or alternate but nearby framing.

5. Primitive/composite confusion failure.
   The trace reuses derived structure as if it were primitive, or conflates repeated evidence with independent support.

If an example only supports three candidates, prioritize `gold coherent`, `local-plausibility failure`, and `shortcut failure`.

## Benchmark shape

The first serious version should contain 50 grouped examples.

Recommended split:

- 30 train groups,
- 10 development groups,
- 10 evaluation groups.

Recommended candidate count:

- 4 candidates per group on average.

That yields roughly 200 candidate traces total, which is enough for the first useful comparison without turning into a full data-engineering project.

## Problem families

Use five families with 10 examples each.

### Family A: Symbolic equation consistency

Task shape:

- solve or transform an equation,
- include candidate derivations that differ by hidden invalid transformations,
- make the wrong derivations look algebraically plausible.

What the oracle might help with:

- rejecting traces that are locally neat but globally inconsistent.

### Family B: Constraint satisfaction / logic chains

Task shape:

- short logic or rule systems,
- multiple candidate chains of inference,
- some candidates satisfy most local rules but violate a global constraint.

What the oracle might help with:

- consistency and stability rather than token-level plausibility.

### Family C: Program execution traces

Task shape:

- small pseudocode or simple Python-style execution,
- multiple candidate trace explanations,
- negatives should contain subtle state-tracking mistakes.

What the oracle might help with:

- preserving coherent latent state across multiple steps.

### Family D: Graph or planning near-miss traces

Task shape:

- shortest-path, scheduling, or multi-step planning,
- wrong candidates take plausible but globally invalid routes.

What the oracle might help with:

- decomposing local factors while preserving whole-plan coherence.

### Family E: Proof-step verification

Task shape:

- short theorem or lemma-step selection,
- candidate proof sketches where negatives use plausible but non-permitted transitions.

What the oracle might help with:

- distinguishing allowed global structure from locally persuasive but invalid jumps.

## Annotation schema

Each group should be annotated with:

- `problem_id`
- `family`
- `difficulty`
- `prompt`
- `gold_candidate_index`
- `candidate_type` for each trace
- `failure_mode` for each negative
- `oracle_relevance` on a `low/medium/high` scale

Recommended negative failure-mode tags:

- `global_constraint_break`
- `later_state_mismatch`
- `shortcut_without_justification`
- `primitive_composite_confusion`
- `branch_instability`
- `symmetry_break`
- `goal_drift`

## Oracle relevance rubric

This should be annotated before training.

### High

Two or more candidates are textually plausible, but one is clearly better only because it preserves deeper consistency across steps.

### Medium

The correct trace is textually somewhat better, but there is still a real latent consistency distinction.

### Low

The correct trace is obviously better from wording or direct local correctness alone.

For the first benchmark, prioritize `high` and `medium`. Avoid filling the dataset with `low` items.

## Query policy for the frozen oracle

Do not broaden the oracle yet.

Use a fixed anchored query policy for the first real benchmark:

- `u` in a small discrete set such as `{0.16, 0.21, 0.24}`,
- canonical `A3_exact_frontend`,
- anchored completion only,
- fixed feature groups: `closure`, `packet`, `zero`, `fe_stability`.

The translator may choose among these, but the oracle itself stays fixed.

## Evaluation plan

The permanent comparison frame should be:

1. `text`
2. `oracle`
3. `text+oracle`

and then feature-family ablations inside the oracle path.

Primary metrics:

- grouped top-1 accuracy,
- per-family grouped accuracy,
- accuracy on `high oracle_relevance` subset.

Secondary metrics:

- calibration gap between score and correctness,
- gain over text-only on near-miss subsets,
- gain over text-only on instability-tagged negatives.

## Success condition

The first real positive result is either:

1. `text+oracle` beats `text` on the full evaluation split, or
2. at least one anchored oracle feature family beats `text` on the `high oracle_relevance` subset.

That is enough to justify the inductive-bias claim.

## What not to do yet

Do not do any of the following before this benchmark exists:

- do not widen the frozen oracle,
- do not add new oracle variants,
- do not train end-to-end through the oracle,
- do not change the canonical A3 anchored pipeline,
- do not interpret smoke-test failures as architecture failures.

The next work is benchmark/interface design, not architecture mutation.

## Dataset record format

Recommended JSONL shape:

```json
{
  "problem_id": "bench-001",
  "family": "symbolic_equation",
  "difficulty": "medium",
  "oracle_relevance": "high",
  "prompt": "...",
  "candidates": [
    {
      "text": "candidate trace 1",
      "label": 1,
      "candidate_type": "gold_coherent",
      "failure_mode": "",
      "oracle_query": {
        "u": 0.24,
        "feature_families": ["closure", "packet", "zero", "fe_stability"],
        "sigma_mode": "anchored_default",
        "cluster_window": "canonical_t28",
        "include_perturbation_features": true,
        "pipeline_tag": "anchored_a3_v1"
      }
    },
    {
      "text": "candidate trace 2",
      "label": 0,
      "candidate_type": "local_plausibility_failure",
      "failure_mode": "global_constraint_break",
      "oracle_query": {
        "u": 0.24,
        "feature_families": ["closure", "packet", "zero", "fe_stability"],
        "sigma_mode": "anchored_default",
        "cluster_window": "canonical_t28",
        "include_perturbation_features": true,
        "pipeline_tag": "anchored_a3_v1"
      }
    }
  ]
}
```

## Practical construction rule

If a candidate can be rejected instantly by a strong text-only model on wording alone, it is a weak benchmark item.

If two candidates both look reasonable in language but one breaks the deeper structure, it is a strong benchmark item.

That is the benchmark regime this anchored oracle is supposed to address.