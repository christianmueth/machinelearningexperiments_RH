# AI Benchmarks

This directory defines the benchmark surface for the AI layer that wraps the frozen anchored oracle.

## Standard benchmarks

Use these as the first external evaluation set:

| Benchmark | Purpose | Metric |
| --- | --- | --- |
| GSM8K | grade-school math reasoning | accuracy |
| MATH | harder mathematical reasoning | accuracy |
| ProofWriter | logical and multi-hop reasoning | accuracy |
| HumanEval | program synthesis / code reasoning | pass@k |

Recommended expansion set after the first pass:

- AQuA-RAT
- CLUTRR
- MiniF2F
- MBPP
- ARC
- OGB graph reasoning tasks

## Custom reranking benchmark

The most important repo-specific benchmark is grouped candidate-trace reranking.

Each example should contain:

- one prompt or problem,
- several candidate traces,
- one best trace label,
- optional precomputed oracle queries or oracle feature vectors.

The current loader is `src/ml_oracle/datasets.py` and the template file is [ai/datasets/templates/reasoning_rerank_template.jsonl](ai/datasets/templates/reasoning_rerank_template.jsonl).

Primary metrics:

- Top-1 trace accuracy,
- MRR,
- nDCG.

The current evaluator in `src/ml_oracle/eval_reranker.py` reports all three metrics directly.

Minimum baseline comparisons:

- text only,
- oracle only,
- text plus oracle.

## Out-of-domain evaluation

After the first in-domain result, add at least three stress cases:

- unseen reasoning domains,
- longer reasoning traces,
- different reasoning styles such as symbolic versus code versus arithmetic.

The key question is not just whether the oracle helps average accuracy. The key question is whether it improves robustness once problem type or trace length shifts.

Current caution from the repo's present synthetic GSM8K candidate generator:

- text-only and text-plus-oracle baselines can saturate because the distractors are still simple,
- so this benchmark currently validates infrastructure more than scientific oracle lift,
- and harder trace generation is the next necessary step.