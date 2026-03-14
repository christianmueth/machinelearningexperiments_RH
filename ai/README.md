# AI Workspace

This directory is the repo's separate working section for AI/ML development built on top of the frozen anchored RH/scattering architecture.

Scope for this section:

- build the AI layer around the existing oracle,
- keep the mathematical backend frozen,
- focus on offline datasets, reranking, and evaluation,
- ignore website and frontend deployment concerns for now.

The core engineering rule remains the same as the rest of the repo: learn around the oracle, not through it.

## What already exists in the repo

The main AI wrapper code is already in `src/ml_oracle/`:

- `src/ml_oracle/frozen_oracle_client.py`
- `src/ml_oracle/feature_registry.py`
- `src/ml_oracle/translator.py`
- `src/ml_oracle/datasets.py`
- `src/ml_oracle/reranker.py`
- `src/ml_oracle/train_reranker.py`
- `src/ml_oracle/eval_reranker.py`

This `ai/` directory is the operational workspace around those modules: requirements, benchmark definitions, dataset staging, and practical experiment configs.

## Recommended stack

For training and experimentation, use:

- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Accelerate
- Weights & Biases

The starter dependency file is [ai/requirements-training.txt](ai/requirements-training.txt).

## Suggested build architecture

The target AI pipeline is:

`problem -> candidate traces -> frozen oracle queries -> oracle features -> reranker/controller -> final answer`

Concretely:

1. Use a frozen or lightly tuned LLM to generate several candidate traces.
2. Translate each trace into an anchored-oracle query.
3. Pull oracle features from the frozen artifact layer.
4. Score candidates with a small reranker using text plus oracle features.
5. Compare against text-only and oracle-only baselines.

## MVP target

The first useful system should do only five things:

1. load a base reasoning model,
2. generate about 5 candidate traces,
3. compute oracle features for each trace,
4. rerank the traces,
5. return the top-ranked answer.

If `text+oracle` beats `text` alone on grouped reranking accuracy, that is a meaningful first result.

## Datasets

Dataset staging guidance and direct download locations live in [ai/datasets/README.md](ai/datasets/README.md).

The important dataset classes are:

- language fluency corpora,
- math and reasoning benchmarks,
- symbolic or theorem-style reasoning sets,
- program reasoning sets,
- custom candidate-trace reranking datasets.

The custom JSONL schema used by the current reranker code is shown in [ai/datasets/templates/reasoning_rerank_template.jsonl](ai/datasets/templates/reasoning_rerank_template.jsonl).

To prepare a larger grouped dataset from a local JSONL file or a Hugging Face dataset, use [ai/scripts/prepare_reasoning_dataset.py](ai/scripts/prepare_reasoning_dataset.py).

That prep script shuffles candidate order to avoid positional leakage from always writing the correct trace first.

## Benchmarks

Benchmark definitions and the recommended evaluation matrix live in [ai/benchmarks/README.md](ai/benchmarks/README.md).

The minimum benchmark set should include:

- GSM8K accuracy,
- MATH accuracy,
- ProofWriter accuracy,
- HumanEval pass@k,
- custom reranking metrics: Top-1, MRR, nDCG.

## Model backbones

Do not train a full language model from scratch here.

Recommended open starting points:

- Phi-3 Mini
- Mistral 7B
- Llama 3
- DistilBERT as a lighter frozen-encoder fallback for local validation

Practical default:

- start with Phi-3 Mini if you want faster local iteration,
- move to Mistral 7B once the reranking loop is stable.

Practical note from this repo's current environment:

- `distilbert-base-uncased` is the validated frozen-transformer path in the current Windows venv,
- `microsoft/Phi-3-mini-4k-instruct` downloads successfully here but crashes the Python process during `AutoModel.from_pretrained`, so it should be treated as a heavier optional target rather than the current default.

## First training loop

The current repo already supports a simple grouped candidate-trace reranker.

Example baseline commands:

```powershell
python ai/scripts/prepare_reasoning_dataset.py --hf_dataset gsm8k --hf_subset main --hf_split train --prompt_field question --answer_field answer --limit 200 --out_jsonl ai/datasets/processed/gsm8k_train_candidates.jsonl
python ai/scripts/prepare_reasoning_dataset.py --hf_dataset gsm8k --hf_subset main --hf_split test --prompt_field question --answer_field answer --limit 100 --out_jsonl ai/datasets/processed/gsm8k_test_candidates.jsonl
python src/ml_oracle/train_reranker.py --dataset ai/datasets/templates/reasoning_rerank_template.jsonl --out_model out/ai/reranker_text_only.npz --feature_mode text --epochs 100
python src/ml_oracle/train_reranker.py --dataset ai/datasets/templates/reasoning_rerank_template.jsonl --out_model out/ai/reranker_oracle_only.npz --feature_mode oracle --epochs 100
python src/ml_oracle/train_reranker.py --dataset ai/datasets/templates/reasoning_rerank_template.jsonl --out_model out/ai/reranker_text_oracle.npz --feature_mode text+oracle --epochs 100
python src/ml_oracle/train_reranker.py --dataset ai/datasets/processed/gsm8k_train_candidates.jsonl --out_model out/ai/reranker_distilbert_text_oracle.npz --feature_mode text+oracle --text_encoder hf --hf_model distilbert-base-uncased --epochs 30
```

Evaluation example:

```powershell
python src/ml_oracle/eval_reranker.py --dataset ai/datasets/templates/reasoning_rerank_template.jsonl --model out/ai/reranker_text_oracle.npz --feature_mode text+oracle
```

The evaluator now reports grouped Top-1 accuracy, MRR, and nDCG.

The text side supports two modes:

- `--text_encoder hashed` for the lightweight baseline,
- `--text_encoder hf --hf_model <model>` for a frozen Hugging Face encoder.

The Hugging Face path requires `torch` and `transformers` from [ai/requirements-training.txt](ai/requirements-training.txt).

## Current smoke-test results

On the current synthetic GSM8K-derived candidate benchmark built by [ai/scripts/prepare_reasoning_dataset.py](ai/scripts/prepare_reasoning_dataset.py):

- hashed `text` baseline on `200` train / `100` test examples: Top-1 `1.0`, MRR `1.0`, nDCG `1.0`
- hashed `oracle` baseline on the same split: Top-1 `0.21`, MRR `0.4925`, nDCG `0.618938`
- hashed `text+oracle` baseline on the same split: Top-1 `1.0`, MRR `1.0`, nDCG `1.0`
- frozen `distilbert-base-uncased` plus oracle on a smaller `20`/`20` GSM8K subset: Top-1 `1.0`, MRR `1.0`, nDCG `1.0`

Interpretation:

- the pipeline works,
- the metric layer is stable,
- but the current candidate generator is still too easy and too text-dominated to show meaningful oracle lift.

The next nontrivial milestone is a harder candidate generator or real candidate traces from an LLM, not another round of tuning on these synthetic distractors.

## Starter config

The starter experiment config is [ai/configs/reranker_mvp.yaml](ai/configs/reranker_mvp.yaml).

It captures:

- the frozen oracle operating point,
- the recommended feature groups,
- the first reranker hyperparameters,
- the baseline comparison targets.

## Near-term roadmap

Phase 1:

- build a small candidate-trace dataset in the current JSONL schema,
- run `text`, `oracle`, and `text+oracle` baselines,
- report grouped accuracy, MRR, and nDCG.

Phase 2:

- replace hashed text embeddings with embeddings from a frozen Hugging Face backbone,
- keep the oracle fixed,
- compare backbone-only versus backbone-plus-oracle.

Phase 3:

- add controlled out-of-domain evaluation,
- add longer-trace stress tests,
- study whether oracle features improve calibration and reranking robustness.

## Important constraints

- Do not backpropagate into the anchored oracle.
- Do not mutate the frozen backend or completion defaults during AI experiments.
- Do not start with giant datasets or full-model pretraining.
- Always compare against a no-oracle baseline.
- Treat oracle outputs as interpretable scientific features, not latent labels.