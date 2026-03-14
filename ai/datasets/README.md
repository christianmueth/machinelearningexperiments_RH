# AI Datasets

This directory is the staging area for AI datasets used with the frozen-oracle reranking stack.

Do not commit large raw corpora into git. Keep only small templates, schemas, and lightweight metadata here.

## Dataset layout

Recommended local layout:

```text
ai/datasets/
  README.md
  templates/
  raw/
  processed/
```

Suggested use:

- `templates/`: checked-in schema examples and small toy files,
- `raw/`: downloaded external datasets, kept out of git,
- `processed/`: transformed candidate-trace datasets for reranking experiments.

## Language competency corpora

Use these only to support fluency if needed. They are not the first place to spend effort.

- The Pile
  - https://pile.eleuther.ai/
  - Hugging Face: `datasets.load_dataset("EleutherAI/pile")`
- OpenWebText2
  - https://skylion007.github.io/OpenWebTextCorpus/
- RedPajama
  - https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T

## Math and reasoning datasets

- GSM8K
  - https://huggingface.co/datasets/gsm8k
  - Hugging Face: `datasets.load_dataset("gsm8k", "main")`
- MATH
  - https://huggingface.co/datasets/hendrycks/competition_math
- AQuA-RAT
  - https://huggingface.co/datasets/aqua_rat

## Symbolic and theorem-style reasoning datasets

- ProofWriter
  - https://huggingface.co/datasets/proofwriter
- CLUTRR
  - https://github.com/facebookresearch/clutrr
- MiniF2F
  - https://github.com/openai/miniF2F

## Program reasoning datasets

- HumanEval
  - https://huggingface.co/datasets/openai_humaneval
- MBPP
  - https://huggingface.co/datasets/mbpp

## Graph and planning datasets

- ARC
  - https://huggingface.co/datasets/ai2_arc
- OGB
  - https://ogb.stanford.edu/

## Custom reranking dataset

The first repo-native dataset should be a grouped candidate-trace file that matches the current loader in `src/ml_oracle/datasets.py`.

Each row should include:

- `problem_id`
- `prompt`
- `candidates`
  - `text`
  - `label`
  - optional `oracle_query`
  - optional `oracle_features`

Use [ai/datasets/templates/reasoning_rerank_template.jsonl](ai/datasets/templates/reasoning_rerank_template.jsonl) as the starting schema.

To convert a local JSONL file or a Hugging Face split into this grouped format, use [ai/scripts/prepare_reasoning_dataset.py](ai/scripts/prepare_reasoning_dataset.py).

The prep script intentionally shuffles candidate order so the gold trace is not always in position 1. That avoids a simple positional label leak in reranking benchmarks.