# Remote GPU Setup

This repo is ready to use a remote GPU machine without changing the scientific workflow.

The goal is simple:

- keep the frozen beta23 oracle unchanged,
- keep the reranker and viability tooling unchanged,
- move only candidate generation to faster hardware.

## What To Persist On The Remote Machine

Keep these paths on persistent disk if the provider supports it:

- the repo checkout
- the Hugging Face cache directory
- `out/ai/`
- `ai/datasets/processed/`

That prevents repeated model downloads and preserves chunk summaries between sessions.

## Recommended Remote Workflow

1. Create a remote Linux GPU machine with persistent storage.
2. Install Python 3.11 and git.
3. Clone this repo.
4. Create a venv and install dependencies.
5. Run the GPU smoke test before any generation job.
6. Resume generation only through chunked scripts that write summary artifacts after each chunk.

## Minimal Bootstrap Commands

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r ai/requirements-training.txt
```

If the remote machine already has a provider-specific CUDA PyTorch build, keep it. Do not replace a working GPU torch install just to match local CPU behavior.

## First Smoke Test

Run:

```bash
python ai/scripts/gpu_smoke_test.py --model Qwen/Qwen2.5-0.5B-Instruct --load_model
```

This checks:

- torch import
- CUDA visibility
- device name and memory
- optional tokenizer/model load timing

## First Remote Generation Sequence

Use the strict instruct-aligned generator regime that performed best locally:

```bash
python ai/scripts/generate_candidate_traces.py \
  --input_jsonl ai/datasets/processed/gsm8k_qwen_probe25_source.jsonl \
  --prompt_field question \
  --answer_field answer \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --out_jsonl ai/datasets/processed/gsm8k_qwen25_probe25_strict_chunks.jsonl \
  --out_config_json out/ai/candidate_generation/gsm8k_qwen25_remote_chunk0_config.json \
  --out_summary_json out/ai/candidate_generation/gsm8k_qwen25_remote_chunk0_summary.json \
  --start_index 0 \
  --limit 10 \
  --num_candidates 3 \
  --max_input_length 256 \
  --max_new_tokens 96 \
  --temperature 0.7 \
  --top_p 0.9 \
  --seed 7 \
  --flush_every 1 \
  --prompt_style strict \
  --use_chat_template
```

Then summarize immediately:

```bash
python ai/scripts/summarize_candidate_groups.py \
  --input_jsonl ai/datasets/processed/gsm8k_qwen25_probe25_strict_chunks.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset gsm8k_probe25_remote_chunk0 \
  --output_json out/ai/dataset_summaries/gsm8k_qwen25_remote_chunk0_summary.json
```

Then inspect a sample:

```bash
python ai/scripts/inspect_candidate_groups.py \
  --input_jsonl ai/datasets/processed/gsm8k_qwen25_probe25_strict_chunks.jsonl \
  --limit 10 \
  --seed 7 \
  --output_json out/ai/hardness_samples/gsm8k_qwen25_remote_chunk0_sample.json
```

## Scientific Guardrails

Do not change these while moving to remote GPU:

- frozen beta23 backend
- oracle feature definitions
- reranker architecture
- viability thresholds

The remote GPU is a hardware substitution, not an experimental redesign.

## Viability Thresholds

Minimum viable benchmark:

- positive coverage >= 0.35
- 1p2n coverage >= 0.25

Preferred benchmark:

- positive coverage >= 0.50
- 1p2n coverage >= 0.40

Do not start reranker evaluation until the generation run clears the minimum gate.

## Current Best Local Regime

The best generator regime discovered locally was:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `--prompt_style strict`
- `--use_chat_template`
- `--num_candidates 3`
- short outputs instead of long chain-of-thought

That regime improved small-sample viability, but it still failed the broader 25-row local gate on CPU. The next remote objective is to test stronger instruct models faster under the same protocol.