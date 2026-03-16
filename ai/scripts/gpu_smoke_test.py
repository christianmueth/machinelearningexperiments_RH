from __future__ import annotations

import argparse
import json
import os
import time


def _device_summary() -> dict[str, object]:
    import torch

    summary: dict[str, object] = {
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        summary.update(
            {
                "current_device": int(index),
                "device_name": str(torch.cuda.get_device_name(index)),
                "total_memory_gb": round(float(props.total_memory) / (1024**3), 2),
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            }
        )
    return summary


def _load_model(model_name: str, *, local_files_only: bool) -> dict[str, object]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=bool(local_files_only))
    tokenizer_secs = time.perf_counter() - load_start

    model_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=bool(local_files_only))
    model_secs = time.perf_counter() - model_start

    device = "cuda" if torch.cuda.is_available() else "cpu"
    move_start = time.perf_counter()
    model.to(device)
    move_secs = time.perf_counter() - move_start
    model.eval()

    return {
        "model": str(model_name),
        "tokenizer_load_seconds": round(tokenizer_secs, 3),
        "model_load_seconds": round(model_secs, 3),
        "model_move_seconds": round(move_secs, 3),
        "device": device,
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0) or 0),
        "parameter_count": int(sum(param.numel() for param in model.parameters())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check remote GPU readiness for this repo's AI generation workflow.")
    parser.add_argument("--model", default="")
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    result: dict[str, object] = {
        "environment": _device_summary(),
    }

    if bool(args.load_model):
        if not str(args.model).strip():
            raise SystemExit("--model is required when --load_model is set")
        result["model_check"] = _load_model(str(args.model), local_files_only=bool(args.local_files_only))

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())