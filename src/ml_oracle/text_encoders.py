from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np


def hashed_text_embedding(text: str, *, dim: int) -> np.ndarray:
    vec = np.zeros((int(dim),), dtype=np.float64)
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    if not tokens:
        return vec
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], byteorder="little", signed=False) % int(dim)
        sign = 1.0 if (digest[4] % 2 == 0) else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 0.0:
        vec /= norm
    return vec


@dataclass
class TextEncoder:
    name: str
    output_dim: int

    def encode(self, text: str) -> np.ndarray:
        raise NotImplementedError


@dataclass
class HashedTextEncoder(TextEncoder):
    dim: int = 256

    def __init__(self, *, dim: int = 256) -> None:
        super().__init__(name="hashed", output_dim=int(dim))
        self.dim = int(dim)

    def encode(self, text: str) -> np.ndarray:
        return hashed_text_embedding(text, dim=self.dim)


@dataclass
class HuggingFaceTextEncoder(TextEncoder):
    model_name: str
    max_length: int = 256

    def __init__(self, *, model_name: str, max_length: int = 256) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Hugging Face encoder requested but torch/transformers is not installed. Install ai/requirements-training.txt first."
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()
        hidden_size = int(getattr(self._model.config, "hidden_size", 0) or 0)
        if hidden_size <= 0:
            raise ValueError(f"unable to determine hidden size for model {model_name}")
        super().__init__(name=f"hf:{model_name}", output_dim=hidden_size)
        self.model_name = str(model_name)
        self.max_length = int(max_length)

    def encode(self, text: str) -> np.ndarray:
        batch = self._tokenizer(
            str(text),
            return_tensors="pt",
            truncation=True,
            max_length=int(self.max_length),
            padding=False,
        )
        with self._torch.no_grad():
            outputs = self._model(**batch)
            hidden = outputs.last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return pooled[0].detach().cpu().numpy().astype(np.float64)


def build_text_encoder(*, text_encoder: str = "hashed", text_dim: int = 256, hf_model: str = "", max_length: int = 256) -> TextEncoder:
    key = str(text_encoder).strip().lower()
    if key == "hashed":
        return HashedTextEncoder(dim=int(text_dim))
    if key in {"hf", "huggingface", "transformers"}:
        model_name = str(hf_model).strip()
        if not model_name:
            raise ValueError("hf_model must be provided when text_encoder=hf")
        return HuggingFaceTextEncoder(model_name=model_name, max_length=int(max_length))
    raise ValueError("text_encoder must be one of: hashed, hf")