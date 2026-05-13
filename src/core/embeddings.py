"""Embedding backends for dense skill retrieval."""

from __future__ import annotations

import hashlib
import math
from typing import Protocol


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class TextEmbedder(Protocol):
    model_name: str

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class NullEmbedder:
    model_name = "none"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[] for _ in texts]


class FakeSemanticEmbedder:
    """Small deterministic embedder for tests.

    It intentionally knows a few retrieval concepts so tests can exercise
    semantic matches without downloading a model.
    """

    model_name = "fake-semantic"

    _CONCEPT_ALIASES = {
        "pdf": {"pdf", "portable", "document", "file", "extractor"},
        "paper_analysis": {"paper", "academic", "article", "claim", "method", "finding", "contribution", "evidence"},
        "data": {"csv", "data", "dataframe", "table", "rows", "columns"},
        "chart": {"chart", "plot", "visualize", "graph"},
        "summary": {"summarize", "summary", "abstract", "condense"},
        "image": {"image", "screenshot", "ocr", "picture"},
    }

    def __init__(self, dimensions: int = 48):
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [_normalize(self._embed_one(text)) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0 for _ in range(self.dimensions)]
        tokens = _tokens(text)
        concept_tokens = set()
        for concept_index, aliases in enumerate(self._CONCEPT_ALIASES.values()):
            overlap = len(tokens & aliases)
            if overlap:
                vector[concept_index] += float(overlap)
                concept_tokens.update(tokens & aliases)
        for token in tokens:
            if token in concept_tokens:
                continue
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = 12 + digest[0] % max(self.dimensions - 12, 1)
            vector[index] += 0.05
        return vector


class LocalHFEmbedder:
    """Hugging Face encoder with mean pooling and L2-normalized outputs."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        batch_size: int = 8,
        max_length: int = 512,
        device: str | None = None,
    ):
        try:
            import torch  # type: ignore[import-not-found]
            import torch.nn.functional as torch_functional  # type: ignore[import-not-found]
            from transformers import AutoModel, AutoTokenizer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ValueError(
                "HF embedding search requires torch and transformers. "
                "Install with `uv sync --extra cpu` or `uv sync --extra cu128`."
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._torch = torch
        self._torch_functional = torch_functional
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        all_embeddings = []
        with self._torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                output = self.model(**encoded)
                token_embeddings = output.last_hidden_state
                attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                pooled = (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)
                pooled = self._torch_functional.normalize(pooled.float(), p=2, dim=1).cpu()
                all_embeddings.extend(pooled.tolist())
        return all_embeddings


def build_embedder(
    backend: str,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 8,
    max_length: int = 512,
    device: str | None = None,
) -> TextEmbedder:
    if backend == "none":
        return NullEmbedder()
    if backend == "fake":
        return FakeSemanticEmbedder()
    if backend == "hf-transformers":
        return LocalHFEmbedder(
            model_name=model_name,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )
    raise ValueError(f"Unsupported embedding backend: {backend}")


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    if dot <= 0:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _tokens(text: str) -> set[str]:
    normalized = "".join(char.lower() if char.isalnum() else " " for char in text)
    return {token for token in normalized.split() if token}
