"""Local ONNX embeddings (fastembed, BAAI/bge-small-en-v1.5, 384-d) and cross-encoder reranker."""
from __future__ import annotations

import asyncio
from typing import Sequence

import numpy as np

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
DIM = 384


class Embedder:
    def __init__(self) -> None:
        from fastembed import TextEmbedding

        self._m = TextEmbedding(EMBED_MODEL)
        list(self._m.embed(["warmup"]))

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        return [np.asarray(v, dtype=np.float32) for v in self._m.embed(list(texts))]

    def embed_query(self, text: str) -> np.ndarray:
        # bge recommends an instruction prefix for short search queries
        return self.embed(["Represent this sentence for searching relevant passages: " + text])[0]

    async def aembed_query(self, text: str) -> np.ndarray:
        return await asyncio.to_thread(self.embed_query, text)


class Reranker:
    def __init__(self) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._m = TextCrossEncoder(RERANK_MODEL)
        list(self._m.rerank("warmup", ["warmup"]))

    def rerank(self, query: str, docs: Sequence[str]) -> list[float]:
        return [float(s) for s in self._m.rerank(query, list(docs))]
