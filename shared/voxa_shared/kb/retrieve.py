"""Hybrid retrieval: Postgres full-text (BM25-style ts_rank_cd) + pgvector cosine -> RRF -> cross-encoder."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass

from ..db import tenant_conn
from .embed import Embedder, Reranker

_WORD = re.compile(r"[A-Za-z0-9]{2,}")
_STOP = frozenset("the a an and or of to in on for with is are was were be been it its this that what how do does "
                  "can i you your my me we our they their at as by from about please tell".split())


@dataclass
class Hit:
    id: int
    content: str
    score: float


def or_query(text: str) -> str:
    """'what is your return policy' -> 'return | policy' (natural questions are rarely AND-matchable)."""
    words = [w.lower() for w in _WORD.findall(text) if w.lower() not in _STOP]
    return " | ".join(dict.fromkeys(words)) or ""


def sanitize_chunk(text: str) -> str:
    """Retrieved text is DATA. Neutralise anything that could close our delimiter or fake a role."""
    text = re.sub(r"</?\s*knowledge_base_excerpts?[^>]*>", "", text, flags=re.I)
    text = re.sub(r"(?im)^\s*(system|assistant|user)\s*:", r"\1 -", text)
    return text.strip()


class Retriever:
    def __init__(self, pool, embedder: Embedder, reranker: Reranker | None, redis=None,
                 cache_ttl_s: int = 300, candidates: int = 20, fused: int = 8, top_k: int = 2) -> None:
        self.pool, self.emb, self.rr, self.redis = pool, embedder, reranker, redis
        self.ttl, self.cand, self.fused, self.top_k = cache_ttl_s, candidates, fused, top_k
        self.last_ms = 0.0

    @staticmethod
    def _key(tenant_id: str, query: str) -> str:
        norm = " ".join(w.lower() for w in _WORD.findall(query))
        return f"kb:{tenant_id}:{hashlib.sha1(norm.encode()).hexdigest()}"

    async def search(self, tenant_id: str, query: str) -> list[Hit]:
        t0 = time.perf_counter()
        key = self._key(tenant_id, query)
        if self.redis:
            cached = await self.redis.get(key)
            if cached:
                self.last_ms = (time.perf_counter() - t0) * 1000
                return [Hit(**h) for h in json.loads(cached)]
        qv = await self.emb.aembed_query(query)
        tsq = or_query(query)

        async def vec():
            async with tenant_conn(self.pool, tenant_id) as c:
                return await c.fetch(
                    "SELECT id, content FROM kb_chunks WHERE tenant_id=$1 "
                    "ORDER BY embedding <=> $2 LIMIT $3", tenant_id, qv, self.cand)

        async def fts():
            if not tsq:
                return []
            async with tenant_conn(self.pool, tenant_id) as c:
                return await c.fetch(
                    "SELECT id, content FROM kb_chunks WHERE tenant_id=$1 AND tsv @@ to_tsquery('english',$2) "
                    "ORDER BY ts_rank_cd(tsv, to_tsquery('english',$2)) DESC LIMIT $3", tenant_id, tsq, self.cand)

        v, f = await asyncio.gather(vec(), fts())
        scores: dict[int, float] = {}
        text: dict[int, str] = {}
        for rows in (v, f):                                   # reciprocal rank fusion
            for rank, r in enumerate(rows):
                scores[r["id"]] = scores.get(r["id"], 0.0) + 1.0 / (60 + rank)
                text[r["id"]] = r["content"]
        top = sorted(scores, key=scores.get, reverse=True)[: self.fused]
        if self.rr and len(top) > 1:
            rs = await asyncio.to_thread(self.rr.rerank, query, [text[i] for i in top])
            order = sorted(zip(top, rs), key=lambda x: x[1], reverse=True)[: self.top_k]
            hits = [Hit(i, text[i], s) for i, s in order]
        else:
            hits = [Hit(i, text[i], scores[i]) for i in top[: self.top_k]]
        if self.redis and hits:
            await self.redis.set(key, json.dumps([h.__dict__ for h in hits]), ex=self.ttl)
        self.last_ms = (time.perf_counter() - t0) * 1000
        return hits


def format_context(hits: list[Hit], max_chars: int = 1300) -> str:
    """Block injected before the user turn. The system prompt tells the model it is untrusted data."""
    if not hits:
        return "<knowledge_base_excerpts>\n(no relevant excerpts found)\n</knowledge_base_excerpts>"
    body, used = [], 0
    for i, h in enumerate(hits, 1):
        t = sanitize_chunk(h.content)[: max_chars - used]
        if not t:
            break
        body.append(f"[{i}] {t}")
        used += len(t)
    return "<knowledge_base_excerpts>\n" + "\n\n".join(body) + "\n</knowledge_base_excerpts>"


async def ingest_documents(pool, embedder: Embedder, tenant_id: str, docs, batch: int = 32) -> dict[str, int]:
    """Chunk, embed, store. Returns counts. Runs in the API container."""
    from .chunking import chunk_faq, chunk_text

    n_docs = n_chunks = 0
    for d in docs:
        chunks = chunk_faq(d.faq) if d.faq else chunk_text(d.text)
        if not chunks:
            continue
        vecs = []
        for i in range(0, len(chunks), batch):
            vecs += await asyncio.to_thread(embedder.embed, chunks[i:i + batch])
        async with tenant_conn(pool, tenant_id) as c:
            doc_id = await c.fetchval(
                "INSERT INTO kb_documents(tenant_id, source, title) VALUES($1,$2,$3) RETURNING id",
                tenant_id, d.source, d.title)
            await c.executemany(
                "INSERT INTO kb_chunks(tenant_id, document_id, ord, content, embedding) VALUES($1,$2,$3,$4,$5)",
                [(tenant_id, doc_id, i, ch, v) for i, (ch, v) in enumerate(zip(chunks, vecs))])
        n_docs += 1
        n_chunks += len(chunks)
    return {"documents": n_docs, "chunks": n_chunks}
