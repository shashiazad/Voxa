"""Streaming LLM interface + Groq (OpenAI-compatible) provider over a persistent HTTP/2 client."""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator

import re

import httpx

log = logging.getLogger("voxa.llm")


@dataclass
class LLMDelta:
    text: str = ""
    tool_call: dict[str, Any] | None = None  # {"id","name","arguments"(json str)} once complete
    finish: str | None = None


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
               max_tokens: int = 160) -> AsyncIterator[LLMDelta]:
        """Yield deltas as they arrive. Cancelling the consuming task closes the HTTP stream."""

    async def warm(self) -> None:
        """Open connections ahead of the first turn."""


class FallbackLLM(LLMProvider):
    """Ordered chain with per-provider cool-downs. A provider that fails *before producing output* (rate limit,
    5xx, timeout) is skipped for a while and the next one answers the same request, so one 429 never means silence.
    A failure after output has started cannot be spliced and is raised."""

    _COOLDOWNS: dict[str, float] = {}

    def __init__(self, providers: list[LLMProvider], cooldown_s: float = 20.0, cooldowns: dict[str, float] | None = None) -> None:
        self.providers, self.cooldown_s = providers, cooldown_s
        # shared by every call in this process: a rate limit on a provider is a property of the API key, not of one call
        self._until = FallbackLLM._COOLDOWNS if cooldowns is None else cooldowns
        self.served_by = ""
        self.name = "+".join(p.name for p in providers)

    async def warm(self) -> None:
        import asyncio

        await asyncio.gather(*(p.warm() for p in self.providers), return_exceptions=True)

    async def stream(self, messages, tools=None, max_tokens=160) -> AsyncIterator[LLMDelta]:
        import asyncio
        import time

        last: Exception | None = None
        for p in self.providers:
            if self._until.get(p.name, 0) > time.monotonic():
                continue
            started = False
            try:
                async for d in p.stream(messages, tools, max_tokens):
                    started = True
                    self.served_by = p.name
                    yield d
                return
            except (asyncio.CancelledError, GeneratorExit):
                raise
            except Exception as e:  # noqa: BLE001
                if started:
                    raise
                last = e
                self._until[p.name] = time.monotonic() + (self.cooldown_s if "429" in str(e) else 5.0)
                log.warning("llm %s failed before output (%s); falling back", p.name, str(e)[:80])
        # every provider cooling down: try the one whose cool-down ends first rather than failing outright
        if last is None and self.providers:
            async for d in min(self.providers, key=lambda p: self._until.get(p.name, 0)).stream(messages, tools, max_tokens):
                yield d
            return
        raise last or RuntimeError("no LLM available")


class GroqLLM(LLMProvider):
    name = "groq"

    def __init__(self, api_key: str, model: str = "qwen/qwen3.8-27b",
                 base_url: str = "https://api.groq.com/openai/v1", timeout_s: float = 8.0,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.model = model
        self.name = f"groq:{model}"
        self._client = httpx.AsyncClient(
            base_url=base_url, headers={"Authorization": f"Bearer {api_key}"}, http2=transport is None,
            timeout=httpx.Timeout(timeout_s, connect=2.0), transport=transport,
            limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=120))

    async def warm(self) -> None:
        try:
            await self._client.get("/models")  # TLS + HTTP/2 handshake
        except httpx.HTTPError as e:
            log.warning("llm warmup failed: %s", e)

    async def stream(self, messages, tools=None, max_tokens=160) -> AsyncIterator[LLMDelta]:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "stream": True,
                                "max_tokens": max_tokens, "temperature": 0.3}
        if tools:
            body["tools"] = tools
        if self.model.startswith("openai/gpt-oss"):
            body["reasoning_effort"] = "low"        # reasoning models: keep thinking short for voice latency
        elif self.model.startswith("qwen/qwen3"):
            body["reasoning_effort"] = "none"
        pending: dict[int, dict[str, str]] = {}
        async with self._client.stream("POST", "/chat/completions", json=body) as r:
            if r.status_code != 200:
                raise RuntimeError(f"groq {r.status_code}: {(await r.aread())[:420]!r}")
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                chunks = json.loads(data).get("choices")
                if not chunks:      # usage / keep-alive chunks carry no choices
                    continue
                choice = chunks[0]
                d = choice.get("delta", {})
                if d.get("content"):
                    yield LLMDelta(text=d["content"])
                for tc in d.get("tool_calls") or []:
                    slot = pending.setdefault(tc["index"], {"id": "", "name": "", "arguments": ""})
                    slot["id"] = tc.get("id") or slot["id"]
                    fn = tc.get("function", {})
                    slot["name"] += fn.get("name") or ""
                    slot["arguments"] += fn.get("arguments") or ""
                if choice.get("finish_reason"):
                    for slot in pending.values():
                        yield LLMDelta(tool_call=slot)
                    yield LLMDelta(finish=choice["finish_reason"])

    async def aclose(self) -> None:
        await self._client.aclose()


class MockLLM(LLMProvider):
    """Deterministic streaming LLM for perf/CI runs and key-less local dev (LLM_PROVIDER=mock)."""

    name = "mock"

    def __init__(self, reply: str | None = None,
                 ttft_s: float = 0.25, tok_s: float = 0.015) -> None:
        import os

        self.reply = reply or os.environ.get("MOCK_REPLY") or "Sure, let me check that for you. One moment please."
        self.ttft_s, self.tok_s = ttft_s, tok_s

    async def stream(self, messages, tools=None, max_tokens=160) -> AsyncIterator[LLMDelta]:
        import asyncio

        await asyncio.sleep(self.ttft_s)
        for w in self.reply.split(" "):
            yield LLMDelta(text=w + " ")
            await asyncio.sleep(self.tok_s)
        yield LLMDelta(finish="stop")


class RuleBasedMockLLM(LLMProvider):
    """Key-less dev/CI stand-in that exercises the *whole* pipeline (RAG context, tool calls, read-back
    confirmation, tool errors) with simple rules. Not a language model: replace with Groq in real use."""

    name = "rulebased-mock"
    _ORDER = re.compile(r"\b([A-Za-z]{2}[\s-]?\d[\d,]{3,7})\b")

    def __init__(self, ttft_s: float = 0.2, tok_s: float = 0.012) -> None:
        self.ttft_s, self.tok_s = ttft_s, tok_s
        self._ticket_args: dict[str, Any] | None = None

    async def stream(self, messages, tools=None, max_tokens=160) -> AsyncIterator[LLMDelta]:
        import asyncio

        await asyncio.sleep(self.ttft_s)
        names = {t["function"]["name"] for t in (tools or [])}
        last = messages[-1]
        call: dict[str, Any] | None = None
        if last["role"] == "tool":
            text = self._after_tool(last["content"])
        else:
            text, call = self._decide(messages, names)
        for w in text.split(" "):
            if w:
                yield LLMDelta(text=w + " ")
                await asyncio.sleep(self.tok_s)
        if call:
            yield LLMDelta(tool_call=call)
        yield LLMDelta(finish="tool_calls" if call else "stop")

    def _decide(self, messages, names):
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        low = user.lower()
        m = self._ORDER.search(user)
        if "order_status" in names and m and ("order" in low or "status" in low or "where" in low):
            oid = m.group(1).replace(" ", "").replace("-", "").replace(",", "").upper()
            return "", {"id": "call_o1", "name": "order_status", "arguments": json.dumps({"order_id": oid})}
        if "create_ticket" in names:
            if self._ticket_args and re.search(r"\b(yes|yeah|yep|sure|please do|go ahead|confirm)\b", low):
                return "", {"id": "call_t2", "name": "create_ticket", "arguments": json.dumps(self._ticket_args)}
            if self._ticket_args and re.search(r"\b(no|nope|don't|cancel)\b", low):
                self._ticket_args = None
                return "Okay, I won't create a ticket. Anything else I can help with?", None
            if re.search(r"\b(ticket|complaint|broken|damaged|not working|doesn't work|defective)\b", low):
                subject = " ".join(user.replace(".", "").split()[:8])
                self._ticket_args = {"subject": subject}
                return "", {"id": "call_t1", "name": "create_ticket", "arguments": json.dumps(self._ticket_args)}
        kb = next((m["content"] for m in messages if m["role"] == "system" and m["content"].startswith("<knowledge_base_excerpts>")), "")
        body = re.sub(r"</?knowledge_base_excerpts>|\(no relevant excerpts found\)", "", kb).strip()
        if body:
            first = re.sub(r"^\[\d+\]\s*", "", body.split("\n\n")[0])
            first = re.sub(r"^Q:.*?\nA:\s*", "", first, flags=re.S)
            sent = re.split(r"(?<=[.!?])\s+", first.replace("\n", " "))
            return " ".join(sent[:2]), None
        return "I'm not sure about that. I can connect you with a person if you'd like.", None

    def _after_tool(self, content: str):
        data = json.loads(content.split(":", 1)[1]) if ":" in content else {}
        if data.get("status") == "confirmation_required":
            subj = (self._ticket_args or {}).get("subject", "your issue")
            return f"I'll open a support ticket titled {subj}. Is that okay?"
        if data.get("ticket_id"):
            self._ticket_args = None
            return f"Done. Your ticket {' '.join(data['ticket_id'])} is open and the team will follow up."
        if data.get("spoken"):
            return data["spoken"]
        if data.get("error") == "not_found":
            return "I couldn't find that order. Could you read the order number again?"
        if data.get("order_id"):
            eta = data.get("eta")
            carrier = f" with {data['carrier']}" if data.get("carrier") else ""
            return f"Your order is {data['status']}{carrier}, estimated delivery {eta}."
        return "I wasn't able to get that information."
