"""One agent response: LLM tokens -> normalizer -> clause chunker -> TTS -> playout, fully streamed."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import numpy as np

from .audio.playout import Playout
from .audio.resample import Upsampler16to48
from .metrics import TurnTimings, errors
from .providers.llm import LLMProvider
from .providers.tts import TTSProvider
from .text.chunker import ClauseChunker
from .tools import ToolOutcome, ToolRuntime, tool_message
from .text.normalizer import normalize
from .turn.history import SpokenChunk

log = logging.getLogger("voxa.responder")


@dataclass
class ResponderConfig:
    pronunciations: dict[str, str] = field(default_factory=dict)
    locale: str = "en"
    first_chunk_words: int = 10
    non_interruptible_phrases: tuple[str, ...] = ()
    error_line: str = "Sorry, I'm having trouble right now. Could you try again in a moment?"


class Responder:
    """Owns generation ids. start() cancels any in-flight response and begins a new one;
    every downstream write is tagged with the id and dropped if stale."""

    def __init__(self, llm: LLMProvider, tts: TTSProvider, playout: Playout,
                 cfg: ResponderConfig | None = None, tools: "ToolRuntime | None" = None,
                 fillers: list[str] | None = None,
                 on_tool: Callable[[str, str, "ToolOutcome | None"], Awaitable[None]] | None = None) -> None:
        self.llm, self.tts, self.playout, self.cfg = llm, tts, playout, cfg or ResponderConfig()
        self.tools, self.fillers, self._on_tool = tools, fillers or [], on_tool
        self._filler_i = -1
        self.generation_id = 0
        self._task: asyncio.Task[None] | None = None
        self.last_text = ""
        self.last_text_original = ""
        self.chunks: list[SpokenChunk] = []   # what was synthesized for the current generation
        self.non_interruptible = False

    async def cancel(self) -> float:
        """Stop LLM+TTS+audio; returns ms of audio that had been played for the cancelled generation."""
        self.generation_id += 1
        if self._task and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        return self.playout.flush()

    async def start(self, messages: list[dict[str, Any]], timings: TurnTimings, hold: bool = False) -> int:
        """hold=True: preemptive generation - audio is synthesized but not played until playout.release()."""
        await self.cancel()
        self.generation_id += 1
        gid = self.generation_id
        timings.generation_id = gid
        self._reset_turn()
        self.playout.begin_generation(gid, hold=hold)
        self._task = asyncio.create_task(self._run(gid, messages, timings), name=f"respond-{gid}")
        return gid

    async def say(self, text: str, timings: TurnTimings | None = None, interruptible: bool = True) -> int:
        """Speak scripted text (greeting, acknowledgments) through the same TTS/playout path."""
        await self.cancel()
        self.generation_id += 1
        gid = self.generation_id
        self._reset_turn()
        self.non_interruptible = not interruptible
        self.playout.begin_generation(gid)
        self._task = asyncio.create_task(self._speak([text], gid, timings or TurnTimings(gid)))
        return gid

    async def play_cached(self, text: str, pcm48: np.ndarray, words, interruptible: bool = True) -> int:
        """Play pre-synthesized audio (greeting) with no TTS round trip."""
        await self.cancel()
        self.generation_id += 1
        gid = self.generation_id
        self._reset_turn()
        self.non_interruptible = not interruptible
        self.playout.begin_generation(gid)
        if words:
            self.chunks.append(SpokenChunk(text, words, words[0].start_ms, words[-1].end_ms))
        self.last_text = text
        loop = asyncio.get_running_loop()
        self._task = loop.create_task(self.playout.enqueue(gid, pcm48, words))
        return gid

    def _reset_turn(self) -> None:
        self.chunks, self.last_text, self.last_text_original, self.non_interruptible = [], "", "", False

    async def wait(self) -> None:
        if self._task:
            await asyncio.gather(self._task, return_exceptions=True)

    async def _run(self, gid: int, messages: list[dict[str, Any]], t: TurnTimings) -> None:
        chunker = ClauseChunker(self.cfg.first_chunk_words)
        t.mark("llm_request")
        # TTS synthesis for clause N overlaps LLM generation of clause N+1 via this queue.
        q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=8)
        spoken: list[str] = []
        tts_task = asyncio.create_task(self._tts_worker(q, gid, t, spoken))
        try:
            msgs = list(messages)
            schemas = self.tools.schemas() if self.tools and self.tools.tools else None
            for _round in range(4):                      # LLM -> tools -> LLM, bounded
                calls: list[dict[str, Any]] = []
                said = ""
                async for d in self.llm.stream(msgs, tools=schemas):
                    if gid != self.generation_id:
                        return
                    if (d.text or d.tool_call) and "llm" not in t.info and getattr(self.llm, "served_by", ""):
                        t.info["llm"] = self.llm.served_by.replace("groq:", "")
                    if d.text:
                        t.mark("llm_first_token")
                        said += d.text
                        for c in chunker.feed(d.text):
                            await q.put(c)
                    if d.tool_call:
                        calls.append(d.tool_call)
                for c in chunker.flush():
                    await q.put(c)
                if not calls or not self.tools:
                    break
                t.mark("llm_first_token")
                msgs.append({"role": "assistant", "content": said or None, "tool_calls": [
                    {"id": c["id"] or f"call_{i}", "type": "function",
                     "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
                    for i, c in enumerate(calls)]})
                for i, c in enumerate(calls):
                    cid = c["id"] or f"call_{i}"
                    outcome = await self._run_tool(gid, c, q, said_something=bool(said.strip()))
                    msgs.append(tool_message(cid, outcome))
                    if gid != self.generation_id:
                        return
            await q.put(None)
            await tts_task
            self.last_text = " ".join(spoken)
            self.last_text_original = " ".join(c.original for c in self.chunks)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            errors.labels("llm").inc()
            log.exception("response failed")
            if gid == self.generation_id and not tts_task.done():
                # never leave the caller in silence: say the tenant's graceful failure line
                await q.put(self.cfg.error_line)
                await q.put(None)
                try:
                    await tts_task
                    self.last_text = " ".join(spoken)
                    self.last_text_original = " ".join(c.original for c in self.chunks)
                except Exception:  # noqa: BLE001
                    log.exception("could not speak the failure line")
        finally:
            if not tts_task.done():
                tts_task.cancel()

    async def _run_tool(self, gid: int, call: dict[str, Any], q: "asyncio.Queue[str | None]",
                        said_something: bool) -> "ToolOutcome":
        name = call["name"]
        if self._on_tool:
            await self._on_tool("start", name, None)
        # Mask latency: a slow tool gets a short spoken acknowledgment while it runs in parallel.
        tool = self.tools.tools.get(name)
        if not said_something and self.tools.expected_latency_ms(name) > 300 and (tool.filler or self.fillers):
            if tool.filler:
                await q.put(tool.filler)
            else:
                self._filler_i = (self._filler_i + 1) % len(self.fillers)
                await q.put(self.fillers[self._filler_i])
        if tool and tool.side_effect:
            await self.playout.wait_release()      # never act on an uncommitted (preemptive) turn
            if gid != self.generation_id:
                raise asyncio.CancelledError
        outcome = await self.tools.execute(name, call["arguments"])
        if self._on_tool:
            await self._on_tool("end", name, outcome)
        return outcome

    async def _speak(self, chunks: list[str], gid: int, t: TurnTimings) -> None:
        q: asyncio.Queue[str | None] = asyncio.Queue()
        for c in chunks:
            q.put_nowait(c)
        q.put_nowait(None)
        await self._tts_worker(q, gid, t, [])

    async def _tts_worker(self, q: asyncio.Queue[str | None], gid: int, t: TurnTimings,
                          spoken: list[str]) -> None:
        ctx = self.tts.context()   # one context per turn: consistent prosody + running word clock
        up = Upsampler16to48() if self.tts.sample_rate == 16_000 else None
        try:
            while (text := await q.get()) is not None:
                if gid != self.generation_id:
                    return
                speakable = normalize(text, self.cfg.locale, self.cfg.pronunciations)
                if not speakable:
                    continue
                spoken.append(speakable)
                if any(p.lower() in text.lower() for p in self.cfg.non_interruptible_phrases):
                    self.non_interruptible = True   # e.g. legal disclosure: held until the turn ends
                async for ch in ctx.synthesize(speakable):
                    if gid != self.generation_id:
                        return
                    t.mark("tts_first_byte")
                    pcm = ch.pcm
                    if up is not None:
                        pcm = np.clip(up.process(pcm), -32768, 32767).astype(np.int16)
                    elif ch.sample_rate != 48_000:
                        raise ValueError(f"unsupported TTS rate {ch.sample_rate}")
                    if ch.words:
                        self.chunks.append(SpokenChunk(text, ch.words, ch.words[0].start_ms, ch.words[-1].end_ms))
                    await self.playout.enqueue(gid, pcm, ch.words)
        except asyncio.CancelledError:
            raise
        except Exception:
            errors.labels("tts").inc()
            raise
        finally:
            await ctx.aclose()
