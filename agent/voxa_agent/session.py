"""Conversation orchestrator: state machine + endpointing + preemptive generation + barge-in + recovery.

Pure asyncio, no LiveKit imports: inputs are `on_vad` / `on_stt`, outputs go through `send` (data channel).
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Awaitable, Callable

from .audio.playout import Playout
from .bargein import BargeInController, Outcome
from .config import TurnConfig
from . import tracing
from .metrics import TurnTimings, calls_total, errors, handoffs_total, interruptions, session_seconds
from .pipeline import VadSignal
from .audio.vad import VadEvent
from .providers.stt import STTEvent
from .responder import Responder
from .state import State, StateMachine
from .turn.detector import TurnDetector, is_complete
from .turn.history import history_message
from voxa_shared.redact import redact

log = logging.getLogger("voxa.session")
Send = Callable[[str, dict], Awaitable[None]]


_FUNCTION_WORDS = frozenset("a an the i you he she it we they me my your our their his her its to of and or but so in on at "
                            "for with from by as is are was were be been am do does did have has had that this these those "
                            "can will would could should just not no yes there here what who how why when".split())


def _norm(t: str) -> list[str]:
    return re.sub(r"[^\w\s']", "", t.lower()).split()


def materially_different(a: str, b: str) -> bool:
    """Transcript changed enough that a preemptive generation must be restarted."""
    wa, wb = _norm(a), _norm(b)
    if wa == wb:
        return False
    common = sum(1 for x, y in zip(wa, wb) if x == y)
    return common / max(len(wa), len(wb), 1) < 0.85


class Session:
    def __init__(self, responder: Responder, playout: Playout, detector: TurnDetector, cfg: TurnConfig,
                 system_prompt: str, send: Send,
                 set_vad: Callable[[float, int], None] = lambda thr, ms: None,
                 on_end: Callable[[], Awaitable[None]] | None = None,
                 greeting_audio: tuple | None = None,
                 kb: Callable[[str], Awaitable[tuple[str, int]]] | None = None,
                 escalation: Any = None,
                 tools: Any = None,
                 summarize: Callable[[list[dict[str, str]]], Awaitable[str]] | None = None,
                 on_handoff: Callable[[str, str], Awaitable[None]] | None = None,
                 on_turn: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
                 redact_pii: bool = True,
                 peek: Callable[[], Awaitable[None]] | None = None,
                 on_consent: Callable[[bool], Awaitable[None]] | None = None,
                 recorder: Any = None,
                 resume_history: list[dict[str, str]] | None = None,
                 on_history: Callable[[list[dict[str, str]]], None] | None = None) -> None:
        self._resume_history, self._on_history = resume_history, on_history
        self._on_consent, self.recorder = on_consent, recorder
        self._consent_asked = False
        self._awaiting_consent = False
        self._consent_tries = 0
        self._greeting_gid = -1
        self._greeting_audio = greeting_audio
        self.peek = peek
        self.level_ratio: Callable[[], float | None] | None = None   # set by the worker from the pipeline
        self._kb, self.esc, self.tools = kb, escalation, tools
        self._summarize, self._on_handoff, self._on_turn = summarize, on_handoff, on_turn
        self._redact = redact if redact_pii else (lambda x: x)
        self._pre_kb = ("", 0)
        self._turn_failed = False
        self._logged: set[int] = set()
        self._prefetch_text = ""
        self.handoff_summary: str | None = None
        self.r, self.playout, self.det, self.cfg = responder, playout, detector, cfg
        self.system_prompt, self.send, self._set_vad, self._on_end = system_prompt, send, set_vad, on_end
        self.sm = StateMachine(self._changed)
        self.history: list[dict[str, str]] = []
        self.turns: dict[int, TurnTimings] = {}
        self.metrics_log: list[dict[str, Any]] = []
        self.bi = BargeInController(cfg, self._bi_pause, self._bi_confirm, self._bi_resume, self._is_echo,
                                    peek=lambda: self.peek(), peek_available=lambda: self.peek is not None,
                                    level_ratio=lambda: self.level_ratio() if self.level_ratio else None)
        # per-user-turn accumulation
        self._finals: list[str] = []
        self._interim = ""
        self._conf: list[float] = []
        self._open_segments = 0           # SPEECH_END seen but final not yet delivered
        self._timing: TurnTimings | None = None
        self._pre_gid: int | None = None  # preemptive generation in flight (audio held)
        self._pre_text = ""
        self._endpoint_task: asyncio.Task[None] | None = None
        self._silence_task: asyncio.Task[None] | None = None
        self._done_task: asyncio.Task[None] | None = None
        self._wake = asyncio.Event()
        self._reprompted = False
        self._silence_started = 0.0
        self._cur_gid = 0

    # ---------------------------------------------------------------- helpers
    def _is_echo(self, text: str, confidence: float = 1.0) -> bool:
        """Heuristic echo detector: the 'user' transcript is (nearly) a subset of what we are saying.
        Echo that went through speaker + room is garbled, so STT gets a few words wrong and is unsure:
        a partial match with low confidence counts too."""
        words = _norm(text)
        if not words:
            return False
        said = {w for c in self.r.chunks
                for w in _norm(c.original) + _norm(" ".join(x.text for x in c.words))}   # "30" and "thirty"
        if len(said) < 3:
            return False
        # function words ("I have a...") appear in any reply and prove nothing: match on content words only
        words = [w for w in words if w not in _FUNCTION_WORDS]
        if not words:
            return False
        match = sum(1 for w in words if w in said) / len(words)
        if len(words) == 1:
            return match == 1.0 and len(words[0]) >= 3 and confidence < 0.9
        return match >= 0.8 or (match >= 0.5 and confidence < 0.75)

    def _changed(self, old: State, new: State, reason: str) -> None:
        asyncio.get_running_loop().create_task(self.send("voxa.state", {"state": new.value, "from": old.value}))
        speaking = new in (State.AGENT_SPEAKING, State.INTERRUPT_PENDING)
        if speaking and old not in (State.AGENT_SPEAKING, State.INTERRUPT_PENDING):
            self._set_vad(self.cfg.vad_threshold_speaking, self.cfg.stage1_min_speech_ms)  # echo safety
        elif not speaking and old in (State.AGENT_SPEAKING, State.INTERRUPT_PENDING):
            self._set_vad(self.cfg.vad_threshold, self.cfg.vad_min_speech_ms)

    def _text(self) -> str:
        return " ".join([*self._finals, self._interim] if self._interim else self._finals).strip()

    def _messages(self, user_text: str, kb_block: str = "") -> list[dict[str, Any]]:
        """system | history | retrieved excerpts (as untrusted data) | current user turn"""
        kb = [{"role": "system", "content": kb_block}] if kb_block else []
        return [{"role": "system", "content": self.system_prompt}, *self.history[-12:], *kb,
                {"role": "user", "content": user_text}]

    async def _retrieve(self, text: str) -> tuple[str, int]:
        if not self._kb or not text.strip():
            return "", 0
        try:
            return await asyncio.wait_for(self._kb(text), timeout=1.5)
        except Exception:  # noqa: BLE001 - retrieval must never stall or break a turn
            log.warning("retrieval failed/timed out; answering without excerpts")
            # say so explicitly: an empty block would read as "the knowledge base has nothing on this"
            return ("<knowledge_base_excerpts>\n(the knowledge base lookup is temporarily unavailable; do not guess, "
                    "tell the caller you cannot check right now)\n</knowledge_base_excerpts>"), 0

    async def _log(self, role: str, text: str, **extra: Any) -> None:
        if self._on_turn and text:
            try:
                await self._on_turn({"role": role, "text": self._redact(text), **extra})
            except Exception:  # noqa: BLE001
                log.exception("turn logging failed")

    async def _on_tool_event(self, kind: str, name: str, outcome: Any) -> None:
        if kind == "start":
            if self.sm.state in (State.THINKING, State.AGENT_SPEAKING):
                self.sm.to(State.TOOL_CALL, f"tool {name}")
        else:
            if self.sm.state is State.TOOL_CALL:
                self.sm.to(State.THINKING, f"tool {name} done")
            if outcome and not outcome.ok:
                self._turn_failed = True
                errors.labels("tool").inc()
            await self._log("tool", f"{name} ok={outcome.ok if outcome else None}", tool=name,
                            duration_ms=outcome.duration_ms if outcome else None)

    def _cancel_task(self, name: str) -> None:
        t = getattr(self, name)
        if t and not t.done() and t is not asyncio.current_task():
            t.cancel()
        setattr(self, name, None)

    def _reset_user_turn(self) -> None:
        self._finals, self._interim, self._conf, self._open_segments = [], "", [], 0
        self._timing, self._pre_gid, self._pre_text = None, None, ""

    # ---------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        calls_total.inc()
        self._t_start = time.monotonic()
        self._ended_metrics = False
        self.sm.to(State.AGENT_SPEAKING, "greeting")
        if self._resume_history is not None:          # re-dispatched after a crash: no greeting, no second consent prompt
            self.history = list(self._resume_history)
            self._consent_asked = True
            self._cur_gid = await self.r.say(self.cfg.resume_line)
            self._greeting_gid = self._cur_gid
            self._watch_done()
            return
        if self._greeting_audio:   # pre-synthesized per tenant+voice: no TTS round trip on connect
            pcm, words = self._greeting_audio
            self._cur_gid = await self.r.play_cached(self.greeting_text(), pcm, words,
                                                     interruptible=not self.cfg.disclaimers)
            self._greeting_gid = self._cur_gid
        else:
            self._cur_gid = await self.r.say(self.greeting_text(), interruptible=not self.cfg.disclaimers)
        self._greeting_gid = self._cur_gid
        self._watch_done()

    async def _log_unfinished_reply(self) -> None:
        """The call ended while the agent was mid-reply: keep what the caller actually heard, with its latency."""
        gid = self.r.generation_id
        t = self.turns.get(gid)
        if not t or gid in self._logged or "first_audio_played" not in t.marks:
            return
        msg = history_message(self.r.chunks, self.playout.played_ms(gid))
        self._logged.add(gid)
        if msg:
            self.history.append({"role": "assistant", "content": msg})
        await self._log("assistant", msg or "(call ended before the reply was heard)", gen=gid,
                        interrupted=bool(msg and msg.endswith("[interrupted]")), latency={"gen": gid, **t.publish()})

    def greeting_text(self) -> str:
        return " ".join([self.cfg.greeting, *self.cfg.disclaimers])

    async def end(self, reason: str = "") -> None:
        if self.sm.state is State.ENDED:
            return
        for n in ("_endpoint_task", "_silence_task", "_done_task"):
            self._cancel_task(n)
        self.bi.cancel()
        await self._log_unfinished_reply()
        self.sm.to(State.ENDED, reason)
        if getattr(self, "_t_start", None) and not self._ended_metrics:
            self._ended_metrics = True
            session_seconds.observe(time.monotonic() - self._t_start)
        await self.r.cancel()
        if self._on_end:
            try:
                await asyncio.wait_for(self._on_end(), 3.0)
            except Exception:  # noqa: BLE001 - the room may already be gone
                pass

    def _watch_done(self) -> None:
        """When the current generation's audio has fully played, AGENT_SPEAKING -> LISTENING."""
        self._cancel_task("_done_task")
        gid = self._cur_gid

        async def run() -> None:
            await self.r.wait()
            if gid != self.r.generation_id:
                return
            if await self.playout.drain(gid):
                await self._agent_finished(gid)

        self._done_task = asyncio.create_task(run())

    async def _agent_finished(self, gid: int) -> None:
        if (self.cfg.record_calls and self.recorder and not self._consent_asked and gid == self._greeting_gid
                and self.sm.state is State.AGENT_SPEAKING):
            self._consent_asked = self._awaiting_consent = True      # ask for consent before anything is captured
            self._cur_gid = await self.r.say(self.cfg.recording_prompt)
            self._watch_done()
            return
        t = self.turns.get(gid)
        m = {"gen": gid, **t.publish()} if t else None
        if t:
            tracing.record_turn(gid, t.marks, m)
        if self.r.last_text and self.sm.state in (State.AGENT_SPEAKING, State.INTERRUPT_PENDING, State.THINKING):
            if gid in self.turns:
                said = self.r.last_text_original or self.r.last_text
                self.history.append({"role": "assistant", "content": said})
                self._logged.add(gid)
                await self._log("assistant", said, gen=gid, latency=m)
                if self.esc:   # consecutive-failure accounting happens once per answered turn
                    (self.esc.record_failure if self._turn_failed else self.esc.record_success)()
                self._turn_failed = False
        if self._on_history:
            self._on_history(self.history)
        if m:
            self.metrics_log.append(m)
            await self.send("voxa.turn_metrics", m)
        if self.sm.state is State.ENDED:
            return
        if self.sm.state is State.INTERRUPT_PENDING:
            return  # decision pending; resume/confirm handlers drive the next transition
        if self.sm.can(State.LISTENING) and self.sm.state is State.AGENT_SPEAKING:
            self.sm.to(State.LISTENING, "agent done")
            self._start_silence_watch()

    # ---------------------------------------------------------------- inputs
    async def on_vad(self, sig: VadSignal) -> None:
        await self.send("voxa.vad", {"event": sig.event.value})
        st = self.sm.state
        if st is State.ENDED:
            return
        if sig.event is VadEvent.SPEECH_START:
            await self._speech_start()
        else:
            await self._speech_end(sig)

    async def _speech_start(self) -> None:
        st = self.sm.state
        if st in (State.AGENT_SPEAKING, State.INTERRUPT_PENDING):
            if self.r.non_interruptible or (st is State.AGENT_SPEAKING and self.bi.suppressed):
                return
            if st is State.AGENT_SPEAKING:
                self.sm.to(State.INTERRUPT_PENDING, "vad stage1")
            await self.bi.on_speech_start()
            return
        if st in (State.TOOL_CALL, State.THINKING, State.ENDPOINTING):
            await self._abandon_preemptive()
            if st is not State.ENDPOINTING:
                self._cancel_task("_done_task")
                await self.r.cancel()          # committed but not yet audible: user kept talking
            self._cancel_task("_endpoint_task")
            if st is not State.ENDPOINTING and self.history and self.history[-1]["role"] == "user":
                # user kept talking after the turn was committed: merge instead of answering twice
                self._finals = [self.history.pop()["content"], *self._finals]
        self._cancel_task("_silence_task")
        self._reprompted = False
        if self.sm.can(State.USER_SPEAKING):
            self.sm.to(State.USER_SPEAKING, "vad start")

    async def _speech_end(self, sig: VadSignal) -> None:
        st = self.sm.state
        if st is State.INTERRUPT_PENDING or self.bi.active:
            await self.bi.on_speech_end()
            return
        if st is State.USER_SPEAKING:
            t = TurnTimings(0)
            t.mark("eos_detected", sig.mono_s)
            self._timing = t
            self._open_segments += 1
            self.sm.to(State.ENDPOINTING, "vad end")
            self._cancel_task("_endpoint_task")
            self._wake = asyncio.Event()
            self._endpoint_task = asyncio.create_task(self._endpoint(sig.mono_s), name="endpoint")

    async def on_stt(self, ev: STTEvent) -> None:
        await self.send("voxa.transcript", {"text": ev.text, "final": ev.is_final, "confidence": ev.confidence})
        st = self.sm.state
        if st in (State.AGENT_SPEAKING, State.INTERRUPT_PENDING) or self.bi.active:
            await self.bi.on_stt(ev)
            return
        if st not in (State.USER_SPEAKING, State.ENDPOINTING, State.THINKING):
            return
        if ev.is_final:
            self._open_segments = max(0, self._open_segments - 1)
            if ev.text.strip():
                self._finals.append(ev.text.strip())
                self._conf.append(ev.confidence)
            self._interim = ""
        else:
            self._interim = ev.text.strip()
            if self._kb and len(self._interim.split()) >= 4 and self._interim != self._prefetch_text:
                self._prefetch_text = self._interim      # warm the retrieval cache while they are still talking
                asyncio.create_task(self._retrieve(self._interim))
        self._wake.set()

    # ---------------------------------------------------------------- endpointing + preemptive
    async def _endpoint(self, eos_t: float) -> None:
        cfg = self.cfg
        silence_already = cfg.vad_min_silence_ms / 1000
        try:
            while True:
                text = self._text()
                if text and cfg.preemptive_generation and not self._awaiting_consent and (
                        self._pre_gid is None or materially_different(self._pre_text, text)):
                    await self._start_preemptive(text)
                complete = bool(text) and is_complete(self.det, text, self._last_agent())
                wait_s = (cfg.endpoint_complete_ms if complete else cfg.endpoint_incomplete_ms) / 1000
                deadline = eos_t + max(0.0, wait_s - silence_already)
                now = time.monotonic()
                have_final = self._open_segments == 0 and bool(self._finals)
                if now >= deadline and have_final:
                    await self._commit()
                    return
                if now >= deadline + cfg.stt_final_timeout_ms / 1000:
                    if text:
                        await self._commit()
                    else:
                        self.sm.to(State.LISTENING, "empty turn")
                        self._reset_user_turn()
                        self._start_silence_watch()
                    return
                self._wake.clear()
                timeout = max(0.005, deadline - now) if now < deadline else 0.05
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            pass

    async def _start_preemptive(self, text: str) -> None:
        t = self._timing or TurnTimings(0)
        self._pre_text = text
        t.marks.pop("llm_first_token", None); t.marks.pop("tts_first_byte", None)
        kb_block, n = await self._retrieve(text)
        self._pre_kb = (kb_block, n)
        if self.tools:
            self.tools.set_context(text, self._last_agent())
        self._pre_gid = await self.r.start(self._messages(text, kb_block), t, hold=True)
        self.turns[self._pre_gid] = t

    async def _abandon_preemptive(self) -> None:
        if self._pre_gid is not None:
            await self.r.cancel()
            self.turns.pop(self._pre_gid, None)
            self._pre_gid = None

    async def _commit(self) -> None:
        text = self._text()
        t = self._timing or TurnTimings(0)
        t.mark("stt_final"); t.mark("turn_committed")
        conf = sum(self._conf) / len(self._conf) if self._conf else 0.0
        self.sm.to(State.THINKING, "turn committed")
        if not text or conf < self.cfg.low_confidence:
            await self._abandon_preemptive()
            self._reset_user_turn()
            if self.esc:
                self.esc.record_failure()
            self.sm.to(State.AGENT_SPEAKING, "clarify")
            self._cur_gid = await self.r.say(self.cfg.clarify_line)
            self._watch_done()
            return
        if self._awaiting_consent:
            await self._handle_consent(text)
            return
        if self.tools:
            self.tools.new_user_turn()
        decision = self.esc.evaluate(text) if self.esc else None
        if decision:
            await self._abandon_preemptive()
            self.history.append({"role": "user", "content": text})
            await self._log("user", text)
            self._reset_user_turn_keep()
            await self._escalate(decision)
            return
        self.history.append({"role": "user", "content": text})
        await self._log("user", text)
        if self._pre_gid is not None and not materially_different(self._pre_text, text):
            gid, n_hits = self._pre_gid, self._pre_kb[1]
            self.playout.release()                      # preemptive audio goes out now
        else:
            await self._abandon_preemptive()
            kb_block, n_hits = await self._retrieve(text)
            if self.tools:
                self.tools.set_context(text, self._last_agent())
            gid = await self.r.start(self._messages_committed(kb_block), t)
            self.turns[gid] = t
        if self._kb and not n_hits and not self.tools_available():
            self._turn_failed = True        # nothing to ground an answer on
        self._cur_gid = gid
        self._reset_user_turn_keep()
        self._watch_done()

    _YES = re.compile(r"\b(yes|yeah|yep|yup|sure|okay|ok|fine|go ahead|that's fine|i agree|you may|you can|please do)\b", re.I)
    _NO = re.compile(r"\b(no|nope|don't|do not|not|never|rather not|no thanks|decline)\b", re.I)

    async def _handle_consent(self, text: str) -> None:
        """Explicit consent only: 'yes' starts recording, 'no' declines, anything else is asked once more and then
        treated as a no. Ambiguity never records."""
        yes, no = bool(self._YES.search(text)) and not self._NO.search(text), bool(self._NO.search(text))
        await self._abandon_preemptive()
        self._reset_user_turn_keep()
        self._consent_tries += 1
        if not (yes or no) and self._consent_tries < 2:
            line = self.cfg.recording_prompt
        else:
            self._awaiting_consent = False
            granted = yes and not no
            if granted and self.recorder:
                self.recorder.start()
            if self._on_consent:
                await self._on_consent(granted)
            line = self.cfg.recording_yes if granted else self.cfg.recording_no
        await self._log("user", text)
        self.sm.to(State.AGENT_SPEAKING, "consent")
        self._cur_gid = await self.r.say(line)
        self._watch_done()

    def tools_available(self) -> bool:
        return bool(self.tools and self.tools.tools)

    async def _escalate(self, d: Any) -> None:
        """Escalation rule fired: warm handoff (with summary) or an after-hours alternative."""
        log.info("escalation: %s handoff=%s", d.reason, d.handoff)
        if not d.handoff:
            self.sm.to(State.AGENT_SPEAKING, "after hours")
            self.history.append({"role": "assistant", "content": d.line})
            self._cur_gid = await self.r.say(d.line)
            self._watch_done()
            return
        handoffs_total.labels(d.reason).inc()
        self.sm.to(State.HANDOFF, d.reason)
        gid = await self.r.say(d.line, interruptible=False)
        self.history.append({"role": "assistant", "content": d.line})
        summary = ""
        if self._summarize:
            try:
                summary = await asyncio.wait_for(self._summarize(self.history), timeout=4)
            except Exception:  # noqa: BLE001
                log.exception("summary failed")
        self.handoff_summary = summary or "Caller requested a human. " + " ".join(
            m["content"] for m in self.history if m["role"] == "user")[-300:]
        await self.send("voxa.handoff", {"status": "connecting", "reason": d.reason})
        if self._on_handoff:
            await self._on_handoff(self.handoff_summary, d.reason)
        await self.r.wait()
        await self.playout.drain(gid)

    def _messages_committed(self, kb_block: str = "") -> list[dict[str, Any]]:
        kb = [{"role": "system", "content": kb_block}] if kb_block else []
        return [{"role": "system", "content": self.system_prompt}, *self.history[-12:-1], *kb, self.history[-1]]

    def _reset_user_turn_keep(self) -> None:
        self._finals, self._interim, self._conf, self._open_segments = [], "", [], 0
        self._pre_gid, self._pre_text, self._timing = None, "", None

    def _last_agent(self) -> str:
        for m in reversed(self.history):
            if m["role"] == "assistant":
                return m["content"]
        return ""

    async def on_first_audio(self, gid: int) -> None:
        """Playout handed the first frame of `gid` to WebRTC."""
        t = self.turns.get(gid)
        if t:
            t.mark("first_audio_played")
        await self.send("voxa.agent_audio_start", {"gen": gid})
        if gid == self.r.generation_id and self.sm.state in (State.THINKING, State.TOOL_CALL):
            self.sm.to(State.AGENT_SPEAKING, "first audio")

    def on_client_ack(self, gid: int, kind: str) -> None:
        t = self.turns.get(gid)
        if t and kind == "first_audio_played":
            t.marks["first_audio_played"] = time.monotonic()

    # ---------------------------------------------------------------- barge-in callbacks
    async def _bi_pause(self) -> None:
        await self.playout.pause()
        await self.send("voxa.interrupt", {"stage": 1})

    async def _bi_resume(self, outcome: Outcome) -> None:
        interruptions.labels(outcome.value).inc()
        self.playout.resume()
        await self.send("voxa.interrupt", {"stage": "resume", "outcome": outcome.value})
        if self.sm.state is State.INTERRUPT_PENDING:
            self.sm.to(State.AGENT_SPEAKING, f"{outcome.value}")
        self._watch_done()

    async def _bi_confirm(self, text: str) -> None:
        interruptions.labels("confirmed").inc()
        gid = self.r.generation_id
        heard_ms = await self.r.cancel()
        msg = history_message(self.r.chunks, heard_ms)
        if msg:
            self.history.append({"role": "assistant", "content": msg})
            await self._log("assistant", msg, gen=gid, interrupted=True,
                            latency=({"gen": gid, **self.turns[gid].publish()} if gid in self.turns else None))
            self._logged.add(gid)
        self._cancel_task("_done_task")
        self.playout.resume()
        self._reset_user_turn()
        self._interim = text
        self.sm.to(State.USER_SPEAKING, "interrupt confirmed")
        await self.send("voxa.interrupt", {"stage": 2, "gen": gid, "heard_ms": heard_ms})

    # ---------------------------------------------------------------- silence handling
    def _start_silence_watch(self) -> None:
        self._cancel_task("_silence_task")
        self._silence_started = time.monotonic()
        self._silence_task = asyncio.create_task(self._silence())

    async def _silence(self) -> None:
        cfg = self.cfg
        try:
            if not self._reprompted:
                await asyncio.sleep(cfg.reprompt_after_s)
                if self.sm.state is not State.LISTENING:
                    return
                self._reprompted = True
                self.sm.to(State.AGENT_SPEAKING, "reprompt")
                self._cur_gid = await self.r.say(cfg.reprompt_line)
                self._watch_done()
                return
            await asyncio.sleep(max(0.0, cfg.close_after_s - cfg.reprompt_after_s))
            if self.sm.state is State.LISTENING:
                self.sm.to(State.AGENT_SPEAKING, "closing")
                self._cur_gid = await self.r.say(cfg.closing_line, interruptible=False)
                await self.r.wait()
                await self.playout.drain(self._cur_gid)
                await self.end("silence timeout")
        except asyncio.CancelledError:
            pass
