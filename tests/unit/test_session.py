"""Scenario tests: full Session (endpointing, preemptive generation, barge-in, recovery) with fake
STT/LLM/TTS/sink running in real time. These double as the in-process behavioural spec."""
import asyncio
import time

import numpy as np
import pytest

from voxa_agent.audio.playout import FRAME, FRAME_MS, Playout
from voxa_agent.audio.vad import VadEvent
from voxa_agent.config import TurnConfig
from voxa_agent.pipeline import VadSignal
from voxa_agent.providers.llm import LLMDelta, LLMProvider
from voxa_agent.providers.stt import STTEvent
from voxa_agent.providers.tts import TTSChunk, TTSContext, TTSProvider, WordTiming
from voxa_agent.responder import Responder
from voxa_agent.session import Session, materially_different
from voxa_agent.state import State
from voxa_agent.turn.detector import HeuristicTurnDetector

WORD_MS = 250


class RealtimeSink:
    def __init__(self):
        self.end = time.monotonic()
        self.sent: list[float] = []     # capture timestamps
        self.clears: list[float] = []

    async def capture(self, pcm):
        assert len(pcm) == FRAME
        now = time.monotonic()
        self.end = max(self.end, now) + FRAME_MS / 1000
        self.sent.append(now)

    def clear(self):
        self.end = time.monotonic()
        self.clears.append(self.end)

    @property
    def queued_ms(self):
        return max(0.0, (self.end - time.monotonic()) * 1000)


class FakeTTS(TTSProvider):
    name, sample_rate = "fake", 48_000

    def context(self):
        class C(TTSContext):
            def __init__(self): self.clock = 0.0

            async def synthesize(self, text):
                await asyncio.sleep(0.01)
                words = text.split()
                dur = len(words) * WORD_MS
                ws = [WordTiming(w, self.clock + i * WORD_MS, self.clock + (i + 1) * WORD_MS) for i, w in enumerate(words)]
                self.clock += dur
                yield TTSChunk(np.full(48 * dur, 1000, dtype=np.int16), 48_000, ws)
        return C()


class FakeLLM(LLMProvider):
    name = "fake"

    def __init__(self, reply="Sure, I can help with that. Your order ships tomorrow morning."):
        self.reply, self.calls = reply, []

    async def stream(self, messages, tools=None, max_tokens=160):
        self.calls.append(messages[-1]["content"])
        await asyncio.sleep(0.03)
        for w in self.reply.split(" "):
            yield LLMDelta(text=w + " ")
            await asyncio.sleep(0.005)


class Rig:
    def __init__(self, reply=None, llm=None, session_kw=None, responder_kw=None, **cfg):
        base = dict(vad_min_silence_ms=50, endpoint_complete_ms=150, endpoint_incomplete_ms=700,
                    stage1_min_speech_ms=100, confirm_speech_ms=300, confirm_hard_speech_ms=600,
                    false_interrupt_timeout_ms=500, reprompt_after_s=0.6, close_after_s=1.2,
                    greeting="Hello there.", stt_final_timeout_ms=400)
        base.update(cfg)
        self.cfg = TurnConfig(**base)
        self.sink = RealtimeSink()
        self.sent_events: list[tuple[str, dict]] = []
        self.vad_calls: list[tuple[float, int]] = []
        self.ended = asyncio.Event()
        self.llm = llm or (FakeLLM(reply) if reply else FakeLLM())

        async def send(topic, payload):
            self.sent_events.append((topic, payload))

        async def on_end():
            self.ended.set()

        self.playout = Playout(self.sink, on_first_audio=lambda g: self.session.on_first_audio(g))
        self.responder = Responder(self.llm, FakeTTS(), self.playout, **(responder_kw or {}))
        self.session = Session(self.responder, self.playout, HeuristicTurnDetector(), self.cfg,
                               "sys", send, lambda t, m: self.vad_calls.append((t, m)), on_end,
                               **(session_kw or {}))
        if responder_kw and responder_kw.get("on_tool") is None and session_kw is not None:
            self.responder._on_tool = self.session._on_tool_event

    async def __aenter__(self):
        self.playout.start()
        await self.session.start()
        return self

    async def __aexit__(self, *a):
        await self.session.end("test")
        await self.playout.stop()

    @property
    def state(self): return self.session.sm.state

    async def until(self, pred, timeout=4.0):
        t0 = time.monotonic()
        while not pred():
            if time.monotonic() - t0 > timeout:
                raise AssertionError(f"timeout; state={self.state} hist={self.session.history}")
            await asyncio.sleep(0.01)

    async def listening(self):
        await self.until(lambda: self.state is State.LISTENING, timeout=12)

    # --- simulated user -------------------------------------------------
    async def vad(self, ev): await self.session.on_vad(VadSignal(ev, 0, time.monotonic()))
    async def stt(self, text, final=False, conf=0.9):
        await self.session.on_stt(STTEvent(text, final, conf))

    async def user_says(self, text, speak_ms=300, conf=0.9, final_delay=0.08, wait_final=True):
        await self.vad(VadEvent.SPEECH_START)
        await asyncio.sleep(speak_ms / 2000)
        await self.stt(text, False, conf)
        await asyncio.sleep(speak_ms / 2000)
        await self.vad(VadEvent.SPEECH_END)
        if wait_final:
            await asyncio.sleep(final_delay)
            await self.stt(text, True, conf)


async def test_clean_turn_flows_through_states_and_records_history():
    async with Rig() as r:
        await r.listening()
        await r.user_says("what is the status of my order.")
        await r.until(lambda: r.state is State.AGENT_SPEAKING)
        await r.until(lambda: r.state is State.LISTENING, timeout=6)
        assert r.session.history[0] == {"role": "user", "content": "what is the status of my order."}
        assert r.session.history[1]["content"].startswith("Sure, I can help")
        await asyncio.sleep(0.1)    # state events are fire-and-forget tasks
        states = [p["state"] for t, p in r.sent_events if t == "voxa.state"]
        want = ["USER_SPEAKING", "ENDPOINTING", "THINKING", "AGENT_SPEAKING", "LISTENING"]
        it = iter(states[2:])   # skip greeting transitions
        assert all(w in it for w in want), states   # ordered subsequence
        m = r.session.metrics_log[-1]
        assert m["voice_to_voice_ms"] is not None and m["voice_to_voice_ms"] < 900


async def test_incomplete_utterance_waits_then_merges():
    async with Rig() as r:
        await r.listening()
        await r.user_says("my order number is", speak_ms=200)
        await asyncio.sleep(0.4)                       # > complete(150) but < incomplete(700)
        assert r.state is State.ENDPOINTING
        assert r.llm.calls and len(r.sink.sent) < 100  # preemptive generation ran...
        before = len(r.sink.sent)
        await asyncio.sleep(0.1)
        assert len(r.sink.sent) == before              # ...but no audio was released
        await r.user_says("four five six seven.", speak_ms=200)
        await r.until(lambda: r.state is State.AGENT_SPEAKING)
        assert "my order number is four five six seven." == r.session.history[0]["content"]
        assert len([c for c in r.llm.calls if "four five six seven" in c]) >= 1


async def test_preemptive_restarts_when_final_differs_materially():
    async with Rig() as r:
        await r.listening()
        await r.vad(VadEvent.SPEECH_START)
        await r.stt("cancel my order", False)
        await r.vad(VadEvent.SPEECH_END)
        await asyncio.sleep(0.05)
        await r.stt("check my order status please.", True, 0.9)
        await r.until(lambda: r.state is State.AGENT_SPEAKING)
        assert r.llm.calls[0] == "cancel my order"
        assert r.llm.calls[-1] == "check my order status please."
        assert r.session.history[0]["content"] == "check my order status please."


def test_materially_different():
    assert not materially_different("check my order", "Check my order.")
    assert materially_different("cancel my order", "check my order status")


async def test_confirmed_interruption_stops_audio_fast_and_truncates_history():
    async with Rig() as r:
        await r.listening()
        await r.user_says("status please.", speak_ms=200)
        await r.until(lambda: r.state is State.AGENT_SPEAKING)
        await asyncio.sleep(0.9)                      # ~3-4 words heard
        await r.vad(VadEvent.SPEECH_START)
        assert r.state is State.INTERRUPT_PENDING and r.playout.is_paused
        t_stt = time.monotonic()
        await r.stt("wait stop that please", False, 0.9)
        assert r.state is State.USER_SPEAKING
        last_sent = r.sink.sent[-1]
        await asyncio.sleep(0.3)
        assert r.sink.sent[-1] == last_sent             # nothing more sent
        assert r.sink.clears and (r.sink.clears[-1] - t_stt) < 0.2
        a = [m for m in r.session.history if m["role"] == "assistant"][-1]["content"]
        assert a.endswith("[interrupted]") and "tomorrow morning" not in a
        assert a.startswith("Sure,")


async def test_cough_is_false_interruption_and_playback_resumes_fully():
    async with Rig(reply="One two three four five six seven eight.") as r:
        await r.listening()
        await r.user_says("status please.", speak_ms=200)
        await r.until(lambda: r.state is State.AGENT_SPEAKING)
        await asyncio.sleep(0.5)
        await r.vad(VadEvent.SPEECH_START)
        assert r.state is State.INTERRUPT_PENDING
        await asyncio.sleep(0.15)
        await r.vad(VadEvent.SPEECH_END)                # a cough: no words ever recognised
        await r.until(lambda: r.state is State.AGENT_SPEAKING, timeout=2)
        assert len(r.sink.clears) >= 1   # pause clawed audio back
        await r.until(lambda: r.state is State.LISTENING, timeout=6)
        a = [m for m in r.session.history if m["role"] == "assistant"][-1]["content"]
        assert a == "One two three four five six seven eight." and "[interrupted]" not in a
        # 8 words * 250 ms = 2000 ms of audio => ~100 frames; resume replays the clawed-back frames
        assert len(r.sink.sent) >= 100


async def test_backchannel_does_not_interrupt():
    async with Rig() as r:
        await r.listening()
        await r.user_says("status please.", speak_ms=200)
        await r.until(lambda: r.state is State.AGENT_SPEAKING)
        await asyncio.sleep(0.4)
        await r.vad(VadEvent.SPEECH_START)
        await r.stt("mm-hmm", False, 0.9)
        await r.vad(VadEvent.SPEECH_END)
        await r.until(lambda: r.state is State.AGENT_SPEAKING, timeout=1)
        assert not r.playout.is_paused
        assert [m for m in r.session.history if m["role"] == "assistant"] == []   # still speaking, untouched
        await r.until(lambda: r.state is State.LISTENING, timeout=6)
        assert "[interrupted]" not in r.session.history[-1]["content"]


async def test_low_confidence_short_transcript_while_speaking_is_ignored():
    async with Rig() as r:
        await r.listening()
        await r.user_says("status please.", speak_ms=200)
        await r.until(lambda: r.state is State.AGENT_SPEAKING)
        await r.vad(VadEvent.SPEECH_START)
        await r.stt("tea", False, 0.2)                   # echo-ish garbage
        assert r.state is State.INTERRUPT_PENDING
        await r.vad(VadEvent.SPEECH_END)
        await r.until(lambda: r.state is State.AGENT_SPEAKING, timeout=2)


async def test_long_noise_burst_without_words_confirms_only_after_hard_limit():
    async with Rig() as r:
        await r.listening()
        await r.user_says("status please.", speak_ms=200)
        await r.until(lambda: r.state is State.AGENT_SPEAKING)
        await r.vad(VadEvent.SPEECH_START)
        await asyncio.sleep(0.4)                         # > confirm_speech_ms(300) but no transcript
        assert r.state is State.INTERRUPT_PENDING
        await asyncio.sleep(0.4)                         # > hard limit (600) in total
        assert r.state is State.USER_SPEAKING


async def test_non_interruptible_line_ignores_barge_in():
    async with Rig() as r:
        await r.listening()
        await r.responder.say("This call is recorded for quality.", interruptible=False)
        r.session.sm.to(State.AGENT_SPEAKING, "test")
        await asyncio.sleep(0.3)
        await r.vad(VadEvent.SPEECH_START)
        await r.stt("stop talking now", False, 0.95)
        assert r.state is State.AGENT_SPEAKING and not r.playout.is_paused


async def test_vad_threshold_raised_while_agent_speaks_and_restored():
    async with Rig() as r:
        await r.listening()
        assert r.vad_calls[0] == (r.cfg.vad_threshold_speaking, r.cfg.stage1_min_speech_ms)
        assert r.vad_calls[-1] == (r.cfg.vad_threshold, r.cfg.vad_min_speech_ms)


async def test_silence_reprompt_then_polite_close():
    async with Rig() as r:
        await r.listening()
        await r.until(lambda: r.state is State.AGENT_SPEAKING, timeout=2)   # reprompt at ~0.6 s
        await r.listening()
        await asyncio.wait_for(r.ended.wait(), 4)
        assert r.state is State.ENDED


async def test_low_asr_confidence_triggers_clarifying_question_not_llm():
    async with Rig() as r:
        await r.listening()
        await r.user_says("blah blah", conf=0.2)
        await r.until(lambda: r.state is State.AGENT_SPEAKING)
        assert r.llm.calls == [] or all(c == "blah blah" for c in r.llm.calls)
        await r.listening()
        assert r.session.history == []                    # nothing committed for an unintelligible turn


async def test_user_resuming_during_thinking_cancels_and_merges():
    async with Rig() as r:
        await r.listening()
        await r.user_says("I need help with", speak_ms=200)
        await asyncio.sleep(0.05)
        await r.user_says("my invoice please.", speak_ms=200)
        await r.until(lambda: r.state is State.AGENT_SPEAKING)
        assert r.session.history[0]["content"] == "I need help with my invoice please."


async def test_own_voice_echo_is_recognised_and_does_not_interrupt():
    async with Rig() as r:
        await r.listening()
        await r.user_says("status please.", speak_ms=200)
        await r.until(lambda: r.state is State.AGENT_SPEAKING)
        await asyncio.sleep(0.8)
        await r.vad(VadEvent.SPEECH_START)
        assert r.state is State.INTERRUPT_PENDING
        await r.stt("sure I can help with that", False, 0.95)   # transcript of our own audio
        await r.until(lambda: r.state is State.AGENT_SPEAKING, timeout=1)
        assert not r.playout.is_paused
        await r.vad(VadEvent.SPEECH_START)                        # echo keeps coming: must not re-pause
        await asyncio.sleep(0.1)
        assert r.state is State.AGENT_SPEAKING and not r.playout.is_paused
        assert [m for m in r.session.history if m["content"].endswith("[interrupted]")] == []


async def test_scraps_of_low_confidence_text_never_hard_confirm():
    async with Rig() as r:
        await r.listening()
        await r.user_says("status please.", speak_ms=200)
        await r.until(lambda: r.state is State.AGENT_SPEAKING)
        await r.vad(VadEvent.SPEECH_START)
        await r.stt("bye", False, 0.2)
        await asyncio.sleep(0.9)            # past confirm_hard_speech_ms (600)
        assert r.state is State.INTERRUPT_PENDING


async def test_call_ended_mid_reply_still_logs_what_was_heard_with_latency():
    rows = []

    async def on_turn(rec): rows.append(rec)

    async with Rig(session_kw={"on_turn": on_turn}) as r:
        await r.listening()
        await r.user_says("status please.", speak_ms=200)
        await r.until(lambda: r.state is State.AGENT_SPEAKING)
        await asyncio.sleep(0.8)
        await r.session.end("hangup")
    a = [x for x in rows if x["role"] == "assistant"]
    assert len(a) == 1 and a[0]["interrupted"] and a[0]["text"].endswith("[interrupted]")
    assert a[0]["latency"]["voice_to_voice_ms"] is not None
    assert "tomorrow morning" not in a[0]["text"]


async def test_echo_matches_spoken_form_of_numbers():
    async with Rig(reply="Returns take 30 days and refunds take 5 to 7 business days.") as r:
        await r.listening()
        await r.user_says("status please.", speak_ms=200)
        await r.until(lambda: r.state is State.AGENT_SPEAKING)
        await asyncio.sleep(0.9)
        await r.vad(VadEvent.SPEECH_START)
        await r.stt("returns take thirty days", False, 0.95)    # STT writes the words, the LLM wrote digits
        await r.until(lambda: r.state is State.AGENT_SPEAKING, timeout=1)
        assert [m for m in r.session.history if m["content"].endswith("[interrupted]")] == []


async def test_garbled_low_confidence_echo_and_single_word_echo_do_not_interrupt():
    async with Rig(reply="First your order is packed at the warehouse then it ships within two days") as r:
        await r.listening()
        await r.user_says("status please.", speak_ms=200)
        await r.until(lambda: r.state is State.AGENT_SPEAKING)
        await asyncio.sleep(0.6)
        for text, conf in (("first year", 0.6), ("order", 0.8), ("hello", 0.8)):
            await r.vad(VadEvent.SPEECH_START)
            await r.stt(text, False, conf)
            await r.vad(VadEvent.SPEECH_END)
            await asyncio.sleep(0.2)
            assert r.state is not State.USER_SPEAKING, text
        await r.until(lambda: r.state is State.AGENT_SPEAKING, timeout=2)


async def test_clear_confident_two_word_interruption_still_confirms():
    async with Rig(reply="First your order is packed at the warehouse then it ships within two days") as r:
        await r.listening()
        await r.user_says("status please.", speak_ms=200)
        await r.until(lambda: r.state is State.AGENT_SPEAKING)
        await asyncio.sleep(0.6)
        await r.vad(VadEvent.SPEECH_START)
        await r.stt("actually cancel everything", False, 0.93)
        assert r.state is State.USER_SPEAKING


async def test_with_cloud_peek_a_transcriptless_burst_needs_a_very_long_time_to_confirm():
    async def peek(): pass
    async with Rig(session_kw={"peek": peek}) as r:
        await r.listening()
        await r.user_says("status please.", speak_ms=200)
        await r.until(lambda: r.state is State.AGENT_SPEAKING)
        await r.vad(VadEvent.SPEECH_START)
        await asyncio.sleep(1.5)                       # > hard limit without peek (600 ms in this rig)
        assert r.state is State.INTERRUPT_PENDING


class FakeRecorder:
    def __init__(self): self.active = False; self.started = 0
    def start(self): self.active = True; self.started += 1
    def user_frame(self, f): pass
    def agent_frame(self, f): pass


async def _consent_rig(**kw):
    rec = FakeRecorder(); consent = []

    async def on_consent(g): consent.append(g)

    rig = Rig(session_kw={"recorder": rec, "on_consent": on_consent}, record_calls=True,
              recording_prompt="May I record this call please", recording_yes="Recording now", recording_no="Not recording", **kw)
    return rig, rec, consent


async def test_recording_starts_only_after_explicit_yes():
    rig, rec, consent = await _consent_rig()
    async with rig as r:
        await r.until(lambda: any(p["state"] == "LISTENING" for t, p in r.sent_events if t == "voxa.state"), timeout=8)
        await r.listening()
        assert not rec.active                       # asked, but nothing captured yet
        await r.user_says("yes that's fine.")
        await r.until(lambda: consent == [True], timeout=4)
        assert rec.started == 1 and r.llm.calls == []     # the consent turn never reaches the LLM


@pytest.mark.parametrize("answer", ["no thanks.", "I'd rather not.", "hmm maybe, not sure."])
async def test_decline_or_ambiguity_never_records(answer):
    rig, rec, consent = await _consent_rig()
    async with rig as r:
        await r.listening()
        await r.user_says(answer)
        if "maybe" in answer:                       # ambiguous: asked once more, then treated as a no
            await r.listening()
            await r.user_says("hmm I don't know.")
        await r.until(lambda: consent == [False], timeout=8)
        assert rec.started == 0


async def test_resumed_session_skips_greeting_and_keeps_history():
    prior = [{"role": "user", "content": "where is my order"}, {"role": "assistant", "content": "It ships tomorrow."}]
    async with Rig(session_kw={"resume_history": prior}) as r:
        await r.listening()
        assert r.session.history == prior
        spoken = " ".join(c.original for c in r.responder.chunks) if r.responder.chunks else r.responder.last_text
        assert "Hello there" not in spoken


async def test_wait_with_modest_confidence_interrupts_and_function_word_matches_are_not_echo():
    async with Rig(reply="I can help with a different order for you now") as r:
        await r.listening()
        await r.user_says("status please.", speak_ms=200)
        await r.until(lambda: r.state is State.AGENT_SPEAKING)
        await asyncio.sleep(0.6)
        assert not r.session._is_echo("I have a", 0.9)            # only function words overlap the reply: not echo
        await r.vad(VadEvent.SPEECH_START)
        await r.stt("Wait.", True, 0.61)                          # moderate confidence, but it is a command
        assert r.state is State.USER_SPEAKING


async def test_intelligible_words_far_quieter_than_the_caller_do_not_interrupt():
    ratio = {"v": 0.12}
    async with Rig() as r:
        r.session.level_ratio = lambda: ratio["v"]
        await r.listening()
        await r.user_says("status please.", speak_ms=200)
        await r.until(lambda: r.state is State.AGENT_SPEAKING)
        await asyncio.sleep(0.5)
        await r.vad(VadEvent.SPEECH_START)
        await r.stt("markets closed higher on tuesday", False, 0.95)      # the TV
        assert r.state is State.INTERRUPT_PENDING
        ratio["v"] = 0.9                                                  # same words at the caller's own level
        await r.stt("markets closed higher on tuesday", False, 0.95)
        assert r.state is State.USER_SPEAKING


async def test_long_quiet_background_speech_never_confirms_by_duration():
    async with Rig() as r:
        r.session.level_ratio = lambda: 0.15
        await r.listening()
        await r.user_says("status please.", speak_ms=200)
        await r.until(lambda: r.state is State.AGENT_SPEAKING)
        await asyncio.sleep(0.4)
        await r.vad(VadEvent.SPEECH_START)
        await r.stt("something about the markets", False, 0.2)      # a transcript exists, so no hard limit applies
        await asyncio.sleep(1.2)                                    # way past confirm_speech_ms (300 in this rig)
        assert r.state is State.INTERRUPT_PENDING
