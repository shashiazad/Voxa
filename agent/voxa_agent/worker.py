"""Voxa agent worker: per-call tenant runtime around the mic -> NS -> VAD -> STT -> Session -> LLM -> TTS pipeline."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import time

import asyncpg
import numpy as np
import redis.asyncio as aioredis
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, JobExecutorType, JobProcess, WorkerOptions, cli

from voxa_shared.db import create_pool, tenant_conn
from voxa_shared.kb.embed import Embedder, Reranker
from voxa_shared.kb.retrieve import Retriever, format_context
from voxa_shared.prompt import render_system_prompt
from voxa_shared.secrets import DbSecretStore
from voxa_shared import voices as vc
from voxa_shared.tenant import TenantConfig, migrate

from .audio.noise import RNNoiseSuppressor
from .audio.playout import Playout, RtcSink
from .audio.resample import Upsampler16to48
from .audio.vad import SileroVAD, VadConfig, VadSegmenter
from .config import from_tenant
from .convlog import ConversationLog
from .escalation import EscalationEngine
from . import tracing
from .metrics import active_sessions, serve_metrics
from .pipeline import InboundPipeline
from .recorder import Recorder
from .resume import SessionStore
from .providers.llm import FallbackLLM, GroqLLM, LLMProvider, MockLLM, RuleBasedMockLLM
from .providers.stt import FasterWhisperSTT, GroqWhisperSTT, StubSTT
from .providers.tts import PiperTTS, WordTiming
from .responder import Responder, ResponderConfig
from .session import Session
from .summary import summarize
from .tools import ToolRuntime
from .turn.detector import HeuristicTurnDetector, LiveKitTurnDetector

log = logging.getLogger("voxa.agent")
SAMPLE_RATE = 48_000
PING_TOPIC = "voxa.ping"
DB_URL = os.environ.get("DATABASE_URL", "postgresql://voxa_app:voxa_app@postgres:5432/voxa")
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
_TENANT_RE = re.compile(r"^[a-z0-9_]{2,40}$")


def _greeting_key(cfg: TenantConfig, text: str) -> str:
    rate = cfg.identity.speaking_rate
    return hashlib.sha1(f"{cfg.tenant_id}|{vc.resolve(cfg.identity.voice_id).id}|{rate}|{text}".encode()).hexdigest()


def _tts_for(ud, cfg: TenantConfig) -> PiperTTS:
    return PiperTTS(voice=ud["voices"][vc.resolve(cfg.identity.voice_id).id], rate=cfg.identity.speaking_rate)


def _synth_greeting(tts: PiperTTS, text: str) -> tuple[np.ndarray, list[WordTiming]]:
    async def go():
        chunks = [c async for c in tts.context().synthesize(text)]
        pcm = np.concatenate([c.pcm for c in chunks])
        return np.clip(Upsampler16to48().process(pcm), -32768, 32767).astype(np.int16), chunks[0].words
    return asyncio.run(go())


_SHARED: dict = {}
_LOCK = threading.Lock()


def _stt_mode() -> str:
    return os.environ.get("STT_PROVIDER") or ("groq" if os.environ.get("GROQ_API_KEY") else "local")


def _load_shared() -> dict:
    """Models shared by every call in this OS process (loaded once, even in thread-executor mode)."""
    with _LOCK:
        if _SHARED:
            return _SHARED
        SileroVAD().prob(np.zeros(512, dtype=np.float32))
        ud: dict = {}
        ud["local_stt"] = FasterWhisperSTT() if _stt_mode() in ("groq", "local") else None
        ud["embedder"], ud["reranker"] = Embedder(), Reranker()
        try:
            ud["detector"] = LiveKitTurnDetector()
        except Exception as e:  # noqa: BLE001
            log.warning("turn-detector model unavailable (%s); using heuristic detector", e)
            ud["detector"] = HeuristicTurnDetector()
        ud["voices"] = {vid: PiperTTS.load_voice(vc.path(v)) for vid, v in vc.VOICES.items()}
        ud["greetings"] = {}

        async def tenants():
            c = await asyncpg.connect(DB_URL, timeout=5)
            try:
                return await c.fetch("SELECT id, config FROM all_tenant_configs()")
            finally:
                await c.close()

        try:
            for row in asyncio.run(tenants()):
                cfg = TenantConfig.model_validate(migrate(json.loads(row["config"])))
                text = " ".join([cfg.scripts.greeting, *cfg.guardrails.compliance_disclaimers])
                ud["greetings"][_greeting_key(cfg, text)] = _synth_greeting(_tts_for(ud, cfg), text)
        except Exception as e:  # noqa: BLE001 - DB not up yet: greetings are synthesized on first call instead
            log.warning("greeting prewarm skipped: %s", e)
        _SHARED.update(ud)
        return _SHARED


def build_llm() -> LLMProvider:
    """Per call: HTTP clients are bound to the call's event loop (the worker may run calls in threads)."""
    provider = os.environ.get("LLM_PROVIDER") or ("groq" if os.environ.get("GROQ_API_KEY") else "rulebased")
    if provider == "groq":
        models = [m.strip() for m in (os.environ.get("GROQ_MODELS") or os.environ.get("GROQ_MODEL")
                  or "qwen/qwen3.8-27b,openai/gpt-oss-20b,openai/gpt-oss-120b").split(",") if m.strip()]
        return FallbackLLM([GroqLLM(os.environ["GROQ_API_KEY"], m) for m in models])
    if provider == "mock":
        return MockLLM()
    log.warning("LLM_PROVIDER=%s: using RuleBasedMockLLM (set GROQ_API_KEY for a real model)", provider)
    return RuleBasedMockLLM()


def build_stt(local: FasterWhisperSTT | None):
    mode = _stt_mode()
    if mode == "stub":
        return StubSTT()
    if mode == "groq":
        return GroqWhisperSTT(os.environ["GROQ_API_KEY"], local, os.environ.get("GROQ_STT_MODEL", "whisper-large-v3-turbo"))
    return local


def prewarm(proc: JobProcess) -> None:
    """Load every model before this process can accept a call, and pre-synthesize each tenant's greeting."""
    proc.userdata.update(_load_shared())
    tracing.init()


async def load_config(pool, redis, tenant_id: str) -> TenantConfig:
    async with tenant_conn(pool, tenant_id) as c:
        row = await c.fetchrow("SELECT config, config_version FROM tenants WHERE id=$1", tenant_id)
    if not row:
        raise LookupError(f"unknown tenant {tenant_id}")
    return TenantConfig.model_validate(migrate(json.loads(row["config"])))


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    room, ud = ctx.room, ctx.proc.userdata
    tenant_id = room.name.rsplit("-", 1)[0]
    if not _TENANT_RE.match(tenant_id):
        log.error("room %s has no valid tenant prefix", room.name)
        await room.disconnect()
        return
    pool = await create_pool(DB_URL, min_size=1, max_size=4)
    redis = aioredis.from_url(REDIS_URL)
    try:
        cfg = await load_config(pool, redis, tenant_id)
    except LookupError:
        log.error("unknown tenant %s", tenant_id)
        await room.disconnect()
        return
    tcfg = from_tenant(cfg)
    tts = _tts_for(ud, cfg)
    llm: LLMProvider = build_llm()
    stt_provider = build_stt(ud["local_stt"])
    asyncio.create_task(llm.warm())
    if hasattr(stt_provider, "warm"):
        asyncio.create_task(stt_provider.warm())

    secrets = DbSecretStore(pool)
    tools = ToolRuntime(cfg, secrets)
    retriever = Retriever(pool, ud["embedder"], ud["reranker"], redis)

    async def kb(query: str) -> tuple[str, int]:
        hits = await retriever.search(tenant_id, query)
        return format_context(hits), len(hits)

    store = SessionStore(redis, room.name, tenant_id)
    resumed = await store.load()
    conv = ConversationLog(pool, tenant_id, room.name, consent_recording=False)
    await conv.start()

    source = rtc.AudioSource(SAMPLE_RATE, 1, queue_size_ms=240)
    await room.local_participant.publish_track(
        rtc.LocalAudioTrack.create_audio_track("agent-audio", source),
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))

    async def send(topic: str, payload: dict, reliable: bool = True) -> None:
        try:
            await room.local_participant.publish_data(json.dumps(payload).encode(), reliable=reliable, topic=topic)
        except Exception:  # noqa: BLE001 - the client left; there is no one to tell
            pass

    session: Session

    async def on_word(gid: int, w: WordTiming) -> None:
        await send("voxa.agent_word", {"gen": gid, "word": w.text}, reliable=False)

    recorder = Recorder(tenant_id, room.name) if tcfg.record_calls else None
    playout = Playout(RtcSink(source), on_word=on_word, on_first_audio=lambda g: session.on_first_audio(g),
                      on_frame=recorder.agent_frame if recorder else None)
    playout.start()
    responder = Responder(
        llm, tts, playout,
        ResponderConfig(pronunciations=cfg.pronunciations, non_interruptible_phrases=tcfg.non_interruptible_phrases,
                        error_line=cfg.scripts.tool_failure),
        tools=tools, fillers=cfg.scripts.fillers, on_tool=lambda k, n, o: session._on_tool_event(k, n, o))
    vad = VadSegmenter(SileroVAD(), VadConfig(tcfg.vad_threshold, tcfg.vad_min_speech_ms, tcfg.vad_min_silence_ms))

    gtext = " ".join([cfg.scripts.greeting, *cfg.guardrails.compliance_disclaimers])
    gkey = _greeting_key(cfg, gtext)
    if gkey not in ud["greetings"]:   # tenant created/edited after worker start: pay the synthesis once
        ud["greetings"][gkey] = await asyncio.to_thread(_synth_greeting, tts, gtext)

    handoff_done = asyncio.Event()

    async def on_handoff(summary: str, reason: str) -> None:
        rec = {"room": room.name, "tenant": tenant_id, "reason": reason, "summary": summary, "ts": time.time()}
        await redis.hset(f"handoff:{tenant_id}", room.name, json.dumps(rec))
        await redis.publish("voxa.handoffs", json.dumps(rec))
        conv.finish(handoff=True, summary=summary)
        if cfg.escalation.sip_transfer_uri and any(
                p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP for p in room.remote_participants.values()):
            await _sip_transfer(room, cfg.escalation.sip_transfer_uri)   # untested against a live trunk

        async def timeout() -> None:
            try:
                await asyncio.wait_for(handoff_done.wait(), 180)
            except asyncio.TimeoutError:
                await responder.say("I'm sorry, nobody is available right now. We'll follow up by email.")
                await responder.wait()
                await session.end("handoff timeout")
        asyncio.create_task(timeout())

    @room.on("participant_connected")
    def _joined(p: rtc.RemoteParticipant) -> None:
        if p.identity.startswith("human-") and session.sm.state.value == "HANDOFF":
            async def deliver() -> None:
                await room.local_participant.publish_data(
                    json.dumps({"summary": session.handoff_summary}).encode(), reliable=True,
                    topic="voxa.handoff_summary", destination_identities=[p.identity])
                handoff_done.set()
                await redis.hdel(f"handoff:{tenant_id}", room.name)
                await responder.say("You're now connected with a colleague.", interruptible=False)
                await responder.wait()
                await session.end("human joined")
            asyncio.create_task(deliver())

    async def on_end() -> None:
        conv.finish()
        await room.disconnect()

    session = Session(
        responder, playout, ud["detector"], tcfg, render_system_prompt(cfg), send,
        set_vad=vad.set_threshold, on_end=on_end, greeting_audio=ud["greetings"][gkey], kb=kb,
        escalation=EscalationEngine(cfg), tools=tools, summarize=lambda h: summarize(llm, h),
        on_handoff=on_handoff, on_turn=lambda rec: _log_turn(conv, rec), redact_pii=cfg.guardrails.pii_policy == "redact",
        on_consent=conv.set_consent, recorder=recorder if not resumed else None,
        resume_history=resumed["history"] if resumed else None, on_history=lambda h: store.update(h))
    store.start_heartbeat()
    stt_stream = await stt_provider.open(cfg.identity.languages[0])
    pipe = InboundPipeline(RNNoiseSuppressor(), vad, stt_stream, session.on_vad, session.on_stt,
                           ref_level=playout.level, echo_gate_ratio=tcfg.echo_gate_ratio,
                           near_field_db=tcfg.near_field_gate_db)
    session.peek = pipe.peek
    session.level_ratio = pipe.segment_ratio
    pipe.start()
    span_cm = tracing.session_span(room.name, tenant_id)
    span_cm.__enter__()
    await session.start()

    async def max_duration() -> None:
        await asyncio.sleep(cfg.guardrails.max_session_minutes * 60)
        await responder.say(cfg.scripts.closing, interruptible=False)
        await responder.wait()
        await session.end("max duration")

    dur_task = asyncio.create_task(max_duration())

    async def ingest(track: rtc.Track) -> None:
        stream = rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=1)
        try:
            async for ev in stream:  # mic ingestion never pauses while the agent speaks (full duplex)
                frame = np.frombuffer(ev.frame.data, dtype=np.int16).copy()
                if recorder:
                    recorder.user_frame(frame)
                pipe.feed(frame)
        finally:
            await stream.aclose()

    ingest_tasks: set[asyncio.Task[None]] = set()

    def start_ingest(track: rtc.Track, participant: rtc.RemoteParticipant) -> None:
        if track.kind == rtc.TrackKind.KIND_AUDIO and not participant.identity.startswith("human-"):
            t = asyncio.create_task(ingest(track))
            ingest_tasks.add(t)
            t.add_done_callback(ingest_tasks.discard)

    @room.on("track_subscribed")
    def _on_track(track: rtc.Track, _p: rtc.RemoteTrackPublication, rp: rtc.RemoteParticipant) -> None:
        start_ingest(track, rp)

    # The server may have subscribed us to the caller's mic while this job was still loading the tenant:
    # events fired before the handler existed are gone, so pick up tracks that are already subscribed.
    for rp in room.remote_participants.values():
        for pub in rp.track_publications.values():
            if pub.track is not None and pub.subscribed:
                start_ingest(pub.track, rp)

    @room.on("data_received")
    def _on_data(pkt: rtc.DataPacket) -> None:
        if pkt.topic == PING_TOPIC:
            asyncio.create_task(room.local_participant.publish_data(pkt.data, reliable=False, topic="voxa.pong"))
        elif pkt.topic == "voxa.csat":
            try:
                score = int(json.loads(pkt.data)["score"])
                if 1 <= score <= 5:
                    asyncio.create_task(conv.set_csat(score))
            except (ValueError, KeyError, TypeError):
                pass
        elif pkt.topic == "voxa.ack":
            m = json.loads(pkt.data)
            session.on_client_ack(m.get("gen", -1), m.get("type", ""))

    async def _step(name: str, coro, timeout: float = 3.0) -> None:
        """One cleanup step must never hang or fail the others: a stuck job holds a thread, memory and CPU forever."""
        try:
            await asyncio.wait_for(coro, timeout)
        except Exception as e:  # noqa: BLE001 - includes TimeoutError
            log.warning("shutdown step %s failed: %r", name, e)

    async def shutdown() -> None:
        dur_task.cancel()
        await _step("session", session.end("shutdown"))
        span_cm.__exit__(None, None, None)
        await _step("playout", playout.stop())
        await _step("pipeline", pipe.aclose())
        if recorder:
            async def _rec():
                rec = await recorder.finish()
                if rec:
                    await conv.add_recording(*rec)
            await _step("recording", _rec(), 20.0)
        conv.finish()
        await _step("conversation log", conv.close(), 6.0)
        await _step("tools", tools.aclose())
        await _step("session store", store.close(clean_end=session.sm.state.value == "ENDED" and not crashed["v"]))
        await _step("redis", redis.aclose())
        try:
            await asyncio.wait_for(pool.close(), 3.0)
        except Exception:  # noqa: BLE001
            pool.terminate()          # never wait on connections a wedged writer still holds
        for t in list(ingest_tasks):
            t.cancel()

    crashed = {"v": False}
    ctx.add_shutdown_callback(shutdown)


def _log_turn(conv: ConversationLog, rec: dict) -> "asyncio.Future[None]":
    conv.turn(rec)
    fut = asyncio.get_running_loop().create_future()
    fut.set_result(None)
    return fut


async def _sip_transfer(room: rtc.Room, uri: str) -> None:
    from livekit import api

    try:
        sip = next(p for p in room.remote_participants.values() if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP)
        async with api.LiveKitAPI() as lk:
            await lk.sip.transfer_sip_participant(api.TransferSIPParticipantRequest(
                participant_identity=sip.identity, room_name=room.name, transfer_to=uri))
    except Exception:  # noqa: BLE001
        log.exception("SIP transfer failed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    threads = os.environ.get("JOB_EXECUTOR", "process") == "thread"
    max_sessions = int(os.environ.get("MAX_SESSIONS", "100"))
    serve_metrics(int(os.environ.get("METRICS_PORT", "9100")))   # from the long-lived main process, not a job process

    def load(w) -> float:
        n = len(w.active_jobs)
        active_sessions.set(n)
        return n / max_sessions

    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint, prewarm_fnc=prewarm, initialize_process_timeout=180,
        agent_name=os.environ.get("AGENT_NAME", "voxa"),   # named: calls are dispatched explicitly (also lets the API re-dispatch)
        num_idle_processes=0 if threads else 2,
        job_executor_type=JobExecutorType.THREAD if threads else JobExecutorType.PROCESS,
        # report load as concurrent calls / capacity so the SFU stops dispatching to a full node (CPU is a lagging signal)
        load_fnc=load, load_threshold=0.95,
        drain_timeout=int(os.environ.get("DRAIN_TIMEOUT_S", "600"))))
