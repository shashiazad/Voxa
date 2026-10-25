# Architecture

## Decisions and why

| Area | Choice | Why / trade-off |
|---|---|---|
| Transport | LiveKit SFU (v1.13.10) + coturn, Opus 48 kHz / 20 ms | One stack for WebRTC, tokens, agent dispatch, SIP. An older server (v1.9) stalled browser mic negotiation for ~15 s with `livekit-client` 2.23: keep server and client versions current. |
| Agent runtime | Python 3.11, asyncio, LiveKit Agents 1.8.9 for worker lifecycle only | Pipecat rejected: LiveKit gives SFU, SIP, dispatch and drain in one place. The turn-taking logic (preemptive generation, two-stage barge-in, `generation_id`, history truncation) is our own `Session`, because `AgentSession` does not expose two-stage interruption. |
| LLM | Groq: `qwen/qwen3.8-27b` (measured first-token ≈ 120–190 ms warm) → `openai/gpt-oss-20b` → `openai/gpt-oss-120b` fallback chain | No Anthropic key was available; `LLMProvider` keeps Claude a drop-in. Free-tier token limits are per model, so the chain also absorbs 429s. |
| STT | Local faster-whisper for interims; Groq `whisper-large-v3-turbo` for finals (≈ 250 ms) | Whisper is not streaming. Interims drive preemptive generation and barge-in text; the cloud final is accurate and fast. Local final is the fallback (circuit breaker). |
| TTS | Local Piper (3 voices) | Free. No word alignment (timings estimated), first audio ≈ 200–350 ms. A streaming cloud TTS is the main remaining latency lever. |
| Noise suppression | RNNoise (pyrnnoise) before VAD | Krisp (incl. background-voice cancellation) is licensed; not included. |
| VAD / turn detection | Silero VAD ONNX (512-sample windows) + LiveKit turn-detector ONNX + a trailing-function-word veto | The model trusts ASR punctuation, and Whisper punctuates fragments ("My order number is."). |
| Data | Postgres 17 + pgvector, Redis, S3-compatible (SeaweedFS in compose) | RLS on every tenant table (forced); the app role cannot bypass it. |
| RAG | fastembed bge-small (384-d) + Postgres FTS → RRF → cross-encoder rerank; Redis cache; prefetch on interim transcripts | uncached p95 ≈ 67 ms (budget 150 ms) |
| Frontend | Next.js 16 console (httpOnly cookie session, API proxy), esbuild widget bundle (Shadow DOM) | The widget bundle includes `livekit-client` (≈ 590 kB, ≈ 150 kB gzip). |
| Observability | Prometheus (multiprocess-safe), Grafana dashboard, OTel traces to Jaeger | Metrics are served by the long-lived worker process; the active-session gauge is set from the worker's load function. |

## The Session (agent/voxa_agent/session.py)

States are explicit (`state.py`, diagram in `STATE_MACHINE.md`); every legal edge and every illegal edge is unit-tested.

* **Endpointing**: VAD end of speech starts a timer; a semantically complete utterance commits after ~300 ms of silence, an
  unfinished one waits up to 1.5 s. Commit also waits for the final transcript.
* **Preemptive generation**: at end of speech the LLM + TTS start on the best transcript with audio *held*; commit releases it,
  a materially different final restarts it, the caller resuming cancels it. Side-effecting tools never run on an uncommitted turn.
* **Barge-in (two stages)**: ≥ 160 ms of speech while the agent talks → soft pause (20 ms fade; unheard audio is clawed back so
  resume is lossless). Confirm on ≥ 2 content words, an interrupt command ("wait", "stop"), or long speech; otherwise (cough, door
  slam, backchannel, echo, background voice) resume. Cloud STT is *asked* what is being said once a pause is pending ("peek").
* **Not-a-caller defences while the agent speaks**: raised VAD threshold; echo gate (mic far below our own output level);
  near-field gate on speech onsets (far below the caller's learned level); echo-text match (transcript ⊂ what we are saying,
  compared in spoken form, function words ignored); segment-loudness check before confirming by words.
* **History truncation**: TTS word timings + the playout position cut the assistant message to what was heard, plus `[interrupted]`.
* **Recovery**: low-confidence → clarifying question; 8 s / 20 s silence handling; LLM failure → spoken failure line; STT/LLM fallback
  chains with circuit breakers; conversation state in Redis (60 s TTL, heartbeat) so a new worker can resume the call.

## Multi-tenancy

`TenantConfig` (Pydantic v2, versioned) is the single source of truth: identity, voice, scripts, prompt template, tools,
escalation, turn-taking, guardrails. The agent loads it per call from the room-name prefix; the API validates every edit
(including SSRF checks on tool URLs). Retrieved text and tool results are wrapped as untrusted data in the prompt.

## Scaling model

One worker process runs many calls as asyncio tasks (thread executor): models are loaded once per process, per-call state is small.
The worker reports load as `active_calls / MAX_SESSIONS`, so the SFU stops dispatching to a full pod. Capacity per pod comes from the
load test (see the report); autoscale pods on `voxa_active_sessions`. Python's GIL bounds one process, so scale out with replicas
rather than bigger pods.
