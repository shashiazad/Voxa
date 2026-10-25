"""Per-turn timing marks and Prometheus metrics."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from prometheus_client import Counter, Gauge, Histogram

_B = (.05, .1, .2, .3, .5, .8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0)
voice_to_voice = Histogram("voice_to_voice_latency_seconds", "eos_detected -> first_audio_played", buckets=_B)
llm_ttft = Histogram("llm_ttft_seconds", "LLM time to first token", buckets=_B)
tts_ttfb = Histogram("tts_ttfb_seconds", "TTS time to first audio", buckets=_B)
stt_latency = Histogram("stt_latency_seconds", "eos_detected -> stt_final", buckets=_B)
interruptions = Counter("voxa_interruption_total", "Barge-in outcomes", ["outcome"])
errors = Counter("voxa_error_total", "Errors by stage (stt, llm, tts, tool)", ["stage"])
calls_total = Counter("voxa_calls_total", "Calls handled")
handoffs_total = Counter("voxa_handoffs_total", "Calls handed to a human", ["reason"])
active_sessions = Gauge("voxa_active_sessions", "Concurrent calls on this worker", multiprocess_mode="livemax")
session_seconds = Histogram("voxa_session_seconds", "Call duration", buckets=(10, 30, 60, 120, 300, 600, 1800))
dropped_frames = Counter("voxa_audio_frames_dropped_total", "Inbound audio frames shed under overload")
ns_vad_frame = Histogram("voxa_ns_vad_frame_seconds", "Noise suppression + VAD time per 10 ms frame",
                         buckets=(.0005, .001, .002, .004, .008, .016, .03, .06))

MARKS = ("eos_detected", "turn_committed", "stt_final", "llm_request", "llm_first_token", "tts_first_byte",
         "first_audio_played")


@dataclass
class TurnTimings:
    generation_id: int
    marks: dict[str, float] = field(default_factory=dict)  # monotonic seconds
    info: dict[str, str] = field(default_factory=dict)     # e.g. which LLM answered (the chain may have fallen back)

    def mark(self, name: str, t: float | None = None) -> None:
        assert name in MARKS, name
        self.marks.setdefault(name, time.monotonic() if t is None else t)

    def delta_ms(self, a: str, b: str) -> float | None:
        if a in self.marks and b in self.marks:
            return (self.marks[b] - self.marks[a]) * 1000
        return None

    def publish(self) -> dict[str, float | None]:
        out = {"voice_to_voice_ms": self.delta_ms("eos_detected", "first_audio_played"), **self.info,
               "stt_ms": self.delta_ms("eos_detected", "stt_final"),
               "llm_ttft_ms": self.delta_ms("llm_request", "llm_first_token"),
               "tts_ttfb_ms": self.delta_ms("llm_first_token", "tts_first_byte")}
        for hist, key in ((voice_to_voice, "voice_to_voice_ms"), (llm_ttft, "llm_ttft_ms"),
                          (tts_ttfb, "tts_ttfb_ms"), (stt_latency, "stt_ms")):
            if out[key] is not None:
                hist.observe(out[key] / 1000)  # type: ignore[operator]
        return out


def serve_metrics(port: int) -> None:
    """Expose /metrics once per container. Job processes are separate, so with
    PROMETHEUS_MULTIPROC_DIR set the first process to bind aggregates all of them."""
    import os

    from prometheus_client import CollectorRegistry, multiprocess, start_http_server

    registry = None
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
    try:
        start_http_server(port, registry=registry) if registry else start_http_server(port)
    except OSError:
        pass  # another worker process already serves it
