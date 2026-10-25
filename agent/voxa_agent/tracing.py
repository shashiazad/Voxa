"""OpenTelemetry traces: one span per session, one child span per turn with the latency marks as events.
Off (no-op) unless OTEL_EXPORTER_OTLP_ENDPOINT is set, so tests and local runs pay nothing."""
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Iterator

log = logging.getLogger("voxa.tracing")
_tracer = None
_session_ctx: dict[str, Any] = {}


def init(service: str = "voxa-agent") -> None:
    global _tracer
    ep = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if _tracer is not None or not ep:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create({"service.name": service}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=ep.rstrip("/") + "/v1/traces")))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("voxa")
    except Exception:  # noqa: BLE001 - tracing must never break a call
        log.exception("tracing init failed; continuing without traces")


@contextmanager
def session_span(room: str, tenant: str) -> Iterator[None]:
    if _tracer is None:
        yield
        return
    from opentelemetry import trace
    from opentelemetry.context import attach, detach, get_current

    span = _tracer.start_span("voxa.session", attributes={"voxa.room": room, "voxa.tenant": tenant})
    _session_ctx["ctx"] = trace.set_span_in_context(span)
    try:
        yield
    finally:
        span.end()
        _session_ctx.clear()


def record_turn(gen: int, marks: dict[str, float], metrics: dict[str, Any]) -> None:
    """Emit a finished turn as a span. `marks` are time.monotonic() seconds; converted to wall-clock ns."""
    if _tracer is None or "eos_detected" not in marks:
        return
    off = time.time() - time.monotonic()
    ns = lambda t: int((t + off) * 1e9)  # noqa: E731
    end = max(marks.values())
    span = _tracer.start_span("voxa.turn", context=_session_ctx.get("ctx"), start_time=ns(marks["eos_detected"]),
                              attributes={"voxa.generation": gen, **{f"voxa.{k}": v for k, v in metrics.items() if v is not None and k != "gen"}})
    for name, t in sorted(marks.items(), key=lambda kv: kv[1]):
        span.add_event(name, timestamp=ns(t))
    span.end(end_time=ns(end))
