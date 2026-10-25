"""Tenant-configurable turn-taking parameters (a slice of TenantConfig, finalized in phase 5)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TurnConfig:
    # --- VAD ---
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 96
    vad_min_silence_ms: int = 160
    # --- endpointing (measured from the start of silence) ---
    endpoint_complete_ms: int = 300        # semantically complete utterance
    endpoint_incomplete_ms: int = 1500     # "my order number is..."
    preemptive_generation: bool = True
    stt_final_timeout_ms: int = 2500       # give up waiting for a final after the deadline
    # --- barge-in ---
    vad_threshold_speaking: float = 0.65   # raised while the agent talks (echo safety)
    stage1_min_speech_ms: int = 160        # soft-pause trigger
    confirm_min_words: int = 2             # stage 2: >= N non-backchannel words
    confirm_speech_ms: int = 600           # ...or speech this long
    confirm_hard_speech_ms: int = 1200     # duration confirm without any transcript
    false_interrupt_timeout_ms: int = 1200 # no words recognized -> resume playback
    min_interrupt_confidence: float = 0.65 # ignore low-confidence short transcripts while agent speaks
    near_field_gate_db: float = 9.0       # while the agent talks, ignore sound this far below the caller's usual level (0 = off)
    background_level_ratio: float = 0.4   # speech this much quieter than the caller's usual level never confirms an interruption
    echo_gate_ratio: float = 0.05          # mic RMS below this fraction of our own output RMS is treated as echo
    # --- silence / recovery ---
    reprompt_after_s: float = 8.0
    close_after_s: float = 20.0
    low_confidence: float = 0.45
    languages: tuple[str, ...] = ("en",)
    extra_backchannels: frozenset[str] = frozenset()
    non_interruptible_phrases: tuple[str, ...] = ()
    record_calls: bool = False             # ask for consent, record only after an explicit yes
    recording_prompt: str = "Before we start: may I record this call to improve our service? Please say yes or no."
    recording_yes: str = "Thank you. This call is now being recorded. How can I help?"
    recording_no: str = "No problem, I won't record this call. How can I help?"
    resume_line: str = "Sorry about that, I lost the connection for a moment. Where were we?"
    # --- scripted lines ---
    greeting: str = "Hi, thanks for calling. How can I help you today?"
    disclaimers: tuple[str, ...] = ()       # spoken after the greeting, non-interruptible
    reprompt_line: str = "Are you still there?"
    closing_line: str = "I haven't heard anything, so I'll end the call. Goodbye."
    clarify_line: str = "Sorry, I didn't quite catch that. Could you say it again?"


def from_tenant(t) -> TurnConfig:
    """Build the runtime TurnConfig from a validated TenantConfig."""
    p, sc = t.turn, t.scripts
    return TurnConfig(
        vad_threshold=p.vad_threshold, vad_threshold_speaking=p.vad_threshold_speaking,
        vad_min_speech_ms=p.vad_min_speech_ms, vad_min_silence_ms=p.vad_min_silence_ms,
        endpoint_complete_ms=p.endpoint_complete_ms, endpoint_incomplete_ms=p.endpoint_incomplete_ms,
        stage1_min_speech_ms=p.stage1_min_speech_ms, echo_gate_ratio=p.echo_gate_ratio, near_field_gate_db=p.near_field_gate_db, confirm_min_words=p.interruption_min_words,
        reprompt_after_s=p.reprompt_after_s, close_after_s=p.close_after_s,
        languages=tuple(t.identity.languages), extra_backchannels=frozenset(p.extra_backchannels),
        non_interruptible_phrases=tuple(p.non_interruptible_phrases),
        greeting=sc.greeting, disclaimers=tuple(t.guardrails.compliance_disclaimers), record_calls=t.guardrails.record_calls,
        recording_prompt=sc.recording_prompt, recording_yes=sc.recording_yes, recording_no=sc.recording_no, resume_line=sc.resume_line,
        reprompt_line=sc.reprompt, closing_line=sc.silence_closing, clarify_line=sc.clarify)
