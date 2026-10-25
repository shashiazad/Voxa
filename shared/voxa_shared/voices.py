"""Voice registry shared by the agent (synthesis) and API (preview). Files live in $VOICE_DIR."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Voice:
    id: str
    name: str
    description: str
    file: str            # Piper .onnx file name
    lang: str = "en"


VOICES: dict[str, Voice] = {v.id: v for v in (
    Voice("amy", "Amy", "Warm, clear American English (female)", "en_US-amy-low.onnx"),
    Voice("ryan", "Ryan", "Calm, steady American English (male)", "en_US-ryan-low.onnx"),
    Voice("lessac", "Lessac", "Bright, articulate American English (female)", "en_US-lessac-low.onnx"),
)}
DEFAULT = "amy"


def resolve(voice_id: str) -> Voice:
    """'default' and unknown ids map to the default voice (a bad config must never break a call)."""
    return VOICES.get(voice_id) or VOICES[DEFAULT]


def path(voice: Voice) -> str:
    return os.path.join(os.environ.get("VOICE_DIR", "/app/models"), voice.file)
