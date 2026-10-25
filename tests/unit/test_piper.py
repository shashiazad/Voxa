from pathlib import Path

import numpy as np
import pytest

MODEL = Path(__file__).parents[2] / "models" / "en_US-amy-low.onnx"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="piper voice not downloaded")


async def test_piper_synthesizes_with_running_clock():
    from voxa_agent.providers.tts import PiperTTS
    tts = PiperTTS(str(MODEL))
    ctx = tts.context()
    a = [c async for c in ctx.synthesize("Sure, let me check.")]
    b = [c async for c in ctx.synthesize("One moment please.")]
    assert a[0].sample_rate == 16000 and len(a[0].pcm) > 8000
    assert np.abs(a[0].pcm).max() > 1000
    assert b[0].words[0].start_ms == pytest.approx(a[0].words[-1].end_ms, abs=1)
