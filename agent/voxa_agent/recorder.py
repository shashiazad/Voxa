"""Consent-gated call recording: caller + agent audio mixed to mono 16 kHz WAV and uploaded to S3-compatible storage.

Audio is appended to two time-aligned raw files on disk (never held in RAM), mixed at the end. Nothing is captured
until `start()`, which the session calls only after an explicit yes."""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
import wave

import numpy as np

from voxa_shared import storage

from .audio.resample import Decimator48to16

log = logging.getLogger("voxa.recorder")


class _Track:
    """Time-aligned raw int16 writer: silence is inserted for gaps so both tracks share one timeline."""

    def __init__(self, path: str, t0: float) -> None:
        self.f, self.t0, self.samples = open(path, "wb"), t0, 0
        self.dec = Decimator48to16()

    def add(self, pcm48: np.ndarray, t: float) -> None:
        want = int((t - self.t0) * 16000)
        if want > self.samples:                              # gap (agent silent / caller muted): pad
            self.f.write(np.zeros(want - self.samples, dtype=np.int16).tobytes()); self.samples = want
        pcm16 = (np.clip(self.dec.process(pcm48), -1, 1) * 32767).astype(np.int16)
        self.f.write(pcm16.tobytes()); self.samples += len(pcm16)

    def close(self) -> None:
        self.f.close()


class Recorder:
    def __init__(self, tenant_id: str, room: str) -> None:
        self.tenant_id, self.room = tenant_id, room
        self.active = False
        self._dir = tempfile.mkdtemp(prefix="voxa-rec-")
        self._user: _Track | None = None
        self._agent: _Track | None = None

    def start(self) -> None:
        t0 = time.monotonic()
        self._user = _Track(os.path.join(self._dir, "user.raw"), t0)
        self._agent = _Track(os.path.join(self._dir, "agent.raw"), t0)
        self.active = True
        log.info("recording started (consent given) room=%s", self.room)

    def user_frame(self, pcm48: np.ndarray) -> None:
        if self.active:
            self._user.add(pcm48, time.monotonic())

    def agent_frame(self, pcm48: np.ndarray) -> None:
        if self.active:
            self._agent.add(pcm48, time.monotonic())

    async def finish(self) -> tuple[str, float, int] | None:
        """Mix, upload, and delete the temp files. Returns (object_key, seconds, bytes) or None if nothing recorded."""
        if not self.active:
            self._cleanup()
            return None
        self.active = False
        self._user.close(); self._agent.close()
        key = f"recordings/{self.tenant_id}/{self.room}.wav"
        try:
            wav = os.path.join(self._dir, "mix.wav")
            n = await asyncio.to_thread(self._mix, wav)
            await asyncio.to_thread(storage.ensure_bucket)
            await asyncio.to_thread(storage.put_file, key, wav)
            return key, n / 16000, os.path.getsize(wav)
        except Exception:  # noqa: BLE001
            log.exception("recording upload failed; recording discarded")
            return None
        finally:
            self._cleanup()

    def _mix(self, out: str) -> int:
        u = np.memmap(os.path.join(self._dir, "user.raw"), dtype=np.int16, mode="r") if self._user.samples else np.zeros(0, np.int16)
        a = np.memmap(os.path.join(self._dir, "agent.raw"), dtype=np.int16, mode="r") if self._agent.samples else np.zeros(0, np.int16)
        n = max(len(u), len(a))
        with wave.open(out, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            for i in range(0, n, 16000 * 30):                 # 30 s at a time
                x = np.zeros(min(16000 * 30, n - i), dtype=np.int32)
                x[:len(u[i:i + len(x)])] += u[i:i + len(x)]
                x[:len(a[i:i + len(x)])] += a[i:i + len(x)]
                w.writeframes(np.clip(x, -32768, 32767).astype(np.int16).tobytes())
        return n

    def _cleanup(self) -> None:
        import shutil
        shutil.rmtree(self._dir, ignore_errors=True)
