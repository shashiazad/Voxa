"""Two-stage barge-in controller.

Stage 1 (soft pause): VAD speech held >= stage1_min_speech_ms while the agent talks -> pause playout.
Stage 2 (confirm): >= N non-backchannel words, or speech >= confirm_speech_ms (needs some transcript, or
confirm_hard_speech_ms without one) -> cancel everything. Otherwise (cough, door slam, backchannel,
echo blip) -> resume from the paused position.
"""
from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Awaitable, Callable

from .config import TurnConfig
from .providers.stt import STTEvent
from .turn.backchannel import content_words, is_backchannel, is_interrupt_command, tokens

log = logging.getLogger("voxa.bargein")


class Outcome(str, Enum):
    CONFIRMED = "confirmed"
    FALSE = "false"
    BACKCHANNEL = "backchannel"
    ECHO = "echo"


class BargeInController:
    def __init__(self, cfg: TurnConfig, pause: Callable[[], Awaitable[None]],
                 confirm: Callable[[str], Awaitable[None]],
                 resume: Callable[[Outcome], Awaitable[None]],
                 is_echo: Callable[..., bool] = lambda text, confidence=1.0: False,
                 echo_suppress_s: float = 1.0,
                 peek: Callable[[], Awaitable[None]] | None = None,
                 peek_available: Callable[[], bool] | None = None,
                 level_ratio: Callable[[], float | None] | None = None) -> None:
        self._level_ratio = level_ratio or (lambda: None)
        self._peek, self._last_peek = peek, 0.0
        self._peek_ok = peek_available or (lambda: peek is not None)
        self.cfg, self._pause, self._confirm, self._resume = cfg, pause, confirm, resume
        self._is_echo, self._echo_suppress_s = is_echo, echo_suppress_s
        self._suppress_until = 0.0
        self._any_text = False
        self.active = False
        self._speaking = False
        self._text = ""
        self._t0 = 0.0
        self._timer: asyncio.Task[None] | None = None

    @property
    def suppressed(self) -> bool:
        return time.monotonic() < self._suppress_until

    async def on_speech_start(self) -> None:
        """VAD fired (already gated by the raised threshold + stage-1 min speech)."""
        if self.active:
            self._speaking = True
            return
        if time.monotonic() < self._suppress_until:
            return  # we just identified our own voice coming back; do not re-pause for it
        self.active, self._speaking, self._text, self._t0 = True, True, "", time.monotonic()
        self._any_text = False
        await self._pause()
        self._timer = asyncio.create_task(self._watch())

    async def on_speech_end(self) -> None:
        if not self.active:
            return
        self._speaking = False
        if self._text and is_backchannel(self._text, self.cfg.languages, self.cfg.extra_backchannels):
            await self._finish(Outcome.BACKCHANNEL)

    async def on_stt(self, ev: STTEvent) -> None:
        if not self.active or not ev.text.strip():
            return
        words = tokens(ev.text)
        self._any_text = True
        if self._is_echo(ev.text, ev.confidence):
            # the mic is hearing the agent itself: not a user interruption
            self._suppress_until = time.monotonic() + self._echo_suppress_s
            await self._finish(Outcome.ECHO)
            return
        ratio0 = self._level_ratio()
        if (ev.confidence >= 0.4 and (ratio0 is None or ratio0 >= self.cfg.background_level_ratio)
                and is_interrupt_command(ev.text, self.cfg.languages)
                and not self._is_echo(ev.text, ev.confidence)):
            self._text = ev.text
            await self._finish(Outcome.CONFIRMED)      # "wait" / "stop": a command, even if the recogniser is unsure
            return
        if ev.confidence < self.cfg.min_interrupt_confidence and len(words) < self.cfg.confirm_min_words + 1:
            return  # low-confidence short transcript while agent speaks: treat as noise
        self._text = ev.text
        n = len(content_words(ev.text, self.cfg.languages, self.cfg.extra_backchannels))
        ratio = self._level_ratio()
        if ratio is not None and ratio < self.cfg.background_level_ratio:
            return    # intelligible words, but far quieter than this caller: a TV / crowd / other room, not them
        if n >= self.cfg.confirm_min_words or is_interrupt_command(ev.text, self.cfg.languages):
            await self._finish(Outcome.CONFIRMED)
        elif ev.is_final and not self._speaking:
            await self._finish(Outcome.BACKCHANNEL if is_backchannel(
                ev.text, self.cfg.languages, self.cfg.extra_backchannels) else Outcome.FALSE)

    async def _watch(self) -> None:
        """Duration-based confirmation and the false-interruption timeout."""
        try:
            while self.active:
                await asyncio.sleep(0.02)
                el = (time.monotonic() - self._t0) * 1000 + self.cfg.stage1_min_speech_ms  # VAD already held it
                if self._peek and self._peek_ok() and self._speaking and el >= 450 and time.monotonic() - self._last_peek >= 0.6:
                    self._last_peek = time.monotonic()      # ask the accurate (cloud) STT what is being said so far
                    asyncio.create_task(self._peek())
                if self._speaking:
                    # No transcript at all: with a cloud STT peek, silence from the recogniser means noise or garbled
                    # echo, so only a very long burst counts; with local interims keep the shorter limit.
                    hard = 3000 if self._peek_ok() else self.cfg.confirm_hard_speech_ms
                    need = self.cfg.confirm_speech_ms if self._text else hard
                    if not self._text and self._any_text:
                        continue  # STT heard only low-confidence scraps: noise/echo, not a user
                    ratio = self._level_ratio()
                    if ratio is not None and ratio < self.cfg.background_level_ratio:
                        continue  # long, but far quieter than this caller: a TV / crowd, never a duration-based confirm
                    if el >= need and not (self._text and is_backchannel(
                            self._text, self.cfg.languages, self.cfg.extra_backchannels)):
                        await self._finish(Outcome.CONFIRMED)
                        return
                elif el >= self.cfg.false_interrupt_timeout_ms:
                    await self._finish(Outcome.FALSE)
                    return
        except asyncio.CancelledError:
            pass

    async def _finish(self, outcome: Outcome) -> None:
        if not self.active:
            return
        self.active = False
        if self._timer and self._timer is not asyncio.current_task():
            self._timer.cancel()
        text = self._text
        log.info("barge-in %s text=%r", outcome.value, text)
        if outcome is Outcome.CONFIRMED:
            await self._confirm(text)
        else:
            await self._resume(outcome)

    def cancel(self) -> None:
        self.active = False
        if self._timer and self._timer is not asyncio.current_task():
            self._timer.cancel()
