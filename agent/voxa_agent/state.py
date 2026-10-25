"""Explicit conversation state machine. Every legal transition is listed in TRANSITIONS;
anything else raises, so a logic bug can never silently put the call in an impossible state."""
from __future__ import annotations

import logging
from enum import Enum
from typing import Callable

log = logging.getLogger("voxa.state")


class State(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    USER_SPEAKING = "USER_SPEAKING"
    ENDPOINTING = "ENDPOINTING"
    THINKING = "THINKING"
    TOOL_CALL = "TOOL_CALL"
    AGENT_SPEAKING = "AGENT_SPEAKING"
    INTERRUPT_PENDING = "INTERRUPT_PENDING"
    HANDOFF = "HANDOFF"
    ENDED = "ENDED"


S = State
TRANSITIONS: dict[State, frozenset[State]] = {
    S.IDLE: frozenset({S.LISTENING, S.AGENT_SPEAKING, S.ENDED}),
    S.LISTENING: frozenset({S.USER_SPEAKING, S.AGENT_SPEAKING, S.HANDOFF, S.ENDED}),
    S.USER_SPEAKING: frozenset({S.ENDPOINTING, S.LISTENING, S.ENDED}),
    S.ENDPOINTING: frozenset({S.USER_SPEAKING, S.THINKING, S.AGENT_SPEAKING, S.LISTENING, S.ENDED}),
    S.THINKING: frozenset({S.AGENT_SPEAKING, S.TOOL_CALL, S.USER_SPEAKING, S.HANDOFF, S.ENDED}),
    S.TOOL_CALL: frozenset({S.THINKING, S.AGENT_SPEAKING, S.USER_SPEAKING, S.HANDOFF, S.ENDED}),
    S.AGENT_SPEAKING: frozenset({S.LISTENING, S.INTERRUPT_PENDING, S.TOOL_CALL, S.HANDOFF, S.ENDED}),
    S.INTERRUPT_PENDING: frozenset({S.AGENT_SPEAKING, S.USER_SPEAKING, S.LISTENING, S.ENDED}),
    S.HANDOFF: frozenset({S.ENDED}),
    S.ENDED: frozenset(),
}


class IllegalTransition(RuntimeError):
    pass


class StateMachine:
    def __init__(self, on_change: Callable[[State, State, str], None] | None = None) -> None:
        self.state = State.IDLE
        self._on_change = on_change

    def to(self, new: State, reason: str = "") -> bool:
        """Transition; returns False (no-op) for a self-transition, raises if illegal."""
        if new is self.state:
            return False
        if new not in TRANSITIONS[self.state]:
            raise IllegalTransition(f"{self.state.value} -> {new.value} ({reason})")
        old, self.state = self.state, new
        log.debug("state %s -> %s (%s)", old.value, new.value, reason)
        if self._on_change:
            self._on_change(old, new, reason)
        return True

    def can(self, new: State) -> bool:
        return new in TRANSITIONS[self.state]


def mermaid() -> str:
    lines = ["stateDiagram-v2", "  [*] --> IDLE"]
    for a, bs in TRANSITIONS.items():
        lines += [f"  {a.value} --> {b.value}" for b in sorted(bs, key=lambda x: x.value)]
    return "\n".join(lines)
