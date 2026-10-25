"""Escalation rules: when the agent should hand the caller to a person."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from voxa_shared.prompt import business_open
from voxa_shared.tenant import TenantConfig

_NEG = {"terrible": 2, "awful": 2, "horrible": 2, "worst": 2, "ridiculous": 2, "useless": 2, "angry": 2,
        "furious": 3, "unacceptable": 3, "scam": 3, "fraud": 3, "sick": 1, "tired": 1, "frustrated": 2,
        "frustrating": 2, "annoyed": 1, "annoying": 1, "hate": 2, "stupid": 2, "waste": 1, "never": 1,
        "again": 1, "cancel": 1, "refund": 0, "damn": 2, "hell": 1, "crap": 2}
_PROFANE = re.compile(r"\b(fuck\w*|shit\w*|bullshit|asshole)\b", re.I)


@dataclass
class Decision:
    reason: str            # human_request | sentiment | failures | restricted_topic
    line: str              # what to say
    handoff: bool = True   # False => after-hours: offer a ticket instead of transferring


class EscalationEngine:
    def __init__(self, cfg: TenantConfig, clock=None) -> None:
        self.cfg, self._clock = cfg, clock
        self._human = [re.compile(p, re.I) for p in cfg.escalation.human_request_patterns]
        self._restricted = [t.lower() for t in cfg.escalation.restricted_topics]
        self.neg_score = 0
        self.failures = 0

    def record_failure(self) -> None:
        self.failures += 1

    def record_success(self) -> None:
        self.failures = 0

    def sentiment(self, text: str) -> int:
        words = re.findall(r"[a-z']+", text.lower())
        score = sum(_NEG.get(w, 0) for w in words) + 3 * len(_PROFANE.findall(text))
        if "!" in text and score:
            score += 1
        return score

    def evaluate(self, text: str, now: datetime | None = None) -> Decision | None:
        e = self.cfg.escalation
        low = text.lower()
        for topic in self._restricted:
            if topic in low:
                return Decision("restricted_topic", e.restricted_line)
        asks_human = any(p.search(text) for p in self._human)
        self.neg_score = max(0, self.neg_score // 2) + self.sentiment(text)   # decays between turns
        reason = None
        if asks_human:
            reason = "human_request"
        elif self.neg_score >= e.negative_sentiment_threshold:
            reason = "sentiment"
        elif self.failures >= e.max_consecutive_failures:
            reason = "failures"
        if not reason:
            return None
        if not business_open(self.cfg, now or (self._clock() if self._clock else None)):
            return Decision(reason, e.after_hours_line, handoff=False)
        return Decision(reason, e.handoff_line)
