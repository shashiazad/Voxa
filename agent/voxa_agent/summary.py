"""Auto-generated conversation summary for human handoff."""
from __future__ import annotations

import logging

from .providers.llm import LLMProvider

log = logging.getLogger("voxa.summary")
PROMPT = ("You write handoff notes for a human support agent. Summarize this call in at most four short "
          "sentences: the customer's issue, what has been tried or looked up, the customer's mood, and what they "
          "need next. Include order numbers or ticket IDs. Do not include card numbers or passwords.")


def rule_summary(history: list[dict[str, str]]) -> str:
    users = [m["content"] for m in history if m["role"] == "user"]
    agent = [m["content"] for m in history if m["role"] == "assistant"]
    return ("Customer said: " + " / ".join(users[-3:])[:400] + ". Last agent reply: " + (agent[-1] if agent else "none"))[:700]


async def summarize(llm: LLMProvider, history: list[dict[str, str]]) -> str:
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-20:])
    out = ""
    try:
        async for d in llm.stream([{"role": "system", "content": PROMPT}, {"role": "user", "content": convo}],
                                  max_tokens=160):
            out += d.text
    except Exception:  # noqa: BLE001
        log.warning("LLM summary failed; using rule-based summary")
    out = out.strip()
    if llm.name.endswith("mock") or not out:
        return rule_summary(history)   # mocks are not summarizers
    return out
