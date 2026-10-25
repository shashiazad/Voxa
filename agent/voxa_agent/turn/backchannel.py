"""Backchannel / acknowledgment classifier: these never interrupt the agent."""
from __future__ import annotations

import re
import unicodedata

# Single-token acknowledgments per language. Extend per tenant via `extra`.
LEXICON: dict[str, frozenset[str]] = {
    "en": frozenset("mm mmm hmm hm mhm uh um uhh uhm huh yeah yep yes yup ok okay right sure aha ah oh "
                    "alright fine cool great good nice gotcha hi hello hey thanks thank".split()),
    "es": frozenset("sí si vale ajá aja claro bueno okey ya entiendo mhm eh".split()),
    "fr": frozenset("oui ouais ok d'accord daccord mhm hum voilà voila bon ah ben".split()),
    "de": frozenset("ja jap jo genau okay ok mhm hm aha klar gut stimmt".split()),
    "hi": frozenset("haan han ji achha accha theek thik hmm ok okay haanji".split()),
    "pt": frozenset("sim ok certo tá ta entendi aham uhum".split()),
}
PHRASES: frozenset[str] = frozenset({
    "uh huh", "mm hmm", "i see", "got it", "go on", "of course", "i understand", "makes sense",
    "that's right", "thats right", "sounds good", "no problem", "d accord", "ok ok", "yeah yeah",
})


def tokens(text: str) -> list[str]:
    t = unicodedata.normalize("NFKC", text.lower()).replace("’", "'")
    t = re.sub(r"[^\w'\s]", " ", t)  # hyphens too: "mm-hmm" -> "mm hmm"
    return [w.strip("-'") for w in t.split() if w.strip("-'")]


def _vocab(langs: tuple[str, ...], extra: frozenset[str]) -> frozenset[str]:
    return frozenset().union(*(LEXICON.get(l, frozenset()) for l in langs), extra)


def content_words(text: str, langs: tuple[str, ...] = ("en",), extra: frozenset[str] = frozenset()) -> list[str]:
    """Words that are NOT acknowledgment filler."""
    toks = tokens(text)
    joined = " ".join(toks)
    for ph in PHRASES:
        joined = re.sub(rf"\b{re.escape(ph)}\b", " ", joined)
    vocab = _vocab(langs, extra)
    return [w for w in joined.split() if w not in vocab]


def is_backchannel(text: str, langs: tuple[str, ...] = ("en",), extra: frozenset[str] = frozenset()) -> bool:
    """True when the utterance is non-empty and consists only of acknowledgments."""
    return bool(tokens(text)) and not content_words(text, langs, extra)


# One-word commands that interrupt on their own (the caller is taking the floor, not acknowledging).
COMMANDS: dict[str, frozenset[str]] = {
    "en": frozenset("stop wait hold pause sorry excuse actually hang no nope cancel enough".split()),
    "es": frozenset("espera espere para alto perdón perdon momento no basta".split()),
    "fr": frozenset("attendez attends arrête arrete stop pardon non attention".split()),
    "de": frozenset("warte warten stopp halt moment nein entschuldigung".split()),
    "hi": frozenset("ruko rukiye ruk nahi".split()),
    "pt": frozenset("espera pare pará calma não nao".split()),
}


def is_interrupt_command(text: str, langs: tuple[str, ...] = ("en",)) -> bool:
    toks = tokens(text)
    vocab = frozenset().union(*(COMMANDS.get(l, frozenset()) for l in langs))
    return bool(toks) and any(t in vocab for t in toks[:3])   # "wait", "hold on", "sorry, actually..."
