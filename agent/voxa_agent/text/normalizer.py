"""Text normalizer: LLM text -> speakable text for TTS (en locale; locale hook for others)."""
from __future__ import annotations

import re
from datetime import date

from num2words import num2words

_MD_LINK = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")
_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
_CURRENCY = re.compile(r"\$(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d{2}))?")
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_ORDER_ID = re.compile(r"\b(?=[A-Z0-9-]*\d)(?=[A-Z0-9-]*[A-Z])[A-Z0-9]{2,}(?:-[A-Z0-9]+)*\b")
_PERCENT = re.compile(r"(\d+)(?:\.(\d+))?%")
_NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?![\w])")
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
           "October", "November", "December"]


def strip_markdown(t: str) -> str:
    t = _MD_LINK.sub(r"\1", t)
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"(?m)^\s*(?:[-*+]|\d+\.)\s+", "", t)   # list markers
    t = re.sub(r"(?m)^\s*#{1,6}\s*", "", t)
    t = re.sub(r"[*_`~>]+", "", t)
    t = re.sub(r"[\U0001F300-\U0001FAFF☀-➿]", "", t)  # emoji
    return t


def _spell_id(m: re.Match[str]) -> str:
    raw = m.group(0)
    chars = [c for c in raw if c != "-"]
    groups = ["".join(chars[i:i + 3]) for i in range(0, len(chars), 3)]
    return ", ".join(" ".join(g) for g in groups)  # "AC-48213" -> "A C 4, 8 2 1, 3"


def _date(m: re.Match[str]) -> str:
    y, mo, d = int(m[1]), int(m[2]), int(m[3])
    try:
        date(y, mo, d)
    except ValueError:
        return m[0]
    return f"{_MONTHS[mo - 1]} {num2words(d, to='ordinal')}, {num2words(y, to='year')}"


def _money(m: re.Match[str]) -> str:
    whole = int(m[1].replace(",", ""))
    cents = int(m[2]) if m[2] else 0
    out = f"{num2words(whole)} dollar{'s' if whole != 1 else ''}"
    if cents:
        out += f" and {num2words(cents)} cent{'s' if cents != 1 else ''}"
    return out


def _number(m: re.Match[str]) -> str:
    whole = num2words(int(m[1].replace(",", "")))
    if m[2]:
        return whole + " point " + " ".join(num2words(int(c)) for c in m[2])
    return whole


_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), "-")
_QUOTES = dict.fromkeys(map(ord, "\u2018\u2019\u201b"), "'")


def normalize(text: str, locale: str = "en", pronunciations: dict[str, str] | None = None) -> str:
    # LLMs emit typographic dashes and quotes ("AC\u201148213"); without this an order id is read as a cardinal number
    t = strip_markdown(text.translate(_DASHES).translate(_QUOTES))
    t = _URL.sub("", t)
    t = _EMAIL.sub(lambda m: f"{m[1].replace('.', ' dot ')} at {m[2].replace('.', ' dot ')}", t)
    t = _ISO_DATE.sub(_date, t)
    t = _CURRENCY.sub(_money, t)
    t = _ORDER_ID.sub(_spell_id, t)
    t = _PERCENT.sub(lambda m: f"{_number(m)} percent", t)
    t = _NUMBER.sub(_number, t)
    for k, v in (pronunciations or {}).items():
        t = re.sub(rf"\b{re.escape(k)}\b", v, t, flags=re.I)
    return re.sub(r"[ \t]+", " ", t).strip()
