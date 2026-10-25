import pytest

from voxa_agent.text.chunker import ClauseChunker
from voxa_agent.text.normalizer import normalize, strip_markdown


@pytest.mark.parametrize("src,expected", [
    ("**Sure**, here's [the link](https://x.com/a)!", "Sure, here's the link!"),
    ("- item one\n- item two", "item one\nitem two"),
    ("See https://acme.com/help now", "See now"),
    ("It costs $1,234.50 today", "It costs one thousand, two hundred and thirty-four dollars and fifty cents today"),
    ("Pay $1", "Pay one dollar"),
    ("Mail jo.doe@acme.com please", "Mail jo dot doe at acme dot com please"),
    ("On 2026-10-24 it ships", "On October twenty-fourth, twenty twenty-six it ships"),
    ("Order AC-48213 shipped", "Order A C four, eight two one, three shipped"),
    ("Save 15% now", "Save fifteen percent now"),
    ("You have 3 items", "You have three items"),
    ("Version 3.5 is out", "Version three point five is out"),
    ("Great 😀 news", "Great news"),
    ("Order AC\u201148213 shipped", "Order A C four, eight two one, three shipped"),     # non-breaking hyphen from an LLM
    ("It\u2019s on its way", "It's on its way"),
])
def test_normalize(src, expected):
    assert normalize(src) == expected


def test_pronunciation_dictionary():
    assert normalize("Call Acme support", pronunciations={"Acme": "Ack-mee"}) == "Call Ack-mee support"


def test_strip_markdown_headers_and_code():
    assert "code" not in strip_markdown("```code```ok")


def feed_all(tokens, **kw):
    c = ClauseChunker(**kw)
    out = []
    for t in tokens:
        out += c.feed(t)
    return out + c.flush()


def test_first_chunk_at_punctuation():
    toks = ["Sure", ",", " I", " can", " help", " you", " with", " that", ".", " What", " is", " it", "?"]
    out = feed_all(toks)
    assert out == ["Sure, I can help you with that.", "What is it?"]


def test_short_lead_in_waits_for_more_words():
    out = feed_all(["Sure", ", ", "let", " me", " check", ".", " "])
    assert out[0] == "Sure, let me check."


def test_word_limit_cuts_long_clause():
    words = [f" w{i}" for i in range(30)]
    out = feed_all(["Start"] + words, first_max_words=10)
    assert len(out[0].split()) == 10


def test_money_and_decimals_never_split():
    out = feed_all(["It", " is", " $1", ",234", ".50", " total", " today", ". ", "Ok"])
    assert out[0].startswith("It is $1,234.50 total today.")


def test_flush_returns_tail():
    c = ClauseChunker()
    assert c.feed("Hello there") == []
    assert c.flush() == ["Hello there"]
