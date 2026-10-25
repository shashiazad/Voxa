import pytest

from voxa_agent.providers.tts import WordTiming
from voxa_agent.turn.backchannel import content_words, is_backchannel
from voxa_agent.turn.detector import HeuristicTurnDetector
from voxa_agent.turn.history import SpokenChunk, heard_text, history_message


@pytest.mark.parametrize("t", ["mm-hmm", "yeah", "Okay.", "uh huh", "right, right", "yeah okay sure", "got it",
                               "mhm", "I see"])
def test_backchannels(t):
    assert is_backchannel(t)


@pytest.mark.parametrize("t", ["wait stop", "no that's wrong", "yeah but I need a refund", "", "hold on"])
def test_not_backchannels(t):
    assert not is_backchannel(t)


@pytest.mark.parametrize("lang,t", [("es", "sí claro"), ("fr", "oui d'accord"), ("de", "ja genau"),
                                    ("hi", "haan ji"), ("pt", "sim certo")])
def test_multilingual_backchannels(lang, t):
    assert is_backchannel(t, (lang,))
    assert not is_backchannel(t)  # English-only config does not swallow them


def test_content_words_ignores_filler():
    assert content_words("yeah okay but wait") == ["but", "wait"]


def test_tenant_extra_backchannels():
    assert is_backchannel("roger", extra=frozenset({"roger"}))


def test_heuristic_detector():
    d = HeuristicTurnDetector()
    assert d.p_complete("my order number is") < 0.1
    assert d.p_complete("I want to check my order.") > 0.5
    assert d.p_complete("") == 0


def chunks():
    # "Sure, I can help." (4 words, 0-800ms) | "Your order ships $5 today." normalized has 7 words 800-2200
    c1 = SpokenChunk("Sure, I can help.", [WordTiming(w, i * 200, (i + 1) * 200) for i, w in
                                           enumerate("Sure, I can help.".split())], 0, 800)
    ws = "Your order ships five dollars today now".split()
    c2 = SpokenChunk("Your order ships $5 today.", [WordTiming(w, 800 + i * 200, 1000 + i * 200)
                                                    for i, w in enumerate(ws)], 800, 2200)
    return [c1, c2]


def test_heard_everything():
    text, cut = heard_text(chunks(), 5000)
    assert text == "Sure, I can help. Your order ships $5 today." and not cut


def test_cut_inside_second_clause_is_proportional_to_original_words():
    # heard 1400 ms -> 3 of 7 normalized words -> floor(3/7*5)=2 original words
    text, cut = heard_text(chunks(), 1400)
    assert text == "Sure, I can help. Your order" and cut


def test_cut_at_clause_boundary_drops_unheard_clause():
    text, cut = heard_text(chunks(), 800)
    assert text == "Sure, I can help." and cut


def test_history_message_marker_and_nothing_heard():
    assert history_message(chunks(), 1400).endswith("[interrupted]")
    assert history_message(chunks(), 5000) == "Sure, I can help. Your order ships $5 today."
    assert history_message(chunks(), 0) is None


from voxa_agent.turn.backchannel import is_interrupt_command


@pytest.mark.parametrize("t", ["wait", "Stop.", "hold on a second", "sorry, actually", "no that's wrong"])
def test_single_word_commands_interrupt(t):
    assert is_interrupt_command(t)


@pytest.mark.parametrize("t", ["yeah okay", "mm-hmm", "I see", "my order is late please stop by"])
def test_acknowledgments_and_late_words_are_not_commands(t):
    assert not is_interrupt_command(t)
