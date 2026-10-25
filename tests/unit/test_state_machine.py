import itertools

import pytest

from voxa_agent.state import TRANSITIONS, IllegalTransition, State, StateMachine, mermaid

ALL = list(State)
LEGAL = [(a, b) for a, bs in TRANSITIONS.items() for b in bs]
ILLEGAL = [(a, b) for a, b in itertools.product(ALL, ALL) if a is not b and b not in TRANSITIONS[a]]


def machine_in(state: State) -> StateMachine:
    sm = StateMachine()
    sm.state = state
    return sm


@pytest.mark.parametrize("a,b", LEGAL, ids=[f"{a.value}->{b.value}" for a, b in LEGAL])
def test_every_legal_transition(a, b):
    seen = []
    sm = StateMachine(lambda o, n, r: seen.append((o, n, r)))
    sm.state = a
    assert sm.to(b, "why") is True and sm.state is b
    assert seen == [(a, b, "why")]


@pytest.mark.parametrize("a,b", ILLEGAL, ids=[f"{a.value}->{b.value}" for a, b in ILLEGAL])
def test_every_illegal_transition_raises(a, b):
    sm = machine_in(a)
    with pytest.raises(IllegalTransition):
        sm.to(b)
    assert sm.state is a


@pytest.mark.parametrize("s", ALL)
def test_self_transition_is_noop(s):
    sm = machine_in(s)
    assert sm.to(s) is False


def test_ended_is_terminal_and_all_states_reachable_from_idle():
    assert TRANSITIONS[State.ENDED] == frozenset()
    reach, todo = {State.IDLE}, [State.IDLE]
    while todo:
        for n in TRANSITIONS[todo.pop()]:
            if n not in reach:
                reach.add(n); todo.append(n)
    assert reach == set(State)


def test_mermaid_lists_every_transition():
    m = mermaid()
    assert all(f"{a.value} --> {b.value}" in m for a, b in LEGAL)
