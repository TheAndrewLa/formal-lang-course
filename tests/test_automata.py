import pytest  # noqa: F401
from project.library import (
    automata,
)  # on import will print something from __init__ file # noqa: F401


def setup_module():
    pass


def teardown_module():
    pass


def assert_equivalence(regex: str):
    # Checks that function `regex_to_dfa` returns a finite automaton which is equivalent to the NFA derived from the graph

    dfa = automata.regex_to_dfa(regex)
    assert dfa is not None

    graph = dfa.to_networkx()
    assert graph is not None

    nfa = automata.graph_to_nfa(graph, dfa.start_states, dfa.final_states)
    assert nfa is not None
    assert dfa.is_equivalent_to(nfa)


def test_automata1():
    assert_equivalence("(a|b)*a(a|b)*b(a|b)*a(a|b)*")


def test_automata2():
    assert_equivalence("((a|b)*a(a|b)*b(a|b)*)*")


def test_automata3():
    assert_equivalence("(a|b)*(aa|bb)(a|b)*(aa|bb)(a|b)*")


def test_automata4():
    assert_equivalence("(a*ba*ba*)* & (a|b)*a(a|b)*")


def test_automata5():
    assert_equivalence("~(a*b*)")
