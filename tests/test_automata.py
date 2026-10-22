import cfpq_data
import pytest  # noqa: F401
from project.library import (
    automata,
)  # on import will print something from __init__ file # noqa: F401


def setup_module():
    pass


def teardown_module():
    pass


def roundtrip(regex: str):
    # Regex -> DFA -> Graph -> NFA
    #
    # Checking that DFA built by `regex_to_dfa` is equivalent to NFA built by `graph_to_nfa`
    # Assuming that `dfa.to_networkx()` works correctly

    dfa = automata.regex_to_dfa(regex)
    assert dfa is not None

    graph = dfa.to_networkx()
    assert graph is not None

    nfa = automata.graph_to_nfa(graph, dfa.start_states, dfa.final_states)
    assert nfa is not None
    assert dfa.is_equivalent_to(nfa)


def test_roundtrip1():
    roundtrip("(a|b)*a(a|b)*b(a|b)*a(a|b)*")


def test_roundtrip_2():
    roundtrip("(a|b)*(aa|bb)(a|b)*(aa|bb)(a|b)*")


def test_roundtrip_3():
    roundtrip("(a*ba*ba*)* & (a|b)*a(a|b)*")


def test_roundtrip_4():
    roundtrip("~(a*b*)")


def test_graph_to_nfa_pr():
    # Information about `pr` graph are taken from 'https://formallanguageconstrainedpathquerying.github.io/CFPQ_Data/graphs/data/pr.html#pr'

    path = cfpq_data.download("pr")
    graph = cfpq_data.graph_from_csv(path)

    nfa = automata.graph_to_nfa(graph, {0}, {1})

    assert len(nfa.states) == 815

    assert nfa.symbols == {"d", "a"}
    assert nfa.start_states == {0}
    assert nfa.final_states == {1}

    assert nfa.get_number_transitions() == 692


def test_graph_to_nfa_gzip():
    # Information about `gzip` graph are taken from 'https://formallanguageconstrainedpathquerying.github.io/CFPQ_Data/graphs/data/gzip.html#gzip'

    path = cfpq_data.download("gzip")
    graph = cfpq_data.graph_from_csv(path)

    nfa = automata.graph_to_nfa(graph, {0, 1, 2}, {453, 12, 988})

    assert len(nfa.states) == 2687

    assert nfa.symbols == {"d", "a"}
    assert nfa.start_states == {0, 1, 2}
    assert nfa.final_states == {453, 12, 988}

    assert nfa.get_number_transitions() == 2293
