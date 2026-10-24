from typing import Iterable

import pytest
from networkx import MultiDiGraph
import cfpq_data

from project.library.automata import tensor_based_rpq, ms_bfs_based_rpq
from project.library.automata import graph_to_nfa, regex_to_dfa


RPQ_IMPLEMENTATIONS = [tensor_based_rpq, ms_bfs_based_rpq]


def setup_module():
    pass


def teardown_module():
    pass


def make_graph(edges: Iterable[tuple[int, int, str]]):
    g = MultiDiGraph()
    for u, v, label in edges:
        g.add_edge(u, v, label=label)
    return g


@pytest.mark.parametrize("rpq", RPQ_IMPLEMENTATIONS)
def test_single_edge(rpq):
    g = make_graph([(0, 1, "a")])
    assert rpq("a", g, {0}, {1}) == {(0, 1)}


@pytest.mark.parametrize("rpq", RPQ_IMPLEMENTATIONS)
def test_union(rpq):
    g = make_graph([(0, 1, "a"), (1, 2, "b"), (1, 2, "c")])
    assert rpq("a (b|c)", g, {0}, {2}) == {(0, 2)}


@pytest.mark.parametrize("rpq", RPQ_IMPLEMENTATIONS)
def test_star_1(rpq):
    g = make_graph([(0, 1, "a"), (1, 2, "a"), (2, 3, "a")])
    assert rpq("a*", g, {0}, {0, 1, 2, 3}) == {
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
    }


@pytest.mark.parametrize("rpq", RPQ_IMPLEMENTATIONS)
def test_star_2(rpq):
    g = make_graph([(0, 1, "a"), (1, 2, "b"), (2, 3, "a")])
    assert rpq("(a b)* a", g, {0}, {3}) == {(0, 3)}


# Some corner cases


@pytest.mark.parametrize("rpq", RPQ_IMPLEMENTATIONS)
def test_empty_graph(rpq):
    g = make_graph([])
    assert not rpq("a", g, set(), set())


@pytest.mark.parametrize("rpq", RPQ_IMPLEMENTATIONS)
def test_empty_start(rpq):
    g = make_graph([(0, 1, "a")])
    assert not rpq("a", g, set(), set([1]))


@pytest.mark.parametrize("rpq", RPQ_IMPLEMENTATIONS)
def test_empty_final(rpq):
    g = make_graph([(0, 1, "a")])
    assert not rpq("a", g, set([0]), set())


def roundtrip(regex: str):
    # Regex -> DFA -> Graph -> NFA
    #
    # Checking that DFA built by `regex_to_dfa` is equivalent to NFA built by `graph_to_nfa`
    # Assuming that `dfa.to_networkx()` works correctly

    dfa = regex_to_dfa(regex)
    assert dfa is not None

    graph = dfa.to_networkx()
    assert graph is not None

    nfa = graph_to_nfa(graph, dfa.start_states, dfa.final_states)
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

    nfa = graph_to_nfa(graph, {0}, {1})

    assert len(nfa.states) == 815

    assert nfa.symbols == {"d", "a"}
    assert nfa.start_states == {0}
    assert nfa.final_states == {1}

    assert nfa.get_number_transitions() == 692


def test_graph_to_nfa_gzip():
    # Information about `gzip` graph are taken from 'https://formallanguageconstrainedpathquerying.github.io/CFPQ_Data/graphs/data/gzip.html#gzip'

    path = cfpq_data.download("gzip")
    graph = cfpq_data.graph_from_csv(path)

    nfa = graph_to_nfa(graph, {0, 1, 2}, {453, 12, 988})

    assert len(nfa.states) == 2687

    assert nfa.symbols == {"d", "a"}
    assert nfa.start_states == {0, 1, 2}
    assert nfa.final_states == {453, 12, 988}

    assert nfa.get_number_transitions() == 2293
