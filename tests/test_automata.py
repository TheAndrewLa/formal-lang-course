from typing import Iterable

import pytest
from networkx import MultiDiGraph

from project.library.automata import tensor_based_rpq, ms_bfs_based_rpq


RPQ_IMPLEMENTATIONS = [tensor_based_rpq, ms_bfs_based_rpq]


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
    assert not rpq("a", g, {}, {})


@pytest.mark.parametrize("rpq", RPQ_IMPLEMENTATIONS)
def test_empty_start(rpq):
    g = make_graph([(0, 1, "a")])
    assert not rpq("a", g, {}, {1})


@pytest.mark.parametrize("rpq", RPQ_IMPLEMENTATIONS)
def test_empty_final(rpq):
    g = make_graph([(0, 1, "a")])
    assert not rpq("a", g, {0}, {})
