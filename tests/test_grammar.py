from pyformlang.cfg import CFG, Variable
import pytest
from networkx import DiGraph, MultiDiGraph

from project.library.grammar import (
    cfg_to_rsm,
    gll_based_cfpq,
    hellings_based_cfpq,
    matrix_based_cfpq,
    tensor_based_cfpq,
)


def setup_module():
    pass


def teardown_module():
    pass


def wrap_tensor_based_cfpq(
    cfg: CFG, graph: MultiDiGraph, starts: set[int], finals: set[int]
):
    rsm = cfg_to_rsm(cfg)
    return tensor_based_cfpq(rsm, graph, starts, finals)


def wrap_gll_based_cfpq(
    cfg: CFG, graph: MultiDiGraph, starts: set[int], finals: set[int]
):
    rsm = cfg_to_rsm(cfg)
    return gll_based_cfpq(rsm, graph, starts, finals)


CFPQ_IMPLEMENTATIONS = [
    hellings_based_cfpq,
    matrix_based_cfpq,
    wrap_tensor_based_cfpq,
    wrap_gll_based_cfpq,
]


@pytest.mark.parametrize("cfpq", CFPQ_IMPLEMENTATIONS)
def test_palindrome(cfpq):
    cfg = CFG.from_text("S -> a S a | b S b | a | b | $", start_symbol=Variable("S"))

    graph = DiGraph()
    graph.add_edge(1, 2, label="a")
    graph.add_edge(2, 3, label="b")
    graph.add_edge(3, 4, label="b")
    graph.add_edge(4, 5, label="a")

    res = cfpq(cfg, graph, {1}, {4, 5})
    assert res == {(1, 5)}

    res = cfpq(cfg, graph, {1, 2}, {4, 5})
    assert res == {(1, 5), (2, 4)}


@pytest.mark.parametrize("cfpq", CFPQ_IMPLEMENTATIONS)
def test_ab(cfpq):
    cfg = CFG.from_text(
        "S -> A B | $\nA -> a A | a\nB -> b B | b", start_symbol=Variable("S")
    )

    graph = MultiDiGraph()
    graph.add_edge(1, 2, label="a")
    graph.add_edge(2, 3, label="a")
    graph.add_edge(3, 4, label="b")
    graph.add_edge(4, 5, label="b")

    all_nodes = set(range(1, 6))

    res = cfpq(cfg, graph, all_nodes, all_nodes)

    assert res == {
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
        (1, 4),
        (1, 5),
        (2, 4),
        (2, 5),
    }


@pytest.mark.parametrize("cfpq", CFPQ_IMPLEMENTATIONS)
def test_parentheses(cfpq):
    cfg = CFG.from_text("S -> S S | a S b | $", start_symbol=Variable("S"))

    graph = DiGraph()
    graph.add_edge(1, 2, label="a")
    graph.add_edge(2, 3, label="b")
    graph.add_edge(3, 4, label="a")
    graph.add_edge(4, 5, label="b")

    res = cfpq(cfg, graph, {1, 3, 4}, {3, 4, 5})

    assert res == {(1, 3), (1, 5), (3, 3), (3, 5), (4, 4)}


@pytest.mark.parametrize("cfpq", CFPQ_IMPLEMENTATIONS)
def test_abcd(cfpq):
    cfg = CFG.from_text("S -> a S d | A\nA -> b c A | $")

    graph = MultiDiGraph()
    graph.add_edge(1, 2, label="a")
    graph.add_edge(2, 3, label="a")
    graph.add_edge(3, 4, label="b")
    graph.add_edge(4, 5, label="c")
    graph.add_edge(5, 6, label="b")
    graph.add_edge(6, 3, label="c")
    graph.add_edge(3, 2, label="d")
    graph.add_edge(2, 1, label="d")
    graph.add_edge(3, 7, label="d")
    graph.add_edge(7, 8, label="d")

    res = cfpq(cfg, graph, {1, 2}, {1, 2, 7, 8})

    assert res == {(1, 1), (1, 8), (2, 2), (2, 7)}
