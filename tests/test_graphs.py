import pytest  # noqa: F401
from project.library import (
    graphs,
)  # on import will print something from __init__ file # noqa: F401
from os import path
from networkx import nx_pydot


def setup_module():
    pass


def teardown_module():
    pass


def test_get_info_bzip():
    # Expected values for `bzip` graph are taken from 'https://formallanguageconstrainedpathquerying.github.io/CFPQ_Data/graphs/data/bzip.html#bzip'

    expected_bzip_vertices = 632
    expected_bzip_edges = 556
    expected_bzip_labels = {"a", "d"}

    bzip = graphs.get_graph_info("bzip")

    assert bzip.vertices == expected_bzip_vertices
    assert bzip.edges == expected_bzip_edges
    assert bzip.edgeLabels == expected_bzip_labels


def test_get_info_ls():
    # Expected values for `ls` graph are taken from 'https://formallanguageconstrainedpathquerying.github.io/CFPQ_Data/graphs/data/ls.html#ls'

    expected_ls_vertices = 1687
    expected_ls_edges = 1453
    expected_ls_labels = {"a", "d"}

    ls = graphs.get_graph_info("ls")

    assert ls.vertices == expected_ls_vertices
    assert ls.edges == expected_ls_edges
    assert ls.edgeLabels == expected_ls_labels


def test_get_info_pathways():
    # Expected values for `pathways` graph are taken from 'https://formallanguageconstrainedpathquerying.github.io/CFPQ_Data/graphs/data/pathways.html#pathways'

    expected_pathways_vertices = 6238
    expected_pathways_edges = 12363
    expected_pathways_labels = {"type", "subClassOf", "label", "narrower", "imports"}

    pathways = graphs.get_graph_info("pathways")

    assert pathways.vertices == expected_pathways_vertices
    assert pathways.edges == expected_pathways_edges
    assert pathways.edgeLabels == expected_pathways_labels


def test_get_info_foaf():
    # Expected values for `foaf` graph are taken from 'https://formallanguageconstrainedpathquerying.github.io/CFPQ_Data/graphs/data/foaf.html#foaf'

    expected_foaf_vertices = 256
    expected_foaf_edges = 631
    expected_foaf_labels = {
        "type",
        "label",
        "term_status",
        "comment",
        "isDefinedBy",
        "domain",
        "range",
        "subPropertyOf",
        "subClassOf",
        "disjointWith",
        "inverseOf",
        "equivalentClass",
        "title",
        "description",
        "equivalentProperty",
    }

    foaf = graphs.get_graph_info("foaf")

    assert foaf.vertices == expected_foaf_vertices
    assert foaf.edges == expected_foaf_edges
    assert foaf.edgeLabels == expected_foaf_labels


def test_labeled_two_cycle_graph_to_dot_1():
    v1, v2 = 134, 76

    graph1 = graphs.labeled_two_cycle_graph_to_dot(v1, v2, ("l1", "l2"))

    assert graph1 is not None

    assert graph1.number_of_nodes() == (v1 + v2 + 1)
    assert graph1.number_of_edges() == (v1 + v2 + 2)

    assert graph1.has_node(0)

    assert graph1.has_edge(0, 1)
    assert graph1.has_edge(0, 135)

    assert graph1.has_edge(1, 2)
    assert graph1.has_edge(135, 136)


def test_labeled_two_cycle_graph_to_dot_2(tmp_path):
    tmp_file = path.join(tmp_path, "graph2.dot")

    v1, v2 = 845, 771

    graph1 = graphs.labeled_two_cycle_graph_to_dot(v1, v2, ("a", "b"), tmp_file)

    assert graph1 is not None
    assert path.exists(tmp_file)

    graph2 = nx_pydot.read_dot(tmp_file)

    assert graph2 is not None

    assert graph1.number_of_nodes() == graph2.number_of_nodes()
    assert graph1.number_of_edges() == graph2.number_of_edges()


def test_labeled_two_cycle_graph_to_dot_3(tmp_path):
    tmp_file = path.join(tmp_path, "graph3.dot")

    v1, v2 = 3867, 1491

    graph1 = graphs.labeled_two_cycle_graph_to_dot(
        v1, v2, ("veryLongEdgeLabel1", "l2"), tmp_file
    )

    assert graph1 is not None
    assert path.exists(tmp_file)

    graph2 = nx_pydot.read_dot(tmp_file)

    assert graph2 is not None

    assert graph1.number_of_nodes() == graph2.number_of_nodes()
    assert graph1.number_of_edges() == graph2.number_of_edges()
