import cfpq_data
from typing import NamedTuple
from networkx import MultiDiGraph, nx_pydot


class GraphInfo(NamedTuple):
    vertices: int
    edges: int
    edgeLabels: set[str]


def get_graph_info(name: str) -> GraphInfo:
    path = cfpq_data.download(name)
    graph = cfpq_data.graph_from_csv(path)

    labels = set()

    for _, _, label in graph.edges(data="label"):
        labels.add(label)

    return GraphInfo(
        vertices=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
        edgeLabels=labels,
    )


def labeled_two_cycle_graph_to_dot(
    n: int, m: int, labels: tuple[str, str] = ("a", "b"), filename: str | None = None
) -> MultiDiGraph:
    graph = cfpq_data.labeled_two_cycles_graph(n, m, labels=labels)

    if filename is not None:
        graph_dot = nx_pydot.to_pydot(graph)
        graph_dot.write(filename)

    return graph
