import numpy as np
import pyformlang.cfg as cfg
from networkx import MultiDiGraph
from pyformlang.cfg import CFG, Epsilon, Production, Terminal, Variable
from scipy import sparse


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    grammar = cfg.to_normal_form()

    if cfg.generate_epsilon():
        start = grammar.start_symbol
        productions = set(grammar.productions)
        productions.add(Production(start, []))
        grammar = CFG(grammar.variables, grammar.terminals, start, productions)

    return grammar


def hellings_based_cfpq(
    cfg: cfg.CFG, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    if not start_nodes or not final_nodes:
        return set()

    node_to_index = {node: i for i, node in enumerate(graph.nodes)}
    index_to_node = {i: node for node, i in node_to_index.items()}

    start_indices = {node_to_index[n] for n in start_nodes}
    final_indices = {node_to_index[n] for n in final_nodes}

    grammar = cfg_to_weak_normal_form(cfg)

    prod_epsilon: list[Variable] = []
    prod_terminal: list[tuple[Variable, Terminal]] = []
    prod_binary: list[tuple[Variable, Variable, Variable]] = []

    for production in grammar.productions:
        s = production.head
        body = production.body

        match body:
            case [] | [Epsilon()]:
                prod_epsilon.append(s)
            case [Terminal() as t]:
                prod_terminal.append((s, t))
            case [Variable() as a, Variable() as b]:
                prod_binary.append((s, a, b))
            case _:
                raise ValueError(
                    f"Invalid production in weak normal form grammar: {production}"
                )

    result: set[tuple[Variable, int, int]] = set()

    for var in prod_epsilon:
        for i in index_to_node:
            result.add((var, i, i))

    for var, terminal in prod_terminal:
        for u, v, label in graph.edges(data="label"):
            if str(label) == terminal.to_text():
                u1 = node_to_index[u]
                v1 = node_to_index[v]
                result.add((var, u1, v1))

    worklist = result.copy()

    while worklist:
        var, u, v = worklist.pop()

        for left, t, end in list(result):
            if end != u:
                continue

            for s, a, b in prod_binary:
                if a == left and b == var:
                    new = (s, t, v)
                    if new not in result:
                        result.add(new)
                        worklist.add(new)

        for right, start, t in list(result):
            if start != v:
                continue

            for s, a, b in prod_binary:
                if a == var and b == right:
                    new = (s, u, t)
                    if new not in result:
                        result.add(new)
                        worklist.add(new)

    return set(
        [
            (index_to_node[u], index_to_node[v])
            for s, u, v in result
            if s == grammar.start_symbol and u in start_indices and v in final_indices
        ]
    )


def matrix_based_cfpq(
    cfg: CFG, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    if not start_nodes or not final_nodes or not graph.nodes:
        return set()

    node_to_index = {node: i for i, node in enumerate(graph.nodes)}
    index_to_node = {i: node for node, i in node_to_index.items()}

    start_indices = {node_to_index[n] for n in start_nodes if n in node_to_index}
    final_indices = {node_to_index[n] for n in final_nodes if n in node_to_index}

    grammar = cfg_to_weak_normal_form(cfg)

    prod_epsilon: list[Variable] = []
    prod_terminal: list[tuple[Variable, Terminal]] = []
    prod_binary: list[tuple[Variable, Variable, Variable]] = []

    for production in grammar.productions:
        head = production.head
        body = production.body

        match body:
            case [] | [Epsilon()]:
                prod_epsilon.append(head)
            case [Terminal() as terminal]:
                prod_terminal.append((head, terminal))
            case [Variable() as a, Variable() as b]:
                prod_binary.append((head, a, b))
            case _:
                raise ValueError(
                    f"Invalid production in weak normal form grammar: {production}"
                )

    n = len(node_to_index)

    matrices = {var: sparse.csr_matrix((n, n), dtype=bool) for var in grammar.variables}

    edge_rows = {var: [] for var in grammar.variables}
    edge_cols = {var: [] for var in grammar.variables}

    for u, v, label in graph.edges(data="label"):
        for var, terminal in prod_terminal:
            if str(label) == terminal.to_text():
                edge_rows[var].append(node_to_index[u])
                edge_cols[var].append(node_to_index[v])

    for var, rows in edge_rows.items():
        if rows:
            matrices[var] = sparse.csr_matrix(
                (np.ones(len(rows), dtype=bool), (rows, edge_cols[var])),
                shape=(n, n),
            )

    for var in prod_epsilon:
        matrices[var] = matrices[var] + sparse.eye(n, dtype=bool, format="csr")

    queue = set(matrices.keys())

    while queue:
        var = queue.pop()

        for head, left, right in prod_binary:
            if var not in (left, right):
                continue

            product = (matrices[left] @ matrices[right]).astype(bool)
            new_matrix = matrices[head] + product
            new_matrix.eliminate_zeros()

            if new_matrix.nnz != matrices[head].nnz:
                matrices[head] = new_matrix
                queue.add(head)

    start_matrix = matrices[grammar.start_symbol]
    rows, cols = start_matrix.nonzero()

    return {
        (index_to_node[u], index_to_node[v])
        for u, v in zip(rows, cols)
        if u in start_indices and v in final_indices
    }
