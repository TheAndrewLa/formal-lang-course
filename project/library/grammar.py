import pyformlang.cfg as cfg
from networkx import MultiDiGraph
from pyformlang.cfg import CFG, Epsilon, Production, Terminal, Variable


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
