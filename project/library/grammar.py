from collections import deque, defaultdict
import numpy as np
import pyformlang.cfg as cfg
from networkx import MultiDiGraph
from pyformlang.finite_automaton import State
from pyformlang.cfg import CFG, Epsilon, Production, Terminal, Variable
from pyformlang.rsa import RecursiveAutomaton
from pyformlang.rsa.recursive_automaton import Symbol
from project.library.automata import AdjacencyMatrixFA, graph_to_nfa
from project.library import gss
from scipy import sparse


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    grammar = cfg.to_normal_form()

    if cfg.generate_epsilon():
        start = grammar.start_symbol
        productions = set(grammar.productions)
        productions.add(Production(start, []))
        grammar = CFG(grammar.variables, grammar.terminals, start, productions)

    return grammar


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    start = Symbol(cfg.start_symbol.to_text())
    return RecursiveAutomaton.from_text(cfg.to_text(), start_symbol=start)


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    return RecursiveAutomaton.from_text(ebnf)


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


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    if not start_nodes or not final_nodes:
        return set()

    rsm_matrix, starts, finals = AdjacencyMatrixFA.from_rsm(rsm)
    graph_matrix = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))

    nonterminals = set(starts)

    m = rsm_matrix.len
    n = graph_matrix.len

    mat2: dict[str, sparse.csr_matrix] = {}

    for symbol in rsm_matrix.transitions:
        if symbol in graph_matrix.transitions and symbol not in nonterminals:
            mat2[symbol] = graph_matrix.transitions[symbol].copy().tocsr()
        else:
            mat2[symbol] = sparse.csr_matrix((n, n), dtype=bool)

    diagonal = sparse.eye(n, dtype=bool, format="csr")

    for symbol in nonterminals:
        mat2[symbol] = sparse.csr_matrix((n, n), dtype=bool)
        if starts[symbol] & finals[symbol]:
            mat2[symbol] = (mat2[symbol] + diagonal).astype(bool).tocsr()

    # `state_map[(u, v)] = [A, B, ...]` means that u in starts[A] & v in finals[B] and etc.

    state_map: dict[tuple[int, int], list[str]] = {}

    for symbol in nonterminals:
        for i in starts[symbol]:
            for j in finals[symbol]:
                state_map.setdefault((i, j), []).append(symbol)

    changed = True

    while changed:
        mat3 = sparse.csr_matrix((m * n, m * n), dtype=bool)

        for symbol, mat1 in rsm_matrix.transitions.items():
            if mat2[symbol].nnz == 0:
                continue

            prod = sparse.kron(mat1, mat2[symbol]).astype(bool).tocsr()
            mat3 = (mat3 + prod).astype(bool).tocsr()

        if mat3.nnz == 0:
            break

        closure = AdjacencyMatrixFA.from_fields(
            len=m * n,
            start_indices=[],
            final_indices=[],
            transitions={"M": mat3},
        )

        closure = closure.transitive_closure()
        closure = closure.tocoo()

        rows = closure.row
        cols = closure.col

        qi = rows // n
        i = rows % n

        qj = cols // n
        j = cols % n

        new_edges: dict[str, tuple[list[int], list[int]]] = {}

        for k in range(len(rows)):
            key = (int(qi[k]), int(qj[k]))
            for symbol in state_map.get(key, []):
                new_edges.setdefault(symbol, ([], []))
                new_edges[symbol][0].append(int(i[k]))
                new_edges[symbol][1].append(int(j[k]))

        changed = False

        for symbol, (r, c) in new_edges.items():
            if not r:
                continue

            block = sparse.csr_matrix(
                (np.ones(len(r), dtype=bool), (r, c)),
                shape=(n, n),
                dtype=bool,
            )

            change = (mat2[symbol] + block).astype(bool).tocsr()

            if change.nnz > mat2[symbol].nnz:
                mat2[symbol] = change
                changed = True

    start = str(rsm.initial_label)
    mat_start = mat2[start].tocsr()

    state_to_index = {s: i for i, s in enumerate(graph_matrix.states)}
    node_to_index = {}

    for s, i in state_to_index.items():
        val = str(s)
        node_to_index[val] = i
        node_to_index[str(val)] = i

    result = set()

    for u_node in start_nodes:
        for v_node in final_nodes:
            i = state_to_index[State(u_node)]
            j = state_to_index[State(v_node)]
            if mat_start[i, j]:
                result.add((u_node, v_node))

    return result


def gll_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    if not start_nodes or not final_nodes:
        return set()

    rsm_matrix, starts, finals = AdjacencyMatrixFA.from_rsm(rsm)

    nonterminals = set(starts)
    initial = str(rsm.initial_label)

    succ: list[dict[str, list[int]]] = list()

    for i in range(rsm_matrix.len):
        row_succ: dict[str, list[int]] = {}
        for symbol, matrix in rsm_matrix.transitions.items():
            row = matrix.getrow(i)
            if len(row.indices) > 0:
                row_succ[str(symbol)] = list(row.indices)
        succ.append(row_succ)


    graph_edges: dict[int, list[tuple[str, int]]] = dict()

    for node, vertex, label in graph.edges(data="label"):
        graph_edges.setdefault(node, [])
        graph_edges[node].append((str(label), vertex))

    index_to_nonterminal = {
        i: s.value[0] for i, s in enumerate(rsm_matrix.states)
    }

    result: set[tuple[int, int]] = set()

    def try_result(u: int, v: int):
        if u in start_nodes and v in final_nodes:
            result.add((u, v))

    for start_v in start_nodes:
        stack = gss.Stack()
        root = gss.Node(None, start_v)

        queue = deque()
        popped = defaultdict(set)
        visited = set()

        def push(state: int | None, vertex: int, node: gss.Node):
            desc = (state, vertex, node)
            if desc not in visited:
                visited.add(desc)
                queue.append(desc)

        for s in starts[initial]:
            push(s, start_v, root)

        while queue:
            state, vertex, node = queue.popleft()

            current_nt = index_to_nonterminal[state]

            if state in finals.get(current_nt, set()):
                if node.state is None:
                    try_result(start_v, vertex)
                else:
                    if vertex not in popped[node]:
                        popped[node].add(vertex)
                        for pred in node.preds:
                            push(node.state, vertex, pred)

            for symbol, successors in succ[state].items():
                if symbol in nonterminals:
                    for next_state in successors:
                        new_node = stack.get_or_create(symbol, next_state, vertex)
                        if node not in new_node.preds:
                            new_node.preds.add(node)
                            if new_node in popped:
                                for v_end in popped[new_node]:
                                    push(new_node.state, v_end, node)
                        for s in starts[symbol]:
                            push(s, vertex, new_node)
                else:
                    edges = graph_edges.get(vertex, [])
                    for label, target in edges:
                        if label == symbol:
                            for next_state in successors:
                                push(next_state, target, node)

    return result
