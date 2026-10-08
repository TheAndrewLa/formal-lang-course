from enum import auto
from typing import Iterable

import scipy.sparse as sparse

from networkx import MultiDiGraph

from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
)
from pyformlang.finite_automaton.state import State
from pyformlang.finite_automaton.symbol import Symbol
from pyformlang.regular_expression import Regex

# helper methods, will be removed, after PR#2 merged


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    nfa = Regex(regex).to_epsilon_nfa()
    if nfa is None:
        raise ValueError("Failed to convert regex to epsilon NFA!")
    else:
        dfa = nfa.to_deterministic()
        return dfa.minimize()


def graph_to_nfa(
    graph: MultiDiGraph, startStates: set[int] | None, finalStates: set[int] | None
) -> NondeterministicFiniteAutomaton:
    nfa = NondeterministicFiniteAutomaton()

    if startStates is None:
        start = set(map(int, graph.nodes()))
    else:
        start = startStates

    if finalStates is None:
        final = set(map(int, graph.nodes()))
    else:
        final = finalStates

    for u, v, label in graph.edges(data="label"):
        symbol = Symbol(str(label))
        i = State(u)
        j = State(v)
        nfa.add_transition(i, symbol, j)

    for s in start:
        nfa.add_start_state(State(s))

    for f in final:
        nfa.add_final_state(State(f))

    return nfa


class AdjacencyMatrixFA:
    def __init__(
        self, automaton: DeterministicFiniteAutomaton | NondeterministicFiniteAutomaton
    ):
        self.states = automaton.states

        state_to_index = {state: i for i, state in enumerate(self.states)}

        self.start_indices = set(state_to_index[s] for s in automaton.start_states)
        self.final_indices = set(state_to_index[s] for s in automaton.final_states)

        self.transitions = dict()

        self.len = len(self.states)

        transitions = automaton.to_dict()

        for s in automaton.symbols:
            self.transitions[s] = sparse.csr_matrix((self.len, self.len), dtype=bool)

        for state, transitionsFromState in transitions.items():
            index = state_to_index[state]
            for symbol, next in transitionsFromState.items():
                match next:
                    case State():
                        next_index = state_to_index[next]
                        self.transitions[str(symbol.value)][index, next_index] = True
                    case _:
                        for n in set(next):
                            next_index = state_to_index[n]
                            self.transitions[str(symbol.value)][index, next_index] = (
                                True
                            )

        for s in self.transitions:
            self.transitions[s] = self.transitions[s].tocsr()

    def accepts(self, word: Iterable[Symbol]) -> bool:
        vec = sparse.csr_matrix(
            ([True], ([0], list(self.start_indices))), shape=(1, self.len), dtype=bool
        )

        def process_symbol(s) -> str:
            if isinstance(s, str):
                return s
            else:
                return str(s.value)

        for symbol in map(process_symbol, word):
            if symbol not in self.transitions:
                return False
            vec = vec @ self.transitions[symbol]

        vec.eliminate_zeros()

        return bool(self.final_indices.intersection(vec.indices))

    def transitive_closure(self) -> sparse.csr_matrix:
        any = sparse.csr_matrix((self.len, self.len), dtype=bool)
        for symbol in self.transitions:
            any += self.transitions[symbol]

        closure = sparse.identity(self.len, dtype="bool")
        closure = closure + any.astype(bool)
        closure = closure.astype(bool).tocsr()

        for _ in range(self.len.bit_length()):
            closure = (closure @ closure).astype(bool)
            closure.eliminate_zeros()

        return closure

    def is_empty(self) -> bool:
        closure = self.transitive_closure()

        for i in self.start_indices:
            for j in self.final_indices:
                if closure[i, j]:
                    return False

        return True

    @classmethod
    def from_fields(
        cls, len, start_indices, final_indices, transitions
    ) -> "AdjacencyMatrixFA":
        obj = cls.__new__(cls)
        obj.len = len
        obj.start_indices = set(start_indices)
        obj.final_indices = set(final_indices)
        obj.transitions = transitions
        return obj


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    symbols = set(automaton1.transitions).union(automaton2.transitions)

    len1 = automaton1.len
    len2 = automaton2.len

    transitions = dict()

    for s in symbols:
        mat1 = automaton1.transitions.get(
            s, sparse.csr_matrix((len1, len1), dtype=bool)
        )
        mat2 = automaton2.transitions.get(
            s, sparse.csr_matrix((len2, len2), dtype=bool)
        )
        transitions[s] = sparse.kron(mat1, mat2).astype(bool).tocsr()

    start_indices = set()
    final_indices = set()

    for s1 in automaton1.start_indices:
        for s2 in automaton2.start_indices:
            start_indices.add(s1 * len2 + s2)

    for f1 in automaton1.final_indices:
        for f2 in automaton2.final_indices:
            final_indices.add(f1 * len2 + f2)

    return AdjacencyMatrixFA.from_fields(
        len1 * len2, start_indices, final_indices, transitions
    )


def tensor_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    dfa = AdjacencyMatrixFA(regex_to_dfa(regex))
    nfa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))

    automaton = intersect_automata(dfa, nfa)
    closure = automaton.transitive_closure()

    nfa_states = list(nfa.states)
    nfa_len = nfa.len

    return {
        (nfa_states[start % nfa_len], nfa_states[final % nfa_len])
        for start in automaton.start_indices
        for final in automaton.final_indices
        if closure[start, final]
    }
