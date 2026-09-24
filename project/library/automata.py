from typing import Set
from networkx import MultiDiGraph
from pyformlang.regular_expression import Regex
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    State,
    Symbol,
)


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    """
    Converts a regular expression to minimal deterministic finite automaton (DFA).
    Throws `ValueError` if the regex can not be converted to an finite automaton.

    Args:
        - `regex`: The regular expression to convert.
    """
    nfa = Regex(regex).to_epsilon_nfa()
    if nfa is None:
        raise ValueError("Failed to convert regex to epsilon NFA!")
    else:
        dfa = nfa.to_deterministic()
        dfa.minimize()
        return dfa


def graph_to_nfa(
    graph: MultiDiGraph, startStates: Set[int], finalStates: Set[int]
) -> NondeterministicFiniteAutomaton:
    """
    Converts a graph to a nondeterministic finite automaton (NFA).

    Args:
        - `graph`: The graph to convert.
        - `start_states`: Set of start states, if it is empty, all graph's nodes are considered to be start.
        - `final_states`: Set of final states, if it is empty, all graph's nodes are considered to be final.
    """
    nfa = NondeterministicFiniteAutomaton()

    start = startStates if startStates else set(map(int, graph.nodes()))
    final = finalStates if finalStates else set(map(int, graph.nodes()))

    for u, v, label in graph.edges(data="label"):
        symbol = Symbol(str(label))
        i = State(int(u))
        j = State(int(v))
        nfa.add_transition(i, symbol, j)

    for s in start:
        nfa.start_states.add(State(s))

    for f in final:
        nfa.final_states.add(State(f))

    return nfa
