from collections import deque
from itertools import product

import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton

from project.task3 import AdjacencyMatrixFA, tensor_based_rpq
from project.task4 import ms_bfs, ms_bfs_based_rpq


def _graph(edges: list[tuple[int, int, str]]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_edges_from((u, v, {"label": label}) for u, v, label in edges)
    return graph


def _automaton(
    transitions: list[tuple[int, str, int]],
    starts: list[int],
    finals: list[int],
    alphabet: list[str],
) -> AdjacencyMatrixFA:
    """Automaton over integer states: `(from, symbol, to)` transitions."""
    fa = NondeterministicFiniteAutomaton()
    for symbol in alphabet:
        fa.add_symbol(symbol)
    for source, symbol, target in transitions:
        fa.add_transition(source, symbol, target)
    for start in starts:
        fa.add_start_state(start)
    for final in finals:
        fa.add_final_state(final)
    return AdjacencyMatrixFA(fa)


# ---------- ms_bfs ----------


def _reference_product_bfs(first_fa, second_fa) -> sp.csr_matrix:
    first_size = first_fa.states_number
    second_size = second_fa.states_number
    first_starts = np.flatnonzero(first_fa.start_states)
    second_starts = np.flatnonzero(second_fa.start_states)

    successors: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for symbol in first_fa.alphabet & second_fa.alphabet:
        first_matrix = first_fa.boolean_decomposition[symbol]
        second_matrix = second_fa.boolean_decomposition[symbol]
        for (i, i_next), (j, j_next) in product(
            zip(*first_matrix.nonzero()), zip(*second_matrix.nonzero())
        ):
            successors.setdefault((i, j), set()).add((i_next, j_next))

    visited = np.zeros((len(first_starts) * second_size, first_size), dtype=bool)
    for block, first_start in enumerate(first_starts):
        block_visited = visited[block * second_size : (block + 1) * second_size]
        queue = deque()
        for j in second_starts:
            block_visited[j, first_start] = True
            queue.append((first_start, j))
        while queue:
            i, j = queue.popleft()
            for i_next, j_next in successors.get((i, j), ()):
                if not block_visited[j_next, i_next]:
                    block_visited[j_next, i_next] = True
                    queue.append((i_next, j_next))
    return sp.csr_matrix(visited)


_SELF_LOOP = ([(0, "a", 0)], [0], [0], ["a"])
_CHAIN_A = ([(0, "a", 1), (1, "a", 2)], [0], [0, 1, 2], ["a"])
_CHAIN_AB = ([(0, "a", 1), (1, "b", 2)], [0], [0, 1, 2], ["a", "b"])
_CYCLE = ([(0, "a", 1), (1, "a", 2), (2, "a", 0)], [0], [0, 1, 2], ["a"])
_TWO_STARTS = ([(0, "a", 1), (1, "a", 2), (1, "a", 3)], [0, 1], [2, 3], ["a"])
_DEAD_END = ([(0, "a", 1)], [0], [1], ["a"])


@pytest.mark.parametrize(
    ("first_spec", "second_spec"),
    [
        pytest.param(_SELF_LOOP, _SELF_LOOP, id="self-loop"),
        pytest.param(_CHAIN_A, _CHAIN_A, id="chain"),
        pytest.param(_CHAIN_AB, _CHAIN_AB, id="chain-ab"),
        pytest.param(_CHAIN_AB, _CHAIN_A, id="chain-ab_x_chain-a"),
        pytest.param(_TWO_STARTS, _CHAIN_A, id="two-starts_x_chain"),
        pytest.param(_CHAIN_A, _TWO_STARTS, id="chain_x_two-starts"),
        pytest.param(_CYCLE, _CYCLE, id="cycle"),
        pytest.param(_DEAD_END, _DEAD_END, id="dead-end"),
        pytest.param(_CHAIN_AB, _DEAD_END, id="chain-ab_x_dead-end"),
    ],
)
def test_ms_bfs_matches_reference_bfs(first_spec, second_spec):
    first = _automaton(*first_spec)
    second = _automaton(*second_spec)

    actual = ms_bfs(first, second)
    expected = _reference_product_bfs(first, second)

    assert actual.shape == expected.shape
    assert (actual != expected).nnz == 0


def test_ms_bfs_empty_automaton():
    empty = _automaton([], [], [], [])
    assert ms_bfs(empty, empty).shape == (0, 0)


def test_ms_bfs_no_start_states():
    first = _automaton([(0, "a", 1)], [], [1], ["a"])
    second = _automaton([(0, "a", 1)], [0], [1], ["a"])
    assert ms_bfs(first, second).shape[0] == 0


def test_ms_bfs_no_transitions():
    first = _automaton([], [0], [0], ["a"])
    second = _automaton([], [0], [0], ["a"])
    result = ms_bfs(first, second).toarray()
    assert result.sum() == 1
    assert result[0, 0]


def test_ms_bfs_disjoint_alphabets():
    first = _automaton([(0, "a", 1)], [0], [1], ["a"])
    second = _automaton([(0, "b", 1)], [0], [1], ["b"])
    result = ms_bfs(first, second).toarray()
    assert result.sum() == 1
    assert result[0, 0]


# ---------- ms_bfs_based_rpq ----------


_GRAPHS = {
    "mixed-labels": _graph([(0, 1, "a"), (1, 2, "b"), (2, 0, "a"), (0, 3, "b")]),
    "cycle": _graph([(0, 1, "b"), (1, 2, "b"), (2, 0, "b")]),
    "self-loop": _graph([(0, 0, "a"), (0, 1, "b"), (1, 0, "a")]),
    "dead-end": _graph([(0, 1, "a"), (2, 3, "a")]),
    "parallel": _graph([(0, 1, "a"), (0, 1, "b"), (0, 1, "a"), (1, 2, "a")]),
    "isolated": _graph([]),
}

_RPQ_CASES = [
    pytest.param(
        graph_name,
        regex,
        starts,
        finals,
        id=f"{graph_name}|{regex or 'empty'}|{sorted(starts) or 'all'}|{sorted(finals) or 'all'}",
    )
    for graph_name, regex, starts, finals in [
        ("mixed-labels", "a b", {0}, {2}),
        ("mixed-labels", "b a", {0}, {2}),
        ("mixed-labels", "a | b", {0}, {2}),
        ("mixed-labels", "a*", {0}, {0}),
        ("mixed-labels", "", {0}, {2}),
        ("mixed-labels", "(a | b)*", {0}, set()),
        ("mixed-labels", "a b", set(), {2}),
        ("mixed-labels", "(a | b)*", set(), set()),
        ("mixed-labels", "a* b a*", {0, 1}, {1, 3}),
        ("cycle", "a", {0}, {2}),
        ("cycle", "(a b)*", {0}, {0, 1, 2}),
        ("self-loop", "a*", {0}, set()),
        ("dead-end", "a a", {0}, {3}),
        ("parallel", "a", {0}, {1}),
        ("parallel", "a a", {0}, {2}),
        ("isolated", "a*", set(), set()),
    ]
]


@pytest.mark.parametrize(
    ("graph_name", "regex", "start_nodes", "final_nodes"), _RPQ_CASES
)
def test_ms_bfs_based_rpq_matches_tensor_based_rpq(
    graph_name, regex, start_nodes, final_nodes
):
    graph = _GRAPHS[graph_name]
    assert ms_bfs_based_rpq(regex, graph, start_nodes, final_nodes) == tensor_based_rpq(
        regex, graph, start_nodes, final_nodes
    )
