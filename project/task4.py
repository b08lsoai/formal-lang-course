import numpy as np
import scipy.sparse as sp
from networkx import MultiDiGraph

from project.task2 import graph_to_nfa, regex_to_dfa
from project.task3 import AdjacencyMatrixFA


def ms_bfs(first_fa: AdjacencyMatrixFA, second_fa: AdjacencyMatrixFA) -> sp.csr_matrix:
    """Multiple source BFS on the product of two finite automata.

    Returns the reached product states as a sparse boolean matrix of shape
    ``(starts * second_fa.states_number, first_fa.states_number)``, where
    ``starts`` is the number of start states of ``first_fa``.  Row block ``i``
    corresponds to the ``i``-th start state of ``first_fa``: inside the block,
    row ``q`` and column ``v`` are True iff the product state ``(q, v)`` is
    reachable from that start state.
    """
    first_size = first_fa.states_number
    second_size = second_fa.states_number
    first_starts = np.flatnonzero(first_fa.start_states)
    starts_count = len(first_starts)
    second_starts = np.flatnonzero(second_fa.start_states)

    # front[i * second_size + q, first_starts[i]] = True
    # for every i and every start state q of second_fa.
    rows = (np.arange(starts_count)[:, None] * second_size + second_starts).ravel()
    columns = np.repeat(first_starts, len(second_starts))
    front = sp.csr_matrix(
        (np.ones(len(rows), dtype=bool), (rows, columns)),
        shape=(starts_count * second_size, first_size),
    )
    visited = front

    # kron(I, M^T) applies M^T to every block independently: one step of the
    # second automaton for all blocks at once.
    identity = sp.identity(starts_count, format="csr", dtype=bool)
    second_steps = {
        symbol: sp.kron(identity, matrix.transpose(), format="csr").astype(
            bool, copy=False
        )
        for symbol, matrix in second_fa.boolean_decomposition.items()
    }

    while front.data.any():
        next_front = None
        for symbol, second_step in second_steps.items():
            first_step = first_fa.boolean_decomposition.get(symbol)
            if first_step is None:
                continue
            # One step of the product: front @ first_step advances the first_fa
            # state, then second_step advances the second_fa state. Both
            # components consume the same symbol.
            moved = second_step @ (front @ first_step)
            next_front = moved if next_front is None else next_front.maximum(moved)
        if next_front is None:
            break
        front = next_front - next_front.multiply(visited)
        visited = visited.maximum(front)

    return visited


def ms_bfs_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    """Multiple-source regular path query on a labelled graph.

    Returns pairs ``(start, final)`` connected by a path whose label matches
    ``regex``.

    An empty ``start_nodes`` or ``final_nodes`` means "every vertex",
    matching :func:`graph_to_nfa`.
    """
    regex_fa = AdjacencyMatrixFA(regex_to_dfa(regex))
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))

    if graph_fa.states_number == 0 or regex_fa.states_number == 0:
        return set()

    regex_finals = np.flatnonzero(regex_fa.final_states)
    if len(regex_finals) == 0:
        return set()

    visited = ms_bfs(graph_fa, regex_fa)

    graph_starts = np.flatnonzero(graph_fa.start_states)
    graph_finals = np.flatnonzero(graph_fa.final_states)
    starts_count = len(graph_starts)

    # regex_selector[0, q] = True iff q is a final state of the regex automaton.
    regex_selector = sp.csr_matrix(
        (
            np.ones(len(regex_finals), dtype=bool),
            (np.zeros(len(regex_finals), dtype=int), regex_finals),
        ),
        shape=(1, regex_fa.states_number),
    )
    # graph_selector[v, k] = True iff v = graph_finals[k].
    graph_selector = sp.csr_matrix(
        (
            np.ones(len(graph_finals), dtype=bool),
            (graph_finals, np.arange(len(graph_finals))),
        ),
        shape=(graph_fa.states_number, len(graph_finals)),
    )
    # reached[i, k] = True iff (graph_starts[i], graph_finals[k]) is reachable.
    reached = (
        sp.kron(
            sp.identity(starts_count, format="csr", dtype=bool),
            regex_selector,
        )
        @ visited
        @ graph_selector
    )

    result = set()
    for start_index, final_position in zip(*reached.nonzero()):
        source = graph_fa.index_to_state[graph_starts[start_index]].value
        target = graph_fa.index_to_state[graph_finals[final_position]].value
        result.add((source, target))

    return result
