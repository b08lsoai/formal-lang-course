import networkx as nx
import numpy as np
import scipy.sparse as sp
from pyformlang.rsa import RecursiveAutomaton

from project.automata import graph_to_nfa
from project.matrix_automaton import AdjacencyMatrixFA
from project.ms_bfs_rpq import ms_bfs


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """Vertex pairs connected by a path in the language of ``rsm``.

    An empty ``start_nodes`` / ``final_nodes`` means every vertex.
    """
    if graph.number_of_nodes() == 0 or rsm.initial_label not in rsm.boxes:
        return set()

    starts = set(graph.nodes) if not start_nodes else set(start_nodes)
    finals = set(graph.nodes) if not final_nodes else set(final_nodes)

    box_fas = {label: AdjacencyMatrixFA(box.dfa) for label, box in rsm.boxes.items()}

    # Intermediate paths may begin anywhere, regardless of the output filters.
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, set(), set()))
    graph_size = graph_fa.states_number
    graph_matrices = graph_fa.boolean_decomposition
    # Box labels denote inferred calls, not original graph edges.
    for label in rsm.boxes:
        graph_matrices[label] = sp.csr_matrix((graph_size, graph_size), dtype=bool)
    graph_fa.alphabet.update(rsm.boxes)

    changed = True
    while changed:
        changed = False
        for label, box_fa in box_fas.items():
            box_size = box_fa.states_number
            visited = ms_bfs(graph_fa, box_fa)
            updated = graph_matrices[label]
            # Each row block corresponds to one graph source; select box exits.
            for final in np.flatnonzero(box_fa.final_states):
                updated = updated + visited[final::box_size, :]
            if updated.nnz != graph_matrices[label].nnz:
                # Later boxes in this pass can already use the new relation.
                graph_matrices[label] = updated
                changed = True

    rows, columns = graph_matrices[rsm.initial_label].nonzero()
    nodes = graph_fa.index_to_state
    return {
        (nodes[source].value, nodes[target].value)
        for source, target in zip(rows, columns)
        if nodes[source].value in starts and nodes[target].value in finals
    }
