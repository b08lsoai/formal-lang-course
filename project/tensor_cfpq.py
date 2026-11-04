import networkx as nx
import numpy as np
import scipy.sparse as sp
from pyformlang.rsa import RecursiveAutomaton

from project.automata import graph_to_nfa
from project.matrix_automaton import AdjacencyMatrixFA


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

    # Intermediate paths may begin anywhere, so every relation is all-pairs.
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, set(), set()))
    graph_size = graph_fa.states_number
    terminals = graph_fa.boolean_decomposition

    box_fas = {label: AdjacencyMatrixFA(box.dfa) for label, box in rsm.boxes.items()}
    relations = {
        label: sp.csr_matrix((graph_size, graph_size), dtype=bool) for label in box_fas
    }

    changed = True
    while changed:
        changed = False
        for label, box_fa in box_fas.items():
            box_size = box_fa.states_number
            # Product states are (graph vertex, box state).
            closure = sp.eye(graph_size * box_size, format="csr", dtype=bool)
            for symbol, box_matrix in box_fa.boolean_decomposition.items():
                relation = relations.get(symbol, terminals.get(symbol))
                if relation is not None and relation.nnz:
                    closure = closure + sp.kron(relation, box_matrix, format="csr")

            # The identity also accounts for empty paths through a box.
            previous_nnz = -1
            while closure.nnz != previous_nnz:
                previous_nnz = closure.nnz
                closure = closure @ closure

            updated = relations[label]
            for start in np.flatnonzero(box_fa.start_states):
                for final in np.flatnonzero(box_fa.final_states):
                    updated = updated + closure[start::box_size, final::box_size]

            if updated.nnz != relations[label].nnz:
                # Later boxes in this pass can already use the new relation.
                relations[label] = updated
                changed = True

    rows, columns = relations[rsm.initial_label].nonzero()
    node_values = [graph_fa.index_to_state[i].value for i in range(graph_size)]
    return {
        (node_values[source], node_values[target])
        for source, target in zip(rows.tolist(), columns.tolist())
        if node_values[source] in starts and node_values[target] in finals
    }
