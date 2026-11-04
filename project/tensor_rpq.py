from networkx import MultiDiGraph

from project.automata import graph_to_nfa, regex_to_dfa
from project.matrix_automaton import AdjacencyMatrixFA, intersect_automata


def tensor_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    """Vertex pairs (start, final) connected by a path in the regex language."""
    regex_fa = AdjacencyMatrixFA(regex_to_dfa(regex))
    regex_starts = [
        index for index, is_start in enumerate(regex_fa.start_states) if is_start
    ]
    if not regex_starts:
        return set()

    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    intersection = intersect_automata(regex_fa, graph_fa)

    closure = intersection.transitive_closure()

    # All pairs of states between which there is a path.
    rows, columns = closure.nonzero()
    # Boolean mask keeping only paths from a valid start state to a valid final state.
    accepting = intersection.start_states[rows] & intersection.final_states[columns]
    rows, columns = rows[accepting], columns[accepting]

    n_graph = graph_fa.states_number
    graph_states = graph_fa.index_to_state

    result = set()
    for index_from, index_to in zip(rows, columns):
        graph_from = graph_states[index_from % n_graph].value
        graph_to = graph_states[index_to % n_graph].value
        result.add((graph_from, graph_to))

    return result
