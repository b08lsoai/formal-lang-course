from collections.abc import Iterable

import numpy as np
import scipy.sparse as sp
from networkx import MultiDiGraph
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State, Symbol


class AdjacencyMatrixFA:
    """Finite automaton as a sparse boolean decomposition of its adjacency
    matrix and start/final state information.
    """

    def __init__(self, fa: NondeterministicFiniteAutomaton) -> None:
        self.states = fa.states
        self.state_to_index = {state: index for index, state in enumerate(fa.states)}
        self.index_to_state = {
            index: state for state, index in self.state_to_index.items()
        }
        states_number = len(self.states)
        self.states_number = states_number
        self.alphabet = fa.symbols

        self.start_states = np.zeros(states_number, dtype=bool)
        for state in fa.start_states:
            self.start_states[self.state_to_index[state]] = True

        self.final_states = np.zeros(states_number, dtype=bool)
        for state in fa.final_states:
            self.final_states[self.state_to_index[state]] = True

        self.boolean_decomposition = self._build_boolean_decomposition(fa)

    def _build_boolean_decomposition(
        self, fa: NondeterministicFiniteAutomaton
    ) -> dict[Symbol, sp.csr_matrix]:

        states_number = self.states_number

        # Build each CSR matrix in a single call instead of element-wise.
        edges: dict[Symbol, list[tuple[int, int]]] = {
            symbol: [] for symbol in fa.symbols
        }

        for state_from, transitions in fa.to_dict().items():
            source_index = self.state_to_index[state_from]
            for symbol, states_to in transitions.items():
                # NFA gives a set of states, DFA a single State.
                states_to = states_to if isinstance(states_to, set) else {states_to}
                for state_to in states_to:
                    edges[symbol].append((source_index, self.state_to_index[state_to]))

        decomposition: dict[Symbol, sp.csr_matrix] = {}
        for symbol, symbol_edges in edges.items():
            if symbol_edges:
                rows, cols = zip(*symbol_edges)
                data = np.ones(len(symbol_edges), dtype=bool)
                decomposition[symbol] = sp.csr_matrix(
                    (data, (rows, cols)), shape=(states_number, states_number)
                )
            else:
                decomposition[symbol] = sp.csr_matrix(
                    (states_number, states_number), dtype=bool
                )
        return decomposition

    @classmethod
    def from_components(
        cls,
        state_to_index: dict[State, int],
        start_states: np.ndarray,
        final_states: np.ndarray,
        alphabet: set[Symbol],
        boolean_decomposition: dict[Symbol, sp.csr_matrix],
    ) -> "AdjacencyMatrixFA":
        """Build directly from matrices and vectors.

        ``state_to_index`` must align states with matrix rows/columns.
        """
        result = cls.__new__(cls)
        result.states = set(state_to_index)
        result.state_to_index = state_to_index
        result.index_to_state = {
            index: state for state, index in state_to_index.items()
        }
        result.alphabet = alphabet
        result.start_states = start_states
        result.final_states = final_states
        result.boolean_decomposition = {}
        states_number = len(state_to_index)
        result.states_number = states_number

        for symbol in alphabet:
            # Missing symbols get a zero matrix to keep the invariant.
            if symbol in boolean_decomposition:
                result.boolean_decomposition[symbol] = boolean_decomposition[
                    symbol
                ].astype(bool)
            else:
                result.boolean_decomposition[symbol] = sp.csr_matrix(
                    (states_number, states_number), dtype=bool
                )
        return result

    def _adjacency_matrix(self) -> sp.csr_matrix:
        """Union of the symbol matrices: one step along any symbol."""
        adjacency = sp.csr_matrix((self.states_number, self.states_number), dtype=bool)
        for matrix in self.boolean_decomposition.values():
            adjacency = (adjacency + matrix).astype(bool)
        return adjacency

    def transitive_closure(self) -> sp.csr_matrix:
        """Boolean matrix T where T[i, j] is True iff j is reachable from i."""
        states_number = self.states_number
        closure = (
            sp.eye(states_number, format="csr", dtype=bool) + self._adjacency_matrix()
        ).astype(bool)

        while True:
            new_closure = (closure + closure @ closure).astype(bool)
            if (new_closure != closure).nnz == 0:
                return new_closure
            closure = new_closure

    def is_empty(self) -> bool:
        """True if no final state is reachable from any start state."""
        if not np.any(self.start_states) or not np.any(self.final_states):
            return True
        closure = self.transitive_closure()
        reachable = (self.start_states @ closure).astype(bool)
        return not np.any(reachable & self.final_states)

    def accepts(self, word: Iterable[Symbol]) -> bool:
        reachable = self.start_states.copy()
        is_empty_word = True

        for symbol in word:
            is_empty_word = False
            matrix = self.boolean_decomposition.get(symbol)
            if matrix is None:
                return False
            reachable = (reachable @ matrix).astype(bool)

            # Empty reachable set is a dead end.
            if not np.any(reachable):
                return False

        # The empty word is accepted iff some state is both start and final.
        if is_empty_word:
            return bool(np.any(self.start_states & self.final_states))

        return bool(np.any(reachable & self.final_states))


def intersect_automata(
    fa1: AdjacencyMatrixFA, fa2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    """Computes the intersection of two automata using the Kronecker product."""
    n2 = fa2.states_number
    if fa1.states_number == 0 or n2 == 0:
        return AdjacencyMatrixFA(NondeterministicFiniteAutomaton())

    result_alphabet = fa1.alphabet & fa2.alphabet

    result_state_to_index = {
        State((state1, state2)): index1 * n2 + index2
        for state1, index1 in fa1.state_to_index.items()
        for state2, index2 in fa2.state_to_index.items()
    }

    boolean_decomposition = {
        symbol: sp.kron(
            fa1.boolean_decomposition[symbol],
            fa2.boolean_decomposition[symbol],
            format="csr",
        )
        for symbol in result_alphabet
    }

    start_states = np.kron(fa1.start_states, fa2.start_states)
    final_states = np.kron(fa1.final_states, fa2.final_states)

    return AdjacencyMatrixFA.from_components(
        result_state_to_index,
        start_states,
        final_states,
        result_alphabet,
        boolean_decomposition,
    )


def tensor_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    """Vertex pairs (start, final) connected by a path in the regex language."""
    if not start_nodes:
        return set()

    from project.task2 import graph_to_nfa, regex_to_dfa

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
