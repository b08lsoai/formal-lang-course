from collections import deque
from typing import NamedTuple

import networkx as nx
from pyformlang.finite_automaton import State, Symbol
from pyformlang.rsa import RecursiveAutomaton

from project.automata import graph_to_nfa

RSMState = tuple[Symbol, State]


class GSSNode:
    """A shared call (nonterminal, graph vertex) and its continuations."""

    def __init__(self, label: Symbol, start_vertex: int) -> None:
        self.label = label
        self.start_vertex = start_vertex
        self.edges: set[tuple[RSMState, GSSNode]] = set()
        self.popped: set[int] = set()

    def add_edge(self, return_state: RSMState, caller: "GSSNode") -> set["Descriptor"]:
        """Give a new continuation all previously found results of this call."""
        edge = (return_state, caller)
        if edge in self.edges:
            return set()
        self.edges.add(edge)
        return {Descriptor(return_state, caller, target) for target in self.popped}

    def pop(self, vertex: int) -> set["Descriptor"]:
        """Save a new result and resume every waiting continuation."""
        if vertex in self.popped:
            return set()
        self.popped.add(vertex)
        return {
            Descriptor(return_state, caller, vertex)
            for return_state, caller in self.edges
        }


class Descriptor(NamedTuple):
    slot: RSMState
    gss_node: GSSNode
    vertex: int


def gll_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """GLL reachability over RSM boxes using descriptors and a shared stack.

    An empty ``start_nodes`` / ``final_nodes`` means every vertex.
    """
    nodes = set(graph.nodes)
    starts = nodes if not start_nodes else nodes.intersection(start_nodes)
    finals = nodes if not final_nodes else nodes.intersection(final_nodes)
    if not starts or not finals or rsm.initial_label not in rsm.boxes:
        return set()

    # Calls may visit any graph vertex, regardless of the endpoint filters.
    graph_transitions = {
        source.value: {
            symbol: {target.value for target in targets}
            for symbol, targets in transitions.items()
        }
        for source, transitions in graph_to_nfa(graph, set(), set()).to_dict().items()
    }
    box_labels = set(rsm.boxes)
    box_starts = {label: box.dfa.start_states for label, box in rsm.boxes.items()}
    box_finals = {label: box.dfa.final_states for label, box in rsm.boxes.items()}
    box_transitions = {label: box.dfa.to_dict() for label, box in rsm.boxes.items()}

    # Shared GSS: one node per (nonterminal, graph vertex) call.
    gss: dict[tuple[Symbol, int], GSSNode] = {}
    queue: deque[Descriptor] = deque()  # descriptors waiting to be processed
    seen: set[Descriptor] = set()  # all descriptors already created

    def get_node(label: Symbol, start_vertex: int) -> GSSNode:
        node = gss.get((label, start_vertex))
        if node is None:
            node = GSSNode(label, start_vertex)
            gss[(label, start_vertex)] = node
        return node

    def add(descriptor: Descriptor) -> None:
        if descriptor not in seen:
            seen.add(descriptor)
            queue.append(descriptor)

    initial_calls = [get_node(rsm.initial_label, source) for source in starts]
    for gss_node in initial_calls:
        for state in box_starts[gss_node.label]:
            add(Descriptor((gss_node.label, state), gss_node, gss_node.start_vertex))

    while queue:
        slot, gss_node, vertex = queue.popleft()
        label, state = slot
        for symbol, next_state in box_transitions[label].get(state, {}).items():
            if symbol not in box_labels:
                # Terminal: advance both the graph vertex and the RSM state.
                for target in graph_transitions.get(vertex, {}).get(symbol, ()):
                    add(Descriptor((label, next_state), gss_node, target))
            else:
                # Call: save the continuation and enter the called box.
                callee = get_node(symbol, vertex)
                for descriptor in callee.add_edge((label, next_state), gss_node):
                    add(descriptor)
                for entry in box_starts[symbol]:
                    add(Descriptor((symbol, entry), callee, vertex))

        # Return: final states may also have outgoing transitions.
        if state in box_finals[label]:
            for descriptor in gss_node.pop(vertex):
                add(descriptor)

    return {
        (gss_node.start_vertex, target)
        for gss_node in initial_calls
        for target in gss_node.popped
        if target in finals
    }
