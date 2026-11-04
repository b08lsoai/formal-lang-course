from collections import defaultdict

import networkx as nx
import numpy as np
import scipy.sparse as sp
from pyformlang.cfg import CFG

from project.grammar import WCNFGrammar


def matrix_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """Vertex pairs connected by a path whose label belongs to ``cfg``.

    An empty ``start_nodes`` / ``final_nodes`` means every vertex.
    """
    grammar = WCNFGrammar(cfg)

    nodes = list(graph.nodes)
    nodes_number = len(nodes)
    if nodes_number == 0:
        return set()

    starts = set(nodes) if not start_nodes else set(start_nodes)
    finals = set(nodes) if not final_nodes else set(final_nodes)
    node_to_index = {node: index for index, node in enumerate(nodes)}
    shape = (nodes_number, nodes_number)

    # matrices[A][i, j] is True iff A derives some path nodes[i] -> nodes[j].
    matrices = {
        variable: sp.csr_matrix(shape, dtype=bool) for variable in grammar.variables
    }
    edges: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for source, target, data in graph.edges(data=True):
        edges[data["label"]].append((node_to_index[source], node_to_index[target]))

    identity = sp.eye(nodes_number, format="csr", dtype=bool)
    for head in grammar.epsilon_productions:
        # N -> epsilon derives the empty path i -> i for every vertex.
        matrices[head] = matrices[head] + identity

    for terminal, heads in grammar.terminal_productions.items():
        terminal_edges = edges.get(terminal)
        if not terminal_edges:
            continue
        rows, columns = zip(*terminal_edges)
        # Build once per terminal and reuse for every A -> terminal rule.
        terminal_matrix = sp.csr_matrix(
            (np.ones(len(rows), dtype=bool), (rows, columns)), shape=shape
        )
        for head in heads:
            matrices[head] = matrices[head] + terminal_matrix

    changed = True
    while changed:
        changed = False
        for (left, right), heads in grammar.binary_productions.items():
            # left @ right composes i -> k with k -> j into i -> j.
            product = matrices[left] @ matrices[right]
            for head in heads:
                updated = matrices[head] + product
                # Matrices only gain True entries, so nnz detects the fixed point.
                if updated.nnz != matrices[head].nnz:
                    matrices[head] = updated
                    changed = True

    rows, columns = matrices[grammar.start_symbol].nonzero()
    return {
        (nodes[source], nodes[target])
        for source, target in zip(rows, columns)
        if nodes[source] in starts and nodes[target] in finals
    }
