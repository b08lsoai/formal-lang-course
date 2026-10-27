from collections import defaultdict

import networkx as nx
import numpy as np
import scipy.sparse as sp
from pyformlang.cfg import CFG, Variable

from project.task6 import cfg_to_weak_normal_form


def matrix_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """Vertex pairs connected by a path whose label belongs to ``cfg``.

    An empty ``start_nodes`` / ``final_nodes`` means every vertex.
    """
    cfg = cfg_to_weak_normal_form(cfg)

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
        variable: sp.csr_matrix(shape, dtype=bool) for variable in cfg.variables
    }
    edges: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for source, target, data in graph.edges(data=True):
        edges[data["label"]].append((node_to_index[source], node_to_index[target]))

    binary_productions: list[tuple[Variable, Variable, Variable]] = []
    terminal_matrices: dict[str, sp.csr_matrix] = {}
    identity = sp.eye(nodes_number, format="csr", dtype=bool)

    for production in cfg.productions:
        head, body = production.head, production.body
        if len(body) == 0:
            # N -> epsilon derives the empty path i -> i for every vertex.
            matrices[head] = matrices[head] + identity
        elif len(body) == 1:
            terminal = body[0].value
            if terminal not in terminal_matrices:
                terminal_edges = edges.get(terminal)
                if not terminal_edges:
                    continue
                rows, columns = zip(*terminal_edges)
                terminal_matrices[terminal] = sp.csr_matrix(
                    (np.ones(len(rows), dtype=bool), (rows, columns)), shape=shape
                )
            matrices[head] = matrices[head] + terminal_matrices[terminal]
        else:
            binary_productions.append((head, body[0], body[1]))

    changed = True
    while changed:
        changed = False
        for head, left, right in binary_productions:
            # left @ right composes i -> k with k -> j into i -> j.
            updated = matrices[head] + matrices[left] @ matrices[right]
            # Matrices only gain True entries, so nnz detects the fixed point.
            if updated.nnz != matrices[head].nnz:
                matrices[head] = updated
                changed = True

    rows, columns = matrices[cfg.start_symbol].nonzero()
    return {
        (nodes[source], nodes[target])
        for source, target in zip(rows, columns)
        if nodes[source] in starts and nodes[target] in finals
    }
