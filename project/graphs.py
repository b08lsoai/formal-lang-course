from dataclasses import dataclass

import cfpq_data as cfpq
import networkx as nx
from networkx import MultiDiGraph


@dataclass
class GraphMetadata:
    nodes_number: int
    edges_number: int
    labels: set[str]


def load_graph(graph_name: str) -> MultiDiGraph:
    graph_path = cfpq.download(graph_name)
    return cfpq.graph_from_csv(graph_path)


def save_graph_to_dot(graph: MultiDiGraph, path: str) -> None:
    pydot_graph = nx.drawing.nx_pydot.to_pydot(graph)
    pydot_graph.write_raw(path)


def get_graph_metadata(graph_name: str) -> GraphMetadata:
    graph = load_graph(graph_name)
    return GraphMetadata(
        graph.number_of_nodes(),
        graph.number_of_edges(),
        set(cfpq.get_sorted_labels(graph)),
    )


def create_two_cycles_graph_and_save_to_dot(
    n: int,
    m: int,
    labels: tuple[str, str],
    file_path: str,
) -> None:
    graph = cfpq.labeled_two_cycles_graph(n, m, labels=labels)
    save_graph_to_dot(graph, file_path)
