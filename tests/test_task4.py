import networkx as nx

from project.task3 import tensor_based_rpq
from project.task4 import ms_bfs_based_rpq


def _graph(edges: list[tuple[int, int, str]]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_edges_from((u, v, {"label": label}) for u, v, label in edges)
    return graph


# ---------- ms_bfs_based_rpq ----------


def test_ms_bfs_based_rpq_three_cycle():
    graph = _graph([(0, 1, "b"), (1, 2, "b"), (2, 0, "b")])
    assert ms_bfs_based_rpq("b", graph, {0}, {2}) == set()
    assert ms_bfs_based_rpq("b b", graph, {0}, {2}) == {(0, 2)}


def test_ms_bfs_based_rpq_multiple_sources_attributed():
    graph = _graph([(0, 1, "a"), (1, 2, "a"), (3, 2, "a")])
    result = ms_bfs_based_rpq("a", graph, {0, 3}, {2})
    assert result == {(3, 2)}
    result = ms_bfs_based_rpq("a a", graph, {0, 3}, {2})
    assert result == {(0, 2)}


def test_ms_bfs_based_rpq_all_nodes():
    graph = _graph([(0, 1, "b")])
    assert ms_bfs_based_rpq("b*", graph, set(), set()) == {(0, 0), (0, 1), (1, 1)}


def test_ms_bfs_based_rpq_empty_graph():
    graph = nx.MultiDiGraph()
    assert ms_bfs_based_rpq("a*", graph, set(), set()) == set()


def test_ms_bfs_based_rpq_vertex_outside_graph():
    graph = _graph([(0, 1, "b")])
    assert ms_bfs_based_rpq("b", graph, {7}, {7}) == set()


def test_ms_bfs_based_rpq_matches_tensor_based_rpq():
    graph = _graph([(0, 1, "a"), (1, 2, "b"), (2, 0, "a"), (0, 3, "b")])
    for regex in ["a", "a b", "(a | b)*", "a* b a*"]:
        for starts in [{0}, {0, 1}, {0, 3}]:
            for finals in [{2}, {1, 3}, {0, 2, 3}]:
                start_nodes, final_nodes = set(starts), set(finals)
                assert ms_bfs_based_rpq(
                    regex, graph, start_nodes, final_nodes
                ) == tensor_based_rpq(regex, graph, start_nodes, final_nodes)
