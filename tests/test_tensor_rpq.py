import networkx as nx

from project.tensor_rpq import tensor_based_rpq


def test_tensor_based_rpq_point_graph():
    graph = nx.MultiDiGraph()
    graph.add_node(1)
    assert tensor_based_rpq("a", graph, {1}, {1}) == set()
    assert tensor_based_rpq("a*", graph, {1}, {1}) == {(1, 1)}


def test_tensor_based_rpq_single_edge():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="b")
    assert tensor_based_rpq("a", graph, {0, 1}, {0, 1}) == set()
    assert tensor_based_rpq("b", graph, {0, 1}, {0, 1}) == {(0, 1)}
    assert tensor_based_rpq("b*", graph, {0, 1}, {0, 1}) == {(0, 0), (0, 1), (1, 1)}


def test_tensor_based_rpq_three_cycle():
    graph = nx.MultiDiGraph()
    graph.add_edges_from(
        [(0, 1, {"label": "b"}), (1, 2, {"label": "b"}), (2, 0, {"label": "b"})]
    )
    assert tensor_based_rpq("b", graph, {0}, {2}) == set()
    assert tensor_based_rpq("b b", graph, {0}, {2}) == {(0, 2)}
    assert tensor_based_rpq("b*", graph, {0}, {0, 1, 2}) == {(0, 0), (0, 1), (0, 2)}


def test_tensor_based_rpq_complete_reachability():
    graph = nx.MultiDiGraph()
    graph.add_edges_from(
        [
            (0, 1, {"label": "b"}),
            (1, 2, {"label": "a"}),
            (2, 0, {"label": "b"}),
        ]
    )
    nodes = {0, 1, 2}
    all_pairs = {(u, v) for u in nodes for v in nodes}
    assert tensor_based_rpq("(a | b)*", graph, nodes, nodes) == all_pairs
