import networkx as nx

from project.grammar import cfg_to_rsm, ebnf_to_rsm
from project.tensor_cfpq import tensor_based_cfpq


def test_tensor_based_cfpq_reachability(cfpq_case):
    graph, cfg, start_nodes, final_nodes, expected = cfpq_case

    actual = tensor_based_cfpq(cfg_to_rsm(cfg), graph, start_nodes, final_nodes)

    assert actual == expected


def test_tensor_based_cfpq_nonconsecutive_nodes():
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 30, label="a")
    graph.add_edge(30, 70, label="b")
    rsm = ebnf_to_rsm("S -> A B\nA -> a\nB -> b*")

    assert tensor_based_cfpq(rsm, graph) == {(10, 30), (10, 70)}


def test_tensor_based_cfpq_nullable_boxes_on_isolated_nodes():
    graph = nx.MultiDiGraph()
    graph.add_nodes_from([10, 30])
    rsm = ebnf_to_rsm("S -> A B\nA -> $\nB -> $")

    assert tensor_based_cfpq(rsm, graph) == {(10, 10), (30, 30)}
    assert tensor_based_cfpq(rsm, graph, {10}, {30}) == set()


def test_tensor_based_cfpq_empty_graph():
    graph = nx.MultiDiGraph()
    rsm = ebnf_to_rsm("S -> $")

    assert tensor_based_cfpq(rsm, graph) == set()
