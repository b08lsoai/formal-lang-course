import networkx as nx
import pytest
from pyformlang.cfg import CFG, Variable
from pyformlang.rsa import RecursiveAutomaton

from project.gll_cfpq import gll_based_cfpq
from project.grammar import cfg_to_rsm, ebnf_to_rsm


def test_gll_based_cfpq_reachability(cfpq_case):
    graph, cfg, start_nodes, final_nodes, expected = cfpq_case

    assert gll_based_cfpq(cfg_to_rsm(cfg), graph, start_nodes, final_nodes) == expected


@pytest.mark.parametrize(
    "grammar, expected",
    [
        ("S -> A\nA -> B\nB -> A", set()),
        ("S -> A\nA -> B | $\nB -> A", {(10, 10), (30, 30)}),
    ],
)
def test_gll_based_cfpq_mutually_recursive_boxes(grammar, expected):
    graph = nx.MultiDiGraph()
    graph.add_nodes_from([10, 30])

    assert gll_based_cfpq(ebnf_to_rsm(grammar), graph) == expected


@pytest.mark.parametrize(
    "start_nodes, final_nodes",
    [({999}, None), (None, {999}), ({999}, {999})],
)
def test_gll_based_cfpq_unknown_endpoints(start_nodes, final_nodes):
    graph = nx.MultiDiGraph()
    graph.add_node(10)
    rsm = ebnf_to_rsm("S -> $")

    assert gll_based_cfpq(rsm, graph, start_nodes, final_nodes) == set()


def test_gll_based_cfpq_custom_start_symbol():
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 20, label="a")
    graph.add_edge(20, 30, label="a")
    cfg = CFG.from_text("Root -> A A\nA -> a", start_symbol=Variable("Root"))

    assert gll_based_cfpq(cfg_to_rsm(cfg), graph) == {(10, 30)}


def test_gll_based_cfpq_deep_call_chain():
    graph = nx.DiGraph()
    graph.add_edges_from((v, v + 1, {"label": "a"}) for v in range(1100))
    graph.add_edge(1100, 1101, label="b")
    rsm = ebnf_to_rsm("S -> a S | b")

    assert gll_based_cfpq(rsm, graph, {0}, {1101}) == {(0, 1101)}


def test_gll_based_cfpq_empty_graph():
    assert gll_based_cfpq(ebnf_to_rsm("S -> $"), nx.MultiDiGraph()) == set()


def test_gll_based_cfpq_empty_rsm():
    graph = nx.MultiDiGraph()
    graph.add_node(10)

    assert gll_based_cfpq(RecursiveAutomaton(), graph) == set()
