import cfpq_data as cfpq
import networkx as nx

from project.task2 import graph_to_nfa, regex_to_dfa

# ---------- DFA: build from regular expressions ----------


def test_regex_to_dfa_single_symbol():
    dfa = regex_to_dfa("a")
    assert dfa.is_deterministic()
    assert dfa.accepts(["a"])
    assert not dfa.accepts(["b"])
    assert not dfa.accepts([])


def test_regex_to_dfa_union():
    dfa = regex_to_dfa("a | b")
    assert dfa.is_deterministic()
    assert dfa.accepts(["a"])
    assert dfa.accepts(["b"])
    assert not dfa.accepts(["a", "b"])
    assert not dfa.accepts(["c"])


def test_regex_to_dfa_concatenation():
    dfa = regex_to_dfa("a b")
    assert dfa.is_deterministic()
    assert dfa.accepts(["a", "b"])
    assert not dfa.accepts(["a"])
    assert not dfa.accepts(["b"])
    assert not dfa.accepts(["b", "a"])


def test_regex_to_dfa_concatenation_of_three():
    dfa = regex_to_dfa("a b c")
    assert dfa.accepts(["a", "b", "c"])
    assert not dfa.accepts(["a", "b"])


def test_regex_to_dfa_kleene_star():
    dfa = regex_to_dfa("a*")
    assert dfa.is_deterministic()
    assert dfa.accepts([])
    assert dfa.accepts(["a"])
    assert dfa.accepts(["a", "a", "a"])
    assert not dfa.accepts(["b"])


def test_regex_to_dfa_epsilon():
    dfa = regex_to_dfa("epsilon")
    assert dfa.is_deterministic()
    assert dfa.accepts([])
    assert not dfa.accepts(["a"])


def test_regex_to_dfa_empty_language():
    dfa = regex_to_dfa("")
    assert dfa.is_deterministic()
    assert not dfa.accepts([])
    assert not dfa.accepts(["a"])


def test_regex_to_dfa_grouping():
    dfa = regex_to_dfa("(a | b) c*")
    assert dfa.is_deterministic()
    assert dfa.accepts(["a"])
    assert dfa.accepts(["b", "c", "c"])
    assert not dfa.accepts(["c"])
    assert not dfa.accepts(["a", "b"])


# ---------- NFA: build from graph ----------


def test_graph_to_nfa_with_start_and_final():
    graph = nx.MultiDiGraph()
    graph.add_nodes_from([0, 1, 2, 3])
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")
    graph.add_edge(2, 3, label="a")
    graph.add_edge(0, 3, label="a")

    nfa = graph_to_nfa(graph, {0}, {3})
    assert nfa.accepts(["a", "b", "a"])
    assert nfa.accepts(["a"])
    assert not nfa.accepts(["b"])
    assert not nfa.accepts(["a", "b", "a", "b"])


def test_graph_to_nfa_all_nodes():
    graph = nx.MultiDiGraph()
    graph.add_nodes_from([0, 1, 2])
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")

    nfa = graph_to_nfa(graph, set(), set())
    assert len(nfa.start_states) == 3
    assert len(nfa.final_states) == 3
    assert nfa.accepts(["a"])


def test_graph_to_nfa_accepts_words_from_cycle():
    graph = nx.MultiDiGraph()
    graph.add_nodes_from([0, 1, 2, 3])
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="a")
    graph.add_edge(2, 3, label="a")
    graph.add_edge(3, 0, label="a")

    nfa = graph_to_nfa(graph, {0}, {0})
    assert nfa.accepts([])
    assert nfa.accepts(["a", "a", "a", "a"])
    assert nfa.accepts(["a", "a", "a", "a", "a", "a", "a", "a"])


def test_graph_to_nfa_empty_graph():
    graph = nx.MultiDiGraph()

    nfa = graph_to_nfa(graph, set(), set())
    assert len(nfa.start_states) == 0
    assert len(nfa.final_states) == 0
    assert nfa.is_empty()


def test_graph_to_nfa_from_two_cycles_graph():
    graph = cfpq.labeled_two_cycles_graph(2, 3, labels=("a", "b"))

    nfa = graph_to_nfa(graph, {0}, {0})
    assert nfa.accepts([])
    assert nfa.accepts(["a", "a", "a"])
    assert nfa.accepts(["b", "b", "b", "b"])
    assert not nfa.accepts(["a"])
    assert not nfa.accepts(["b", "b"])
