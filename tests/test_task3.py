import networkx as nx

from project.task2 import graph_to_nfa, regex_to_dfa
from project.task3 import AdjacencyMatrixFA, intersect_automata, tensor_based_rpq


def _fa_from_regex(regex: str) -> AdjacencyMatrixFA:
    return AdjacencyMatrixFA(regex_to_dfa(regex))


def _fa_from_graph(
    edges: list[tuple[int, int, str]], start: set[int], final: set[int]
) -> AdjacencyMatrixFA:
    graph = nx.MultiDiGraph()
    graph.add_edges_from((u, v, {"label": label}) for u, v, label in edges)
    return AdjacencyMatrixFA(graph_to_nfa(graph, start, final))


# ---------- build from automata ----------


def test_fa_from_dfa():
    dfa = regex_to_dfa("a b")
    fa = AdjacencyMatrixFA(dfa)
    assert fa.states_number == 3
    assert sorted(map(str, fa.alphabet)) == ["a", "b"]
    start = next(iter(dfa.start_states))
    final = next(iter(dfa.final_states))
    assert fa.start_states[fa.state_to_index[start]]
    assert fa.final_states[fa.state_to_index[final]]


def test_fa_from_nfa():
    fa = _fa_from_graph([(0, 1, "a"), (1, 2, "b")], {0}, {2})
    assert fa.states_number == 3
    assert sorted(map(str, fa.alphabet)) == ["a", "b"]
    assert fa.start_states.sum() == 1
    assert fa.final_states.sum() == 1


# ---------- accepts ----------


def test_accepts_simple_path():
    fa = _fa_from_graph([(0, 1, "a"), (1, 2, "b")], {0}, {2})
    assert fa.accepts(["a", "b"])
    assert not fa.accepts(["a"])
    assert not fa.accepts(["b", "a"])


def test_accepts_prefix_is_not_enough():
    fa = _fa_from_regex("a b c")
    assert not fa.accepts(["a"])
    assert not fa.accepts(["a", "b"])
    assert fa.accepts(["a", "b", "c"])


def test_accepts_empty_word():
    fa = _fa_from_regex("a*")
    assert fa.accepts([])
    fa = _fa_from_regex("a")
    assert not fa.accepts([])


def test_accepts_symbol_outside_alphabet():
    fa = _fa_from_regex("a b")
    assert not fa.accepts(["c"])


def test_accepts_dead_end_stops_early():
    fa = _fa_from_graph([(0, 1, "a")], {0}, {1})
    assert fa.accepts(["a"])
    assert not fa.accepts(["a", "a"])


# ---------- is_empty ----------


def test_is_empty_with_path():
    assert not _fa_from_graph([(0, 1, "a"), (1, 2, "b")], {0}, {2}).is_empty()


def test_is_empty_no_path():
    assert _fa_from_graph([(0, 1, "a")], {0}, {5}).is_empty()


def test_is_empty_empty_automaton():
    fa = AdjacencyMatrixFA(graph_to_nfa(nx.MultiDiGraph(), set(), set()))
    assert fa.states_number == 0
    assert fa.is_empty()


# ---------- transitive_closure ----------


def test_transitive_closure_path_and_identity():
    fa = _fa_from_graph([(0, 1, "a"), (1, 2, "b")], {0}, {2})
    closure = fa.transitive_closure().toarray()
    assert closure[0, 1]
    assert closure[0, 2]
    assert not closure[2, 0]
    assert closure[0, 0] and closure[1, 1] and closure[2, 2]


# ---------- intersect_automata ----------


def test_intersect_common_word():
    inter = intersect_automata(_fa_from_regex("a b"), _fa_from_regex("a b"))
    assert not inter.is_empty()
    assert inter.accepts(["a", "b"])


def test_intersect_disjoint_languages():
    inter = intersect_automata(_fa_from_regex("a b"), _fa_from_regex("b a"))
    assert inter.is_empty()


def test_intersect_shared_only_empty_word():
    inter = intersect_automata(_fa_from_regex("a*"), _fa_from_regex("b*"))
    assert not inter.is_empty()
    assert inter.accepts([])


def test_intersect_alphabet_is_shared():
    inter = intersect_automata(_fa_from_regex("a b"), _fa_from_regex("a"))
    assert sorted(map(str, inter.alphabet)) == ["a"]
    assert inter.is_empty()


def test_intersect_cycle_automata():
    cycle_4 = _fa_from_graph([(i, (i + 1) % 4, "a") for i in range(4)], {0}, {2})
    cycle_8 = _fa_from_graph([(i, (i + 1) % 8, "a") for i in range(8)], {0}, {2})
    inter = intersect_automata(cycle_4, cycle_8)
    assert not inter.is_empty()
    assert inter.accepts(["a"] * 10)
    assert not inter.accepts(["a"] * 6)


def test_intersect_with_empty_automaton():
    empty = AdjacencyMatrixFA(graph_to_nfa(nx.MultiDiGraph(), set(), set()))
    inter = intersect_automata(_fa_from_regex("a*"), empty)
    assert inter.is_empty()
    assert not inter.accepts([])


# ---------- tensor_based_rpq ----------


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
