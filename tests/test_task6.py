from pathlib import Path

import cfpq_data as cfpq
import networkx as nx
import pytest
from pyformlang.cfg import CFG

from project.task6 import hellings_based_cfpq

_DATA_DIR = Path(__file__).parent / "data" / "cfpq"


def _load_graph(name: str) -> nx.MultiDiGraph:
    return cfpq.graph_from_csv(_DATA_DIR / f"{name}.csv")


_CFPQ_CASES = [
    ("balanced_ab", "S -> a S b | a b", set(), set(), {(0, 4), (1, 3)}),
    (
        "balanced_ab",
        "S -> a S b | $",
        set(),
        set(),
        {(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (0, 4), (1, 3)},
    ),
    ("balanced_ab", "S -> a S b | $", {0}, {4}, {(0, 4)}),
    ("balanced_ab", "S -> a b", {1}, {3}, {(1, 3)}),
    ("abc_chain", "S -> X\nX -> a b c", {0}, {3}, {(0, 3)}),
    ("abc_chain", "S -> a b | b c", {1}, {1, 3}, {(1, 3)}),
    ("diamond_ab", "S -> a b", {0}, {3}, {(0, 3)}),
    ("diamond_ab", "S -> b a", set(), set(), set()),
    ("diamond_ab", "S -> a S b | $", {0}, {0, 3}, {(0, 0), (0, 3)}),
    ("abc_cycle", "S -> a b c", {0}, {3}, {(0, 3)}),
    ("parallel_a", "S -> a a | a", set(), set(), {(0, 1), (1, 2), (0, 2)}),
]


@pytest.mark.parametrize(
    ("graph_name", "cfg_text", "start_nodes", "final_nodes", "expected"), _CFPQ_CASES
)
def test_hellings_based_cfpq_reachability(
    graph_name, cfg_text, start_nodes, final_nodes, expected
):
    graph = _load_graph(graph_name)
    cfg = CFG.from_text(cfg_text)

    assert hellings_based_cfpq(cfg, graph, start_nodes, final_nodes) == expected
