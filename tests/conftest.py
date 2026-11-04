from pathlib import Path

import cfpq_data as cfpq
import pytest
from pyformlang.cfg import CFG

_DATA_DIR = Path(__file__).parent / "data" / "cfpq"

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


@pytest.fixture(params=_CFPQ_CASES)
def cfpq_case(request):
    graph_name, cfg_text, starts, finals, expected = request.param
    graph = cfpq.graph_from_csv(_DATA_DIR / f"{graph_name}.csv")
    cfg = CFG.from_text(cfg_text)
    return graph, cfg, starts.copy(), finals.copy(), expected.copy()
