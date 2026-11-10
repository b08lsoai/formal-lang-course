from project.gll_cfpq import gll_based_cfpq
from project.grammar import cfg_to_rsm


def test_gll_based_cfpq_reachability(cfpq_case):
    graph, cfg, start_nodes, final_nodes, expected = cfpq_case

    assert gll_based_cfpq(cfg_to_rsm(cfg), graph, start_nodes, final_nodes) == expected
