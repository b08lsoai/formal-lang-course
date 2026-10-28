from project.hellings_cfpq import hellings_based_cfpq
from project.matrix_cfpq import matrix_based_cfpq


def test_matrix_based_cfpq_reachability(cfpq_case):
    graph, cfg, start_nodes, final_nodes, expected = cfpq_case

    actual = matrix_based_cfpq(cfg, graph, start_nodes, final_nodes)

    assert actual == expected
    assert actual == hellings_based_cfpq(cfg, graph, start_nodes, final_nodes)
