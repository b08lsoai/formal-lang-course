from project.task6 import hellings_based_cfpq


def test_hellings_based_cfpq_reachability(cfpq_case):
    graph, cfg, start_nodes, final_nodes, expected = cfpq_case
    assert hellings_based_cfpq(cfg, graph, start_nodes, final_nodes) == expected
