import os

import pytest

from project.task1 import create_two_cycles_graph_and_save_to_dot, get_graph_metadata


def test_get_graph_metadata():
    meta_data = get_graph_metadata("bzip")

    assert meta_data.nodes_number == 632
    assert meta_data.edges_number == 556
    assert meta_data.labels == {"a", "d"}


def test_get_graph_metadata_nonexistent_graph():
    with pytest.raises(FileNotFoundError):
        get_graph_metadata("nonexistent_graph")


def test_create_two_cycles_graph_and_save_to_dot(tmp_path):
    file_path = tmp_path / "two_cycles.dot"
    expected_path = "tests/data/expected/two_cycles.dot"
    create_two_cycles_graph_and_save_to_dot(3, 4, ("a", "b"), file_path)

    assert os.path.exists(file_path)
    with open(file_path) as f_actual, open(expected_path) as f_expected:
        assert f_actual.read() == f_expected.read()
