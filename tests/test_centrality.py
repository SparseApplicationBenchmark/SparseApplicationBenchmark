import pytest

import numpy as np

import networkx as nx

from sparseappbench.benchmarks.centrality import betweenness_centrality
from sparseappbench.binsparse_format import BinsparseFormat
from sparseappbench.frameworks.numpy_framework import NumpyFramework


def run_bc(xp, A):
    A_bin = BinsparseFormat.from_numpy(A)
    result_bin = betweenness_centrality(xp, A_bin)
    return xp.from_benchmark(result_bin).ravel()


# Modified the intended results because I am calculating
# unnormalized betweenness centrality.
def test_joels_case():
    xp = NumpyFramework()

    A = np.array(
        [
            [0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ],
        dtype=float,
    )

    result = run_bc(xp, A)
    expected = np.array([0.0, 1.0, 1.0, 3.0, 0.0])

    assert np.allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize(
    "A,expected",
    [
        (np.zeros((3, 3)), np.array([0.0, 0.0, 0.0])),
        (
            np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float),
            np.array([0.0, 1.0, 0.0]),
        ),
        (
            np.array(
                [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]],
                dtype=float,
            ),
            np.array([0.0, 0.0, 0.0, 0.0]),
        ),
    ],
)
def test_basic_bc(A, expected):
    xp = NumpyFramework()
    result = run_bc(xp, A)
    assert np.allclose(result, expected, atol=1e-6)


def test_networkx():
    xp = NumpyFramework()
    G = nx.DiGraph()
    G.add_edges_from(
        [
            (0, 1),
            (1, 2),
            (2, 0),
            (2, 3),
            (3, 4),
            (4, 2),
        ]
    )

    A = nx.to_numpy_array(G, dtype=float)
    result = run_bc(xp, A)

    bc = nx.betweenness_centrality(G, normalized=False)
    expected = np.array([bc[i] for i in range(len(G))])

    assert np.allclose(result, expected, atol=1e-6)
