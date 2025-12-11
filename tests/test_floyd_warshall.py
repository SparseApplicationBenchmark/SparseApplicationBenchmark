import numpy as np

from sparseappbench.benchmarks.floyd_warshall import floyd_warshall
from sparseappbench.binsparse_format import BinsparseFormat
from sparseappbench.frameworks.numpy_framework import NumpyFramework

xp = NumpyFramework()


# Reference Floyd–Warshall
def floyd_warshall_reference(A):
    A = A.copy()
    n = A.shape[0]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                via = A[i, k] + A[k, j]
                if via < A[i, j]:
                    A[i, j] = via
    return A


# SOC-TRIBES SOCIAL NETWORK (16 nodes, 58 edges)
TRIBES_N = 16
TRIBES_EDGES = [
    (0, 1),
    (1, 0),
    (0, 2),
    (2, 0),
    (1, 2),
    (2, 1),
    (0, 3),
    (3, 0),
    (2, 3),
    (3, 2),
    (0, 4),
    (4, 0),
    (1, 4),
    (4, 1),
    (0, 5),
    (5, 0),
    (1, 5),
    (5, 1),
    (2, 5),
    (5, 2),
    (2, 6),
    (6, 2),
    (4, 6),
    (6, 4),
    (5, 6),
    (6, 5),
    (2, 7),
    (7, 2),
    (3, 7),
    (7, 3),
    (5, 7),
    (7, 5),
    (6, 7),
    (7, 6),
    (1, 8),
    (8, 1),
    (4, 8),
    (8, 4),
    (7, 8),
    (8, 7),
    (3, 9),
    (9, 3),
    (8, 9),
    (9, 8),
    (9, 10),
    (10, 9),
    (10, 11),
    (11, 10),
    (8, 11),
    (11, 8),
    (11, 12),
    (12, 11),
    (7, 12),
    (12, 7),
    (12, 13),
    (13, 12),
    (13, 14),
    (14, 13),
    (9, 14),
    (14, 9),
    (14, 15),
    (15, 14),
    (12, 15),
    (15, 12),
]


def build_tribes_matrix():
    A = np.full((TRIBES_N, TRIBES_N), np.inf)
    np.fill_diagonal(A, 0)
    for u, v in TRIBES_EDGES:
        A[u, v] = 1.0
    return A


# CHESAPEAKE ROAD NETWORK (39 nodes, 170 edges)
CHESAPEAKE_N = 39
CHESAPEAKE_EDGES = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (0, 6),
    (0, 7),
    (0, 8),
    (1, 9),
    (1, 10),
    (1, 11),
    (1, 12),
    (1, 13),
    (1, 14),
    (1, 15),
    (1, 16),
    (2, 9),
    (2, 10),
    (2, 17),
    (2, 18),
    (3, 11),
    (3, 12),
    (3, 19),
    (3, 20),
    (3, 21),
    (4, 13),
    (4, 22),
    (4, 23),
    (4, 24),
    (5, 14),
    (5, 22),
    (5, 25),
    (5, 26),
    (6, 15),
    (6, 23),
    (6, 27),
    (6, 28),
    (7, 16),
    (7, 24),
    (7, 29),
    (7, 30),
    (8, 17),
    (8, 18),
    (8, 19),
    (8, 20),
    (8, 21),
    (9, 22),
    (9, 31),
    (9, 32),
    (10, 23),
    (10, 31),
    (10, 33),
    (11, 24),
    (11, 32),
    (11, 34),
    (12, 25),
    (12, 26),
    (12, 35),
    (13, 27),
    (13, 36),
    (14, 28),
    (14, 37),
    (15, 29),
    (15, 38),
    (16, 30),
    (17, 31),
    (18, 32),
    (19, 33),
    (20, 34),
    (21, 35),
    (22, 36),
    (23, 37),
    (24, 38),
    (25, 27),
    (25, 29),
    (26, 28),
    (26, 30),
    (27, 31),
    (27, 33),
    (28, 32),
    (28, 34),
    (29, 35),
    (30, 36),
    (31, 37),
    (32, 38),
    (33, 35),
    (34, 36),
    (35, 37),
    (36, 38),
    (37, 38),
]


def build_chesapeake_matrix():
    A = np.full((CHESAPEAKE_N, CHESAPEAKE_N), np.inf)
    np.fill_diagonal(A, 0)
    for u, v in CHESAPEAKE_EDGES:
        A[u, v] = 1.0
        A[v, u] = 1.0
    return A


# TESTS
def test_floyd_warshall_tribes():
    A = build_tribes_matrix()
    bench = BinsparseFormat.from_numpy(A)
    out = xp.from_benchmark(floyd_warshall(xp, bench))
    ref = floyd_warshall_reference(A)
    assert np.allclose(out, ref, equal_nan=True)


def test_floyd_warshall_chesapeake():
    A = build_chesapeake_matrix()
    bench = BinsparseFormat.from_numpy(A)
    out = xp.from_benchmark(floyd_warshall(xp, bench))
    ref = floyd_warshall_reference(A)
    assert np.allclose(out, ref, equal_nan=True)
