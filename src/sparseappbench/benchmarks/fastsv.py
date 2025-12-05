"""
Name: FastSV Algorithm
Author: Richard Wan
Email: rwan41@gatech.edu

Motivation:
The FastSV algorithm is a graph algorithm used to find the connected components
for a simple graph. This algorithm introduces several optimizations that allow
for faster convergence to a solution compared to the SV algorithm it is based on,
specifically through modifications to the tree hooking and termination condition.

Citation for reference implementation:
Zhang, Y., Azad, A., & Hu, Z. (2020). FastSV: A distributed-memory connected
component algorithm with fast convergence. In Proceedings of the 2020 SIAM Conference on
Parallel Processing for Scientific Computing (pp. 46-57). Society for Industrial and
Applied Mathematics.

Statement on the use of Generative AI: No generative AI was used to construct
the benchmark function itself. Generative AI might have been used to construct tests.
This statement was written by hand.
"""


def benchmark_fastsv(xp, adjacency_matrix):
    edges = xp.from_benchmark(adjacency_matrix)
    (n, m) = edges.shape
    assert n == m

    f = xp.arange(n)

    # for edge traversal
    rows, cols = xp.nonzero(edges)

    while True:
        f_next = xp.copy(f)

        # step 1 (stochastic hooking)
        for u, v in zip(rows, cols, strict=True):
            if f_next[f[u]] > f[f[v]]:
                f_next[f[u]] = f[f[v]]

        # step 2 (aggressive hooking)
        for u, v in zip(rows, cols, strict=True):
            if f_next[u] > f[f[v]]:
                f_next[u] = f[f[v]]

        # step 3 (shortcutting)
        f_next = xp.minimum(f_next, f[f])

        f_prev = f
        f = f_next

        if xp.all(f[f] == f_prev[f_prev]):
            break

    return xp.to_benchmark(f)
