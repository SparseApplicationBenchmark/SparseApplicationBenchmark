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
the benchmark function itself. Generative AI was used for debugging. Generative
AI might have been used to construct tests. This statement was written by hand.
"""


def benchmark_fastsv(xp, adjacency_matrix):
    edges = xp.from_benchmark(adjacency_matrix)

    (n, m) = edges.shape
    assert n == m

    f = xp.arange(n)
    gf = xp.asarray(f, copy=True)

    while True:
        dup = gf

        edges, f, gf = xp.lazy([edges, f, gf])

        # step 1: stochastic hooking
        gf_row = xp.expand_dims(gf, 0)
        neighbor_gp = xp.where(edges != 0, gf_row, n)
        mngf = xp.min(neighbor_gp, axis=1)

        # step 2: aggressive hooking
        f = xp.minimum(f, mngf)

        # step 3: shortcutting
        f = xp.minimum(f, gf)

        # step 4: calculate grandparents
        gf = xp.take(f, f)

        dup, gf = xp.compute([dup, gf])

        # step 5: check termination
        stop = xp.all(dup == gf)

        f, gf, stop = xp.compute([f, gf, stop])

        if stop:
            break

    return xp.to_benchmark(f)
