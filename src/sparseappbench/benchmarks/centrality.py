"""
Name: Betweenness Centrality

Author: Aadharsh Rajkumar

Email: arajkumar34@gatech.edu

What does this code do:
This code is based on the Brandes betweennness centrality algorithm. The
current code for the benchmark takes a two step approach. The first step
involves going layer by layer from each potential starting node to find
the total amount of shortest paths that lead to a node. So for example
4 -> 6 could have 3 diff shortest paths and 4 -> 2 could have only 1
shortest path. The second step is for tracing backwards to see how many
times a node appears in other shortest paths. The number of times this
node is in one of the shortest path divided by total shortest paths between
the two edge nodes gets added to the intermediate nodes bc score. This code
performs lazy calculations before computing at the end of iteration blocks.

Citation for reference implementation:
https://github.com/SparseApplicationBenchmark/SparseApplicationBenchmark/pull/48/files

Citation for importance of the problem:
Matta, J., Ercal, G. & Sinha, K. Comparing the speed and accuracy of
approaches to betweenness centrality approximation.
Comput Soc Netw 6, 2 (2019). https://doi.org/10.1186/s40649-019-0062-5

Statement on the use of Generative AI: No generative AI was used to construct
the benchmark function. This statement is written by hand.
"""


def betweenness_centrality(xp, A_binsparse):
    G = xp.lazy(xp.from_benchmark(A_binsparse))
    n = G.shape[0]
    bc_scores = xp.zeros(n, dtype=float)

    for s in range(n):
        stack = xp.zeros((0,), dtype=int)
        prev = xp.zeros((n, n), dtype=float)

        number_of_paths = xp.zeros(n, dtype=float)
        number_of_paths = number_of_paths + (xp.arange(n) == s).astype(float)

        dist = -xp.ones(n, dtype=int)
        dist = dist + (xp.arange(n) == s).astype(int)

        queue = xp.array([s], dtype=int)

        while len(queue) > 0:
            curr_q = int(queue[0])
            queue = queue[1:]
            stack = xp.concatenate([stack, xp.array([curr_q], dtype=int)])
            row = xp.compute(G[curr_q])
            neighbors = row != 0

            if xp.any(neighbors):
                not_visited = neighbors & (dist == -1)
                if xp.any(not_visited):
                    dist_val = int(dist[curr_q])
                    dist = xp.where(not_visited, dist_val + 1, dist)

                    queue_index = xp.nonzero(not_visited)[0]
                    queue = xp.concatenate([queue, queue_index.astype(int)])

                target_val = neighbors & (dist == (int(dist[curr_q]) + 1))

                if xp.any(target_val):
                    update_val = float(number_of_paths[curr_q])
                    number_of_paths = (
                        number_of_paths + target_val.astype(float) * update_val
                    )

                    col_update = (xp.arange(n) == curr_q).astype(float)
                    prev = prev + xp.outer(target_val.astype(float), col_update)

        total = xp.zeros(n, dtype=float)

        while len(stack) > 0:
            curr_s = int(stack[-1])
            stack = stack[:-1]
            node_val = xp.maximum(number_of_paths[curr_s], 1e-10)

            previous = prev[curr_s, :] != 0

            if xp.any(previous):
                scale_factor = (1.0 + total[curr_s]) / node_val
                path_counter = previous.astype(float) * (number_of_paths * scale_factor)
                total = total + path_counter

            if curr_s != s:
                bc_scores = (
                    bc_scores + (xp.arange(n) == curr_s).astype(float) * total[curr_s]
                )

    return xp.to_benchmark(bc_scores)
