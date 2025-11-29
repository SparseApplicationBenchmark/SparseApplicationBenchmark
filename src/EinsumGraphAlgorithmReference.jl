"""
Name: Graph Algorithms using Finch Einsums
Author: Joel Mathew Cherian
Email: jcherian32@gatech.edu
Reference: Kepner, Jeremy, and John Gilbert, eds. Graph algorithms in the language of linear algebra

This file contains the implementation (and basic tests) for the following algorithms
    1. BFS
    2. Bellman Ford
    3. Floyd Warshall
    4. Transitive Closure
    5. Strongly Connected Components
    6. Page Rank
    7. Markov Clustering
    8. Vertex Betweeness Centrality
"""

using Finch

#region GraphAlgorithms
function bfs(edges,src)
    """
        Returns level id of vertices in a graph during BFS from
        a given source vertex.
    """

    (n, m) = size(edges)
    @assert n == m

    G = Tensor(Dense(SparseList(Element(false))), edges)

    visited = Tensor(SparseByteMap(Pattern()), n)
    frontier = Tensor(SparseByteMap(Pattern()), n)
    @finch frontier[src] = true

    level_idx = 1
    level = Tensor(Dense(Element(0)), n)

    fnz_count = Scalar(1)
    while fnz_count[] > 0
        @einsum level[i] = level[i] | (frontier[i] * level_idx)
        @einsum visited[i] = (visited[i] | frontier[i])
        @einsum frontier[j] <<choose(false)>>= frontier[i] * G[i,j] * !visited[j]

        @einsum fnz_count[] += frontier[i]
        level_idx += 1
    end

    return level
end

function bellman_ford(edges,src)
    """
        Returns single source shortest path i.e. for src vertex, A[i] is
        the shortest from src to i.
    """

    (n, m) = size(edges)
    @assert n == m

    G = Tensor(Dense(SparseList(Element(Inf))), edges)
    D = Tensor(Dense(Element(Inf)), n)
    @finch D[src] = 0

    for _ in 1:n
        @einsum D[i] <<min>>= D[j] + G[j,i]
    end

    return D
end

function floyd_warshall(edges)
    """
        Returns the all pair shortest path i.e. A[i,j] is the shortest
        path from i to j
    """

    (n, m) = size(edges)
    @assert n == m

    G = Tensor(Dense(SparseList(Element(Inf))), edges)

    for _ in 1:n
        @einsum G[i,j] <<min>>= G[i,k] + G[k,j]
    end

    return G
end

function transitive_closure(edges)
    """
        Returns an adjacency matrix where A[i,j] = 1 indicates that
        vertex j is reachable from vertex i.
    """

    (n, m) = size(edges)
    @assert n == m

    G = Tensor(Dense(SparseByteMap(Element(false))), edges)
    @finch for i = _
        G[i,i] |= true
    end

    while true
        @einsum G_next[i,j] |= G[i,k] & G[k,j]

        if (G == G_next) break end
        G = G_next
    end

    return G
end

function scc(edges)
    """
        Returns an adjacency matrix where for a given row k, all j's with
        A[k,j] = 1 are a part of the same strongly connected component.
    """

    (n, m) = size(edges)
    @assert n == m

    G = Tensor(Dense(SparseByteMap(Element(false))), edges)
    @finch for i = _
        G[i,i] |= true
    end

    while true
        @einsum G_next[i,j] |= G[i,k] & G[k,j]

        if (G == G_next) break end
        G = G_next
    end

    @einsum result[i,j] = G[i,j] & G[j,i]
    return result
end

function page_rank(edges, alpha, conv_thres)
    """
        Returns a list of page ranks for each vertex in the graph.
        Alpha in the parameter list is the damping factor and conv_thres
        is a threshold below which we can say that the page rank algorithm has converged.
    """
    (n, m) = size(edges)
    @assert n == m

    G = Tensor(Dense(SparseList(Element(0))), edges)
    pr = Tensor(Dense(Element(1)), n)

    @einsum N_weight[i] += G[i,j]
    @einsum transition_mat[i,j] = (N_weight[i] != 0) * (G[i,j] / N_weight[i])

    while true
        @einsum update[j] += pr[i] * transition_mat[i,j]
        @einsum pr_next[i] =  (1-alpha) + (alpha * update[i])

        @einsum magnitude[] += (pr_next[i] - pr[i])^2
        if (magnitude[] ^ (1/2)) <= conv_thres
            break
        end

        pr, pr_next = pr_next, pr
    end

    return pr
end

function markov_clustering(edges, e, r, conv_thres)
    """
       Returns an adjacency matrix where the A[i,j] = 1 entries identify unique clusters.
       e is the expansion parameter and r the inflation parameter.
       conv_thres is a threshold below which we can say that the algorithm has converged.
    """
    (n, m) = size(edges)
    @assert n == m

    C_i = Tensor(Dense(SparseList(Element(0))), edges)
    C_f = Tensor(Dense(SparseList(Element(0))), edges)

    while true
        # expansion
        for _ in 1:(e-1)
            @einsum C_f[i,j] += C_f[i,k] * C_i[k,j]
        end

        # inflation
        @einsum C_f[i,j] = C_f[i,j] ^ r

        # normalization
        @einsum w[j] += C_f[i,j]
        @einsum C_f[i,j] = C_f[i,j] / w[j]

        # convergence check
        @einsum magnitude[] += (C_f[i,j] - C_i[i,j])^2
        if (magnitude[] ^ (1/2)) <= conv_thres
            break
        end

        @einsum C_i[i,j] = C_f[i,j]
    end

    return C_f
end

function vertex_betweeness_centrality(edges)
    """
        Implementation of Brandes algorithm. Returns a list of vertex betweeness
        centralities for each vertex in the graph.
    """
    (n, m) = size(edges)
    @assert n == m

    G = Tensor(Dense(SparseList(Element(0))), edges)
    bc_score = Tensor(Dense(Element(0)), n)
    bc_update = Tensor(Dense(Element(0)), n)

    for v in 1:n
        frontier_stack = []
        num_short_path = Tensor(SparseByteMap(Element(0)), n)
        @finch num_short_path[v] = 1

        frontier = Tensor(SparseByteMap(Element(0)), n)
        @finch for k in 1:n
            frontier[k] = G[v,k]
        end

        d = 0
        @einsum fnz_count[] += frontier[i]
        while fnz_count[] != 0
            d = d + 1
            push!(frontier_stack, frontier)
            @einsum num_short_path[i] = num_short_path[i] + frontier[i]
            @einsum frontier[j] += frontier[i] * G[i,j] * (num_short_path[j] == 0)
            @einsum fnz_count[] += frontier[i]
        end

        bc_update = Tensor(Dense(Element(0)), n)
        while d > 1
            frontier = pop!(frontier_stack)
            @einsum w[i] = (num_short_path[i] != 0) * frontier[i] * ((1+bc_update[i])/num_short_path[i])
            @einsum w[i] += G[i,j] * w[j]

            frontier_prime = frontier_stack[end]
            @einsum w[i] = w[i] * frontier_prime[i] * num_short_path[i]
            @einsum bc_update[i] = bc_update[i] + w[i]
            d = d-1
        end

        @einsum bc_score[i] = bc_score[i] + bc_update[i]
    end

    return bc_score
end

#endregion GraphAlgorithms

#region Tests
function bfs_test()
    input_src = 1
    input_edges =  [
        0 1 1 0 0 0;
        0 0 1 1 0 0;
        0 0 0 0 1 0;
        0 0 0 0 0 0;
        0 0 0 0 0 1;
        0 0 0 1 0 0;
    ]

    expected_out = Tensor(Dense(Element(0)), [1, 2, 2, 3, 3, 4])
    @assert expected_out == bfs(input_edges,input_src)
end

function bellman_ford_test()
    input_edges = [
        0   1   5   Inf Inf Inf;
        Inf 0   3   12  Inf Inf;
        Inf Inf 0   Inf 2   Inf;
        Inf Inf Inf 0   Inf Inf;
        Inf Inf Inf Inf 0   2;
        Inf Inf Inf 2   Inf 0;
    ]
    input_src = 1

    expected_out = Tensor(Dense(Element(0)), [0, 1, 4, 10, 6, 8])
    @assert expected_out == bellman_ford(input_edges,input_src)
end

function floyd_warshall_test()
    input_edges = [
        0   1   5   Inf Inf Inf;
        Inf 0   3   12  Inf Inf;
        Inf Inf 0   Inf 2   Inf;
        Inf Inf Inf 0   Inf Inf;
        Inf Inf Inf Inf 0   2;
        Inf Inf Inf 2   Inf 0;
    ]

    expected_out = Tensor(
        Dense(Dense(Element(Inf))),
        [
            0   1   4   10  6   8;
            Inf 0   3   9   5   7;
            Inf Inf 0   6   2   4;
            Inf Inf Inf 0   Inf Inf;
            Inf Inf Inf 4   0   2;
            Inf Inf Inf 2   Inf 0;
        ]
    )
    @assert expected_out == floyd_warshall(input_edges)
end

function transitive_closure_test()
    input_edges = [
        0 1 1 0 0 0;
        0 0 1 1 0 0;
        0 0 0 0 1 0;
        0 0 0 0 0 0;
        0 0 0 0 0 1;
        0 0 0 1 0 0;
    ]

    expected_out = Tensor(
        Dense(Dense(Element(false))),
        [
            1 1 1 1 1 1;
            0 1 1 1 1 1;
            0 0 1 1 1 1;
            0 0 0 1 0 0;
            0 0 0 1 1 1;
            0 0 0 1 0 1;
        ]
    )
    @assert expected_out == transitive_closure(input_edges)
end

function scc_test()
    input_edges = [
        0 1 0 0 0 0 0 0;
        0 0 1 0 1 1 0 0;
        0 0 0 1 0 0 1 0;
        0 0 1 0 0 0 0 1;
        1 0 0 0 0 1 0 0;
        0 0 0 0 0 0 1 0;
        0 0 0 0 0 1 0 1;
        0 0 0 0 0 0 0 1
    ]
    expected_out = 4

    res = scc(input_edges)

    # Counting the number of strongly connected components
    processed_dict = Dict{Int, Bool}()
    (n,_) = size(input_edges)
    num_scc = 0
    for i = 1:n
        new_scc = false
        for j = 1:n
            if res[i,j] == true && !haskey(processed_dict,j)
                new_scc = true
                processed_dict[j] = true
            end
        end
        num_scc += new_scc
    end

    @assert expected_out == num_scc
end

function page_rank_test()
    input_edges = [
        0 1 1 0;
        0 0 1 0;
        1 0 0 0;
        0 0 1 0
    ]

    expected_res = Tensor(Dense(Element(0.0)), [1.4901077212, 0.7832957815, 1.5765964972, 0.1500000000])

    conv_thres = 1e-6
    res = page_rank(input_edges,0.85,conv_thres)

    @einsum magnitude[] += (res[i] - expected_res[i])^2
    print(res)
    print(magnitude[])
    @assert (magnitude[] ^ (1/2)) <= conv_thres
end

function markov_cluster_test()
    input_edges = [
        0 1 1 1 0 0 0 0
        1 0 0 0 0 0 0 0
        1 0 0 0 0 0 0 0
        1 0 0 0 0 0 0 0
        0 0 0 0 0 1 1 1
        0 0 0 0 1 0 0 0
        0 0 0 0 1 0 0 0
        0 0 0 0 1 0 0 0
    ]
    expected_out = 2

    res = markov_clustering(input_edges,2,2,1e-6)
    # Counting the number of strongly connected components
    processed_dict = Dict{Int, Bool}()
    (n,_) = size(input_edges)
    num_cluster = 0
    for i = 1:n
        new_cluster = false
        for j = 1:n
            if res[i,j] == 1 && !haskey(processed_dict,j)
                new_cluster = true
                processed_dict[j] = true
            end
        end
        num_cluster += new_cluster
    end

    @assert num_cluster == expected_out
end

function vertex_betweeness_centrality_test()
    input_edges = [
        0 1 1 0 0;
        0 0 0 1 0;
        0 0 0 1 0;
        0 0 0 0 1;
        0 0 0 0 0
    ]

    res = vertex_betweeness_centrality(input_edges)
    expected_res = Tensor(
        Dense(Element(0)),
        [0.0, 5.0, 5.0, 6.0, 0.0]
    )

    @assert expected_res == res
end
#endregion Tests
