# Bellman Ford Algorithm:

# Given a weighted graph with V vertices and E edges and a starting vertex src, 
# Bellman-Ford algoirthm computes the shortest distance from src to all V. 
# If the vertext is unreachable, then infinite distance. 
# If there is a negative weight cycle, then we return -1 since shortest path 
# calculations are not feasible

# TO DO:
# Add Lazy and Compute
# Generate the test cases

import numpy as np
import scipy.sparse as sp

from sparseappbench.binsparse_format import BinsparseFormat

xp = np

def bellman_ford (edges, src): 
    #assume that edges is a square adjacency matrix of edge weights 
    # and src is the index of the starting node
    # check if edges is a valid adjacency matrix
    n = len(edges)
    m = len(edges[0])
    if n!=m:
        return ("This is not a valid adjacency matrix")
    
    G = xp.asarray(edges, dtype=float)
    D = xp.full((n,), xp.inf) # distance vectors initialized to infinity
    D[src] = 0

        # Relax n times (Bellman–Ford)
    for _ in range(n):
        # candidates[j, i] = D[j] + G[j, i]
        candidates = D[:, None] + G  # shape (n, n)

        # For each i: D[i] = min(D[i], min_j candidates[j, i])
        D = xp.minimum(D, candidates.min(axis=0))

    return D # running this using vectorized algebra instead of loops
    # expected return output is an array of shortest-path distances


