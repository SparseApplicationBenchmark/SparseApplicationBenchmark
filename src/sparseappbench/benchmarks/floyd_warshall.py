"""
Name: Floyd-Warshall algorithm
Co-Authors: Aarav Joglekar, Joel Mathew Cherian
Email: ajoglekar32@gatech.edu

Motivation:
The Floyd-Warshall algorithm computes the shortest paths between every pair of vertices
in a weighted directed graph.

Role of sparsity:
Sparse graphs reduce unnecessary computation, as most entries in the adjacency
matrix represent non-edges and begin as inifinity. Efficient sparse representations
allow the backend framework to skip work and minimize memory movement during the
relaxation steps of the algorithm.

Implementation Reference:
J. Kepner and J. Gilbert (eds.), “Graph Algorithms in the Language of Linear Algebra,”
Society for Industrial and Applied Mathematics (SIAM), Philadelphia, 2011.

Data Sources Used for Testing:
Some unit tests use real-world networks, including the Chesapeake road network
and soc-tribes network.
from the Network Repository:
    @inproceedings{nr,
        title={The Network Data Repository with Interactive Graph Analytics
        and Visualization},
        author={Ryan A. Rossi and Nesreen K. Ahmed},
        booktitle={AAAI},
        url={https://networkrepository.com},
        year={2015}
    }

Data Generation:
Graphs for this benchmark may be created manually for testing or generated
procedurally using a Graph500-style R-MAT model. Generated adjacency matrices
are converted into the benchmark format with BinsparseFormat.from_numpy().

Statement on the use of Generative AI: No generative AI was used to construct
the benchmark function itself. Generative AI might have been used to construct
tests. This statement was written by hand.
"""


def floyd_warshall(xp, edges_binsparse):
    """
    Returns the all pair shortest path i.e. A[i,j] is the shortest
    path from i to j
    """
    edges = xp.from_benchmark(edges_binsparse)
    edges = xp.lazy(edges)

    n, m = edges.shape
    assert n == m

    G = xp.array(edges, dtype=float)

    for _ in range(n):
        G = xp.lazy(G)
        G = xp.einsum("G[i,j] min= G[i,k] + G[k,j]", G=G)
        G = xp.compute(G)

    return xp.to_benchmark(G)
