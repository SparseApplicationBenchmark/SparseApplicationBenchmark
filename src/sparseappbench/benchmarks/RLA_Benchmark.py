import scipy.sparse as sp
import numpy as np


"""
Name: Random Numerical Linear Algenra
Author: Vilohith Gokarakonda
Email: vgokarakonda3@gatech.edu
Motivation (Importance of problem with citation):
The purpose of this is to create python tests that are for RLA methods.
Specifically, I will first show the application of the JL Lemma for NN.
My goal is to write benchmarks on applications of RNLA,
for graph algorithms, PDEs, and Scientific Machine Learning

Murray, R., Demmel, J., Mahoney, M. W., Erichson, N. B., 
Melnichenko, M., Malik, O. A., ... & Dongarra, J. (2023). 
Randomized numerical linear algebra: A perspective on the field with an eye to software. 
arXiv preprint arXiv:2302.11474.
Role of sparsity (How sparsity is used in the problem):
The inputs to the matrix multiply are sparse.
Implementation (Where did the reference algorithm come from? With citation.):
Hand-written, direct call to array api function
https://data-apis.org/array-api/latest/API_specification/generated/array_api.matmul.html
Data Generation (How is the data generated? Why is it realistic?):
Sparse-sparse matrix multiplication is sensitive to sparsity patterns and their
interaction. We use random sparsity patterns for now.  Statement on the use of
Generative AI: No generative AI was used to construct the benchmark function
itself. Generative AI might have been used to construct tests. This statement
was written by hand.
"""


def benchmark_johnson_lindenstrauss_nn(xp, data_bench, query_bench, k=5, eps=0.1):
    data = xp.lazy(xp.from_benchmark(data_bench))
    query = xp.lazy(xp.from_benchmark(query_bench))

    n_samples, n_features = data.shape
    #  Johnson Lindenstrauss Theorem Lemmna.
    # The eps represents the disortion of distance by epsilon,
    # between the the original space and the reduced subspace
    target_dim = xp.log(n_samples) / (eps * eps)
    if target_dim > n_features:
        target_dim = n_features

    # --- Sparse Random Projection Stuff -------

    # These values are recommened by Ping Li et al.
    s = np.sqrt(n_features)  # s = 1/density
    density = 1.0 / s  # probability of a nonzero entry = density.
    density_half = density / 2.0  # probability for + or -
    scale = np.sqrt(s / target_dim)  # scale = sqrt(s / n_components)

    # total number of entries in matrix
    n_total = n_features * target_dim

    # 1-D matrix so it is easier to do random with.
    # these random entries represent probs that values will either be -1, 1, or 0.

    rng = np.random.default_rng(seed=10)
    U = rng.random(n_total)

    # Indices for negative entries
    neg_checker = (
        U < density_half  # range of [0, density_half)
    )  # since probability that there will be a negative  value is 1/2s
    one_dimen_neg_indices = np.nonzero(neg_checker)[
        0
    ]  # tells which row and column will have -1, but in 1D array
    neg_rows = one_dimen_neg_indices // target_dim  # tells row of -1 val
    neg_cols = one_dimen_neg_indices % target_dim  # tells col of -1 val
    neg_vals = -np.ones(len(one_dimen_neg_indices))  # the actual negative one value

    # in the end, it would be like
    # neg_rows = [3,4],
    # neg_cols = [0,1],
    # neg_vals  = [-1,-1].
    # Mean -1 vals will be at [3,0] and [4,1].

    # Indices for positive entries
    pos_checker = (U >= density_half) & (
        U < density
    )  # range of [density_half, density), still 1/2s probability.
    # Everything else is sense as negative case.
    one_dimen_pos_indices = np.nonzero(pos_checker)[0]
    pos_rows = one_dimen_pos_indices // target_dim
    pos_cols = one_dimen_pos_indices % target_dim
    pos_vals = np.ones(len(one_dimen_pos_indices))

    # Stack basically all the rows and columns from the negative and
    # positives row and col positions and the nonzero values with the pos.
    rows = np.concatenate([neg_rows, pos_rows])
    cols = np.concatenate([neg_cols, pos_cols])
    vals = np.concatenate([neg_vals, pos_vals])

    # Build sparse matrix
    projection_matrix_sparse = sp.csr_matrix(
        (vals, (rows, cols)), shape=(n_features, target_dim)
    )

    # Scales to  np.sqrt(s / target_dim)
    projection_matrix_sparse_scaled = projection_matrix_sparse * scale

    # --- End of Sparse Random Projection Stuff -------

    final_projection_matrix = xp.array(projection_matrix_sparse_scaled.toarray())

    # Project to lower subspace
    projected_data = xp.matmul(data, final_projection_matrix)
    projected_query = xp.matmul(query, final_projection_matrix)

    # -----K Nearest Neighbour from here on out--------

    # Euclidean distances
    diff = projected_data - projected_query
    distances = xp.sqrt(xp.sum(diff**2, axis=1))

    # Get nearest k neighbors
    nearest_indices = xp.argsort(distances)[:k]
    nearest_distances = xp.sort(distances)[:k]

    # Just puts the results in 3 by k matrix. nearest_indices is scalar
    # that associates to sample point i in original space.
    # Distance is in projected subspace.

    result_indices = xp.stack([xp.arange(k), nearest_indices], axis=0)
    result = xp.stack([result_indices, nearest_distances], axis=0)

    result = xp.compute(result)
    return xp.to_benchmark(result)
