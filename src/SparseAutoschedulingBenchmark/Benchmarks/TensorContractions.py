import numpy as np
import scipy.sparse as sp
from ..BinsparseFormat import BinsparseFormat

#CURRENT TO-DO:
# 1. Figure out BinsparseFormat file importing 
# 2. Figure out why lazy compute is not working 

"""
Personal / Context Notes:

Tensor Networks are multi-dimensional arrays like a matrix is 2 indices, a tensor is 3 indices, and a tensor network connects tensors by "contracting" shared indices
They can be contracted if theyw share an index i.e a dimension in the same label : we can sum over them 
In 2D, this is matrix multiplication : C(ik) = summation(j) A(ij) B(jk)

In Einstein Summation Notation, this can be as follows:
C(ab) = Summ(i,j) A(aij)*B(jib) that can be represented as C = xp.einsum('aij,jib->ab', A, B)

The ITensor Library (https://www.scipost.org/SciPostPhysCodeb.4/pdf) where it is also represented as D + A * B * C 
More terminology:
Outper Product : if no indices are shared, then dimensions are stacked : D(ij), (kl) = A(ij) * B(kl)
Scalar or Inner Product : In essence a dot product i.e. a dot b = Summ (i) a(i) * b(i). It is the simplest possible tensor contraction 

Matrix Product State: applied in quantum physics where each state can be 2**n entries or c(i1 i2 i3.. iN) = Summ (a1 a2 a3... a(N-1)) = A[1] (i1, a1) * A[2] (a1 i2, a2)
Each A[K] represents a small 3D tensory (2D at the edges) and the bonds ak are internal connections
MPS tensor has A(i) [s(i), a(i), a(i+1)] instead of storing one huge tensor, it is a chain of smaller tensors 

Matrix Product Operation: Represents huge operator that is applied to contract the MPS

Singular Vector Decomposition: For any matrix M = USV where is U is left singular vectors, V is right singular vectors and S is the diagonal matrix of singilar values 
Can truncate singular small values to compress the data - keeps MPS efficient

Single MPS is compressed by applying an MPO to each site, which performs SVD compressions with a given truncation rank and returns the new MPS 
This uses einsum, but can also be done with tensordot for higher-order tensors
"""

"""
Name: Matrix Product Operator (MPO) - Matrix Product State (MPS) Contraction
Author: Ilisha Gupta

Motivation:
MPS and MPO are central to tensor-network algorithms like DMRG and TEBD (https://itensor.org/). This benchmark measures performance of contracting an MPO with an MPS. 

Role of Sparsity:
Tensor dimensions scale polynomially with dimension, making contraction structure (index ordering, einsum) critical

Implementation: 
This benchmark performs sequential MPO-MPS contractions using the array-api functions only, enabling fair comparison across libraries

Data Generation:
Synthetic random tensors generated with fixed seed to ensure reproducibility. 

Statement on Generative AI:
No generative AI has been used throughout this project
"""

def generate_mpo_mps_data(num_sites=3, phys_dim=2, bond_dim=3, xp=np):
    """
    Generate random MPS (Matrix Product State) and MPO (Matrix Product Operator) tensors.
    Parameters
    1. num_sites : int - Number of lattice sites.
    2. phys_dim : int - Physical dimension per site.
    bond_dim : int - Bond dimension.

    Returns
    1. mpo_bin, mps_bin : tuple - Serialized benchmark data 
    """

    rng = xp.random.default_rng(0)
    mpo = [
        rng.standard_normal((phys_dim, phys_dim, bond_dim, bond_dim))
        for _ in range(num_sites)
    ]
    mps = [
        rng.standard_normal((phys_dim, bond_dim, bond_dim))
        for _ in range(num_sites)
    ]

    mpo_bin = [BinsparseFormat.from_numpy(W) for W in mpo]
    mps_bin = [BinsparseFormat.from_numpy(A) for A in mps]
    return mpo_bin, mps_bin


def contract_mpo_mps(xp, mpo, mps):
    """
    Perform MPO-MPS contraction:
        B[s', a*w_in, b*w_out] = Summ_s A[s, a, b] * W[s, s', w_in, w_out]

    Parameters
    1. mpo : list of xp.ndarray - MPO tensors of shape (s_in, s_out, w_in, w_out).
    2. mps : list of xp.ndarray - MPS tensors of shape (s, a, b).

    Returns
    1. new_mps : list of xp.ndarray - New MPS tensors after contraction.
    """
    new_mps = []
    for i in range(len(mpo)):
        A = mps[i]
        W = mpo[i]

        B = xp.einsum("sab,ss'ww'->s'aww'b", A, W)
        s_out, a, w_in, w_out, b = B.shape
        B = xp.reshape(B, (s_out, a * w_in, b * w_out))
        new_mps.append(B)
    return new_mps

def print_mpo_mps_summary(mpo, mps):
    print(f"Number of sites: {len(mpo)}")
    print("\nMPO tensors:")
    for i, W in enumerate(mpo):
        print(f"  Site {i}: shape {W.shape}, dtype={W.dtype}")

    print("\nMPS tensors:")
    for i, A in enumerate(mps):
        print(f"  Site {i}: shape {A.shape}, dtype={A.dtype}")

def benchmark_mpo_mps(xp, mpo_bench, mps_bench, num_sites=10):
    """
    Parameters
    1. mpo_bench, mps_bench : list - Benchmark-format MPO and MPS data.
    2. num_sites : int - Number of tensor sites.

    Returns
    1. result_bin : benchmark-format MPS tensors.
    """
    mpo = [xp.lazy(xp.from_benchmark(W)) for W in mpo_bench]
    mps = [xp.lazy(xp.from_benchmark(A)) for A in mps_bench]
    print_mpo_mps_summary(mpo, mps)

    new_mps = contract_mpo_mps(xp, mpo, mps)
    new_mps_eval = [xp.lazy(xp.compute(B)) for B in new_mps]
    result_bin = [BinsparseFormat.from_numpy(B) for B in new_mps_eval]
    return result_bin

def dg_mpo_mps_small():
    return generate_mpo_mps_data(num_sites=2, phys_dim=2, bond_dim=2)
mpo, mps = dg_mpo_mps_small()
print(benchmark_mpo_mps(np, mpo, mps)) 