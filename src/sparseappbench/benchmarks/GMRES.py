"""
Name: GMRES (Generalized Minimal Residual Method)

Author: Aadharsh Rajkumar

Email: arajkumar34@gatech.edu

What does this code do:
This code is implements the GMRES algorithm for solving indefinite
and non-symmetric linear systems. The algorithm follows the Arnoldi
iteration process where a Krylov matrix is maintained at each iteration.
Starting with an initial guess and the residual for that guess, the matrix
A is dot producted with the previous residual to obtain the next basis vector.
This algorithm also uses a similar method to Gram-Schmidt to ensure that
the Kyrlov matrix is orthogonal. I also maintain an upper Hessenberg matrix
which keeps track of the dot products between different basis vectors and
the norm of the new basis vector. The Hessenberg matrix follows the property:
Q_n * A = Q_(n+1) * H_n where Q is the Krylov matrix. This matrix allows
for a simplified least squares problem to be solved at each iteration so that
the residual is minimized at each step. My implementation restarts the Kyrlov
matrix every 50 iterations and will end when the current residual / initial residual
is less than the tolerance level.

Citation for reference implementation:
https://github.com/SparseApplicationBenchmark/SparseApplicationBenchmark/pull/45/files#diff-ba19ac630cf7b27852173e387c91d502f769e88858efa82bda51b1d1e8861a59

Motivation: GMRES is the most widely used and effective method for solving linear
systems that are indefinite, non-symmetric, and are sparse in nature.
https://www.netlib.org/templates/templates.pdf
https://www.netlib.org/utk/people/JackDongarra/PAPERS/sparse-bench.pdf

Statement on the use of Generative AI: No generative AI was used to construct
the benchmark function. This statement is written by hand.
"""


def gmres(
    xp, A_binsparse, b_binsparse, x0_binsparse, restart=50, tol=1e-8, max_iter=1000
):
    A = xp.lazy(xp.from_benchmark(A_binsparse))
    b = xp.lazy(xp.from_benchmark(b_binsparse))
    x0 = xp.lazy(xp.from_benchmark(x0_binsparse))

    itcount = 0
    r0 = b - A @ x0
    beta = xp.compute(xp.linalg.norm(r0))[()]
    rcurr = r0 / beta

    while itcount < max_iter:
        Q = xp.zeros((A.shape[0], restart + 1), dtype=float)
        H = xp.zeros((restart + 1, restart), dtype=float)
        Q[:, 0] = rcurr

        for i in range(restart):
            x0 = xp.lazy(x0)
            rcurr = xp.lazy(rcurr)

            rcurr = A @ Q[:, i]

            # Orthogonalization process without extra loop.
            H[: i + 1, i] = xp.compute(xp.vecdot(Q[:, : i + 1].T, rcurr))
            rcurr = rcurr - Q[:, : i + 1] @ H[: i + 1, i]
            H[i + 1, i] = xp.compute(xp.linalg.norm(rcurr))[()]
            Q[:, i + 1] = rcurr / H[i + 1, i]

            # Orthogonalization with looping. Unsure if my above method works.
            # for j in range(i + 1):
            #     H[j, i] = xp.compute(xp.vecdot(Q[:, j], rcurr))
            #     rcurr = rcurr - H[j, i] * Q[:, j]

            # H[i + 1, i] = xp.compute(xp.linalg.norm(rcurr))[()]
            # Q[:, i + 1] = rcurr / H[i + 1, i]

            e1 = xp.zeros((i + 2,), dtype=float)
            e1[0] = beta

            H_reduced = H[: i + 2, : i + 1]
            coeffs, _, _, _ = xp.linalg.lstsq(H_reduced, e1, rcond=None)
            x0 = x0 + Q[:, : i + 1] @ coeffs

            r0 = b - A @ x0
            r0_norm = xp.compute(xp.linalg.norm(r0))[()]
            rcurr = r0 / r0_norm
            if r0_norm / beta < tol:
                return xp.to_benchmark(xp.compute(x0))

            itcount += 1
            if itcount >= max_iter:
                break

    xsol = xp.compute(x0)
    return xp.to_benchmark(xsol)
