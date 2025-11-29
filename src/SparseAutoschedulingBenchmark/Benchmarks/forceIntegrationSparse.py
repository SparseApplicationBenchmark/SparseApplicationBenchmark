"""
Name: Force Integration Sparse (port from Cyclops)
Author: Vitaly V. Marin
Email: vmarin3@gatech.edu
Motivation: It is a part of Cyclops library implemented in C++ that included in examples. It relevant to research community.

Implementation:

The original code was converted from C++ to Python.

The following Implementation was used: https://solomon2.web.engr.illinois.edu/ctf/group__force__integration__sparse.html

Algorithm is based on: https://github.com/cyclops-community/ctf/blob/master/examples/force_integration_sparse.cxx

Reference implementations used and translated: N/A

Data Generation: yes.

License Type: BSD

version: 0.1

Date: 10/24/2025

TODO Next: integrate fully into https://github.com/SparseAutoschedulingBenchmark/SparseAutoschedulingBenchmark
similar to
https://github.com/SparseAutoschedulingBenchmark/SparseAutoschedulingBenchmark/pull/31/files benchmark

"""


import numpy as np

def get_distance(p1_dx, p1_dy, p2_dx, p2_dy):
    """Compute Euclidean distance between two particles."""
    return np.sqrt((p1_dx - p2_dx)**2 + (p1_dy - p2_dy)**2)

def get_force(p1_dx, p1_dy, p2_dx, p2_dy, p1_coeff, p2_coeff, distance):
    """
    Compute force vector between two particles (directional force along x and y).
    Returns (fx, fy) for the force on p1 due to p2.
    """
    # Avoid division by zero
    if distance < 1e-10:
        return 0.0, 0.0
    # Force magnitude: inverse-square law scaled by coefficients
    force_mag = p1_coeff * p2_coeff / (distance**2 + 1e-10)
    # Directional components: force points along vector from p2 to p1
    dx = p1_dx - p2_dx
    dy = p1_dy - p2_dy
    fx = force_mag * dx / distance
    fy = force_mag * dy / distance
    return fx, fy

def force_integration_sparse_einsum(n):
    """
    Perform sparse force integration for n particles, updating their positions based on forces.
    Uses np.einsum for sparse matrix-vector multiplication.

    Parameters:
    n (int): Number of particles.

    Returns:
    bool: True if test passes (positions change and revert correctly), False otherwise.
    """
    # Initialize particles
    np.random.seed(42)  # For reproducibility
    particles = np.zeros(n, dtype=[('dx', np.float64), ('dy', np.float64), ('coeff', np.float64)])
    particles['dx'] = np.random.rand(n)
    particles['dy'] = np.random.rand(n)
    particles['dx'][0] = 0.5  # First particle in middle
    particles['dy'][0] = 0.5
    particles['coeff'] = np.random.rand(n)  # Increased magnitude for significant force updates

    # Save original particles for testing
    particles_orig = particles.copy()

    # Construct sparse force matrix F (COO format) with vector forces
    rows = []
    cols = []
    vals_x = []  # Force components along x
    vals_y = []  # Force components along y
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = get_distance(particles['dx'][i], particles['dy'][i],
                                  particles['dx'][j], particles['dy'][j])
                if dist < 0.708:
                    fx, fy = get_force(particles['dx'][i], particles['dy'][i],
                                     particles['dx'][j], particles['dy'][j],
                                     particles['coeff'][i], particles['coeff'][j], dist)
                    rows.append(i)
                    cols.append(j)
                    vals_x.append(fx)
                    vals_y.append(fy)
                    # Antisymmetric: F[j,i] = -F[i,j]
                    rows.append(j)
                    cols.append(i)
                    vals_x.append(-fx)
                    vals_y.append(-fy)

    rows   = np.array(rows, dtype=np.int32)
    cols   = np.array(cols, dtype=np.int32)
    vals_x = np.array(vals_x, dtype=np.float64)
    vals_y = np.array(vals_y, dtype=np.float64)

    # Compute F2 = 2 * F (mimicking F2["ij"] += F["ij"] twice)
    vals_x_F2 = 2.0 * vals_x
    vals_y_F2 = 2.0 * vals_y

    # Apply force accumulation using einsum: P[i].dx += sum_j F2[i,j] * P[j].coeff
    # Group contributions by row using einsum with row indices
    dx_update = np.zeros(n, dtype=np.float64)
    dy_update = np.zeros(n, dtype=np.float64)

    # Einsum-based sparse matrix-vector multiplication
    # For each unique row i, compute sum over k where rows[k]=i of (vals_x_F2[k] * particles['coeff'][cols[k]])
    unique_rows = np.unique(rows)
    for i in unique_rows:
        mask = (rows == i)
        # Einsum: sum over k of vals_x_F2[k] * coeff[cols[k]] where rows[k] == i
        dx_update[i] = np.einsum('k,k->', vals_x_F2[mask], particles['coeff'][cols[mask]])
        dy_update[i] = np.einsum('k,k->', vals_y_F2[mask], particles['coeff'][cols[mask]])

    # Update particle positions
    particles['dx'] += dx_update
    particles['dy'] += dy_update

    # Check if positions changed
    diff_dx = np.abs(particles['dx'] - particles_orig['dx'])
    diff_dy = np.abs(particles['dy'] - particles_orig['dy'])
    pass1 = np.any((diff_dx > 1e-6) | (diff_dy > 1e-6))
    if not pass1:
        print(f"Pass1 failed: Max dx change = {np.max(diff_dx):.2e}, Max dy change = {np.max(diff_dy):.2e}")

    # Apply inverse forces (F.addinv() makes F[i,j] = -F[i,j])
    vals_x_F_inv = -vals_x
    vals_y_F_inv = -vals_y

    # Einsum-based inverse force accumulation (applied twice)
    dx_update_inv = np.zeros(n, dtype=np.float64)
    dy_update_inv = np.zeros(n, dtype=np.float64)
    for i in unique_rows:
        mask = (rows == i)
        # Each inverse application: sum over k of vals_x_F_inv[k] * coeff[cols[k]]
        inv_contrib_x = np.einsum('k,k->', vals_x_F_inv[mask], particles['coeff'][cols[mask]])
        inv_contrib_y = np.einsum('k,k->', vals_y_F_inv[mask], particles['coeff'][cols[mask]])
        dx_update_inv[i] = 2.0 * inv_contrib_x  # Apply twice
        dy_update_inv[i] = 2.0 * inv_contrib_y  # Apply twice

    # Update particle positions with inverse forces
    particles['dx'] += dx_update_inv
    particles['dy'] += dy_update_inv

    # Check if positions reverted to original
    diff_dx_inv = np.abs(particles['dx'] - particles_orig['dx'])
    diff_dy_inv = np.abs(particles['dy'] - particles_orig['dy'])
    pass2 = not np.any((diff_dx_inv > 1e-6) | (diff_dy_inv > 1e-6))
    if not pass2:
        print(f"Pass2 failed: Max dx change = {np.max(diff_dx_inv):.2e}, Max dy change = {np.max(diff_dy_inv):.2e}")

    # Test passes if both conditions are met
    test_passed = pass1 and pass2

    return test_passed

def test_force_integration_sparse():
    """Test the sparse force integration function using einsum."""
    n = 5  # Small number of particles, as in C++ main
    passed = force_integration_sparse_einsum(n)
    print(f"{{ P[\"i\"] = uacc(F[\"ij\"]) }} {'passed' if passed else 'failed'}")
    return passed

if __name__ == "__main__":
    test_force_integration_sparse()
