import numpy as np
from scipy.linalg import eigh
import argparse

def build_2d_hamiltonian(N=20, potential='well', boundary=None):
    """
    Build a discretized 2D Hamiltonian on an N x N grid.

    Parameters
    ----------
    N : int
        Number of points in each dimension (N^2 total points).
    potential : str
        Choose the potential. 'well' or 'harmonic' examples.

    Returns
    -------
    H : ndarray of shape (N^2, N^2)
        The Hamiltonian matrix approximating -d^2/dx^2 - d^2/dy^2 + V(x,y).
    """
    dx = 1. / float(N)  # grid spacing, can be arbitrary
    inv_dx2 = float(N * N)  # 1/dx^2
    H = np.zeros((N*N, N*N), dtype=np.float64)

    # Helper function to map (i,j) -> linear index
    def idx(i, j):
        return i * N + j

    # Potential function
    def V(i, j):
        # Example 1: infinite square well -> zero in interior, large outside
        if potential == 'well':
            # No boundary enforcement here, but can skip boundary wavefunction
            return 0.
        # Example 2: 2D harmonic oscillator around center
        elif potential == 'harmonic':
            x = (i - N/2) * dx
            y = (j - N/2) * dx
            # Quadratic potential V = k * (x^2 + y^2)
            return 4. * (x**2 + y**2)
        else:
            return 0.

    # Build the matrix: For each (i,j), set diagonal for 2D Laplacian plus V
    for i in range(N):
        for j in range(N):
            row = idx(i,j)
            # Potential
            H[row, row] = 4. * inv_dx2 + V(i,j)  # Kinetic ~ -4/dx^2 in 2D FD
            # Neighbors (generalized boundary conditions)
            if i > 0:
                H[row, idx(i-1, j)] = inv_dx2
            else:
                if boundary is not None:
                    H[row, row] += inv_dx2 * boundary(i * dx, j * dx)
            if i < N-1:
                H[row, idx(i+1, j)] = inv_dx2
            else:
                if boundary is not None:
                    H[row, row] += inv_dx2 * boundary(i * dx, j * dx)
            if j > 0:
                H[row, idx(i, j-1)] = inv_dx2
            else:
                if boundary is not None:
                    H[row, row] += inv_dx2 * boundary(i * dx, j * dx)
            if j < N-1:
                H[row, idx(i, j+1)] = inv_dx2
            else:
                if boundary is not None:
                    H[row, row] += inv_dx2 * boundary(i * dx, j * dx)

    return H

def solve_eigen(N=20, potential='well', n_eigs=None):
    """
    Build a 2D Hamiltonian and solve for the lowest n_eigs eigenvalues.

    Parameters
    ----------
    N : int
        Grid points in each dimension.
    potential : str
        Potential type.
    n_eigs : int
        Number of eigenvalues to return.

    Returns
    -------
    vals : array_like
        The lowest n_eigs eigenvalues sorted ascending.
    vecs : array_like
        The corresponding eigenvectors.
    """
    H = build_2d_hamiltonian(N, potential)
    # Solve entire spectrum (careful for large N)
    vals, vecs = eigh(H)
    # Sort
    idx_sorted = np.argsort(vals)
    vals_sorted = vals[idx_sorted]
    vecs_sorted = vecs[:, idx_sorted]
    if n_eigs is None:
        return vals_sorted, vecs_sorted
    else:
        return vals_sorted[:n_eigs], vecs_sorted[:, :n_eigs]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2D quantum eigenvalue solver.')
    parser.add_argument('--N', type=int, default=10,
                        help='Grid points per dimension (positive integer)')
    parser.add_argument('--potential', type=str, default='well',
                        choices=['well', 'harmonic'],
                        help='Potential type')
    parser.add_argument('--n_eigs', type=int, default=5,
                        help='Number of eigenvalues to return (positive integer, <= N^2)')
    parser.add_argument('--save_density', action='store_true',
                        help='Save ground-state probability density to file')
    args = parser.parse_args()

    # Sanity checks
    if args.N < 1:
        raise ValueError(f'N must be a positive integer, got {args.N}')
    if args.n_eigs < 1 or args.n_eigs > args.N ** 2:
        raise ValueError(f'n_eigs must be between 1 and N^2={args.N**2}, got {args.n_eigs}')

    vals, vecs = solve_eigen(N=args.N, potential=args.potential, n_eigs=args.n_eigs)
    print("Lowest", args.n_eigs, "eigenvalues:", vals)
    np.savetxt(f'eigs_N{args.N}.txt', vals)
    if args.save_density:
        psi = vecs[:, 0].reshape(args.N, args.N)
        density = np.abs(psi) ** 2
        np.savetxt(f'density_N{args.N}.txt', density)
        print(f"Saved density to density_N{args.N}.txt")