import numpy as np
import argparse

def setup_cell(L, M, N,lattice=np.sqrt(2)*2**(1/6)):

    """
    Generate a computational cell for an FCC lattice of size LxMxM periodic 
    coppies. 

    Parameters
    ----------
    L : int
        Periodic repeats in direction L.
    M : int
        Periodic repeats in direction M.
    N:  int
        Periodic repeats in direction N. 


    Returns
    -------
    atom_positions : ndarray, shape (num_atoms, 3)
        The positions of all atoms in the computational cell.
    lattice_vectors : ndarray, shape (3, 3)
        The lattice vectors defining the computational cell.
    
    """
    # Define the primitive FCC basis (4 atoms per unit cell)
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5]
    ])
   
    l_vals = np.arange(L)
    m_vals = np.arange(M)
    n_vals = np.arange(N)
    
    # Create 3D grids of indices for periodic displacements in L, M, and N directions.
    # 'ij' indexing ensures row-major (matrix-style) order.
    l_grid, m_grid, n_grid = np.meshgrid(l_vals, m_vals, n_vals, indexing='ij')

    # Combine L, M, N grid arrays into a single array of displacement vectors.
    # Reshape into a 2D array where each row is a 3D vector (L, M, N).
    cell_positions = np.stack([l_grid, m_grid, n_grid], axis=-1).reshape(-1, 3, order='C')

    # Add FCC basis vectors to each displacement vector using broadcasting.
    # This generates all atom positions in fractional coordinates.
    atom_positions = cell_positions[:, np.newaxis, :] + basis[np.newaxis, :, :]
    
    # Flatten the array to a 2D array where each row is a 3D atom position.
    atom_positions = atom_positions.reshape(-1, 3)

    # Normalize atom positions by the number of repeats (convert to fractional coordinates)
    atom_positions /= np.array([L, M, N])

    # Define the lattice vectors of the computational cell
    latvec = np.diag([L, M, N]) * lattice

    return atom_positions@latvec,latvec
    
if __name__ == "__main__":
    # Set up command-line argument parsing (its so lovely!!!)
    parser = argparse.ArgumentParser(description="Generate an FCC computational cell.")
    parser.add_argument("--L", type=int, default=2, help="Periodic repeats in the L direction (default: 2)")
    parser.add_argument("--M", type=int, default=2, help="Periodic repeats in the M direction (default: 2)")
    parser.add_argument("--N", type=int, default=2, help="Periodic repeats in the N direction (default: 2)")
    args = parser.parse_args()

    # Call the setup_cell function with arguments from the command line (I love argparse so much)
    atom_positions, lattice_vectors = setup_cell(args.L, args.M, args.N)

    # Printing printing printing
    print("Atom Positions:\n", atom_positions)
    print("\nLattice Vectors:\n", lattice_vectors)
