import numpy as np 
from utils.get_compcell import setup_cell
import argparse

def lightning_mcqueen(atoms, rcut, latvec):
    """
    Compute the Lennard-Jones (LJ) potential energy and forces for the provided computational 
    cell within a cutoff radius. This version uses the minimum image convention. 

    Parameters
    ----------
    atoms : ndarray, shape (num_atoms, 3)
        Coordinates of all atoms in the computational cell.

    rcut : float
        Cutoff radius for the Lennard-Jones interaction. 

    latvec : ndarray, shape (3, 3)
        Lattice vectors defining the simulation cell. The diagonal elements correspond 
        to the lengths of the cell along the L*a, M*a and N*a where a is a lattice constant.

    Returns
    -------
    total_lj : float
        Total Lennard-Jones potential energy of crystal 

    forces : ndarray, shape (num_atoms, 3)
        Forces on each atom within cutoff radius 

    Notes
        -----
        - **Minimum Image Convention (MIC)**:
            The MIC ensures that only the nearest periodic copy of each atom is considered 
            when computing interactions. This is valid only when the cutoff radius satisfies:
            
        r_cut <= min(lattice_lengths) / 2

            where `lattice_lengths` are the lengths of the cell along each direction (scaled by lattice constant)
            extracted from the diagonal of the `latvec` matrix.

        - If r_cut does not satisfy this condition use other function (here). 

        - The Lennard-Jones potential is given by:
            V(r) = 4 * [(1/r)^12 - (1/r)^6]

        - Forces are derived as:
            F(r) = -dV/dr = 24 * [(2/r^14) - (1/r^8)]

        """

    # Extract lattice lengths
    lattice_lengths = np.diagonal(latvec)

     # Check if rcut satisfies the minimum image condition
    # The minimum image convention is valid only if rcut is less than or equal to half
    # the shortest lattice vector length. If this condition is not met, raise an error!
    min_lattice_length = np.min(lattice_lengths)
    if rcut > min_lattice_length / 2:
        raise ValueError(
            f"Oy vey! Cutoff radius rcut={rcut} exceeds the maximum allowed "
            f"value for the minimum image convention!: {min_lattice_length / 2:.2f}."
        )

    # Precompute constants for the shifted Lennard-Jones potential
    rcut_sqr = rcut * rcut
    rcut_6 = rcut_sqr * rcut_sqr * rcut_sqr
    rcut_12 = rcut_6 * rcut_6
    ecut = ((-1 / rcut_6 )+ (1 / rcut_12))

    ## Apply the minimum image convention
    # Calculate pairwise displacement vectors using minimum image convention 
    displacement_vectors = atoms[:, np.newaxis, :] - atoms[np.newaxis, :, :]
    displacement_vectors = (
        displacement_vectors + 0.5 * lattice_lengths
    ) % lattice_lengths - 0.5 * lattice_lengths

    # Compute pairwise distances from displacement vectors
    # set self interactions to zero
    # Set interactions and distances beyond rcut to infinity so they dont blow up in lj computation
    pairwise_distances = np.linalg.norm(displacement_vectors, axis=-1).astype(np.longdouble)
    pairwise_distances[pairwise_distances == 0] = np.inf
    pairwise_distances[pairwise_distances > rcut] = np.inf

    # Calculate the Lennard-Jones potential for all atom pairs  
    # Shift the potential by ecut so there is not a steep cut off to zero at rcut!!! 

    p12=1/(pairwise_distances**12)
    p6= 1/(pairwise_distances**6)
    # lj = p12 - p6 - ecut

    lj = ((1 / (pairwise_distances**12)) - (1 / (pairwise_distances**6)))

    lj[np.isinf(pairwise_distances)] = 0
    total_lj = np.sum(lj)

    # Precompute force constants for Lennard-Jones forces 
    force_const = 24 * (1 / (pairwise_distances**2)) * (2 * p12 - p6)
    force_const[np.isinf(pairwise_distances)] = 0

    # get force on all pairs (dependent on distance away)
    forces = force_const[:, :, np.newaxis] * displacement_vectors

    return total_lj*2 , np.sum(forces, axis=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Lennard-Jones energy and forces.")
    parser.add_argument("--L", type=int, default=3, help="Number of repeats along L (default: 2)")
    parser.add_argument("--M", type=int, default=3, help="Number of repeats along M (default: 2)")
    parser.add_argument("--N", type=int, default=3, help="Number of repeats along N (default: 2)")
    parser.add_argument("--lattice_constant", type=float, default=np.sqrt(2)*2**(1/6), help="Lattice constant (default: sqrt(2)*2**(1/6))")
    parser.add_argument("--rcut", type=float, default=1.3, help="Cutoff radius (default: 1.3)")
    args = parser.parse_args()

    # Generate computational cell
    atoms, latvec = setup_cell(args.L, args.M, args.N, args.lattice_constant)

    # Compute Lennard-Jones potential and forces
    total_lj, forces = lightning_mcqueen(atoms, args.rcut, latvec)

    num_atoms = atoms.shape[0]
    energy_per_atom = total_lj / num_atoms

    print(f"Total Lennard-Jones Energy: {total_lj:.6f}")
    print(f"Total Energy Per Atom: {energy_per_atom:.6f}")

    # print("Forces on each atom:")
    # print(forces)


    # lj = ((1 / (pairwise_distances**12)) - (1 / (pairwise_distances**6)))
    # total_lj = np.sum(lj)