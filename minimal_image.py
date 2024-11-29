import numpy as np 
from get_compcell import setup_cell

def lightning_mcqueen(atoms, rcut, latvec):
    """
    Compute the Lennard-Jones (LJ) potential energy and forces for the provided computational 
    cell within a cutoff radius. This version uses the minimum image convention. 

    Parameters
    ----------
    atoms : ndarray, shape (num_atoms, 3)
        Coordinates of all atoms in the computational cell.

    rcut : float
        Cutoff radius for the Lennard-Jones interaction. Interactions at distances greater than
        this value are excluded from the calculation.

    latvec : ndarray, shape (3, 3)
        Lattice vectors defining the simulation cell. The diagonal elements correspond 
        to the lengths of the cell along the L, M and N directions scaled by the lattice constant.

    Returns
    -------
    total_lj : float
        Total Lennard-Jones potential energy of the system (with a factor of 2 
        for double-counting correction).

    forces : ndarray, shape (num_atoms, 3)
        Forces on each atom due to Lennard-Jones interactions, including all  
        periodic neighbors within the cutoff radius. 


    Notes
        -----
        - **Minimum Image Convention (MIC)**:
            The MIC ensures that only the nearest periodic copy of each atom is considered 
            when computing interactions. This is valid only when the cutoff radius satisfies:
            
        r_cut <= min(lattice_lengths) / 2

            where `lattice_lengths` are the lengths of the cell along each direction, 
            extracted from the diagonal of the `latvec` matrix.

        - If r_cut does not satisfy this condition use other function. 

        - The Lennard-Jones potential is given by:
            V(r) = 4 * [(1/r)^12 - (1/r)^6]

        Here, the potential is shifted to zero at the cutoff distance.

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
    ecut = (-1 / rcut_6 + 1 / rcut_12)

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
    lj = p12 - p6 - ecut
    lj[np.isinf(pairwise_distances)] = 0
    total_lj = np.sum(lj)

    # Precompute force constants for Lennard-Jones forces 
    force_const = 24 * (1 / (pairwise_distances**2)) * (2 * p12 - p6)
    force_const[np.isinf(pairwise_distances)] = 0

    # get force on all pairs (dependent on distance away)
    forces = force_const[:, :, np.newaxis] * displacement_vectors

    return total_lj * 2, np.sum(forces, axis=1)


 