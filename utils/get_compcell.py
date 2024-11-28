import numpy as np

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




def lightning_mcqueen(atoms, rcut,latvec):
    """
    Compute the dot product of two or more arrays in a single function call,
    while automatically selecting the fastest evaluation order.

    `multi_dot` chains `numpy.dot` and uses optimal parenthesization
    of the matrices [1]_ [2]_. Depending on the shapes of the matrices,
    this can speed up the multiplication a lot.

    If the first argument is 1-D it is treated as a row vector.
    If the last argument is 1-D it is treated as a column vector.
    The other arguments must be 2-D.

    Think of `multi_dot` as::

        def multi_dot(arrays): return functools.reduce(np.dot, arrays)


    Parameters
    ----------
    arrays : sequence of array_like
        If the first argument is 1-D it is treated as row vector.
        If the last argument is 1-D it is treated as column vector.
        The other arguments must be 2-D.
    out : ndarray, optional
        Output argument. This must have the exact kind that would be returned
        if it was not used. In particular, it must have the right type, must be
        C-contiguous, and its dtype must be the dtype that would be returned
        for `dot(a, b)`. This is a performance feature. Therefore, if these
        conditions are not met, an exception is raised, instead of attempting
        to be flexible.

        .. versionadded:: 1.19.0

    Returns
    -------
    output : ndarray
        Returns the dot product of the supplied arrays.

    See Also
    --------
    numpy.dot : dot multiplication with two arguments.

    References
    ----------

    .. [1] Cormen, "Introduction to Algorithms", Chapter 15.2, p. 370-378
    .. [2] https://en.wikipedia.org/wiki/Matrix_chain_multiplication

    Examples
    --------
    `multi_dot` allows you to write::

    >>> import numpy as np
    >>> from numpy.linalg import multi_dot
    >>> # Prepare some data
    >>> A = np.random.random((10000, 100))
    >>> B = np.random.random((100, 1000))
    >>> C = np.random.random((1000, 5))
    >>> D = np.random.random((5, 333))
    >>> # the actual dot multiplication
    >>> _ = multi_dot([A, B, C, D])

    instead of::

    >>> _ = np.dot(np.dot(np.dot(A, B), C), D)
    >>> # or
    >>> _ = A.dot(B).dot(C).dot(D)

    Notes
    -----
    The cost for a matrix multiplication can be calculated with the
    following function::

        def cost(A, B):
            return A.shape[0] * A.shape[1] * B.shape[1]

    Assume we have three matrices
    :math:`A_{10x100}, B_{100x5}, C_{5x50}`.

    The costs for the two different parenthesizations are as follows::

        cost((AB)C) = 10*100*5 + 10*5*50   = 5000 + 2500   = 7500
        cost(A(BC)) = 10*100*50 + 100*5*50 = 50000 + 25000 = 75000

    """

    rcut_sqr=rcut*rcut
    rcut_6=rcut_sqr*rcut_sqr*rcut_sqr
    rcut_12=rcut_6*rcut_6
    ecut=(-1/rcut_6+1/rcut_12)
    lattice_lengths = np.diagonal(latvec)
    
    # 
    displacement_vectors = atoms[:, np.newaxis, :] - atoms[np.newaxis, :, :]
    displacement_vectors = (
        displacement_vectors + 0.5 * lattice_lengths
    ) % lattice_lengths - 0.5 * lattice_lengths
    
   
    pairwise_distances = np.linalg.norm(displacement_vectors, axis=-1).astype(np.longdouble)
    
    
    pairwise_distances[pairwise_distances <= 0] = np.inf
    pairwise_distances[pairwise_distances > rcut] = np.inf
    zero_indices = (pairwise_distances == np.inf)
    
    displacement_vectors[zero_indices, :] = 0

    lj = (1 / (pairwise_distances**12)) - (1 / (pairwise_distances**6))-ecut
    lj[zero_indices]=0
    # print(np.shape(lj),'shape')
    total_lj = np.sum(lj)
    # print(np.shape(total_lj),'tot')
    p12 = (1 / (pairwise_distances**12))
    p6 = (1 / (pairwise_distances**6))
    

    force_const = 24 * (1 / (pairwise_distances**2)) * (2 * p12 - p6)
    force_const[zero_indices] = 0  # i know it jank
    forces = force_const[:, :, np.newaxis] * displacement_vectors

    return total_lj*2, np.sum(forces, axis=1)
