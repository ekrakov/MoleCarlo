import numpy as np
import matplotlib.pyplot as plt
import time
from mpmath import mp
from utils.get_compcell import setup_cell
import time


plt.style.use("~/.matplotlib/styles/style.mplstyle")

# np.set_printoptions(precision=17)


def charge(length,con):
    num_positive=int(np.ceil(length*con))
    num_neg=length-num_positive
    array = np.array([2] * num_positive + [-2] * num_neg)
    shuffle=np.random.permutation(array)
    return shuffle

def generate_periodic(atoms,rcut,latvec,lmax=None, mmax=None, nmax=None):
   
    
    if lmax is None:
        lmax = int(np.ceil(rcut / latvec[0, 0]))
    if mmax is None:
        mmax = int(np.ceil(rcut / latvec[1, 1]))
    if nmax is None:
        nmax = int(np.ceil(rcut / latvec[2, 2]))

    l_disp_range = np.arange(-lmax, lmax + 1)
    m_disp_range = np.arange(-mmax, mmax + 1)
    n_disp_range = np.arange(-nmax, nmax + 1)

    l_disp, m_disp, n_disp = np.meshgrid(l_disp_range, m_disp_range, n_disp_range, indexing='ij')
    all_displacements = np.stack([l_disp.ravel(), m_disp.ravel(), n_disp.ravel()], axis=-1)  # Shape: (num_displacements, 3)
    displacement_extended = all_displacements[np.newaxis, :, :]@latvec

    return displacement_extended



def periodic(atoms,rcut,displacement_extended,vacancy=False):
    """
    Periodic boundary conditions when many coppies are required (rcut)
    Parameters:
        atoms (ndarray): Atomi (shape: Nx3).
        rcut (float): Cutoff radius for interactions.
        latvec (ndarray): Lattice vectors (shape: 3x3).
    
    Returns:
        tuple: Total Lennard-Jones potential energy and net forces on each atom.
    """

    if vacancy:
        atoms = np.delete(atoms, 2, axis=0)


    rcut_sqr=rcut*rcut
    rcut_6=rcut_sqr*rcut_sqr*rcut_sqr
    rcut_12=rcut_6*rcut_6
    ecut=(-1/rcut_6+1/rcut_12)

    # Create a 3D grid of displacements
  

    atoms_extended = atoms[:, np.newaxis, :]  # Shape (N, 1, 3) (Expand atoms in computational cell)
   
    displacement_vec =-atoms_extended + displacement_extended #shape (n_atom,disp,3) (all displacements for every atom)
    flattened_displacements = displacement_vec.reshape(-1, 3) # I do not care what atom the displacement came from. 


    ###########################START PAIRWISE###########################################
    ##########Expand vecctors to compute pairwise distance##############################
    ### take for every atom in atoms find difference between every single displacement vector generated in displacement vec (4xdisp)

    disp_expanded_pairwise= flattened_displacements[ np.newaxis, :,:]   # Shape (N*M, 1, 3)
    ####r_i-r_j
    disp_diff=atoms_extended+disp_expanded_pairwise
  
    #### get pairwise distance between all atoms 
    pairwise_distances = np.linalg.norm(disp_diff, axis=-1)
    #janky filtering methods 
    pairwise_distances[pairwise_distances <= 0] = np.inf
    pairwise_distances[pairwise_distances >rcut] = np.inf
    zero_indices = (pairwise_distances == np.inf)

    # Set corresponding entries in A to (0, 0, 0)
    disp_diff[zero_indices, :] = np.zeros_like(disp_diff[zero_indices, :])
    ###################################END OF PAIRWISE################################
    ##################################################################################

    #####################COMPUTE LENNARD JONES POTENTIAL$$#############
    shape = (32, 864)

    # Define the value you want to subtract
    value = ecut

    # Create an array of the same shape filled with the specific value
    ecut_arr = np.full(shape, value)


    lj = ((1 / (pairwise_distances**12)) - (1 / (pairwise_distances**6)))
    total_lj = np.sum(lj)
    p12=(1 / (pairwise_distances**12))
    p6=(1 / (pairwise_distances**6))

    ####################COMPUTE FORCE################################
    ############this is wrong!#######################################
    force_const=24*(1/(pairwise_distances**2))*(2*p12-p6)
    force = disp_diff
    force = force_const[:, :, np.newaxis]*disp_diff #This is what is giving me the issue 

    return ((total_lj)*2),np.sum(force,axis=(1))





#############DEFINE VARIABLES ##################################

# L=1
# M=L
# N=M
# lattice =  np.sqrt(2) * 2**(1/6)
# latvec = np.identity(3, dtype=np.longdouble)  

# latvec = np.array([
#     [L * lattice, 0, 0],
#     [0, M * lattice, 0],
#     [0, 0, N * lattice]
# ])

start_time = time.time()
atoms, lv = setup_cell(2, 2, 2)
setup_cell_time = time.time() - start_time

# Measure time for generate_periodic
start_time = time.time()
disp_extended = generate_periodic(atoms, 1.3, lv, lmax=None, mmax=None, nmax=None)
generate_periodic_time = time.time() - start_time

# Measure time for periodic
start_time = time.time()
pot, force = periodic(atoms, 1.3, disp_extended)
periodic_time = time.time() - start_time

# Output results
print(f"setup_cell time: {setup_cell_time:.6f} seconds")
print(f"generate_periodic time: {generate_periodic_time:.6f} seconds")
print(f"periodic time: {periodic_time:.6f} seconds")
print(f"Potential Energy: {pot}, here")

# # rcut=2.2


# # print(vector_indices)


# num_atoms = atoms.shape[0]

# # start_time = time.time()
# pot,force=periodic(atoms,1.3,lv)
# print(pot/(num_atoms-1))
# print(pot)

# e_arr = []
# L_values = []

# # Loop for m = 1 to 5
# for i in range(8):
#     m = i + 1
#     atoms, lv = setup_cell(m, m, m)
#     num_atoms = atoms.shape[0] - 1
#     pot, force = periodic(atoms, 1.3, lv)
#     L_values.append(m)
#     print(pot/num_atoms)
#     print(num_atoms,'here shape')
#     e_arr.append(pot / num_atoms)

# # Plot L (L=M=N) vs Energy/Atom
# plt.figure(figsize=(8, 6))
# plt.plot(L_values, e_arr, marker='o', linestyle='-')
# plt.axhline(y=-6, color='black', linestyle='--', label="y = -6")  
# plt.xlabel("$(L=M=N)$")
# plt.ylabel("Energy/Atom")
# plt.savefig("energy_vs_lattice.pdf", format="pdf", bbox_inches="tight")
# plt.show()