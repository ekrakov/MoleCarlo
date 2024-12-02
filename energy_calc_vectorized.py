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

def generate_displacements(rcut,latvec,lmax=None, mmax=None, nmax=None):
   
    
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

    return displacement_extended, rcut



def periodic(atoms,rcut,displacement_extended,vacancy=True):
    """
    Periodic boundary conditions when many coppies are required (rcut)
    Parameters:
        atoms (ndarray): Atomi (shape: Nx3).
        rcut (float): Cutoff radius for interactions.
        latvec (ndarray): Lattice vectors (shape: 3x3).
    
    Returns:
        tuple: Total Lennard-Jones potential energy and net forces on each atom.
    """

    # if vacancy:
    # atoms = np.delete(atoms, 0, axis=0)
    # atoms = np.delete(atoms, 2, axis=0)


    # for next assigment
    rcut_sqr=rcut*rcut
    rcut_6=rcut_sqr*rcut_sqr*rcut_sqr
    rcut_12=rcut_6*rcut_6
    ecut=(-1/rcut_6+1/rcut_12)
  
    #expands atom and displacement vector to do computation
    atoms = atoms[:, np.newaxis, :]  # Shape (N, 1, 3) (Expand atoms in computational cell)
    displacement_vec =-atoms + displacement_extended #shape (n_atom,disp,3) (all displacements for every atom)
    displacement_vec = displacement_vec.reshape(-1, 3) # I do not care what atom the displacement came from. 
    displacement_vec= displacement_vec[ np.newaxis, :,:]   # Shape (N*M, 1, 3)
    
    # calculate displacements in squared form to reduce unnecessary sqrt calculation
    disp_diff=atoms+displacement_vec
    pairwise_sqr = np.sum(disp_diff**2, axis=-1)
    #janky filtering methods 
    pairwise_sqr[pairwise_sqr <= 0] = np.inf
    pairwise_sqr[pairwise_sqr >=rcut**2] = np.inf
    zero_indices = (pairwise_sqr == np.inf)


    #####################COMPUTE LENNARD JONES POTENTIAL$$#############
    p12=(1 / (pairwise_sqr**6))
    p6=(1 / (pairwise_sqr**3))
    lj = (p12) - (p6)
    total_lj = np.sum(lj)

    ####################COMPUTE FORCE################################
    force_const=24*(1/(pairwise_sqr))*(2*p12-p6)

    disp_diff[zero_indices, :] = np.zeros_like(disp_diff[zero_indices, :])
    force = force_const[:, :, np.newaxis]*disp_diff 

    return ((total_lj)*2),np.sum(force,axis=(1))