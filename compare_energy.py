import numpy as np
import time 
from utils.get_compcell import setup_cell
from utils.minimal_image import lightning_mcqueen
from utils.energy_calc_vectorized import periodic,generate_displacements


start_time = time.time()
atoms, lv = setup_cell(2,2,2)
setup_cell_time = time.time() - start_time

# Measure time for generate_periodic
start_time = time.time()
rcut = 1.3
disp_extended,_= generate_displacements(rcut, lv)
print(rcut,'here')
generate_periodic_time = time.time() - start_time

# Measure time for periodic
num_atoms = atoms.shape[0]
print(num_atoms,'numatoms')
start_time = time.time()
pot, force = periodic(atoms, rcut, disp_extended,vacancy=True)
periodic_time = time.time() - start_time

# measure time for mic
start_time = time.time()
pot_mic, force_mic = lightning_mcqueen(atoms, rcut, lv)
mic_time = time.time() - start_time



# Output results
print(f"setup_cell time: {setup_cell_time:.6f} seconds")
print(f"generate_periodic time: {generate_periodic_time:.6f} seconds")
print(f"periodic time: {periodic_time:.6f} seconds")
print(f"Potential Energy per atom PERIODIC: {pot/num_atoms}, here")

print(f"MIC time: {mic_time:.6f} seconds")
print(f"Potential Energy per atom MIC: {pot_mic/num_atoms}, here")


