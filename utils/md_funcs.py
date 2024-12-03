import numpy as np
import random 

from minimal_image import lightning_mcqueen

import matplotlib.pyplot as plt
plt.style.use("~/.matplotlib/styles/style.mplstyle")


def mcmove(atoms,mc_max,beta,rcut,latvec,etot):
    atomsc=atoms

    move = random.randint(0, np.shape(atoms)[0]-1)  

    current =atoms[move,:].copy()

    atoms[move, :] += (np.random.rand(3) - 0.5) * mc_max  # 
    print(atoms[move, :],'test')
    enew,_=lightning_mcqueen(atoms, rcut,latvec)
    # print(etot,enew)
    if np.random.rand() < np.exp(beta * (etot - enew)):
    # Accept the move
        etot = enew
        reject=1
    else:
        # print(etot-enew,'etot-enew')

    # Reject the move
        atoms[move, :] = current  
        reject=0
    return reject,atoms,etot


L=2
M=L
N=L

kb_T=.1
rcut=1.3
dt=.02
step=50000

rejected_configs=0
kb_T=4
beta=1/kb_T
nsteps=6000
mc_max=0.07
delta_kb_T=-kb_T/nsteps*2


atoms,latvec=setup_cell(L,M,N)
atomsn,latvec=setup_cell(L,M,N)


etot_arr=[]
reject_sum_arr=[]
etot,force = lightning_mcqueen(atoms,rcut,latvec)
for i in range(step):
    if i > 6000 and i < 15000:
        beta=1/2
    if i >= 15000<=30000:
        beta = 1/.3
    if i >= 30000:
        beta = 1/.01

    reject,atoms,etot=mcmove(atoms,mc_max,beta,rcut,latvec,etot)
    reject_sum_arr.append(reject)
    etot_arr.append(etot)
    # print(reject,'here')
plt.plot(etot_arr)
plt.xlabel("Itteration")
plt.ylabel("Energy")
plt.show()

running_avg = np.cumsum(reject_sum_arr) / np.arange(1, len(reject_sum_arr) + 1)

# Plotting the running average
plt.plot(1-running_avg)
plt.xlabel("Step")
plt.ylabel("Rejection Ratio")
plt.legend()
plt.show()


num=5000
val=np.shape(reject_sum_arr[num:])[0]
print(np.sum(reject_sum_arr[num:])/val)