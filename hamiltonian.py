import numpy as np


def rwa_ham(t):
    return  np.array([
        [-(hbar/2)*(omega_0-omega), g_s*mu_B*B_x],
        [g_s*mu_B*B_x, (hbar/2)*(omega_0-omega)]
        ])



 # hmmmm -  i think i should get rabi chevron to work - then think about bigger hamiltonian