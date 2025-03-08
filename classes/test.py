import numpy as np
from quantum_classes import *
from pauli_matrices import *
import multiprocessing as mp
from functools import partial
from scipy.linalg import expm
from multiprocessing import Pool


def identity_fidelity_multithread(detunings):

    def fidelity_for_detuning(detuning):
        # Create quantum system with the given detuning for second qubit
        hamiltonian_2q = Quantum_Hamiltonian(rabi_freq, 2, np.array([natural_freq, natural_freq + detuning]))
        system_2q = Quantum_System(hamiltonian_2q, 2)
        
        # Calculate unitaries for all times
        _, unitaries = system_2q.calculate_unitaries(evaluation_time, evaluation_points, 
                                            ham_type='ccd_rwa_multiple', 
                                            driving_freq=driving_freq, 
                                            phi_0=phi_0, epsilon_m=epsilon_m, 
                                            phase_freq=phase_freq, theta_m=theta_m, 
                                            coupling=0)
        
        
        fidelities = calculate_fidelitys(np.eye(4), unitaries, 2**2)

        
        return fidelities
    
    # Create a pool of workers

    with Pool() as pool:
        # Map the function to different detunings
        results = pool.map(fidelity_for_detuning, detunings)
    
    return results