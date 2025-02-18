import numpy as np
from scipy.integrate import solve_ivp
from scipy.constants import hbar, physical_constants

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from functools import reduce

from pauli_matrices import *





def calculate_fidelitys(U_ideal, U_array, d):
    """
    Calculate the fidelity between two unitaries U_ideal and U.
    
    Parameters:
    U_ideal (np.array): Ideal unitary matrix (square matrix).
    U (np.array): Actual unitary matrix (square matrix).
    d (int): Dimension of the Hilbert space (e.g., 2^N for N qubits).
    
    Returns:
    float: Fidelity value between 0 and 1.
    """
    # Compute the trace of the product of U_ideal^dagger and U
    fidelities = []
    for hamiltonian in U_array:
        trace_term = np.trace(np.dot(U_ideal.conj().T, hamiltonian))
    
        # Fidelity formula
        fidelities.append((np.abs(trace_term)**2 + d) / (d * (d + 1)))
    
    return fidelities


def calculate_z_expectation(states):
    """
    Takes array of states and returns the expecation of pauli z on them - only single qubit
    """
    z_expectation = []
    for state in states:
        z_expectation.append(state.conj() @ sigma_z @ state)
    return np.array(z_expectation)

from functools import reduce

def kron_multiple_arrays(array_list):
    """
    Compute the Kronecker product of a list of arrays.
    
    Parameters:
        array_list (list of ndarray): List of arrays to compute the Kronecker product.
        
    Returns:
        ndarray: Resulting Kronecker product of the input arrays.
    """
    return reduce(np.kron, array_list)




class Quantum_Hamiltonian:
    # my convention is to have freq and omega for Hz and rads respectively
    def __init__(self, rabi_freq, num_qubits, natural_freqs):

        self.rabi_omega = 2*np.pi*rabi_freq  # Rabi frequency
        self.num_qubits = num_qubits
        self.natural_omegas = 2*np.pi*natural_freqs

    def non_rwa(self):
        # TODO
        pass

    def idle(self):
        return np.zeros((2, 2))

    def rwa(self, driving_omega):
        """
        Time-independent Hamiltonian in the rotating wave approximation (RWA).
        """
        H = np.zeros((2**self.num_qubits, 2**self.num_qubits), dtype=complex)

        for a in range(self.num_qubits):
            single_hamiltonian = hbar*(1/2) * np.array([
            [-(self.natural_omegas[a] - driving_omega), self.rabi_omega],
            [self.rabi_omega, self.natural_omegas[a] - driving_omega]])

            array_list = [identity for b in range(a)] +[single_hamiltonian]+[identity for b in range(a, self.num_qubits-1)]
            H+= kron_multiple_arrays(array_list)

        return H

    def ccd_rwa(self):
        # TODO
        pass

    def ccd_non_rwa(self):
        # TODO
        pass


# Im gonna think that this is permanent stuff - may be better to split up but press on
class Quantum_System:
    def __init__(self, hamiltonian, num_qubits):
        """
        Initialize the quantum system.
        
        Parameters:
        - hamiltonian (Hamiltonian): Instance of the Hamiltonian class.
        - num_qubits (int): Number of qubits in the system.
        """
        self.hamiltonian = hamiltonian
        self.num_qubits = num_qubits

    
    def initialize_state(self):
        """
        Initialize the system's state. Default: |0...0> (ground state).
        """
        basis_state = np.zeros(2**self.num_qubits, dtype=complex)
        basis_state[0] = 1.0  # Ground state |0...0>
        return basis_state
    
    def evolve_state(self, initial_state, time, num_points, driving_freq, ham_type="rwa", **kwargs):
        """
        Evolve the system using the time-dependent Schrödinger equation.
        
        Parameters:
        - t_span (tuple): Time range for the evolution (start, end).
        - t_eval (array): Time points at which to evaluate the solution.
        - omega (float): Driving frequency.
        - ham_type (str): Type of Hamiltonian to use ("rwa", "non_rwa", "custom").
        
        Updates the system's state after evolution.
        """


        def tdse(t, psi):
            if ham_type == "rwa":
                H = self.hamiltonian.rwa(driving_freq*2*np.pi)
            elif ham_type == "idle":
                H = self.hamiltonian.idle()
            else:
                raise ValueError("Unknown Hamiltonian type.")
            return -1j / hbar * (H @ psi)
        

        t_span = (0, time)  # From t=0 to t=10
        t_eval = np.linspace(0, time, num_points)  # Points at which to evaluate the solution


        # !!!! want to add tolerances to this !!!!!
        sol = solve_ivp(tdse, t_span, initial_state, t_eval=t_eval )


        return sol.t, sol.y.T  # Return the full evolution for analysis

    def find_total_hamiltonians(self, time, num_points, driving_freq, ham_type="rwa", **kwargs):

        total_hamiltonian = []
        
        for initial_state_index in range(2**self.num_qubits):
            initial_state = np.zeros(2**self.num_qubits, dtype=complex)
            initial_state[initial_state_index] = 1  # Initialize basis state
            
            t, y = self.evolve_state(initial_state, time, num_points, driving_freq, ham_type, **kwargs)
            
            # Now we stack the evolved states for all basis states at each time step
            total_hamiltonian.append(y)

        # Stack all the evolved states for each basis state along the second axis
        # Result will be (num_points, 2**num_qubits)
        return t, np.stack(total_hamiltonian, axis=1) 



    
    def expectation(self, observable):
        """
        Measure an observable on the current state.
        
        Parameters:
        - observable (np.array): Hermitian operator to measure.
        
        Returns:
        - float: Expectation value of the observable.
        """
        return np.real(self.state.conj().T @ observable @ self.state)


    def rotate_frame(self):
        # TODO
        pass





























