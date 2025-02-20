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

def calculate_expectations(states, obervable):
    """
    Takes array of states and returns the expecation of pauli z on them - only single qubit
    """
    expectations = []
    for state in states:
        expectations.append(state.conj() @ obervable @ state)
    return np.array(expectations)


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
            single_hamiltonian = (1/2) * np.array([
            [-(self.natural_omegas[a] - driving_omega), self.rabi_omega],
            [self.rabi_omega, self.natural_omegas[a] - driving_omega]])

            array_list = [identity for b in range(a)] +[single_hamiltonian]+[identity for b in range(a, self.num_qubits-1)]
            H+= kron_multiple_arrays(array_list)

        return H

    def ccd_rwa(self, phi_0, epsilon_m, phase_freq, theta_m, t, driving_freq):
        """
        Time-independent Hamiltonian in the rotating wave approximation (RWA) for CCD.
        """
        driving_omega = 2 * np.pi * driving_freq
        phase_omega = 2 * np.pi * phase_freq

        delta = self.natural_omegas[0] - driving_omega
        cos_phi_0 = np.cos(phi_0)
        sin_phi_0 = np.sin(phi_0)
        cos_phase = epsilon_m * (phase_omega / self.rabi_omega) * np.cos(phase_omega * t - theta_m)
        # print('delta', delta,'cosphi0', cos_phi_0, 'sin_phio', sin_phi_0, 'cosphase', cos_phase)
        H = - (delta / 2) * sigma_z + (self.rabi_omega / 2) * (cos_phi_0 * sigma_x + sin_phi_0 * sigma_y) +cos_phase * sigma_z
            
        return H

    def ccd_non_rwa(self, phi_0, epsilon_m, phase_freq, theta_m, t, driving_freq):
        """
        Time-dependent Hamiltonian in the non-rotating wave approximation (non-RWA) for CCD.
        """
        driving_omega = 2 * np.pi * driving_freq
        phase_omega = 2 * np.pi * phase_freq

        delta = self.natural_omegas[0] - driving_omega # this definition is the opposite
        cos_term = np.cos(driving_omega * t + phi_0 - (2 * epsilon_m / self.rabi_omega) * np.sin(phase_omega * t - theta_m))
        
        H = (delta / 2) * sigma_z + self.rabi_omega * cos_term * sigma_x
            
        return H


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
    
    def evolve_state(self, initial_state, time, num_points, ham_type="rwa", **kwargs):
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
                H = self.hamiltonian.rwa(kwargs['driving_freq']*2*np.pi)
            elif ham_type == "idle":
                H = self.hamiltonian.idle()
            elif ham_type == "ccd_rwa":
                H = self.hamiltonian.ccd_rwa(**kwargs, t=t)
            else:
                raise ValueError("Unknown Hamiltonian type.")
            return -1j * (H @ psi)
        

        t_span = (0, time)  # From t=0 to t=10
        t_eval = np.linspace(0, time, num_points)  # Points at which to evaluate the solution


        # !!!! want to add tolerances to this !!!!!
        sol = solve_ivp(tdse, t_span, initial_state, t_eval=t_eval )


        return sol.t, sol.y.T  # Return the full evolution for analysis

    def find_total_hamiltonians(self, time, num_points, ham_type="rwa", **kwargs):

        total_hamiltonian = []
        
        for initial_state_index in range(2**self.num_qubits):
            initial_state = np.zeros(2**self.num_qubits, dtype=complex)
            initial_state[initial_state_index] = 1  # Initialize basis state
            
            t, y = self.evolve_state(initial_state, time, num_points, ham_type=ham_type, **kwargs)
            
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

    def rotate_state(self, state, theta, axis):
        """
        Rotate the frame of the quantum state by an angle theta around a given axis.
        
        Parameters:
        - theta (float): Angle by which to rotate the frame.
        - axis (np.array): Axis around which to rotate (must be a 3-element array).
        
        Returns:
        - np.array: Rotated state.
        """
        # Normalize the axis vector
        axis = axis / np.linalg.norm(axis)
        
        # Pauli matrices
        pauli_matrices = [sigma_x, sigma_y, sigma_z]
        
        # Compute the rotation matrix
        rot_mat = np.cos(theta / 2) * np.eye(2) - 1j * np.sin(theta / 2) * sum(axis[i] * pauli_matrices[i] for i in range(3))
        
        return rot_mat @ state
    def rotate_system(self, t, y, theta_func, axis_func):
        """
        Rotate the frame of the quantum system by an angle theta around a given axis.
        
        Parameters:
        - t (array): Time points at which the solution is evaluated.
        - y (array): Solution array with shape (num_points, 2**num_qubits).
        - theta_func (function): Function of time that returns the rotation angle theta.
        - axis_func (function): Function of time that returns the rotation axis (must be a 3-element array).
        
        Returns:
        - array: Rotated solution array with the same shape as y.
        """
        rotated_solution = np.zeros_like(y, dtype=complex)
        
        for i in range(len(t)):
            theta = theta_func(t[i])
            axis = axis_func(t[i])
            rotated_solution[i] = self.rotate_state(y[i], theta, axis)
        
        return rotated_solution
    

    def visualise_solution(self, t, y):
        """
        Visualize the solution of the quantum system on a Bloch sphere.
        
        Parameters:
        - t (array): Time points at which the solution is evaluated.
        - y (array): Solution array with shape (num_points, 2**num_qubits).
        """
        # Calculate expectations
        x_expectation = calculate_expectations(y, sigma_x)
        y_expectation = calculate_expectations(y, sigma_y)
        z_expectation = calculate_expectations(y, sigma_z)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

        # Function to update the Bloch vector
        def update(frame):
            ax.cla()  # Clear the previous frame
            ax.set_title("Animated Bloch Vector")
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])

            # Plot the Bloch vector
            ax.quiver(0, 0, 0, x_expectation[frame], y_expectation[frame], z_expectation[frame], 
                      color='r', arrow_length_ratio=0.1)

        # Create the animation
        ani = FuncAnimation(fig, update, frames=np.arange(0, len(t), 1), interval=50)

        plt.show()
        # ani.save("Bloch_Vector.gif", writer='pillow', fps=30)


























