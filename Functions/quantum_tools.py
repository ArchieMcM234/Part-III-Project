import numpy as np
from scipy.integrate import solve_ivp
from scipy.constants import hbar, physical_constants

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from functools import reduce

from pauli_matrices import *
from IPython.display import HTML



# def calculate_fidelities(U_ideal, U_array, d):
#     # Compute the conjugate transpose of the ideal unitary
#     U_ideal_dag = U_ideal.conj().T
    
#     # Batch matrix multiplication of U_ideal_dag with each matrix in U_array
#     product = np.matmul(U_ideal_dag, U_array)

    
#     # Compute the trace for each matrix in the batch result
#     trace_terms = np.trace(product, axis1=1, axis2=2)
    
#     # Calculate fidelity using vectorized operations
#     fidelities = (np.abs(trace_terms)**2 + d) / (d * (d + 1))
    
#     # Return the real parts to handle any negligible imaginary components
#     return np.real(fidelities)

def calculate_fidelities(U_ideal, U_array, d):
    fids = []

    for U in U_array:
        fid = (np.abs(np.trace(U_ideal.conj().T @ U))**2 +d)/(d*(d+1))
        fids.append(fid)

    return np.real(fids)


def calculate_expectations(states, observable):
    """
    Takes an array of states with shape (n, 2) and returns the expectation of the observable for each state.

    Args:
        states (np.ndarray): Array of shape (n, 2), where each row is a state vector.
        observable (np.ndarray): Matrix of shape (2, 2) representing the observable.

    Returns:
        np.ndarray: Array of shape (n,) containing the expectation values.
    """
    # Compute the expectation values using einsum
    # einsum performs the contraction: state^dagger @ observable @ state
    expectations = np.einsum('si,ij,sj->s', states.conj(), observable, states)
    
    # Return the real part of the expectations
    return np.real(expectations)



def evolve_state(initial_state, time, num_points, hamiltonian_func, rtol=1e-7, atol=1e-7, **kwargs):
    """
    Evolve the system using the time-dependent Schrödinger equation.
    
    Parameters:
    - initial_state (np.array): Initial state vector
    - time (float): Total evolution time
    - num_points (int): Number of time points to evaluate
    - hamiltonian_func (callable): Function that returns the Hamiltonian
    - rtol (float): Relative tolerance for the solver (default: 1e-7)
    - atol (float): Absolute tolerance for the solver (default: 1e-7)
    - **kwargs: Additional parameters passed to the Hamiltonian function
    
    Returns:
    - tuple: (t, y) Time points and evolved state vectors
    """
    def tdse(t, psi):
        H = hamiltonian_func(**kwargs, t=t)
        return -1j * (H @ psi)

    t_span = (0, time)
    t_eval = np.linspace(0, time, num_points)

    # Solver with specified tolerances
    sol = solve_ivp(tdse, t_span, initial_state, 
                    t_eval=t_eval, 
                    method='RK45',
                    rtol=rtol, 
                    atol=atol)

    return sol.t, sol.y.T

def calculate_unitaries(num_qubits, time, num_points, hamiltonian_func, **kwargs):
    """
    Calculate the unitary evolution operators for a quantum system.
    
    Parameters:
    - num_qubits (int): Number of qubits in the system
    - time (float): Total evolution time
    - num_points (int): Number of time points
    - hamiltonian_func (callable): Function that returns the Hamiltonian
    - **kwargs: Additional parameters passed to evolve_state
    
    Returns:
    - tuple: (t, U) Time points and unitary operators
    """
    solutions = []
    total_Us = []

    for initial_state_index in range(2**num_qubits):
        initial_state = np.zeros(2**num_qubits, dtype=complex)
        initial_state[initial_state_index] = 1  # Initialize basis state
        
        t, y = evolve_state(initial_state, time, num_points, hamiltonian_func, **kwargs)
        
        solutions.append(y)

    for i in range(len(t)):
        U = np.array([solutions[j][i] for j in range(2**num_qubits)]).T  # Manually form columns
        total_Us.append(U)

    return t, np.array(total_Us)#np.stack(total_Us, axis=1)

def rotate_state(state, theta, axis):
    """
    Rotate a quantum state by an angle theta around a given axis.
    
    Parameters:
    - state (np.array): Quantum state vector
    - theta (float): Rotation angle
    - axis (np.array): Rotation axis (3-element array)
    
    Returns:
    - np.array: Rotated state
    """
    axis = axis / np.linalg.norm(axis)
    pauli_matrices = [sigma_x, sigma_y, sigma_z]
    rot_mat = np.cos(theta / 2) * np.eye(2) - 1j * np.sin(theta / 2) * sum(axis[i] * pauli_matrices[i] for i in range(3))
    return rot_mat @ state

def rotate_system(t, states, theta_func, axis_func):
    """
    Rotate quantum states over time around specified axes.
    
    Parameters:
    - t (array): Time points
    - y (array): State vectors (shape: num_points x state_dimension)
    - theta_func (callable): Function returning rotation angle at time t
    - axis_func (callable): Function returning rotation axis at time t
    
    Returns:
    - array: Rotated states
    """
    rotated_solution = np.zeros_like(states, dtype=complex)
    
    for i in range(len(t)):
        theta = theta_func(t[i])
        axis = axis_func(t[i])
        rotated_solution[i] = rotate_state(states[i], theta, axis)
    
    return rotated_solution

def rotate_unitary(t, U, R_func):
    """
    Transform unitary operators using a time-dependent rotation matrix.
    
    Parameters:
    - t (array): Time points
    - U (array): Unitary operators (shape: num_points x state_dimension x state_dimension)
    - R_func (callable): Function returning rotation matrix at time t
    
    Returns:
    - array: Transformed unitary operators
    """
    transformed_U = np.zeros_like(U, dtype=complex)

    for i in range(len(t)):
        R = R_func(t[i])  # Get the rotation matrix at time t[i]
        transformed_U[i] = R @ U[i] @ R.conj().T  # Basis transformation

    return transformed_U

def qubit_frame_transformation(U, freq, t):
    omega = 2 * np.pi * freq  # Convert to angular frequency
    
    # Define the rotation matrix R(t) and its conjugate transpose R†(t)
    exp_factor = np.exp(-1j * omega * t / 2)
    R_t = np.array([[exp_factor, 0], [0, np.conj(exp_factor)]])
    
    # Apply the transformation
    transformed_U = R_t @ U    #so implicitly here we assume the initial state alignes with the iniital state in all frames
    
    return transformed_U

def phase_boost_unitaries(U_array, freq, times, t_0=0):
    omega = 2 * np.pi * freq  # Convert to angular frequency
    

    # Define the rotation matrix R(t) and its conjugate transpose R†(t)

    transformed_Us = []
    for t, U in zip(times, U_array):
        R_t = np.array([[np.exp(-1j * omega * t / 2), 0], [0, np.exp(1j * omega * t / 2)]])
        R_t_0_dag = np.array([[np.exp(1j * omega * t_0 / 2), 0], [0, np.exp(-1j * omega * t_0 / 2)]])
        transformed_U = R_t @ U @ R_t_0_dag
        transformed_Us.append(transformed_U)    

    
    return np.array(transformed_Us)



def visualise_solution(t, y, static_vector=None):
    """
    Visualize quantum states on the Bloch sphere.
    
    Parameters:
    - t (array): Time points
    - y (array): State vectors (shape: num_points x state_dimension)
    - static_vector (array, optional): Static vector to display
    
    Returns:
    - HTML: Animation of Bloch sphere
    """
    x_expectation = np.real(calculate_expectations(y, sigma_x))
    y_expectation = np.real(calculate_expectations(y, sigma_y))
    z_expectation = np.real(calculate_expectations(y, sigma_z))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1, 1, 1])

    def update(frame):
        ax.cla()
        ax.set_title("Animated Bloch Vector")
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_box_aspect([1, 1, 1])

        ax.quiver(0, 0, 0, x_expectation[frame], y_expectation[frame], z_expectation[frame], 
                 color='r', arrow_length_ratio=0.1)
        
        if static_vector is not None:
            ax.quiver(0, 0, 0, static_vector[0], static_vector[1], static_vector[2], 
                     color='b', arrow_length_ratio=0.1)

    ani = FuncAnimation(fig, update, frames=np.arange(0, len(t), 1), interval=50)
    plt.close(fig)
    
    return HTML(ani.to_jshtml())