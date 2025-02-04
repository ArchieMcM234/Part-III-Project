import numpy as np
from scipy.integrate import solve_ivp
from scipy.constants import hbar, physical_constants
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation







# Get the values for mu_B (Bohr magneton) and g_s (electron g-factor)
mu_B = physical_constants['Bohr magneton'][0]  # Bohr magneton in J/T
g_s = physical_constants['electron g factor'][0]  # Electron g-factor



# Pauli Spin Matrices
Sx = np.array([[0,1],[1,0]])
Sy = np.array([[0,-1j],[1j,0]])
Sz = np.array([[1,0],[0,-1]])
I = np.array([[1, 0], [0, 1]])

S_plus = Sx+1j*Sy
S_minus = Sx-1j*Sy




################################################################################################

# Helper functions

def rotating_frame_transformation():
    pass



def calculate_fidelity(U_ideal, U, d):
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
    trace_term = np.trace(np.dot(U_ideal.conj().T, U))
    
    # Fidelity formula
    F = (np.abs(trace_term)**2 + d) / (d * (d + 1))
    
    return F




################################################################################################



 

B_0= 2
B_x = 0.2


omega_0 = g_s*mu_B*B_0/hbar

omega = omega_0

omega_array = np.linspace(0.5*omega_0, 1.5*omega_0, 100)


################################################################################################
# Integration
################################################################################################


def non_rwa_ham(t, omega):
    return (g_s*mu_B/(2)) * np.array([
        [B_0, B_x*np.cos(omega * t)],
        [B_x*np.cos(omega * t), -B_0]
    ])



def rwa_ham(t, omega):
    return  np.array([
        [-(hbar/2)*(omega_0-omega), g_s*mu_B*B_x],
        [g_s*mu_B*B_x, (hbar/2)*(omega_0-omega)]
        ])


def model(t, psi, omega):

    return -(rwa_ham(t, omega)@ psi)* 1j/hbar # need to remember to divide by hbar tdse

# Initial conditions
basis_state_1 = np.array([1 + 0j, 0 + 0j])
basis_state_2 = np.array([0 + 0j, 1 + 0j])

# Time span for the integration
num_steps = 1000
t_span = (0, 10**-9)  # From t=0 to t=10
t_eval = np.linspace(0, 10**-9, num_steps)  # Points at which to evaluate the solution

# Solve the ODE

# is there a nice numpy way of doing this? - i can do as for loop - better to be numpy

z_population_matrix = []
fidelity_matrix = []



for omega in omega_array:

    ham_base_1 = solve_ivp(model, t_span, basis_state_1, t_eval=t_eval, args=(omega,)) 
    ham_base_2 = solve_ivp(model, t_span, basis_state_2, t_eval=t_eval, args=(omega,)) 

    omega_z_pop = []
    omega_fid = []
    for a in range(num_steps):

        total_ham = np.column_stack((ham_base_1.y.T[a], ham_base_2.y.T[a]))



        final_state = total_ham@basis_state_1
        z_population = final_state.conj().T @ Sz @ final_state  # this might be bad?
        omega_z_pop.append(np.real(z_population))

        omega_fid.append(calculate_fidelity(I, total_ham, 2))

    z_population_matrix.append(omega_z_pop)
    fidelity_matrix.append(omega_fid)



z_population_matrix = np.array(z_population_matrix).T
fidelity_matrix = np.array(fidelity_matrix).T

plt.figure(figsize=(8, 6))
plt.imshow(z_population_matrix, aspect="auto", 
           extent=[(omega_array[-1]-omega_0)/(2*np.pi), (omega_array[0]-omega_0)/(2*np.pi), t_eval[0], t_eval[-1]], 
           origin="lower", cmap="viridis")


# Add colorbar and labels
plt.colorbar(label="Z Population")
plt.xlabel("Detuning/s^-1")
plt.ylabel("Time/s")
plt.title("Z Population vs Time and Omega")

plt.show()


plt.figure(figsize=(8, 6))
plt.imshow(fidelity_matrix, aspect="auto", 
           extent=[(omega_array[-1]-omega_0)/(2*np.pi), (omega_array[0]-omega_0)/(2*np.pi), t_eval[0], t_eval[-1]], 
           origin="lower", cmap="viridis")


# Add colorbar and labels
plt.colorbar(label="Identitity fidelity")
plt.xlabel("Detuning/s^-1")
plt.ylabel("Time/s")
plt.title("Z Population vs Time and Omega")

plt.show()