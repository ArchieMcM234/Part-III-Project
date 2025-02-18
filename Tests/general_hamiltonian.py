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




num_qubits = 2

# Define Magnetic fields - one for each qubit - and one driving field
B_0= [2, 4]
B_x = 0.2





omega_0 = [g_s*mu_B*B_i/hbar for B_i in B_0]



omega = omega_0[0]


# Coupling parameter
J = g_s*mu_B*B_x*0.0 # just a made up value


# integration variables
num_steps = 1000
t_span = (0, 10**-9)  # From t=0 to t=10
t_eval = np.linspace(0, 10**-9, num_steps)  # Points at which to evaluate the solution



################################################################################################

# TODO

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
# Integration
################################################################################################

# this is not very nice - i want to pass the omegas in properly ideally

#note on basis - the single electron basis states used are up down which is 0, 1 in that order


first_ham = np.array([
    [-(hbar/2)*(omega_0[0]-omega), g_s*mu_B*B_x],
    [g_s*mu_B*B_x, (hbar/2)*(omega_0[1]-omega)]
    ])
second_ham = np.array([
    [-(hbar/2)*(omega_0[1]-omega), g_s*mu_B*B_x],
    [g_s*mu_B*B_x, (hbar/2)*(omega_0[1]-omega)]
    ])

# then the total hamiltonian is the tensor product of the two - this will be in the comp basis
# this was incorrect - you get the total by adding the individual hams tensored with identity(inplace of other)

rwa_hamiltonian = np.kron(first_ham, I) + np.kron(I, second_ham) + 0*J *(np.kron(Sx, Sx)+np.kron(Sy, Sy)+np.kron(Sz, Sz)) # so my understaning is that this only works for the tensor prod stuff


print((np.kron(Sx, Sx)+np.kron(Sy, Sy)+np.kron(Sz, Sz))-np.kron(I, I))
print(np.kron(I, I))



def schrodinger_equation(t, psi):
    # I am timesing and dividing by hbar -redundant
    print(rwa_hamiltonian)
    return -(rwa_hamiltonian @ psi)* 1j/hbar # need to remember to divide by hbar tdse



# Initial conditions
spin_up = np.array([1 + 0j, 0 + 0j])
spin_down = np.array([0 + 0j, 1 + 0j])

# Time span for the integration

# Solve the ODE
initial_state = np.kron(spin_up, spin_up) # this is both in the up state

print(initial_state)

solution= solve_ivp(schrodinger_equation, t_span, initial_state, t_eval=t_eval, ) 



# now to find the populations for each we have to take tensor products of z with I and vice verse



# still not sure what the best practice with passing every variable consievable is
# other thing is that I need to have the hamiltonian for all time steps

def find_total_hamiltonian(number_qubits):
    final_hamiltonian = []
    for initial_state in np.eye(2**number_qubits, dtype=complex):
        final_hamiltonian.append(solve_ivp(schrodinger_equation, t_span, initial_state, t_eval=t_eval,).y.T[-1])

    return np.array(final_hamiltonian)

total_ham = find_total_hamiltonian(num_qubits)
print('fidelity')
print(calculate_fidelity(np.kron(I, I), total_ham, 2**num_qubits))



print(solution.y.T[-1])

print(solution.y.T[-1] @ (np.kron(Sz, I) @ solution.y.T[-1]))


z_expectation_1 = []
z_expectation_2 = []
for a in range(num_steps):
    z_expectation_1.append(solution.y.T[a].conj() @ (np.kron(Sz, I) @ solution.y.T[a]))
    z_expectation_2.append(solution.y.T[a].conj() @ (np.kron(I, Sz) @ solution.y.T[a]))


plt.plot(solution.t, z_expectation_1, label='1')
plt.plot(solution.t, z_expectation_2, label='2')
plt.xlabel('Time (s)')
plt.ylabel('$\\langle\\sigma_z\\rangle$ ')
plt.title('Rabi Oscillations Electron Spin')
plt.legend()
plt.show()





