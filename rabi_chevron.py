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

S_plus = Sx+1j*Sy
S_minus = Sx-1j*Sy


def Ket(a, b):
    """
    Constructs a ket vector |a, b>.
    """
    return np.array([[a], [b]], dtype=complex)

def Bra(a, b=None):
    """
    Constructs a bra vector <a, b|.
    If only one argument is provided, it returns the conjugate transpose of that argument.
    """
    if b is None:
        # Treat a as a column vector (ket), take its conjugate transpose
        return a.conjugate().transpose()
    else:
        # Construct a 1x2 bra matrix explicitly
        return np.array([[a.conjugate(), b.conjugate()]])


def expectation(operator, state):
    return Bra(state).dot((operator.dot(state)))[0,0]






################################################################################################

# TODO

def rotating_frame_transformation():
    pass






################################################################################################





B_0= 2
B_x = 0.2


omega_0 = g_s*mu_B*B_0/hbar

omega_array = np.linspace(0.9*omega_0, 1.1*omega_0, 100)


################################################################################################
# Integration
################################################################################################


def non_rwa_ham(t):
    return (g_s*mu_B/(2)) * np.array([
        [B_0, B_x*np.cos(omega * t)],
        [B_x*np.cos(omega * t), -B_0]
    ])



def rwa_ham(t):
    return  np.array([
        [-(hbar/2)*(omega_0-omega), g_s*mu_B*B_x],
        [g_s*mu_B*B_x, (hbar/2)*(omega_0-omega)]
        ])


def model(t, psi):

    return -(rwa_ham(t)@ psi)* 1j/hbar # need to remember to divide by hbar tdse

# Initial conditions
rwa_spin = np.array([1 + 0j, 0 + 0j]) # start in up state

# Time span for the integration
t_span = (0, 10**-9)  # From t=0 to t=10
t_eval = np.linspace(0, 10**-9, 1000)  # Points at which to evaluate the solution

# Solve the ODE
solution = solve_ivp(model, t_span, rwa_spin, t_eval=t_eval)


# Extract the expectation values

transformed_vectors = Sz @ solution.y # Matrix-vector product over all vectors

# Compute the Hermitian conjugates of the original vectors (transpose and conjugate)
hermitian_conjugates = solution.y.conj().T  # Shape will be (N, 2)

# Dot product between each Hermitian conjugate and transformed vector
x_expectation = np.einsum('ij,ij->i', hermitian_conjugates, (Sx @ solution.y).T)
y_expectation = np.einsum('ij,ij->i', hermitian_conjugates, (Sy @ solution.y).T)
z_expectation = np.einsum('ij,ij->i', hermitian_conjugates, (Sz @ solution.y).T)


plt.plot(solution.t, solution.y[1].imag)
plt.plot(solution.t, z_expectation, label='z_expectation')
plt.xlabel('Time')
plt.ylabel('State values')
plt.title('Integration of ODEs')
plt.legend()
plt.show()



