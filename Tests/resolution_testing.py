import numpy as np
from scipy.integrate import solve_ivp
from scipy.constants import hbar, physical_constants
import matplotlib.pyplot as plt





################################################################################################
# Constants and Operators
################################################################################################


# Get the values for mu_B (Bohr magneton) and g_s (electron g-factor)
mu_B = physical_constants['Bohr magneton'][0]  # Bohr magneton in J/T
g_s = physical_constants['electron g factor'][0]  # Electron g-factor





# Pauli Spin Matrices
Sx = np.array([[0,1],[1,0]])
Sy = np.array([[0,-1j],[1j,0]])
Sz = np.array([[1,0],[0,-1]])

S_plus = Sx+1j*Sy
S_minus = Sx-1j*Sy




################################################################################################
# Some (occasionally) helpful notation
################################################################################################

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
# Define System 
################################################################################################


B_0= 2
B_x = 0.2


omega_0 = g_s*mu_B*B_0/hbar # this is the zeeman splitting
omega = omega_0*1
# Rabi = 2*g_s*mu_B*B_x/hbar
Rabi = g_s*mu_B*B_x/hbar 


# def non_rwa_ham(t):
#     return (g_s*mu_B) * np.array([
#         [B_0/2, 2*B_x*np.cos(omega * t)],
#         [2*B_x*np.cos(omega * t), -B_0/2]
#     ])


# trying to write it interms of pauli matrices
bodge = 1
def non_rwa_ham(t):
    return g_s*mu_B*B_0/2*Sz +bodge*g_s*mu_B*B_x*Sx*np.cos(omega*t)





# way of thinking about some of the /2s is thta the spin is 1/2 so you can only split by a total of 1

# is there a factor of 2 in there - including a factor of 2 made it work - but i dont know why


# def rwa_ham(t):
#     return  np.array([
#         [-(hbar/2)*(omega_0-omega), g_s*mu_B*B_x],
#         [g_s*mu_B*B_x, (hbar/2)*(omega_0-omega)]
#         ])

def rwa_ham(t):
    return (hbar/2)* np.array([
        [-(omega_0-omega), Rabi],
        [Rabi, (omega_0-omega)]
        ])

################################################################################################
# Solve Numerically
################################################################################################


def model(t, psi):
    # returns dpsi/dt from TDSE
    return -( rwa_ham(t)@ psi)* 1j/hbar 



for a in [10, 100, 1000]:


    rwa_spin = np.array([1+ 0j, 0 + 0j]) # so this is in the up down basis 1,0  and 0,1
    # Time span for the integration
    t_span = (0, 10**-10)  # From t=0 to t=10
    t_eval = np.linspace(0, 10**-10, 1000)  # Points at which to evaluate the solution

    solution = solve_ivp(model, t_span, rwa_spin, t_eval=t_eval, atol = 10**-12, rtol=10**-12)

    effective_rabi =  (Rabi**2+(omega_0-omega)**2)**(1/2)

    analytic_solution = 1-2*(Rabi**2/effective_rabi**2)*np.sin(effective_rabi* t_eval/2)**2 


    hermitian_conjugates = solution.y.conj().T  # Shape will be (N, 2)
    z_expectation = np.einsum('ij,ij->i', hermitian_conjugates, (Sz @ solution.y).T)
    abs_error = (analytic_solution-z_expectation)
    plt.plot(solution.t, abs_error)
    plt.xlabel('Time (s)')
    plt.ylabel('$Error$ ')
    plt.title('Error with time - num time points '+str(a))
    plt.legend()
    plt.show()



################################################################################################
# Plotting and animating
################################################################################################

print('rabi ', Rabi, Rabi/(2*np.pi))
print('omega_0', omega_0, omega_0/(2*np.pi))


# Extract the expectation values

transformed_vectors = Sz @ solution.y # Matrix-vector product over all vectors

# Compute the Hermitian conjugates of the original vectors (transpose and conjugate)
hermitian_conjugates = solution.y.conj().T  # Shape will be (N, 2)

# Dot product between each Hermitian conjugate and transformed vector
z_expectation = np.einsum('ij,ij->i', hermitian_conjugates, (Sz @ solution.y).T)


plt.plot(solution.t, analytic_solution, label='Analytic')
plt.plot(solution.t, z_expectation, label='Numerical')
plt.xlabel('Time (s)')
plt.ylabel('$\\langle\\sigma_z\\rangle$ ')
plt.title('Rabi Oscillations Electron Spin')
plt.legend()
plt.show()








