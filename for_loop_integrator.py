import numpy as np
from scipy.constants import hbar, physical_constants
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



# Get the values for mu_B (Bohr magneton) and g_s (electron g-factor)
mu_B = physical_constants['Bohr magneton'][0]  # Bohr magneton in J/T
g_s = physical_constants['electron g factor'][0]  # Electron g-factor





# Pauli Spin Matrices
Sx = np.matrix([[0,1],[1,0]])
Sy = np.matrix([[0,-1j],[1j,0]])
Sz = np.matrix([[1,0],[0,-1]])

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




gyromagnetic_ratio = g_s*mu_B/hbar 
B_0= 2
B_x = 0.2


omega_0 = gyromagnetic_ratio*B_0/hbar
omega = omega_0*0.99

# note the use of hbar is silly here
detuning = omega_0-omega
rabi = gyromagnetic_ratio*B_x/hbar


rwa_ham = (hbar/2)*np.matrix([[-detuning,2*rabi],[2*rabi, detuning]])

def non_rwa_ham(t, hbar, omega):
    return (g_s*mu_B/(2)) * np.array([
        [B_0, B_x*np.cos(omega * t)],
        [B_x*np.cos(omega * t), -B_0]
    ])





 
time_steps = np.arange(0, 1000, 1)

rwa_spin = Ket(1, 0) # start in up state
non_rwa_spin = Ket(1, 0) 
rwa_spins = []
non_rwa_spins = []

for a in time_steps:
    rwa_spin =rwa_spin-rwa_ham.dot(rwa_spin)* (1j/hbar)/10**12
    non_rwa_spin =non_rwa_spin-non_rwa_ham(a/10**12, hbar, omega) @ (non_rwa_spin)* (1j/hbar)/10**12


    rwa_spins.append(rwa_spin)
    # non_rwa_spins.append(non_rwa_spin)





# this appears to be correct? but the integration is unstable



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
    ax.quiver(0, 0, 0, expectation(Sx, rwa_spins[int(frame)]), expectation(Sy, rwa_spins[int(frame)]), expectation(Sz, rwa_spins[int(frame)]), 
              color='r', arrow_length_ratio=0.1)
    # ax.quiver(0, 0, 0, expectation(Sx, non_rwa_spins[int(frame)]), expectation(Sy, non_rwa_spins[int(frame)]), expectation(Sz, non_rwa_spins[int(frame)]), 
              # color='g', arrow_length_ratio=0.1)


# Create the animation
ani = FuncAnimation(fig, update, frames=time_steps, interval=500)


# plt.plot(time_steps, np.array([vec[0] for vec in rwa_spins]), label='ψ1 (State 1)')
# plt.plot(time_steps, np.array([vec[1] for vec in rwa_spins]), label='ψ2 (State 2)')
# plt.xlabel('Time')
# plt.ylabel('State values')
# plt.title('Integration of ODEs')
# plt.legend()
# plt.show()




# Show the animation
plt.show()









