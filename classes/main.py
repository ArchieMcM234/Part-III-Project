import numpy as np
from scipy.integrate import solve_ivp
from scipy.constants import hbar, physical_constants
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



from quantum_classes import *

# a note - i need to extend code to allow different tunings for each qubit - there must be a nice way of doing this

# natural_freq, driving_freq, rabi_freq = 18*10**9, 18*10**9, 5*10**6

# hamiltonian = Quantum_Hamiltonian(natural_freq, rabi_freq)

# system = Quantum_System(hamiltonian, 1)


##########################################################################################
# Rabi Oscillations
##########################################################################################

# # Points at which to evaluate the solution
# initial_state = np.array([1 + 0j, 0 + 0j])
# t, y = system.evolve_state(initial_state, 10**-6, 1000, driving_freq)


# effective_rabi =  (rabi_freq**2+(natural_freq-driving_freq)**2)**(1/2)

# analytic_solution = 1-2*(rabi_freq**2/effective_rabi**2)*np.sin(2*np.pi*effective_rabi* t/2)**2 


# plt.plot(t, analytic_solution, label='Analytic')
# plt.plot(t, calculate_z_expectation(y), label='Numerical')
# plt.xlabel('Time (s)')
# plt.ylabel('$\\langle\\sigma_z\\rangle$ ')
# plt.title('Rabi Oscillations Electron Spin')
# plt.legend(loc='upper right')

# # Add metadata/caption with relevant frequencies
# metadata = (f"Natural Frequency: {natural_freq/1e9} GHz\n"
#             f"Driving Frequency: {driving_freq/1e9} GHz\n"
#             f"Rabi Frequency: {rabi_freq/1e6} MHz")

# # Positioning the metadata in the top-left corner
# plt.text(0.5, -0.2, metadata, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', ha='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
# # Save the figure with the caption
# plt.show()

##########################################################################################
# Rabi Chevrons
##########################################################################################


# U_ideal = np.eye(2, 2)


# driving_freq_array = np.linspace(natural_freq*0.999, natural_freq*1.001, 100)


# freq_fids = []

# for driving_freq in driving_freq_array:
# 	t, hamiltonians = system.find_total_hamiltonians(3*10**-6, 1000, driving_freq, ham_type='rwa')
# 	fidelities = calculate_fidelitys(U_ideal, hamiltonians, 2)

# 	freq_fids.append(fidelities)

# plt.figure(figsize=(8, 6))
# plt.imshow(np.array(freq_fids).T, aspect="auto", 
#            extent=[(driving_freq_array[-1]-natural_freq), (driving_freq_array[0]-natural_freq), t[0], t[-1]], 
#            origin="lower", cmap="viridis")


# # Add colorbar and labels
# plt.colorbar(label="Fidelities")
# plt.xlabel("Detuning/s^-1")
# plt.ylabel("Time/s")
# plt.title("Fidelities vs Time and Omega RWA")



# plt.show()

##########################################################################################
# multiple qubits
##########################################################################################

hamiltonian = Quantum_Hamiltonian(5*10**6, 20*10**9, 2, np.array([20*10**9, 15**9]))

print(hamiltonian.rwa())









#todos
#driving freq should be in the rwa function to start with




