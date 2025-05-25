

import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt

# lets me import from a different directory
import sys 
sys.path.append("/Users/archie/documents/year4/project/code")

import spin_tools.hamiltonians as hamiltonians
from spin_tools.gates import *
import spin_tools.quantum_tools as qt
from spin_tools.floquet_tools import ccd_floquet

steel_blue, plum = "#4682B4", "#B44682"

# Bloch sphere mesh
u, v = np.linspace(0, 2*np.pi, 100), np.linspace(0, np.pi, 100)
xs, ys = np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones_like(u), np.cos(v))

# Define initial state and system parameters
initial_state = (1/np.sqrt(2) )*np.array([1+0j, 1+0j])
natural_freq = 10  # GHz
driving_freq = 10
rabi_freq = 0.00
detuning = 0.001

tol = 1e-12
evaluation_points = 2001
evaluation_time = 8 / 0.005

# Define effective Hamiltonian vector
vec = np.array([rabi_freq, 0, detuning])
vec_norm = np.linalg.norm(vec)

# Generate states by applying effective unitary evolution
times = np.linspace(0, evaluation_time, evaluation_points)
states = []
times, Us = qt.calculate_unitaries(
    1, 
    evaluation_time, 
    evaluation_points, 
    hamiltonians.rabi_rwa, 
    natural_freq=natural_freq-detuning, 
    driving_freq=driving_freq, 
    rabi_freq=rabi_freq, 
    atol=tol, 
    rtol=tol
)

states= []
for U in Us:
    states.append(U @ initial_state)  

states = np.array(states)

# Compute Bloch coordinates
x = np.real([np.vdot(s, sigma_x @ s) for s in states])
y = np.real([np.vdot(s, sigma_y @ s) for s in states])
z = np.real([np.vdot(s, sigma_z @ s) for s in states])

# Plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xs, ys, zs, color='lightgrey', alpha=0.15, linewidth=0)
ax.plot(x, y, z, color=steel_blue, label='Qubit trajectory')

# plt the detuning and the rabi drive as a vector - aswell as them as components making up the drive vector
ax.quiver(0, 0, 0, -vec[0]/vec_norm, -vec[1]/vec_norm, vec[2]/vec_norm, color=steel_blue, label='Drive vector',)
ax.quiver(0, 0, 0, -rabi_freq/vec_norm, 0, 0, color=plum, label='Rabi drive',)
ax.quiver(-rabi_freq/vec_norm, 0, 0, 0, 0, detuning/vec_norm, color='orange', label='Detuning',)




ax.set_box_aspect((1, 1, 1))
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_axis_off()
plt.legend()
plt.show()