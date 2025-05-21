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

# ------------------------------------------------------------
# Generate the Bloch-sphere mesh once
u, v = np.linspace(0, 2*np.pi, 100), np.linspace(0, np.pi, 100)
xs, ys = np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones_like(u), np.cos(v))

# Example dummy trajectories for illustration
t = np.linspace(0, 2*np.pi, 200)
# First qubit: uniform rotation around the equator (z=0)
z_t = np.cos(t)
y_t = np.sin(t)
x_t = np.zeros_like(t)
# Second qubit: small oscillations around the north pole (z≈1)
x_s = 0.2 * np.cos(t)
y_s = 0.2 * np.sin(t)
z_s = np.ones_like(t) * 0.95

# Create real trajectories for the target and spectator qubits in CCD scheme
# ---------------------------------------------------------------------------------
# Create a quantum system with a single qubit
natural_freq = 10# 10ghz   
driving_freq = 10
rabi_freq = 0.005


# CCD parameters
phi_0, epsilon_m, phase_freq, theta_m = 0, rabi_freq/4, rabi_freq, 0

initial_state = np.array([1+0j, 0+0j])# |0> state

tol = 10**-7
evaluation_points = 3500
evaluation_time = 4/rabi_freq 

times, states_target = qt.evolve_state(initial_state, evaluation_time, evaluation_points,  hamiltonians.ccd_rwa, 
                natural_freq = natural_freq,
                driving_freq = driving_freq,
                rabi_freq = rabi_freq,
                phi_0 = phi_0,
                epsilon_m = epsilon_m,
                phase_freq = phase_freq,
                theta_m = theta_m,
                atol=tol,
                rtol=tol)


x_t = np.real(qt.calculate_expectations(states_target, sigma_x))
y_t = np.real(qt.calculate_expectations(states_target, sigma_y))
z_t = np.real(qt.calculate_expectations(states_target, sigma_z))

detuning = 3*rabi_freq

times, states_spectator = qt.evolve_state(initial_state, evaluation_time, evaluation_points,  hamiltonians.ccd_rwa, 
                natural_freq = natural_freq-detuning,
                driving_freq = driving_freq,
                rabi_freq = rabi_freq,
                phi_0 = phi_0,
                epsilon_m = epsilon_m,
                phase_freq = phase_freq,
                theta_m = theta_m,
                atol=tol,
                rtol=tol)


x_s = np.real(qt.calculate_expectations(states_spectator, sigma_x))
y_s = np.real(qt.calculate_expectations(states_spectator, sigma_y))
z_s = np.real(qt.calculate_expectations(states_spectator, sigma_z))

from matplotlib.gridspec import GridSpec

# Build figure with GridSpec
fig = plt.figure(figsize=(10, 12))
gs = GridSpec(2, 2, height_ratios=[4, 1], hspace=0.4, wspace=0.3)

# Top: 3D Bloch spheres
ax = fig.add_subplot(gs[0, :], projection='3d')
offset = 1.5  # sphere spacing

# Draw the two spheres
ax.plot_surface(xs - offset, ys, zs, color='lightgrey', alpha=0.15, linewidth=0)
ax.plot_surface(xs + offset, ys, zs, color='lightgrey', alpha=0.15, linewidth=0)

# Plot trajectories
ax.plot(x_t - offset, y_t, z_t, color=steel_blue, label='Target')
ax.plot(x_s + offset, y_s, z_s, color=plum, label='Spectator')

# Arrow to trajectory endpoints
ax.quiver(-offset, 0, 0, x_t[-1], y_t[-1], z_t[-1],
          length=1.0, normalize=False, arrow_length_ratio=0.2, color=steel_blue)
ax.quiver(offset, 0, 0, x_s[-1], y_s[-1], z_s[-1],
          length=1.0, normalize=False, arrow_length_ratio=0.2, color=plum)

# ESR antenna
x_wire = np.linspace(-offset*1.5, offset*1.5, 200)
y_wire = np.zeros_like(x_wire)
z_wire = np.full_like(x_wire, -1.3)
ax.plot(x_wire, y_wire, z_wire, color='red', linewidth=3)

r = 0.2
theta = np.linspace(0, 2*np.pi, 100)
for x0 in (-offset, 0, offset):
    y_loop = r * np.cos(theta)
    z_loop = z_wire[0] + r * np.sin(theta)
    x_loop = np.full_like(theta, x0)
    ax.plot(x_loop, y_loop, z_loop, color='red', linestyle='--', linewidth=1)
    th0 = np.pi/4
    ax.quiver(
        x0,
        r * np.cos(th0),
        z_wire[0] + r * np.sin(th0),
        0, -np.sin(th0), np.cos(th0),
        length=0.1, normalize=True, arrow_length_ratio=0.3, color='red'
    )

# Labels
ax.text(-offset, 1.2, 0.8, "Target Qubit", color=steel_blue, fontsize=16, ha='center')
ax.text(offset, 1.2, 0.8, "Spectator Qubit", color=plum, fontsize=16, ha='center')
ax.text(0, 0, z_wire[0] - 0.4, "ESR antenna", color='red', fontsize=16, ha='center')

# Aspect and limits
ax.set_box_aspect((1, 1, 1))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)
ax.set_axis_off()

# Bottom left: Target energy splitting
ax_e1 = fig.add_subplot(gs[1, 0])
E0, E1 = 0, 2
ax_e1.hlines([E0, E1], 0.2, 0.8, color=steel_blue, linewidth=2)
ax_e1.annotate("", (0.5, E1), (0.5, E0),
               arrowprops=dict(arrowstyle="<->", color=steel_blue))
ax_e1.text(0.5, (E0 + E1) / 2, r'$\omega_0$', ha='center', va='bottom', color=steel_blue)
ax_e1.set_xticks([]); ax_e1.set_yticks([]); ax_e1.set_ylim(-0.5, 2.5)
ax_e1.set_title("Target", color=steel_blue)

# Bottom right: Spectator energy splitting
ax_e2 = fig.add_subplot(gs[1, 1])
E0s, E1s = 0, 2.6
ax_e2.hlines([E0s, E1s], 0.2, 0.8, color=plum, linewidth=2)
ax_e2.annotate("", (0.5, E1s), (0.5, E0s),
               arrowprops=dict(arrowstyle="<->", color=plum))
ax_e2.text(0.5, (E0s + E1s) / 2, r'$\omega_0 + \delta$', ha='center', va='bottom', color=plum)
ax_e2.set_xticks([]); ax_e2.set_yticks([]); ax_e2.set_ylim(-0.5, 3.1)
ax_e2.set_title("Spectator", color=plum)

plt.tight_layout()
fig.subplots_adjust(bottom=0.1)
plt.show()
