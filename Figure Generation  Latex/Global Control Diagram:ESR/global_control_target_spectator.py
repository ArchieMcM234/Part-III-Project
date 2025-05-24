import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import os
import sys

# --- PATH modification for TinyTeX (BEGIN) ---
tinytex_bin_path = os.path.expanduser("~/Library/TinyTeX/bin/universal-darwin")
if os.path.exists(tinytex_bin_path):
    if tinytex_bin_path not in os.environ['PATH']:
        os.environ['PATH'] = tinytex_bin_path + os.pathsep + os.environ['PATH']
        print(f"Added {tinytex_bin_path} to PATH for this Python session.")
    else:
        print(f"{tinytex_bin_path} already in PATH for this session.")
else:
    print(f"Warning: TinyTeX bin path not found at {tinytex_bin_path}. LaTeX rendering might fail.")
# --- PATH modification for TinyTeX (END) ---

# --- Matplotlib Configuration for LaTeX Integration ---
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 12,
    "axes.titlesize": 10,
})

# --- Figure Dimensions for Your LaTeX Document ---
pt_to_inches = 1.0 / 72.27
latex_textwidth_pts = 469.75502
latex_fig_width_inches = 0.8 * latex_textwidth_pts * pt_to_inches  # Use 0.8 textwidth for good fit
latex_fig_height_inches = latex_fig_width_inches  # Make it square for the 3D plot to fill

# Build figure with GridSpec
fig = plt.figure(figsize=(latex_fig_width_inches, latex_fig_height_inches))
gs = GridSpec(2, 2, height_ratios=[4, 1], hspace=0.4, wspace=0.3)

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
natural_freq = 10  # 10ghz
driving_freq = 10
rabi_freq = 0.005

# CCD parameters
phi_0, epsilon_m, phase_freq, theta_m = 0, rabi_freq/4, rabi_freq, 0

initial_state = np.array([1+0j, 0+0j])  # |0> state

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


# Top: 3D Bloch spheres
ax = fig.add_subplot(gs[0, :], projection='3d') # Still targets the top GridSpec cell
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
theta_esr = np.linspace(0, 2*np.pi, 100)
for x0 in (-offset, 0, offset):
    y_loop = r * np.cos(theta_esr)
    z_loop = z_wire[0] + r * np.sin(theta_esr)
    x_loop = np.full_like(theta_esr, x0)
    ax.plot(x_loop, y_loop, z_loop, color='red', linestyle='--', linewidth=1)
    th0 = np.pi/4
    ax.quiver(
        x0,
        r * np.cos(th0),
        z_wire[0] + r * np.sin(th0),
        0, -np.sin(th0), np.cos(th0),
        length=0.1, normalize=True, arrow_length_ratio=0.3, color='red'
    )

# Labels - Using specified colors. Removed fontsize=16 as rcParams sets the base.
ax.text(-offset, 1.2, 0.8, "Target Qubit", color=steel_blue, ha='center')
ax.text(offset, 1.2, 0.8, "Spectator Qubit", color=plum, ha='center')
ax.text(0, 0, z_wire[0] - 0.4, "ESR antenna", color='red', ha='center')

# Aspect and limits
ax.set_box_aspect((1, 1, 1)) # This is still key to making the data space cubic
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)
ax.set_axis_off()


plt.tight_layout()
fig.subplots_adjust(bottom=0.1)

# --- Save the plot as PDF ---
plt.savefig('global_control_esr_figure.pdf', bbox_inches='tight', dpi=300)
plt.close(fig) # Close the figure to free up memory

print("Figure 'global_control_esr_figure.pdf' has been generated and saved at LaTeX-appropriate size.")
print("Include it in your LaTeX document using: \\includegraphics[width=\\columnwidth]{global_control_esr_figure.pdf}")