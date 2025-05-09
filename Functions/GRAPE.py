#!/usr/bin/env python
"""
GRAPE optimisation of a √X pulse on qubit-0
in the “CCD” drive scheme, with qubit-1 detuned (crosstalk modelled).
Author:  <your-name>, 2025-05-05
"""

import numpy as np
import qutip as qt
from qutip.control import pulseoptim, OptimResult

# ---------------------------------------------------------------------------
# 1)  Physical / control parameters  (change as needed)
# ---------------------------------------------------------------------------

# System parameters (angular frequencies, in 2π·MHz units for concreteness)
delta_0   = 0.0           # qubit-0 detuning in its rotating frame
delta_1   = 0.02          # qubit-1 detuning  (makes it a spectator)
omega_max = 0.005          # maximum drive Rabi rate Ω_max
omega_m   = 2.0           # modulation frequency ω_m  (for CCD term)
eps_m     = 0.2           # modulation depth ε_m  (dimensionless)
theta_m   = 0.0           # modulation phase θ_m
phi_0     = 0.0           # fixed carrier phase φ_0
eta_xtalk = 0.03          # fraction of the drive that leaks to qubit-1

# Optimisation grid
n_ts      = 200           # number of time slices
evo_time  = 0.04          # total gate time in µs  (→ 25 ns per slice)

# GRAPE hyper-parameters
max_iter     = 800
fid_err_targ = 1e-4
max_ctrl_amp = 1.0        # allow ≥ Ω_max after scaling below?

# ---------------------------------------------------------------------------
# 2)  Pauli operators and tensor helpers
# ---------------------------------------------------------------------------

I = qt.qeye(2)
sx, sy, sz = qt.sigmax(), qt.sigmay(), qt.sigmaz()

def op0(op):
    """Operator acting on qubit-0."""
    return qt.tensor(op, I)

def op1(op):
    """Operator acting on qubit-1."""
    return qt.tensor(I, op)

# ---------------------------------------------------------------------------
# 3)  Drift and control Hamiltonians
# ---------------------------------------------------------------------------

# Drift:  −δ/2 · σ_z  on each qubit (lab frame → rotating frame simplification)
H_drift = (-delta_0/2) * op0(sz)  +  (-delta_1/2) * op1(sz)

# Control 0: σ_x  drive on qubit-0  (+ crosstalk on q1)
H_c0 = op0(sx) + eta_xtalk * op1(sx)

# Control 1: σ_y  drive on qubit-0  (+ crosstalk on q1)
H_c1 = op0(sy) + eta_xtalk * op1(sy)

# Control 2: σ_z modulation term used in the CCD scheme
H_c2 = op0(sz)                                  # (no drive on q1)

# Bundle controls
ctrls = [H_c0, H_c1, H_c2]

# ---------------------------------------------------------------------------
# 4)  Target unitary:  √X on qubit-0 ⊗ I on qubit-1
# ---------------------------------------------------------------------------

rootx = (1/np.sqrt(2)) * np.array([[1, 1j],
                                   [1j, 1]], dtype=complex)
U_target = qt.tensor(qt.Qobj(rootx), I)

# ---------------------------------------------------------------------------
# 5)  Build the optimisation problem
# ---------------------------------------------------------------------------

# Initial control amplitudes:  two quadratures at carrier phase φ_0,
# plus a cos(ω_m t − θ_m) guess for the modulation channel
tlist = np.linspace(0.0, evo_time, n_ts, endpoint=False)        # left-point grid
init_amps = np.zeros((len(ctrls), n_ts))

# carrier envelope for σ_x / σ_y (flat-top with Gaussian edges)
gauss_len = int(0.15 * n_ts)
flat_len  = n_ts - 2*gauss_len
edge = np.sin(np.linspace(0, np.pi/2, gauss_len))**2            # smooth edges
envelope = np.concatenate([edge, np.ones(flat_len),
                           edge[::-1]])
init_amps[0, :] = omega_max * np.cos(phi_0) * envelope
init_amps[1, :] = omega_max * np.sin(phi_0) * envelope

# CCD modulation: ε_m (ω_m/Ω) cos(ω_m t − θ_m)
ccd_coeff = eps_m * (omega_m/omega_max)
init_amps[2, :] = ccd_coeff * np.cos(2*np.pi*omega_m*tlist - theta_m)

# ---------------------------------------------------------------------------
# 6)  Run GRAPE
# ---------------------------------------------------------------------------

result: OptimResult = pulseoptim.optimize_pulse_unitary(
    H_drift, ctrls, U_target,
    n_ts, evo_time,
    fid_err_targ=fid_err_targ,
    init_pulse_type='SUPPLIED',
    init_amplitudes=init_amps,
    max_iter=max_iter,
    amp_lbound=-max_ctrl_amp,
    amp_ubound=max_ctrl_amp,
    method='GRAPE'
)

# ---------------------------------------------------------------------------
# 7)  Inspect/output the results
# ---------------------------------------------------------------------------

print("\n— GRAPE finished —")
print(f" Fidelity error target    : {fid_err_targ}")
print(f" Achieved fidelity error  : {result.fid_err: .3e}")
print(f" Completed iterations     : {result.num_iter}")
print(f" Termination reason       : {result.termination_reason}\n")

# Plot the optimised controls (requires matplotlib)
try:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 6))
    labels = [r'$\sigma_x$', r'$\sigma_y$', r'$\sigma_z$ (CCD)']
    for k, ax in enumerate(axes):
        ax.step(tlist, result.final_amps[k], where='post')
        ax.set_ylabel(labels[k])
        ax.grid(True, which='both', ls=':')
    axes[-1].set_xlabel("Time (µs)")
    plt.tight_layout()
    plt.show()
except ImportError:
    print("matplotlib not installed; skipping plots.")

# Save the controls to disk for later use
np.savez("ccd_rootx_controls.npz",
         tlist=tlist,
         amps=result.final_amps)
print("Optimised pulse saved to ccd_rootx_controls.npz")