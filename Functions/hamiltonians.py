import numpy as np

from functools import reduce
from pauli_matrices import *


def hz_to_radians(freq_hz):
    """Convert frequency in Hz to angular frequency in radians/second"""
    return 2 * np.pi * freq_hz

def kron_multiple_arrays(array_list):
    """Compute Kronecker product of multiple arrays"""
    return reduce(np.kron, array_list)



def idle(t):
    """Return idle Hamiltonian"""
    return np.zeros((2, 2))

def rwa(natural_freqs, rabi_freq, driving_freq, num_qubits, t):
    """
    Time-independent Hamiltonian in the rotating wave approximation (RWA).
    All frequencies should be in Hz.
    """
    natural_omegas = hz_to_radians(natural_freqs)
    rabi_omega = hz_to_radians(rabi_freq)
    driving_omega = hz_to_radians(driving_freq)

    H = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)

    for a in range(num_qubits):
        single_hamiltonian = (1/2) * np.array([
            [-(natural_omegas[a] - driving_omega), rabi_omega],
            [rabi_omega, natural_omegas[a] - driving_omega]])

        array_list = [identity for b in range(a)] + [single_hamiltonian] + [identity for b in range(num_qubits-a-1)]
        H += kron_multiple_arrays(array_list)

    return H

def ccd_rwa(natural_freq, rabi_freq, driving_freq, phase_freq, 
            phi_0, epsilon_m, theta_m, t):
    """
    Time-independent Hamiltonian in the RWA for CCD.
    All frequencies should be in Hz.
    """
    natural_omega = hz_to_radians(natural_freq)
    rabi_omega = hz_to_radians(rabi_freq)
    driving_omega = hz_to_radians(driving_freq)
    phase_omega = hz_to_radians(phase_freq)
    epsilon_m = hz_to_radians(epsilon_m)

    delta = driving_omega - natural_omega
    cos_phi_0 = np.cos(phi_0)
    sin_phi_0 = np.sin(phi_0)
    cos_phase = epsilon_m * (phase_omega / rabi_omega) * np.cos(phase_omega * t - theta_m)

    H = -(delta / 2) * sigma_z + (rabi_omega / 2) * (cos_phi_0 * sigma_x + sin_phi_0 * sigma_y) + cos_phase * sigma_z
    return H

def ccd_lab(natural_freq, rabi_freq, driving_freq, phase_freq,
            phi_0, epsilon_m, theta_m, t):
    """
    Time-dependent Hamiltonian in the lab frame for CCD.
    All frequencies should be in Hz.
    """
    natural_omega = hz_to_radians(natural_freq)
    rabi_omega = hz_to_radians(rabi_freq)
    driving_omega = hz_to_radians(driving_freq)
    phase_omega = hz_to_radians(phase_freq)
    epsilon_m = hz_to_radians(epsilon_m)

    cos_term = np.cos(driving_omega * t + phi_0 - (2 * epsilon_m / rabi_omega) * np.sin(phase_omega * t - theta_m))
    H = (natural_omega / 2) * sigma_z + rabi_omega * cos_term * sigma_x
    return H

def ccd_lab_multiple(natural_freqs, rabi_freq, driving_freq, phase_freq,
                     phi_0, epsilon_m, theta_m, t, coupling):
    """
    Time-dependent Hamiltonian in the lab frame for multiple qubits.
    All frequencies should be in Hz.
    """
    num_qubits = len(natural_freqs)
    natural_omegas = hz_to_radians(natural_freqs)
    rabi_omega = hz_to_radians(rabi_freq)
    driving_omega = hz_to_radians(driving_freq)
    phase_omega = hz_to_radians(phase_freq)
    epsilon_m = hz_to_radians(epsilon_m)

    H = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    for a in range(num_qubits):
        cos_term = np.cos(driving_omega * t + phi_0 - (2 * epsilon_m / rabi_omega) * np.sin(phase_omega * t - theta_m))
        single_hamiltonian = (natural_omegas[0] / 2) * sigma_z + rabi_omega * cos_term * sigma_x
        array_list = [identity for b in range(a)] + [single_hamiltonian] + [identity for b in range(a, num_qubits-1)]
        H += kron_multiple_arrays(array_list)
    
    return H + coupling * (np.kron(sigma_x, sigma_x) + np.kron(sigma_y, sigma_y) + np.kron(sigma_z, sigma_z))

def ccd_rwa_multiple(natural_freqs, rabi_freq, driving_freq, phase_freq, phi_0, epsilon_m, theta_m, t, coupling):
    """
    Time-independent Hamiltonian in the rotating wave approximation (RWA) for CCD.
    All frequencies should be in Hz.
    """
    num_qubits = len(natural_freqs)
    natural_omegas = hz_to_radians(natural_freqs)
    rabi_omega = hz_to_radians(rabi_freq)
    driving_omega = hz_to_radians(driving_freq)
    phase_omega = hz_to_radians(phase_freq)
    epsilon_m = hz_to_radians(epsilon_m)
    
    cos_phi_0 = np.cos(phi_0)
    sin_phi_0 = np.sin(phi_0)
    cos_phase = epsilon_m * (phase_omega / rabi_omega) * np.cos(phase_omega * t - theta_m)

    H = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    for a in range(num_qubits):
        delta = driving_omega - natural_omegas[a]

        single_hamiltonian = -(delta / 2) * sigma_z + (rabi_omega / 2) * (cos_phi_0 * sigma_x + sin_phi_0 * sigma_y) + cos_phase * sigma_z
        array_list = [identity for b in range(a)] + [single_hamiltonian] + [identity for b in range(a, num_qubits-1)]
        H += kron_multiple_arrays(array_list)    
    
    return H + coupling * (np.kron(sigma_x, sigma_x) + np.kron(sigma_y, sigma_y) + np.kron(sigma_z, sigma_z))


def smart_lab(natural_freq, rabi_freq, driving_freq, modulation_freq, t):
    """
    Time-dependent Hamiltonian in the lab frame for SMART protocol.
    All frequencies should be in Hz.
    """
    natural_omega = hz_to_radians(natural_freq)
    rabi_omega = hz_to_radians(rabi_freq)
    driving_omega = hz_to_radians(driving_freq)
    modulation_omega = hz_to_radians(modulation_freq)

    H = (1/2) * (natural_omega * sigma_z + 
                 rabi_omega * np.sqrt(2) * np.sin(modulation_omega * t) * 
                 2 * np.cos(driving_omega * t) * sigma_x)
    return H

def smart_rwa(natural_freq, rabi_freq, driving_freq, modulation_freq, t):
    """
    Time-independent Hamiltonian in the RWA for SMART protocol.
    All frequencies should be in Hz.
    """
    natural_omega = hz_to_radians(natural_freq)
    rabi_omega = hz_to_radians(rabi_freq)
    driving_omega = hz_to_radians(driving_freq)
    modulation_omega = hz_to_radians(modulation_freq)

    delta = driving_omega - natural_omega
    H = (1/2) * (delta * sigma_z + rabi_omega * np.sqrt(2) * np.sin(modulation_omega * t) * sigma_x)
    return H

def smart_rwa_multiple(natural_freqs, rabi_freq, driving_freq, modulation_freq, coupling, t):
    """
    Time-independent Hamiltonian in the rotating wave approximation (RWA) for CCD.
    All frequencies should be in Hz.
    """
    num_qubits = len(natural_freqs)
    natural_omegas = hz_to_radians(natural_freqs)
    rabi_omega = hz_to_radians(rabi_freq)
    driving_omega = hz_to_radians(driving_freq)
    modulation_omega = hz_to_radians(modulation_freq)

    

    H = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    for a in range(num_qubits):
        delta = driving_omega - natural_omegas[a]

        single_hamiltonian = (1/2) * (delta * sigma_z + rabi_omega * np.sqrt(2) * np.sin(modulation_omega * t) * sigma_x)
        array_list = [identity for b in range(a)] + [single_hamiltonian] + [identity for b in range(a, num_qubits-1)]
        H += kron_multiple_arrays(array_list)    
    
    return H + coupling * (np.kron(sigma_x, sigma_x) + np.kron(sigma_y, sigma_y) + np.kron(sigma_z, sigma_z))

