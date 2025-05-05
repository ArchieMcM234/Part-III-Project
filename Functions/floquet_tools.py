import numpy as np
from pauli_matrices import *





# def compute_floquet_quasienergies(H_n_dict, omega, N=1):
#     """
#     Compute quasienergies from Fourier components of a time-periodic Hamiltonian.
    
#     Parameters:
#         H_n_dict: dict
#             A dictionary mapping integer n to 2×2 Fourier components H_n (numpy arrays).
#             Should include at least H_0 and a few harmonics like H_{±1}.
#         omega: float
#             Modulation frequency (in same units as Hamiltonian, e.g. GHz).
#         N: int
#             Truncation order: use harmonics from -N to +N.

#     Returns:
#         numpy array of sorted real parts of quasienergies (in same units as omega).
#     """
#     d = 2  # assume 2x2 Hamiltonians
#     dim = (2*N + 1) * d
#     HF = np.zeros((dim, dim), dtype=complex)

#     I2 = np.eye(2, dtype=complex)

#     for m in range(-N, N+1):
#         for n in range(-N, N+1):
#             i = (m + N) * d
#             j = (n + N) * d
#             block = H_n_dict.get(m - n, np.zeros((2,2), dtype=complex))
#             if m == n:
#                 block += m * omega * I2  # diagonal: add ω m I
#             HF[i:i+d, j:j+d] = block

#     eigvals = np.linalg.eigvals(HF)
#     return np.sort(eigvals.real)



# def floquet_for_CCD(delta, Omega, phi0, epsilon_m, theta_m):
#     H0  = -delta/2 * sigma_z + (Omega/2)*(np.cos(phi0)*sigma_x + np.sin(phi0)*sigma_y)
#     H1  = (epsilon_m*Omega/(2*Omega)) * np.exp(-1j*theta_m) * sigma_z
#     Hm1 = (epsilon_m*Omega/(2*Omega)) * np.exp(+1j*theta_m) * sigma_z

#     # Build Fourier component dict and compute all quasienergies
#     Hn = {0: H0, 1: H1, -1: Hm1}
#     eigs = compute_floquet_quasienergies(Hn, omega=Omega, N=1)

#     # For a 2-level system and N=1, the primary band is the middle two eigenvalues
#     d = H0.shape[0]
#     start = 1 * d
#     primary_band = eigs[start:start + d]

#     return primary_band


def ccd_floquet(natural_freq, driving_freq, rabi_freq, phi_0, epsilon_m, phase_freq, theta_m):
    """
    Compute Floquet quasienergies for a driven two-level system.

    Parameters:
        natural_freq: float
            Natural frequency of the system (GHz).
        driving_freq: float
            Driving frequency (GHz).
        rabi_freq: float
            Rabi frequency (GHz).
        phi_0: float
            Initial phase (radians).
        epsilon_m: float
            Modulation amplitude (GHz).
        phase_freq: float
            Modulation frequency (GHz).
        theta_m: float
            Modulation phase (radians).

    Returns:
        numpy.ndarray
            Sorted real parts of the Floquet quasienergies (GHz).
    """
    # Derived parameters
    delta = natural_freq - driving_freq  # Detuning in GHz
    Omega = rabi_freq                    # Rabi frequency in GHz

    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j,  0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    I2      = np.eye(2, dtype=complex)

    # Build H0, H1, H_{-1}
    H0       = -delta/2 * sigma_z \
            + (Omega/2) * (np.cos(phi_0)*sigma_x + np.sin(phi_0)*sigma_y)
    H1       = (epsilon_m*phase_freq/(2*Omega)) * np.exp(-1j*theta_m) * sigma_z
    H_minus1 = (epsilon_m*phase_freq/(2*Omega)) * np.exp( 1j*theta_m) * sigma_z
    omega_I  = phase_freq * I2

    # Assemble the 6×6 Floquet Hamiltonian
    HF = np.block([
        [H0 - omega_I,      H1,                   np.zeros((2,2), dtype=complex)],
        [H_minus1,          H0,                   H1],
        [np.zeros((2,2)),   H_minus1,             H0 + omega_I]
    ])

    # Diagonalize
    eigvals = np.linalg.eigvals(HF)

    # Sort and return
    return np.sort(eigvals.real)