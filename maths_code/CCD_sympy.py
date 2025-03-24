import sympy as sp 




# Pauli matrices
sigma_z = sp.Matrix([[1, 0], [0, -1]])
sigma_x = sp.Matrix([[0, 1], [1, 0]])
sigma_y = sp.Matrix([[0, -sp.I], [sp.I, 0]])

# Define variables
t, omega, omega_0, Omega, phi_0, epsilon, omega_m, theta_m, sig_sym_x, sig_sym_y, sig_sym_z = sp.symbols(
    't omega omega_0 Omega phi_0 epsilon omega_m theta_m sigma_x sigma_y sigma_z', real=True
)
hbar = sp.Symbol('hbar', real=True)  # Reduced Planck constant

# Define the unitary matrix U
# This is the simple to omega frame
# U = sp.Matrix([
#     [sp.exp(-sp.I * omega * t / 2), 0],
#     [0, sp.exp(sp.I * omega * t / 2)]
# ])

# The oscillating rotating frame
t, t_prime, hbar, omega, epsilon, Omega, Theta = sp.symbols('t t_prime hbar omega epsilon Omega Theta', real=True)
sigma_z = sp.Matrix([[1, 0], [0, -1]])

# Define H(t)
frame_ham = hbar*(omega / 2 - (epsilon * omega / Omega) * sp.cos(omega * t - Theta)) * sigma_z

# Define U(t)
U = sp.exp(-sp.I / hbar * sp.integrate(frame_ham, (t_prime, 0, t)))

# Define the lab frame Hamiltonian
# Phase modulated CCD Hamiltonian
cos_argument = omega * t + phi_0 - (2 * epsilon / Omega) * sp.sin(omega_m * t - theta_m)
cos_exp_argument = (sp.exp(sp.I * cos_argument) + sp.exp(-sp.I * cos_argument)) / 2

H_lab = sp.Matrix([
    [hbar * omega_0 / 2, Omega * cos_exp_argument],
    [Omega * cos_exp_argument, -hbar * omega_0 / 2]
])

# Construct the Hamiltonian matrix
gamma, Bx = sp.symbols('gamma Bx')
H_lab = (hbar * gamma * Bx / 2) * sigma_x * (sp.exp(sp.I * omega * t) + sp.exp(-sp.I * omega * t)) / 2 + sp.Matrix([[hbar * omega_0 / 2, 0], [0, -hbar * omega_0 / 2]])

# Compute U^dagger (Hermitian conjugate of U)
U_dagger = U.H

# Compute d(U^dagger)/dt
U_dagger_dt = sp.diff(U_dagger, t)

# Full expression
H = U * H_lab * U_dagger + sp.I * hbar * U * U_dagger_dt

# Expand the Hamiltonian
H = H.expand(trig=True) 

# Convert to LaTeX string
latex_str = sp.latex(H)
print(latex_str)

# Simplify Hamiltonian by removing fast oscillating terms
H[0, 1] = H[0, 1].subs(sp.exp(-2 * sp.I * omega * t), 0)
H[1, 0] = H[1, 0].subs(sp.exp(2 * sp.I * omega * t), 0)

# Display the simplified expression
H = sp.simplify(H)

# Simplify Hamiltonian in terms of Pauli matrices
expression = sp.simplify(sp.trace(H.H * sigma_z)) * sig_sym_z + sp.simplify(sp.trace(H.H * sigma_x)) * sig_sym_x + sp.simplify(sp.trace(H.H * sigma_y)) * sig_sym_y

# Convert to LaTeX string
latex_str = sp.latex(expression)
print(latex_str)






