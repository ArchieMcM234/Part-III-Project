import numpy as np
import scipy as sp


# Pauli Spin Matrices
Sx = 0.5*np.matrix([[0,1],[1,0]])
Sy = 0.5*np.matrix([[0,-1j],[1j,0]])
Sz = 0.5*np.matrix([[1,0],[0,-1]])

S_plus = Sx+1j*Sy
S_minus = Sx-1j*Sy


spin = 1/(2)**(1/2)*np.array([[1], [1]])

def Ket(a, b):
    """
    Constructs a ket vector |a, b>.
    """
    return np.array([[a], [b]])

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

# def Operator(matrix):
#     """
#     Constructs an operator from a given 2D matrix.
#     """
#     return np.array(matrix)

# def Inner(B, K):
#     """
#     Computes the inner product <B|K> between a bra B and a ket K.
#     Returns a scalar.
#     """
#     return np.dot(B, K)[0, 0]  # Extracts the scalar from a 1x1 array

# def Outer(K, B):
#     """
#     Computes the outer product |K><B| between a ket K and a bra B.
#     Returns a 2D array representing an operator.
#     """
#     return np.dot(K, B)



spin_up = Ket(1, 0)
spin_down = Ket(0, 1)




# print('S+')
# print(S_plus)
print(Bra(spin_up).dot(S_plus.dot(spin_down))[0,0])




# this should be the same as expectation
print('expectation')
print(Bra(spin_up).dot((Sy.dot(spin_up))))



from qiskit.visualization import plot_bloch_vector     
import matplotlib.pyplot as plt

def expectation(operator, state):
	return Bra(state).dot((operator.dot(state)))[0,0]

plot_bloch_vector([expectation(Sx, spin_up),expectation(Sy, spin_up),expectation(Sz, spin_up)], title="New Bloch Sphere");
plt.show()

#we probably need to have a normalise thing?


#so we start with a spin
#we need to find the equation we are integrating
#define the mag field, magnetic dipole etc 
#accurate splitting they said was important




#we need two paths rwa and non RWA

#we want a nice way to define the function/hamiltonian we are working with

#they said try heisenberg and shrodinger picture - not too important

#analytic solutions to compare

