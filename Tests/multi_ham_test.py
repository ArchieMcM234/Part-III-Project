import numpy as np



# Pauli Spin Matrices
Sx = np.array([[0,1],[1,0]])
Sy = np.array([[0,-1j],[1j,0]])
Sz = np.array([[1,0],[0,-1]])
I = np.array([[1, 0], [0, 1]])


up = np.array([1, 0]) # define this as up state
down = np.array([0, 1])




state = np.kron(up, up)

				
print(np.kron(Sx, I))

print((np.kron(Sx, I) @ state) )

