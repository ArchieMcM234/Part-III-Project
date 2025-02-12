import numpy as np


Sx = np.array([[0,1],[1,0]])
Sy = np.array([[0,-1j],[1j,0]])
Sz = np.array([[1,0],[0,-1]])

S_plus = Sx+1j*Sy
S_minus = Sx-1j*Sy




v1 = np.array([1, 0])
v2 = np.array([0, 1])

print(np.kron(v2, v1 ))




def rotation_x(theta):
    # Pauli X matrix
    sigma_x = np.array([[0, 1],
                        [1, 0]])
    
    # Rotation operator e^(-i * theta / 2 * sigma_x)
    rotation_matrix = np.cos(theta / 2) * np.eye(2) - 1j * np.sin(theta / 2) * sigma_x
    
    return rotation_matrix

# Example usage:
theta = np.pi  # Example angle, could be any value
rot_mat = rotation_x(theta)
print(rot_mat)


print(print(np.exp(0+Sx*(np.pi/2)*1j)))

theta = np.arccos(5**(-1/4))  # Replace with the value of theta


# Define the vector

for j in range(5):
	vector_1 = np.array([
	    np.sin(theta) * np.sin(((2 * j + 1) % 5) * (2 * np.pi / 5)),
	    np.sin(theta) * np.cos(((2 * j + 1) % 5) * (2 * np.pi / 5)),
	    np.cos(theta)
	])
	vector_2 = np.array([
	    np.sin(theta) * np.sin(((2 * (j+1) + 1) % 5) * (2 * np.pi / 5)),
	    np.sin(theta) * np.cos(((2 * (j+1) + 1) % 5) * (2 * np.pi / 5)),
	    np.cos(theta)
	])

	print(vector_1 @ vector_2)
