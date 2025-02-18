import numpy as np


# was making sure I understood the new basis states as a tensor prod of single ones
# result of this is not useful
# but we can keep them as tensor products in our heads? unless they arent seperable?


v1 = np.array([1, 0])
v2 = np.array([0, 1])




single_basis_states = np.array([v1, v2])



def find_basis_states(num, new_basis_states, single_basis_states):
	new_new_basis_states = []
	if num>1:
		for a in range(len(new_basis_states)):
			for b in range(len(single_basis_states)):
				new_new_basis_states.append(np.kron(new_basis_states[a], single_basis_states[b]))
				print(new_basis_states[a], single_basis_states[b])
				print(np.kron(new_basis_states[a], single_basis_states[b]))
				print('\n')

		new_new_basis_states = np.array(new_new_basis_states)
		new_new_basis_states = find_basis_states(num-1, new_new_basis_states, single_basis_states)

	return new_new_basis_states

find_basis_states(2, single_basis_states, single_basis_states)