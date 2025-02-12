import numpy as np
from scipy.integrate import solve_ivp
from scipy.constants import hbar, physical_constants
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation




def no_damping(t, vec):
  return -np.cross(np.array([1,0, 10]), vec)


def adiabatic(t, vec):
  return -np.cross(np.array([1,0, 10*t/(50*2*np.pi)-10]), vec)

def non_adiabatic(t, vec):
  return -np.cross(np.array([1,0, 10*t/(2.5*2*np.pi)-10]), vec)


def damped(t, vec):
  return -np.cross(np.array([1,0, 0.1]), vec)-0.1*np.array([vec[0]/2, vec[1]/2, vec[2]+1])



def model(t, vec):
    return damped(t, vec)




# Initial conditions
bloch_vector = np.array([0, 0, -1])

# Time span for the integration
t_span = (0, 20*2*np.pi)  # From t=0 to t=10
t_eval = np.linspace(0, 20*2*np.pi, 500)  # Points at which to evaluate the solution

# Solve the ODE
solution = solve_ivp(model, t_span, bloch_vector, t_eval=t_eval)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])


print(solution.y)

# Function to update the Bloch vector
def update(frame):
    ax.cla()  # Clear the previous frame
    ax.set_title("Animated Bloch Vector")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    

    # Plot the Bloch vector
    ax.quiver(0, 0, 0, solution.y[0][frame], solution.y[1][frame], solution.y[2][frame], 
              color='r', arrow_length_ratio=0.1)
    # ax.quiver(0, 0, 0, expectation(Sx, non_rwa_spins[int(frame)]), expectation(Sy, non_rwa_spins[int(frame)]), expectation(Sz, non_rwa_spins[int(frame)]), 
              # color='g', arrow_length_ratio=0.1)


# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(0, 500, 1), interval=1)

plt.show()
