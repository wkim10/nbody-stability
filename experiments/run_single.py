import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from simulation.system import NBodySystem
from simulation.integrator import step

positions = [
    [0, 0],
    [1, 0],
    [-1, 0]
]

# velocities = [
#     [0, 0],
#     [0, 31.6],
#     [0, -30.6]
# ]

velocities = [
    [0, 0],
    [0, 10],
    [0, -11]
]

# masses = [1000, 1, 1]

masses = [100, 1, 1]

system = NBodySystem(positions, velocities, masses)

N = len(masses)

trajectories = []

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

colors = plt.cm.tab10(np.linspace(0, 1, N))

lines = []
points = []

for i in range(N):
    c = colors[i]
    line, = ax.plot([], [], lw=1, color=c)
    point, = ax.plot([], [], 'o', color=c)
    lines.append(line)
    points.append(point)

def update(frame):
    step(system, 0.01)

    snapshot = system.positions.copy()
    trajectories.append(snapshot)

    for i in range(N):
        pos = system.positions[i]

        traj = np.array([t[i] for t in trajectories])

        lines[i].set_data(traj[:, 0], traj[:, 1])
        points[i].set_data([pos[0]], [pos[1]])
    
    return lines + points

ani = FuncAnimation(fig, update, frames=2000, interval=10)
plt.show()

trajectories = np.array(trajectories)

np.save("data/run_single.npy", trajectories)

print("Saved trajectory:", trajectories.shape)