import numpy as np

from simulation.system import NBodySystem
from simulation.integrator import step

def run_simulation(
    positions,
    velocities,
    masses,
    dt=0.01,
    T=2000
):
    system = NBodySystem(positions, velocities, masses)

    trajectories = []

    for _ in range(T):
        step(system, dt)
        trajectories.append(system.positions.copy())
    
    return np.array(trajectories)

if __name__ == "__main__":
    positions = [
        [0, 0],
        [1, 0],
        [-1, 0]
    ]

    velocities = [
        [0, 0],
        [0, 10],
        [0, -11]
    ]

    masses = [100, 1, 1]

    traj = run_simulation(positions, velocities, masses)

    np.save("data/run_000.npy", traj)

    print("Saved trajectory:", traj.shape)