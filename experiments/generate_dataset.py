from generate_run import run_simulation
import numpy as np
import os

os.makedirs("data/dataset", exist_ok=True)

N_SAMPLES = 50

def sample_system():
    positions = [
        [0, 0],
        np.random.uniform(-1, 1, 2).tolist(),
        np.random.uniform(-1, 1, 2).tolist()
    ]

    velocities = [
        [0, 0],
        np.random.uniform(-5, 5, 2).tolist(),
        np.random.uniform(-5, 5, 2).tolist()
    ]

    masses = [100, 1, 1]

    return positions, velocities, masses

dataset = []

for i in range(N_SAMPLES):
    positions, velocities, masses = sample_system()

    traj = run_simulation(positions, velocities, masses)

    dataset.append({
        "positions": positions,
        "velocities": velocities,
        "masses": masses,
        "trajectory": traj
    })

    np.save(f"data/dataset/run_{i:03d}.npy", traj)

print("Done:", len(dataset))