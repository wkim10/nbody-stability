from generate_run import run_simulation
from simulation.constants import G
import numpy as np
import os
from tqdm import tqdm

os.makedirs("data/dataset", exist_ok=True)

np.random.seed(42)

N_SAMPLES = 1000

def sample_planet(M):
    # choose random point in polar coordinates (ring between radius 0.5 and 2)
    r = np.random.uniform(0.5, 2.0)
    theta = np.random.uniform(0, 2 * np.pi)

    # convert to Cartesian coordinates
    pos = np.array([r * np.cos(theta), r * np.sin(theta)])

    # circular orbit velocity where G = 1 and M = 100
    v_mag = np.sqrt(G * M / r)

    # find perpendicular direction as velocity must be tangent to the orbit
    direction = np.array([-pos[1], pos[0]])

    # normalize it so it has length 1
    direction /= np.linalg.norm(direction)

    # randomize orbital direction
    if np.random.rand() < 0.5:
        direction = -direction

    # apply correct speed to the perpendicular direction
    vel = direction * v_mag

    # adds noise to create a mix of stability and instability
    vel *= np.random.uniform(0.6, 1.4)

    return pos.tolist(), vel.tolist()

def sample_system():
    masses = [100, 1, 1]
    M = masses[0]

    pos1, vel1 = sample_planet(M)

    # create second planet that is further than 0.2 away
    while True:
        pos2, vel2 = sample_planet(M)

        if np.linalg.norm(np.array(pos1) - np.array(pos2)) > 0.2:
            break

    positions = [
        [0, 0],
        pos1,
        pos2
    ]

    velocities = [
        [0, 0],
        vel1,
        vel2
    ]

    return positions, velocities, masses

for i in tqdm(range(N_SAMPLES)):
    positions, velocities, masses = sample_system()

    traj = run_simulation(positions, velocities, masses)

    np.savez(
        f"data/dataset/run_{i:03d}.npz",
        trajectory=traj,
        positions=np.array(positions),
        velocities=np.array(velocities),
        masses=np.array(masses)
    )

print("Done:", N_SAMPLES)