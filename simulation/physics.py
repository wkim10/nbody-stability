import numpy as np
from simulation.constants import G

def compute_accelerations(positions, masses):
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]

    dist_sq = np.sum(diff**2, axis=-1)
    dist_sq += np.eye(len(masses)) * 1e9

    dist_cube = dist_sq**1.5

    acc = -G * np.sum(
        masses[np.newaxis, :, np.newaxis] * diff / dist_cube[:, :, np.newaxis],
        axis=1
    )

    return acc