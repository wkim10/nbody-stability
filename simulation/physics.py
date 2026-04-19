import numpy as np
from simulation.constants import G

def compute_accelerations(positions, masses):
    n = len(masses)
    acc = np.zeros_like(positions)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            r_vec = positions[j] - positions[i]
            r = np.linalg.norm(r_vec) + 1e-9

            acc[i] += G * masses[j] * r_vec / (r**3)

    return acc