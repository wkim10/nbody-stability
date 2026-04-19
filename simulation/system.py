import numpy as np

class NBodySystem:
    def __init__(self, positions, velocities, masses):
        self.positions = np.array(positions, dtype=float)
        self.velocities = np.array(velocities, dtype=float)
        self.masses = np.array(masses, dtype=float)

        self.n = len(masses)
    
    def state(self):
        return self.positions, self.velocities