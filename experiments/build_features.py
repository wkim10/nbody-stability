import numpy as np
import os
from simulation.constants import G

DATA_DIR = "data/dataset"

def compute_features(positions, velocities, masses):
    star_pos = np.array(positions[0])
    M = masses[0]

    # will store features for all planets in a flattened vector
    features = []

    planet_rel_positions = []

    for i in range(1, len(positions)):
        # current planet position and velocity
        pos = np.array(positions[i])
        vel = np.array(velocities[i])

        # convert to star-centered coordinate system (so all motion is relative to the star)
        rel_pos = pos - star_pos

        # distance from star (orbital radius)
        r = np.linalg.norm(rel_pos)

        # speed magnitude of the planet
        v = np.linalg.norm(vel)

        # circular orbit speed at this distance
        v_circ = np.sqrt(G * M / r)

        # ratio of actual speed to circular orbit speed
        # 1.0 = perfect circular orbit, >1 = too fast, <1 = too slow
        v_ratio = v / v_circ
        
        # add features for this planet -> [distance from star, speed, velocity ratio]
        features.extend([r, v, v_ratio])
        planet_rel_positions.append(rel_pos)
    
    # inter-planet separation
    for i in range(len(planet_rel_positions)):
        for j in range(i + 1, len(planet_rel_positions)):
            sep = np.linalg.norm(planet_rel_positions[i] - planet_rel_positions[j])
            features.append(sep)
    
    return np.array(features)

def main():
    # load binary stability labels (0 = unstable, 1 = stable)
    labels = np.load("data/labels.npy")

    # fetch all dataset files -> each file is one simulation
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".npz")])

    # will store feature vectors for all samples
    X = []

    for f in files:
        data = np.load(os.path.join(DATA_DIR, f))

        # extract initial conditions to compute features
        positions = data["positions"]
        velocities = data["velocities"]
        masses = data["masses"]

        # convert physics state to a feature vector
        features = compute_features(positions, velocities, masses)

        # store feature vector for current simulation
        X.append(features)
    
    # convert feature vectors list into matrix form -> shape of (num_samples, num_features)
    X = np.array(X)

    print("X shape:", X.shape)
    print("y shape:", labels.shape)

    # save processed dataset for training
    np.save("data/X.npy", X)
    np.save("data/y.npy", labels)

    print("Saved feature dataset.")

if __name__ == "__main__":
    main()