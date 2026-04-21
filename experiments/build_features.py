import numpy as np
import os

DATA_DIR = "data/dataset"

def compute_features(positions, velocities):
    star_pos = np.array(positions[0])

    # will store features for all planets in a flattened vector
    features = []

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

        # alignment between position vector and velocity vector:
        #   +1 -> moving directly away from star (escaping)
        #    0 -> moving perpendicular (ideal circular orbit direction)
        #   -1 -> moving directly toward star
        if r == 0 or v == 0:
            alignment = 0.0
        else:
            alignment = np.dot(rel_pos, vel) / (r * v)
        
        # add features for this planet -> [distance from star, speed, motion-direction alignment]
        features.extend([r, v, alignment])
    
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

        # convert physics state to a feature vector
        features = compute_features(positions, velocities)

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