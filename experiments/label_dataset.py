import numpy as np
import os
from tqdm import tqdm

DATA_DIR = "data/dataset"

def is_unstable(traj, collision_threshold=0.05, escape_factor=5.0):
    T, N, _ = traj.shape

    initial_star = traj[0, 0]
    initial_planets = traj[0, 1:]
    initial_dists = np.linalg.norm(initial_planets - initial_star, axis=1)
    escape_threshold = escape_factor * np.max(initial_dists)

    for t in range(T):
        star = traj[t, 0]
        planets = traj[t, 1:]

        # check if any bodies escaped
        dists = np.linalg.norm(planets - star, axis=1)
        if np.any(dists > escape_threshold):
            return True
        
        # check if any collisions occurred
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(traj[t, i] - traj[t, j])
                if dist < collision_threshold:
                    return True
    
    return False

labels = []

files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".npz")])

for f in tqdm(files):
    data = np.load(os.path.join(DATA_DIR, f))
    traj = data["trajectory"]

    label = 0 if is_unstable(traj) else 1
    labels.append(label)

labels = np.array(labels)

np.save("data/labels.npy", labels)

print("Saved labels:", labels.shape)
print("Stable:", np.sum(labels == 1))
print("Unstable:", np.sum(labels == 0))