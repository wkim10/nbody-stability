import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# -----------------------------
# Load data
# -----------------------------
X = np.load("data/X.npy")
y = np.load("data/y.npy")

# -----------------------------
# Train model (same settings as train_model.py)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
    shuffle=True
)

# scale features so all inputs are on the same scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    class_weight='balanced',  # compensate for unstable/stable imbalance
    random_state=42
)
rf.fit(X_train_scaled, y_train)

# -----------------------------
# Define which features to vary
# We vary v_ratio for planet 1 and v_ratio for planet 2
# because these are the most physically meaningful features -
# they measure how far each planet deviates from a stable circular orbit
# All other features are held fixed at their mean values
# -----------------------------
feature_names = [
    "r1", "v1", "v_ratio1",  # planet 1: distance, speed, velocity ratio
    "r2", "v2", "v_ratio2",  # planet 2: distance, speed, velocity ratio
    "separation"             # distance between the two planets
]

x_idx = 2  # v_ratio for planet 1
y_idx = 5  # v_ratio for planet 2

# -----------------------------
# Build a 2D grid over v_ratio1 and v_ratio2
# -----------------------------
resolution = 200

x_range = np.linspace(0.4, 1.6, resolution)
y_range = np.linspace(0.4, 1.6, resolution)

# create a grid of all (v_ratio1, v_ratio2) combinations
xx, yy = np.meshgrid(x_range, y_range)

# start every grid point at the mean feature values
X_mean = X.mean(axis=0)
grid = np.tile(X_mean, (resolution * resolution, 1))

# then vary only the two features we care about
grid[:, x_idx] = xx.ravel()
grid[:, y_idx] = yy.ravel()

# -----------------------------
# Predict stability probability for every point on the grid
# -----------------------------
grid_scaled = scaler.transform(grid)

# predict_proba returns [P(unstable), P(stable)] for each point
# we take index 1 to get P(stable)
probs = rf.predict_proba(grid_scaled)[:, 1]
probs = probs.reshape(resolution, resolution)

# -----------------------------
# Plot decision boundary
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 6))

# filled contour plot: green = high P(stable), red = low P(stable)
contour = ax.contourf(xx, yy, probs, levels=50, cmap='RdYlGn', alpha=0.8)
plt.colorbar(contour, ax=ax, label='P(stable)')

# overlay actual data points colored by true label
stable_mask = y == 1
unstable_mask = y == 0

ax.scatter(
    X[unstable_mask, x_idx], X[unstable_mask, y_idx],
    c='red', s=8, alpha=0.4, label='Unstable'
)
ax.scatter(
    X[stable_mask, x_idx], X[stable_mask, y_idx],
    c='green', s=8, alpha=0.6, label='Stable'
)

# mark v_ratio = 1.0 for each axis - this is perfect circular orbit speed
# systems near this intersection should be most stable
ax.axvline(x=1.0, color='white', linestyle='--', alpha=0.6, linewidth=1)
ax.axhline(y=1.0, color='white', linestyle='--', alpha=0.6, linewidth=1)

ax.set_xlabel('v_ratio planet 1 (v / v_circ)')
ax.set_ylabel('v_ratio planet 2 (v / v_circ)')
ax.set_title('Stability Decision Boundary\n(all other features fixed at mean)')
ax.legend(markerscale=2)

plt.tight_layout()
plt.show()