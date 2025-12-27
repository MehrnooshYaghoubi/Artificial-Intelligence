import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles, make_moons

np.random.seed(42)

# -------------------
# Dataset Generation
# -------------------

# Linearly separable dataset
X_blobs, y_blobs = make_blobs(
    n_samples=300,
    centers=2,
    cluster_std=1.2,
    random_state=42
)

# Circular non-linear dataset
X_circles, y_circles = make_circles(
    n_samples=300,
    noise=0.05,
    factor=0.5,
    random_state=42
)

# Moon-shaped non-linear dataset
X_moons, y_moons = make_moons(
    n_samples=300,
    noise=0.1,
    random_state=42
)

# -------------------
# Visualization
# -------------------


# Create subplots 1x3 Grid (for each dataset)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

datasets = [
    (X_blobs, y_blobs, "make_blobs"),
    (X_circles, y_circles, "make_circles"),
    (X_moons, y_moons, "make_moons")
]


# Plot each dataset 
for ax, (X, y, title) in zip(axes, datasets):
    ax.scatter(X[:, 0], X[:, 1], c=y)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

plt.tight_layout()
plt.savefig("datasets_visualization.png")
plt.show()
