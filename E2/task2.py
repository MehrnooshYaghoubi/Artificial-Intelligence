import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from task1 import X_blobs, y_blobs, X_circles, y_circles, X_moons, y_moons

def plot_decision_boundary(model, X, y, title):
    """
    Plot the decision boundary of a classifier together with the original 2D data points.
    - A grid resolution `h = 0.02` is used to sample the feature space.
    """
    h = 0.02  # grid resolution

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")


def run_task2():
    """
    Trains Linear SVM on all datasets, plots decision boundaries,
    and returns a list of result dictionaries for Task 5 table.
    """
    results = []

    # Using All Three Datasets (Blobs, Circles, Moons)
    datasets = [
        ("make_blobs", X_blobs, y_blobs),
        ("make_circles", X_circles, y_circles),
        ("make_moons", X_moons, y_moons)
    ]

    # Linear SVM on all datasets
    for name, X, y in datasets:
        model = SVC(kernel="linear") # Linear SVM 
        model.fit(X, y) # Fit model to training data
        y_pred = model.predict(X) # Predict data
        acc = accuracy_score(y, y_pred) 

        # Plot
        plt.figure(figsize=(5, 4))
        plot_decision_boundary(model, X, y, f"Linear SVM on {name} (Accuracy={acc:.2f})")
        plt.savefig(f"svm_{name}_decision_boundary.png")
        plt.close()  # Close figure to prevent display

        # Store results for Task 5
        observation = "Perfect linear separation" if name=="make_blobs" else "Fails on nonlinear structure"
        results.append({
            "dataset": name,
            "kernel": "Linear",
            "params": "—",
            "accuracy": round(acc, 3),
            "observation": observation
        })

    return results