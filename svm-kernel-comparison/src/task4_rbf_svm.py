import os

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from task1_datasets import X_blobs, y_blobs, X_circles, y_circles, X_moons, y_moons
from task2_linear_svm import plot_decision_boundary

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def run_task4():
    """
    Trains RBF SVM with low and high gamma on all datasets,
    plots decision boundaries, and returns results for Task 5 table.
    """
    results = []

    # Using All Three Datasets (Blobs, Circles, Moons)
    datasets = [
        ("make_blobs", X_blobs, y_blobs),
        ("make_circles", X_circles, y_circles),
        ("make_moons", X_moons, y_moons)
    ]

    gammas = [0.1, 10] # Choosing Different Gammas

    # Similar To Previous Tasks, but using RBF kernel with different gamma values
    for name, X, y in datasets:
        for gamma in gammas:
            model = SVC(kernel="rbf", gamma=gamma) # initialize model with rbf kernel and specific gamma
            model.fit(X, y)
            y_pred = model.predict(X)
            acc = accuracy_score(y, y_pred)

            # Plot
            plt.figure(figsize=(5, 4))
            plot_decision_boundary(
                model, X, y,
                f"RBF SVM (gamma={gamma}) on {name} (Accuracy={acc:.2f})"
            )
            plt.savefig(os.path.join(OUTPUT_DIR, f"svm_{name}_rbf_gamma{gamma}.png"))
            plt.close()

            # Observation
            obs = "Smooth boundary; may underfit" if gamma==0.1 else "Flexible boundary; may overfit"

            results.append({
                "dataset": name,
                "kernel": "RBF",
                "params": f"gamma={gamma}",
                "accuracy": round(acc, 3),
                "observation": obs
            })

    return results