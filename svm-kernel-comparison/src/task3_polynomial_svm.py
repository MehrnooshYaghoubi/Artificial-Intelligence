import os

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from task2_linear_svm import plot_decision_boundary
from task1_datasets import X_blobs, y_blobs, X_circles, y_circles, X_moons, y_moons

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

def run_task3():
    """
    Trains Polynomial SVM (degree 2 and 3) on all datasets,
    plots decision boundaries, and returns results for Task 5 table.
    """
    results = []

    # Using All Three Datasets (Blobs, Circles, Moons)
    datasets = [
        ("make_blobs", X_blobs, y_blobs),
        ("make_circles", X_circles, y_circles),
        ("make_moons", X_moons, y_moons)
    ]

    degrees = [2, 3] 

    # Similar to task 2, but using Polynomial kernel with different degrees
    for name, X, y in datasets:
        for degree in degrees:
            model = SVC(kernel="poly", degree=degree) # initialize polynomial SVM
            model.fit(X, y)
            y_pred = model.predict(X)
            acc = accuracy_score(y, y_pred)

            # Plot
            plt.figure(figsize=(5, 4))
            plot_decision_boundary(model, X, y,
                                   f"Polynomial SVM (degree={degree}) on {name} (Accuracy={acc:.2f})")
            plt.savefig(os.path.join(OUTPUT_DIR, f"svm_{name}_poly{degree}.png"))
            plt.close()

            # Observation (simple)
            obs = "Adds global curvature; better than linear" if name != "make_blobs" else "Curvature unnecessary but works"

            results.append({
                "dataset": name,
                "kernel": "Polynomial",
                "params": f"degree={degree}",
                "accuracy": round(acc, 3),
                "observation": obs
            })

    return results