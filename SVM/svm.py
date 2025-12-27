import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

# =============================
# Load Iris dataset
# =============================
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

# =============================
# Train / Test split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# =============================
# SVM classifier (One-vs-All)
# =============================
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        decision_function_shape="ovr"  # One-vs-All
    ))
])

model.fit(X_train, y_train)

# =============================
# Prediction & Accuracy
# =============================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# =============================
# Confusion Matrix
# =============================
cm = confusion_matrix(y_test, y_pred)

plt.figure()
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)
disp.plot()
plt.title("Confusion Matrix - SVM (OvA)")
plt.savefig("cm.png", dpi=300, bbox_inches="tight")
plt.show()

# =============================
# PCA visualization (2D)
# =============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure()
for i, name in enumerate(class_names):
    plt.scatter(
        X_pca[y == i, 0],
        X_pca[y == i, 1],
        label=name
    )

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Iris Dataset - PCA Projection")
plt.legend()
plt.savefig("plot.png", dpi=300, bbox_inches="tight")
plt.show()
