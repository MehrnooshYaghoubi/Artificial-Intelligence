import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# =========================
# Load dataset
# =========================
iris = load_iris(as_frame=True)
df = iris.frame

# =========================
# Target (continuous)
# =========================
y = df["petal length (cm)"]

# =========================
# Features
# =========================
X = df[[
    "sepal length (cm)",
    "sepal width (cm)",
    "petal width (cm)",
    "target"  # species (categorical)
]]

# =========================
# Column Transformer
# =========================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal width (cm)"
        ]),
        ("cat", OneHotEncoder(), ["target"])
    ]
)

# =========================
# Pipeline: preprocessing + linear regression
# =========================
model = Pipeline([
    ("preprocess", preprocessor),
    ("regressor", LinearRegression())
])

# =========================
# Train / Test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42
)

# =========================
# Train model
# =========================
model.fit(X_train, y_train)

# =========================
# Prediction
# =========================
y_pred = model.predict(X_test)

# =========================
# RMSE
# =========================
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)
