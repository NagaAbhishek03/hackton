import pandas as pd
import numpy as np
import os, json, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ==============================
# Load Dataset
# ==============================
df = pd.read_csv("./data/Heart Disease data set.csv")

# Assume last column is target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# ==============================
# Train/Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# Scale features
# ==============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# Train Multiple Models
# ==============================
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, kernel="rbf", random_state=42),
}

best_model = None
best_score = 0
results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

    if acc > best_score:
        best_score = acc
        best_model = model

print(f"\nBest Model: {best_model} with accuracy {best_score:.4f}")

# ==============================
# Save Best Model + Metadata
# ==============================
os.makedirs("artifacts", exist_ok=True)

joblib.dump(best_model, os.path.join("artifacts", "model.pkl"))
joblib.dump(scaler, os.path.join("artifacts", "scaler.pkl"))

features = list(X.columns)

# ✅ Auto-generate feature info
feature_info = {}
for col in features:
    feature_info[col] = {
        "min": float(X[col].min()),
        "max": float(X[col].max()),
        "step": 1 if str(X[col].dtype) in ["int64", "int32"] else 0.1
    }

metadata = {
    "features": features,
    "feature_info": feature_info
}

with open(os.path.join("artifacts", "feature_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

print("\nTraining complete. Model and metadata saved in 'artifacts/' folder.")
