import pandas as pd
import joblib
import os
import json

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from preprocess import build_preprocessor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load
df = pd.read_csv("data/raw/telco_churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)
df["Churn"] = df["Churn"].map({"Yes":1,"No":0})

X = df.drop(columns=["Churn"])
y = df["Churn"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Preprocess + model
preprocessor = build_preprocessor(X_train)
model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", model)
])

# Train
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred)
}

# Save
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/churn_model.pkl")

# Save metrics
with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
