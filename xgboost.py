# XGBoost — Churn Prediction
# Uses XGBClassifier with ColumnTransformer preprocessing and k-Fold Cross Validation.

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier


def main():
    # ── Load dataset (relative to this script) ──────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "Churn_Modelling.csv")
    dataset = pd.read_csv(csv_path)
    X = dataset.iloc[:, 3:13].values
    y = dataset.iloc[:, 13].values

    # ── Encode categorical features ─────────────────────────────────────
    # Label-encode Geography (col 1) and Gender (col 2)
    labelencoder_geo = LabelEncoder()
    X[:, 1] = labelencoder_geo.fit_transform(X[:, 1])

    labelencoder_gender = LabelEncoder()
    X[:, 2] = labelencoder_gender.fit_transform(X[:, 2])

    # One-hot encode Geography (col 1) via ColumnTransformer (modern API)
    ct = ColumnTransformer(
        transformers=[("onehot", OneHotEncoder(drop="first"), [1])],
        remainder="passthrough",
    )
    X = ct.fit_transform(X).astype(float)

    # ── Train / Test split ──────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # ── Fit XGBoost ─────────────────────────────────────────────────────
    classifier = XGBClassifier(
        n_estimators=100,
        eval_metric="logloss",
    )
    classifier.fit(X_train, y_train)

    # ── Predict & evaluate ──────────────────────────────────────────────
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # ── k-Fold Cross Validation ─────────────────────────────────────────
    accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
    print(f"\nCross-Validation Accuracy: {accuracies.mean():.4f} (+/- {accuracies.std():.4f})")


if __name__ == "__main__":
    main()
