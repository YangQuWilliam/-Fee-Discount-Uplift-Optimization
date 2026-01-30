from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_propensity_model(X: np.ndarray, t: np.ndarray):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    model.fit(X, t)
    return model


def predict_propensity(model, X: np.ndarray) -> np.ndarray:
    # clip for numerical stability
    p = model.predict_proba(X)[:, 1]
    return np.clip(p, 1e-3, 1 - 1e-3)
