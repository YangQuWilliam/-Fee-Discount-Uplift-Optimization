from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_two_model_uplift(X: np.ndarray, t: np.ndarray, y: np.ndarray):
    """
    Returns two classifiers: model_t1 and model_t0
    """
    X1, y1 = X[t == 1], y[t == 1]
    X0, y0 = X[t == 0], y[t == 0]

    # Simple, strong baseline model
    def make_clf():
        return Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(max_iter=2000, n_jobs=None)),
        ])

    m1 = make_clf()
    m0 = make_clf()

    m1.fit(X1, y1)
    m0.fit(X0, y0)
    return m1, m0


def predict_uplift(m1, m0, X: np.ndarray) -> np.ndarray:
    """
    uplift = P(Y=1|T=1,X) - P(Y=1|T=0,X)
    """
    p1 = m1.predict_proba(X)[:, 1]
    p0 = m0.predict_proba(X)[:, 1]
    return p1 - p0
