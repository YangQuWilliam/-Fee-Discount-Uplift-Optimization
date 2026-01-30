from __future__ import annotations

import numpy as np


def ips_value(
    y: np.ndarray,
    t: np.ndarray,
    a: np.ndarray,
    prop: np.ndarray,
) -> float:
    """
    IPS estimate of policy value.
    a: policy action (0/1)
    """
    match = (a == t)
    w = np.where(t == 1, 1.0 / prop, 1.0 / (1.0 - prop))
    return float(np.mean(match * w * y))


def dr_value(
    y: np.ndarray,
    t: np.ndarray,
    a: np.ndarray,
    prop: np.ndarray,
    mu1: np.ndarray,
    mu0: np.ndarray,
) -> float:
    """
    Doubly Robust estimator.
    mu1 = P(Y=1|T=1,X)
    mu0 = P(Y=1|T=0,X)
    """
    mu = a * mu1 + (1 - a) * mu0
    correction = (
        (t == a)
        * (y - (t * mu1 + (1 - t) * mu0))
        / np.where(t == 1, prop, 1 - prop)
    )
    return float(np.mean(mu + correction))
