"""Policies for assigning discounts."""


# src/policy.py
from __future__ import annotations

import numpy as np


def select_top_k(scores: np.ndarray, k_frac: float) -> np.ndarray:
    """
    Returns a binary assignment array a[i] in {0,1}, selecting top k_frac by score.
    """
    if not (0 < k_frac < 1):
        raise ValueError("k_frac must be in (0,1)")
    n = len(scores)
    k = max(1, int(round(n * k_frac)))
    idx = np.argsort(scores)[::-1][:k]
    a = np.zeros(n, dtype=int)
    a[idx] = 1
    return a


def random_policy(n: int, k_frac: float, rng: np.random.Generator) -> np.ndarray:
    k = max(1, int(round(n * k_frac)))
    idx = rng.choice(n, size=k, replace=False)
    a = np.zeros(n, dtype=int)
    a[idx] = 1
    return a


def rule_based_policy(activity_score: np.ndarray, k_frac: float) -> np.ndarray:
    """
    Mimics "threshold-based / activity-based" fee discount allocation.
    """
    return select_top_k(activity_score, k_frac)


