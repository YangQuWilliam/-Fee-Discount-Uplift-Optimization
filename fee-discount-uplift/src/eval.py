# src/eval.py
from __future__ import annotations

import numpy as np


def summarize_policy(y: np.ndarray, assigned: np.ndarray, name: str) -> dict:
    """
    Simple summary of the selected set:
    - coverage
    - outcome rate among selected
    - outcome rate overall
    """
    n = len(y)
    sel = assigned.astype(bool)
    if sel.sum() == 0:
        return {"policy": name, "coverage": 0.0, "selected_rate": None, "overall_rate": float(np.mean(y))}
    return {
        "policy": name,
        "coverage": float(sel.mean()),
        "selected_rate": float(y[sel].mean()),
        "overall_rate": float(y.mean()),
    }
