from __future__ import annotations

import numpy as np


def summarize_selected(y: np.ndarray, a: np.ndarray, name: str) -> dict:
    sel = a.astype(bool)
    return {
        "policy": name,
        "coverage": float(sel.mean()),
        "selected_y_rate": float(y[sel].mean()) if sel.any() else None,
        "overall_y_rate": float(y.mean()),
        "selected_count": int(sel.sum()),
    }
