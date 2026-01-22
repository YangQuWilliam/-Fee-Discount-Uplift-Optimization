"""Evaluation metrics for uplift models."""


def dr_score(preds, T, Y):
    """Compute doubly robust (DR) score."""
    raise NotImplementedError("Implement DR score.")


def ips_score(preds, T, Y, propensities):
    """Compute inverse propensity score (IPS)."""
    raise NotImplementedError("Implement IPS score.")

