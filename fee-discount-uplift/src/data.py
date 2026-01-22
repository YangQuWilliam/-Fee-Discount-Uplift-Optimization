"""Data loading and feature construction."""


def load_data(path: str):
    """Load raw data from a file path."""
    raise NotImplementedError("Implement data loading.")


def build_features(raw_df):
    """Build X (features), T (treatment), Y (outcome)."""
    raise NotImplementedError("Implement feature construction.")

