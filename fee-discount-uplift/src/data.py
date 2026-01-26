"""Data loading and feature construction."""

# src/data.py
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline



@dataclass
class DatasetBundle:
    X_train: np.ndarray
    X_test: np.ndarray
    t_train: np.ndarray
    t_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    # For baselines: a simple "historical activity" proxy used by rule-based policy
    activity_train: np.ndarray
    activity_test: np.ndarray
    feature_names: list[str]


def load_bank_marketing_df() -> pd.DataFrame:
    """
    Load UCI Bank Marketing dataset from local CSV.

    Expected path:
        data/bank-full.csv

    The file must be non-empty and semicolon-separated.
    """
    data_path = Path(__file__).resolve().parents[1] / "data" / "bank-full.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing dataset at {data_path}. "
            "Download UCI bank-full.csv and place it under data/."
        )

    if data_path.stat().st_size == 0:
        raise ValueError(
            f"Dataset file is empty: {data_path}. "
            "Re-download bank-full.csv."
        )

    df = pd.read_csv(data_path, sep=";")
    df.columns = [c.strip() for c in df.columns]
    return df


def build_t_y(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, str, str]:
    """
    Treatment (T): whether user was contacted.
    Outcome (Y): whether user subscribed (y == 'yes').
    """
    treatment_col = "contact"
    if treatment_col not in df.columns:
        # Fallback: use campaign intensity as a binary treatment proxy.
        # This keeps the pipeline running across dataset variants.
        treatment_col = "campaign" if "campaign" in df.columns else ""
        if not treatment_col:
            # Last resort: take the first non-outcome column as a proxy treatment.
            candidate_cols = [c for c in df.columns if c not in {"y", "Class"}]
            if not candidate_cols:
                raise ValueError(
                    "No usable treatment column found. "
                    f"Available columns: {list(df.columns)}"
                )
            treatment_col = candidate_cols[0]
    outcome_col = "y" if "y" in df.columns else ("Class" if "Class" in df.columns else "")
    if not outcome_col:
        raise ValueError(
            "Expected outcome column 'y' or 'Class' in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    # Proxy treatment:
    # - If contact exists, treat anyone with contact != 'unknown' as contacted.
    # - If we fall back to campaign, treat campaign > 1 as "more intensive contact".
    if treatment_col == "contact":
        t = (df["contact"].astype(str).str.lower() != "unknown").astype(int)
    elif treatment_col == "campaign":
        t = (pd.to_numeric(df["campaign"], errors="coerce").fillna(0) > 1).astype(int)
    else:
        raw_t = df[treatment_col]
        if raw_t.dtype.kind in {"i", "u", "f"}:
            t = (pd.to_numeric(raw_t, errors="coerce").fillna(0) > raw_t.median()).astype(int)
        else:
            t_str = raw_t.astype(str)
            mode = t_str.mode().iloc[0]
            t = (t_str != mode).astype(int)

    raw_y = df[outcome_col]
    if raw_y.dtype.kind in {"i", "u", "f"}:
        uniq = np.unique(raw_y.dropna())
        if len(uniq) == 2:
            y = (raw_y == uniq.max()).astype(int)
        else:
            y = (raw_y.astype(float) > np.nanmedian(raw_y)).astype(int)
    else:
        y_str = raw_y.astype(str).str.lower()
        if set(y_str.unique()) <= {"yes", "no"}:
            y = (y_str == "yes").astype(int)
        elif set(y_str.unique()) <= {"true", "false"}:
            y = (y_str == "true").astype(int)
        elif set(y_str.unique()) <= {"1", "0"}:
            y = (y_str == "1").astype(int)
        else:
            # Fallback: positive class = most common label is 0, others = 1
            mode = y_str.mode().iloc[0]
            y = (y_str != mode).astype(int)
    return t, y, treatment_col, outcome_col


def choose_features(df: pd.DataFrame, treatment_col: str, outcome_col: str) -> pd.DataFrame:
    """
    Select pre-treatment features; explicitly drop leakage feature 'duration' if present.
    Also drop outcome 'y'. Keep 'contact' for T only (not in X).
    """
    drop_cols = {outcome_col, treatment_col}

    # Known leakage feature in many bank marketing variants
    if "duration" in df.columns:
        drop_cols.add("duration")

    # Some variants have 'pdays' etc. Keep them; they are pre-treatment history.
    feature_cols = [c for c in df.columns if c not in drop_cols]
    return df[feature_cols].copy()


def build_activity_score(df: pd.DataFrame) -> np.ndarray:
    """
    A simple proxy for "historical activity" for rule-based policy.
    We use a conservative, pre-treatment-only signal.

    Prefer columns that exist across dataset variants:
    - balance: account balance
    - campaign: number of contacts during this campaign (can be considered 'effort' proxy)
    - previous: number of contacts before this campaign
    """
    cols = []
    for c in ["balance", "campaign", "previous"]:
        if c in df.columns:
            cols.append(c)

    if not cols:
        # fallback: all numeric columns mean
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] == 0:
            return np.zeros(len(df), dtype=float)
        return numeric.mean(axis=1).to_numpy(dtype=float)

    numeric = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    # Simple linear combo; keep it explainable
    score = (
        1.0 * numeric.get("balance", 0.0)
        + 10.0 * numeric.get("previous", 0.0)
        + 5.0 * numeric.get("campaign", 0.0)
    )
    return score.to_numpy(dtype=float)


def make_preprocessor(X_df: pd.DataFrame) -> Tuple[Pipeline, list[str]]:
    """
    One-hot encode categoricals, pass through numeric.
    Returns a fitted ColumnTransformer pipeline and feature names (post-transform).
    """
    cat_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X_df.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    pipe = Pipeline([("pre", preprocessor)])
    # Fit to extract feature names
    pipe.fit(X_df)

    feature_names: list[str] = []
    try:
        ohe: OneHotEncoder = pipe.named_steps["pre"].named_transformers_["cat"]
        ohe_names = ohe.get_feature_names_out(cat_cols).tolist() if cat_cols else []
        feature_names.extend(ohe_names)
        feature_names.extend(num_cols)
    except Exception:
        # feature names not critical for baseline; keep empty if extraction fails
        feature_names = []

    return pipe, feature_names


def load_and_prepare(test_size: float = 0.2, random_state: int = 42) -> DatasetBundle:
    df = load_bank_marketing_df()
    t, y, treatment_col, outcome_col = build_t_y(df)
    X_df = choose_features(df, treatment_col, outcome_col)
    activity = build_activity_score(df)

    X_train_df, X_test_df, t_train, t_test, y_train, y_test, act_train, act_test = train_test_split(
        X_df, t.to_numpy(), y.to_numpy(), activity,
        test_size=test_size, random_state=random_state, stratify=y
    )

    preproc, feat_names = make_preprocessor(X_train_df)
    X_train = preproc.transform(X_train_df)
    X_test = preproc.transform(X_test_df)

    # Convert sparse -> dense if needed (keep it simple for V1)
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    return DatasetBundle(
        X_train=X_train,
        X_test=X_test,
        t_train=t_train.astype(int),
        t_test=t_test.astype(int),
        y_train=y_train.astype(int),
        y_test=y_test.astype(int),
        activity_train=act_train.astype(float),
        activity_test=act_test.astype(float),
        feature_names=feat_names,
    )
