from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure the project root is on sys.path for "src" imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_and_prepare
from src.eval import summarize_policy
from src.policy import random_policy, rule_based_policy


def main():
    K = 0.10
    rng = np.random.default_rng(42)

    ds = load_and_prepare(test_size=0.2, random_state=42)

    # Baselines on test split (simulate "decisioning" on a holdout)
    a_random = random_policy(n=len(ds.y_test), k_frac=K, rng=rng)
    a_rule = rule_based_policy(activity_score=ds.activity_test, k_frac=K)

    print("=== Baseline Policy Sanity Check (NOT causal yet) ===")
    print(summarize_policy(ds.y_test, a_random, "random"))
    print(summarize_policy(ds.y_test, a_rule, "rule_based_activity"))


if __name__ == "__main__":
    main()
