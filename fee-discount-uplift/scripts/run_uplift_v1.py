from __future__ import annotations

import numpy as np

from src.data import load_and_prepare
from src.policy import random_policy, rule_based_policy, select_top_k
from src.policy_eval_v1 import summarize_selected
from src.train_uplift import fit_two_model_uplift, predict_uplift


def main():
    K = 0.10
    rng = np.random.default_rng(42)

    ds = load_and_prepare(test_size=0.2, random_state=42)

    # Baselines
    a_random = random_policy(n=len(ds.y_test), k_frac=K, rng=rng)
    a_rule = rule_based_policy(activity_score=ds.activity_test, k_frac=K)

    # Uplift model (fit on train, score on test)
    m1, m0 = fit_two_model_uplift(ds.X_train, ds.t_train, ds.y_train)
    uplift_scores = predict_uplift(m1, m0, ds.X_test)
    a_uplift = select_top_k(uplift_scores, K)

    print("=== Policy Sanity Comparison (NOT causal yet) ===")
    print(summarize_selected(ds.y_test, a_random, "random"))
    print(summarize_selected(ds.y_test, a_rule, "rule_based_activity"))
    print(summarize_selected(ds.y_test, a_uplift, "uplift_two_model"))


if __name__ == "__main__":
    main()
