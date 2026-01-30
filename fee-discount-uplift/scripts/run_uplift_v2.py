import numpy as np

from src.data import load_and_prepare
from src.policy import random_policy, rule_based_policy, select_top_k
from src.train_uplift import fit_two_model_uplift, predict_uplift
from src.propensity import fit_propensity_model, predict_propensity
from src.causal_eval import ips_value, dr_value


def main():
    K = 0.10
    rng = np.random.default_rng(42)

    ds = load_and_prepare(test_size=0.2, random_state=42)

    # policies
    a_random = random_policy(len(ds.y_test), K, rng)
    a_rule = rule_based_policy(ds.activity_test, K)

    m1, m0 = fit_two_model_uplift(ds.X_train, ds.t_train, ds.y_train)
    uplift = predict_uplift(m1, m0, ds.X_test)
    a_uplift = select_top_k(uplift, K)

    # nuisance models
    prop_model = fit_propensity_model(ds.X_train, ds.t_train)
    prop = predict_propensity(prop_model, ds.X_test)

    mu1 = m1.predict_proba(ds.X_test)[:, 1]
    mu0 = m0.predict_proba(ds.X_test)[:, 1]

    for name, a in [
        ("random", a_random),
        ("rule_based", a_rule),
        ("uplift", a_uplift),
    ]:
        print(name)
        print("  IPS:", ips_value(ds.y_test, ds.t_test, a, prop))
        print("  DR :", dr_value(ds.y_test, ds.t_test, a, prop, mu1, mu0))


if __name__ == "__main__":
    main()
