# Fee Discount Uplift

Project scaffold for fee discount uplift modeling and evaluation.

## Results & Limitations

### Results
Offline policy evaluation (observational data)

In this project, I evaluate three incentive allocation policies under a fixed budget constraint (top 10% of users):

- Random policy: sanity-check baseline
- Rule-based activity policy: heuristic based on user activity
- Uplift-based policy: two-model approach estimating individual-level treatment effects

Because treatment assignment in the data is non-random, I evaluate policies using Inverse Propensity Scoring (IPS) and Doubly Robust (DR) estimators.

Policy | IPS | DR
--- | --- | ---
Random | 0.143 | -0.130
Rule-based activity | 0.356 | 0.064
Uplift (two-model) | 0.397 | 0.085

#### Interpretation

- The uplift-based policy consistently outperforms both the random baseline and the rule-based heuristic under both IPS and DR.
- IPS estimates are higher but exhibit noticeable variance, while DR provides a more conservative and robust estimate.
- Under the DR estimator, the uplift policy achieves roughly 30% relative improvement over the rule-based activity heuristic.
- While absolute effect sizes are modest, this is expected in incentive systems where many users would convert without intervention.
  From a decision perspective, these gains are incremental but economically meaningful.

#### Propensity overlap diagnostics

Treatment assignment in the dataset is highly structured and depends strongly on user attributes. The estimated propensity score distribution shows that:

- Treated users tend to have propensity scores close to 1.0
- Untreated users are concentrated near 0.0
- Overlap exists primarily in the mid-range of the propensity spectrum

This reflects realistic production settings, where historical targeting policies prioritize high-activity users.

Due to this limited overlap:

- IPS estimates can be optimistic and noisy
- DR estimates are intentionally conservative, relying more heavily on outcome models in regions with weak support

All causal estimates should therefore be interpreted within regions of common support.

### Limitations

1. Limited overlap from historical targeting
   Because treatment assignment was largely deterministic for certain user segments,
   counterfactual outcomes are weakly identified for users with extreme propensity scores.
   This limits the magnitude of reliably estimable causal effects and motivates the use of
   doubly robust estimation rather than naive comparisons.

2. Observational evaluation only
   This project evaluates policies offline using observational data. While IPS and DR
   correct for observed confounding, unobserved confounders may still bias estimates.
   In practice, these methods are best viewed as policy screening tools prior to online
   A/B testing, not replacements for experimentation.

3. Simplified treatment and outcome definition
   Treatment is binary and does not capture incentive intensity. The outcome is short-term
   and binary. Long-term user behavior and strategic effects are not modeled. These
   simplifications are intentional, allowing the project to focus on causal policy
   evaluation mechanics rather than domain-specific tuning.

### Practical takeaway

Despite strong selection bias and limited overlap, the results suggest that uplift-based targeting can outperform heuristic incentive allocation when evaluated with appropriate causal methods. This mirrors real-world decision systems, where improvements are incremental, constrained, and must be justified under uncertainty.
