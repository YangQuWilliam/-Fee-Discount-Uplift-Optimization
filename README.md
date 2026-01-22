# Fee Discount Uplift Optimization

This project studies how to allocate fee discounts under a fixed budget by estimating
the causal uplift of discounts on future user trading behavior, rather than relying on
simple threshold-based rules.

## Problem
Many platforms grant fee discounts based on users' past trading volume.
This project evaluates whether such discounts generate real incremental impact
and explores improved allocation strategies.

## Status
- README created
- Project setup in progress

## Data & Decision Schema

**Decision**: allocate fee discounts under a fixed budget.

**Treatment (T)**: whether a user receives a fee discount in the decision window.

**Outcome (Y)**: binary indicator of whether the user trades in the subsequent window.

**Features (X)**: pre-treatment user features derived from historical activity only.

**Budget (K)**: top 10% of users can receive discounts.

**Baseline Policy**: prioritize users with highest historical activity (threshold-based rule).

The Bank Marketing dataset is used as a structural proxy for fee discount incentives,
focusing on causal decision-making under budget constraints rather than domain semantics.