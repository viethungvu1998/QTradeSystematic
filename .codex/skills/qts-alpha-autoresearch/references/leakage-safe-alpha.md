# Leakage-Safe Alpha Research

Use these rules when proposing or implementing trial ideas:

- Predictors must never include future target columns. Reject names starting with `forward_return`, `future_`, or `target_`.
- Rolling price, volume, volatility, and rank features must be computed from current-or-prior data only.
- Cross-sectional ranks are allowed only for the prediction date and only over data available on that date.
- Fundamentals must use an explicit availability date. If the data only has report period dates, lag conservatively before joining.
- Train windows must end strictly before the rebalance/prediction date.
- Do not improve backtest metrics by changing the evaluation harness, date range, commission assumptions, or benchmark unless that is the declared idea family for the trial.
- If a metric improves because signal count collapses, mark the result as suspicious in the description and do not keep it without a clear reason.
