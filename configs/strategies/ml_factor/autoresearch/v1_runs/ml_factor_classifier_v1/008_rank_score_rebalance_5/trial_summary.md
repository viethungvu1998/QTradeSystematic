# Trial 008_rank_score_rebalance_5

- Status: discard
- Stop reason: continue
- Objective: sharpe = 1.135804104523059
- Best value: 1.4502032808497105
- Criteria hit: false
- Description: rank_col=score rebalance_frequency=5

## Metrics

- cagr: 0.1895430600142285
- max_drawdown: 0.2877184440175244
- sharpe: 1.135804104523059
- sortino: 1.1009153199090056
- win_rate: 0.32633587786259544

## Log Tail

```text
output_dir=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/008_rank_score_rebalance_5
config=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/008_rank_score_rebalance_5/trial_config.yaml
params: run_id=autoresearch_008_rank_score_rebalance_5 fwd=12 cw=120 tw=630 rb=5 max_pos=5 threshold=0.3
loaded raw: 202669 rows, 101 symbols
rows                   200574
dates                    2096
symbols                   100
feature_ready_rows     177774
training_ready_rows    176574
training rebalance 1/269: 2021-01-07
training rebalance 25/269: 2021-07-06
training rebalance 50/269: 2021-12-30
training rebalance 75/269: 2022-07-06
training rebalance 100/269: 2022-12-30
training rebalance 125/269: 2023-07-06
training rebalance 150/269: 2024-01-02
training rebalance 175/269: 2024-07-08
training rebalance 200/269: 2025-01-02
training rebalance 225/269: 2025-07-09
training rebalance 250/269: 2026-01-06
prediction_rows            25977.000000
prediction_dates             266.000000
selected_rows                410.000000
selected_dates               151.000000
top_mean_forward_return        0.045713
class_accuracy                 0.329214
backtest_symbols            65
order_dates                128
first_order_date    2021-01-08
last_order_date     2026-05-11
wrote outputs to /home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/008_rank_score_rebalance_5
```
