# Trial 015_rank_score_threshold_025

- Status: discard
- Stop reason: continue
- Objective: sharpe = 1.3079008439106647
- Best value: 1.4502032808497105
- Criteria hit: false
- Description: rank_col=score probability_threshold=0.25

## Metrics

- cagr: 0.3278063644647189
- max_drawdown: 0.2843357641246119
- sharpe: 1.3079008439106647
- sortino: 1.401869484928288
- win_rate: 0.36211832061068705

## Log Tail

```text
output_dir=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/015_rank_score_threshold_025
config=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/015_rank_score_threshold_025/trial_config.yaml
params: run_id=autoresearch_015_rank_score_threshold_025 fwd=12 cw=120 tw=630 rb=8 max_pos=5 threshold=0.25
loaded raw: 202669 rows, 101 symbols
rows                   200574
dates                    2096
symbols                   100
feature_ready_rows     177774
training_ready_rows    176574
training rebalance 1/168: 2021-01-04
training rebalance 25/168: 2021-10-13
training rebalance 50/168: 2022-08-02
training rebalance 75/168: 2023-05-24
training rebalance 100/168: 2024-03-11
training rebalance 125/168: 2024-12-24
training rebalance 150/168: 2025-10-16
prediction_rows            16315.000000
prediction_dates             167.000000
selected_rows                808.000000
selected_dates               167.000000
top_mean_forward_return        0.028695
class_accuracy                 0.326877
backtest_symbols            86
order_dates                165
first_order_date    2021-01-05
last_order_date     2026-05-21
wrote outputs to /home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/015_rank_score_threshold_025
```
