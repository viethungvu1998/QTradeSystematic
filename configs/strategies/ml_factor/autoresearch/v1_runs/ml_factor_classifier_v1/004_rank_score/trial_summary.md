# Trial 004_rank_score

- Status: keep
- Stop reason: continue
- Objective: sharpe = 1.4502032808497105
- Best value: 1.4502032808497105
- Criteria hit: false max_drawdown
- Description: rank_col=score

## Metrics

- cagr: 0.24725774694677072
- max_drawdown: 0.19920635040227147
- sharpe: 1.4502032808497105
- sortino: 1.496098847689778
- win_rate: 0.3330152671755725

## Log Tail

```text
output_dir=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/004_rank_score
config=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/004_rank_score/trial_config.yaml
params: run_id=autoresearch_004_rank_score fwd=12 cw=120 tw=630 rb=8 max_pos=5 threshold=0.3
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
selected_rows                261.000000
selected_dates                94.000000
top_mean_forward_return        0.054175
class_accuracy                 0.326877
backtest_symbols            63
order_dates                 84
first_order_date    2021-01-05
last_order_date     2026-05-11
wrote outputs to /home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/004_rank_score
```
