# Trial 010_rank_score_forward_10

- Status: discard
- Stop reason: continue
- Objective: sharpe = 0.9568715066318714
- Best value: 1.4502032808497105
- Criteria hit: false
- Description: rank_col=score forward_period=10

## Metrics

- cagr: 0.1892133995468337
- max_drawdown: 0.45056397177103763
- sharpe: 0.9568715066318714
- sortino: 0.9188752313060643
- win_rate: 0.3520992366412214

## Log Tail

```text
output_dir=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/010_rank_score_forward_10
config=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/010_rank_score_forward_10/trial_config.yaml
params: run_id=autoresearch_010_rank_score_forward_10 fwd=10 cw=120 tw=630 rb=8 max_pos=5 threshold=0.3
loaded raw: 202669 rows, 101 symbols
rows                   200574
dates                    2096
symbols                   100
feature_ready_rows     177774
training_ready_rows    176774
training rebalance 1/168: 2021-01-04
training rebalance 25/168: 2021-10-13
training rebalance 50/168: 2022-08-02
training rebalance 75/168: 2023-05-24
training rebalance 100/168: 2024-03-11
training rebalance 125/168: 2024-12-24
training rebalance 150/168: 2025-10-16
prediction_rows            16315.000000
prediction_dates             167.000000
selected_rows                411.000000
selected_dates               132.000000
top_mean_forward_return        0.033614
class_accuracy                 0.314496
backtest_symbols            61
order_dates                111
first_order_date    2021-01-05
last_order_date     2026-04-24
wrote outputs to /home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/010_rank_score_forward_10
```
