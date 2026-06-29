# Trial 014_rank_score_quantiles_tail

- Status: discard
- Stop reason: continue
- Objective: sharpe = -0.17335257283916836
- Best value: 1.4502032808497105
- Criteria hit: false
- Description: rank_col=score class_quantiles=0.10,0.35,0.65,0.90

## Metrics

- cagr: -0.03173821988498671
- max_drawdown: 0.5508893933488674
- sharpe: -0.17335257283916836
- sortino: -0.1442265429233813
- win_rate: 0.23091603053435114

## Log Tail

```text
output_dir=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/014_rank_score_quantiles_tail
config=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/014_rank_score_quantiles_tail/trial_config.yaml
params: run_id=autoresearch_014_rank_score_quantiles_tail fwd=12 cw=120 tw=630 rb=8 max_pos=5 threshold=0.3
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
selected_rows                 68.000000
selected_dates                23.000000
top_mean_forward_return        0.035507
class_accuracy                 0.306528
backtest_symbols            28
order_dates                 20
first_order_date    2022-04-21
last_order_date     2026-03-11
wrote outputs to /home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/014_rank_score_quantiles_tail
```
