# Trial 030_rank_score_fallback_3_class_window_252

- Status: discard
- Stop reason: continue
- Objective: sharpe = 1.2963641540965296
- Best value: 1.7974416674903662
- Criteria hit: false
- Description: rank_col=score fallback_min_positions=3 classification_window=252

## Metrics

- cagr: 0.31626847810779735
- max_drawdown: 0.3287777292143735
- sharpe: 1.2963641540965296
- sortino: 1.3639156042731213
- win_rate: 0.36927480916030536

## Log Tail

```text
output_dir=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/030_rank_score_fallback_3_class_window_252
config=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/030_rank_score_fallback_3_class_window_252/trial_config.yaml
params: run_id=autoresearch_030_rank_score_fallback_3_class_window_252 fwd=12 cw=252 tw=630 rb=8 max_pos=5 threshold=0.3 stop_loss=None take_profit=None
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
selected_rows                834.000000
selected_dates               167.000000
top_mean_forward_return        0.026387
class_accuracy                 0.384432
backtest_symbols            82
order_dates                165
first_order_date    2021-01-05
last_order_date     2026-05-21
wrote outputs to /home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/030_rank_score_fallback_3_class_window_252
```
