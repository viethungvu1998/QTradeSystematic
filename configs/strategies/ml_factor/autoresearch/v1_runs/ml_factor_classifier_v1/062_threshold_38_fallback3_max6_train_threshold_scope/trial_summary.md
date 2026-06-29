# Trial 062_threshold_38_fallback3_max6_train_threshold_scope

- Status: discard
- Stop reason: continue
- Objective: sharpe = 1.081785389932894
- Best value: 1.8280890880755754
- Criteria hit: false
- Description: probability_threshold=0.38 fallback_min_positions=3 max_positions=6 classification_threshold_scope=train

## Metrics

- cagr: 0.258863852068397
- max_drawdown: 0.3174365394834643
- sharpe: 1.081785389932894
- sortino: 1.1488018969418499
- win_rate: 0.3568702290076336

## Log Tail

```text
output_dir=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/062_threshold_38_fallback3_max6_train_threshold_scope
config=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/062_threshold_38_fallback3_max6_train_threshold_scope/trial_config.yaml
params: run_id=autoresearch_062_threshold_38_fallback3_max6_train_threshold_scope fwd=12 cw=120 threshold_scope=train balanced_sample_weight=False tw=630 rb=8 max_pos=6 threshold=0.38 min_predicted_class=None stop_loss=None take_profit=None
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
selected_rows               1002.000000
selected_dates               167.000000
top_mean_forward_return        0.020271
class_accuracy                 0.295372
backtest_symbols            87
order_dates                168
first_order_date    2021-01-05
last_order_date     2026-05-21
wrote outputs to /home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/062_threshold_38_fallback3_max6_train_threshold_scope
```
