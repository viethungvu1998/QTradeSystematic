# Trial 066_threshold_38_rank_score_fallback3_max6_rebalance7

- Status: discard
- Stop reason: continue
- Objective: sharpe = 1.2282190435762788
- Best value: 1.8280890880755754
- Criteria hit: false
- Description: probability_threshold=0.38 rank_col=score fallback_min_positions=3 max_positions=6 rebalance_frequency=7

## Metrics

- cagr: 0.20908971676148513
- max_drawdown: 0.2738282841962049
- sharpe: 1.2282190435762788
- sortino: 1.3952856373345581
- win_rate: 0.35782442748091603

## Log Tail

```text
output_dir=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/066_threshold_38_rank_score_fallback3_max6_rebalance7
config=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/066_threshold_38_rank_score_fallback3_max6_rebalance7/trial_config.yaml
params: run_id=autoresearch_066_threshold_38_rank_score_fallback3_max6_rebalance7 fwd=12 cw=120 threshold_scope=classification_window balanced_sample_weight=False tw=630 rb=7 max_pos=6 threshold=0.38 min_predicted_class=None stop_loss=None take_profit=None
loaded raw: 202669 rows, 101 symbols
rows                   200574
dates                    2096
symbols                   100
feature_ready_rows     177774
training_ready_rows    176574
training rebalance 1/192: 2021-01-08
training rebalance 25/192: 2021-09-15
training rebalance 50/192: 2022-05-31
training rebalance 75/192: 2023-02-10
training rebalance 100/192: 2023-10-20
training rebalance 125/192: 2024-07-05
training rebalance 150/192: 2025-03-19
training rebalance 175/192: 2025-11-27
prediction_rows            18561.000000
prediction_dates             190.000000
selected_rows                570.000000
selected_dates               190.000000
top_mean_forward_return        0.039266
class_accuracy                 0.327569
backtest_symbols            77
order_dates                180
first_order_date    2021-01-11
last_order_date     2026-05-28
wrote outputs to /home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/066_threshold_38_rank_score_fallback3_max6_rebalance7
```
