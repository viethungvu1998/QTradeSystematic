# Trial 179_098base_rank_probsum_max4_sl05_benchmark_filtered_close_no_regime

- Status: discard
- Stop reason: quota_exhausted
- Objective: sharpe = 1.4751711969591064
- Best value: 1.9132302197463205
- Criteria hit: false max_drawdown
- Description: base 098 probability_threshold=0.45 class_quantiles=0.15,0.35,0.55,0.70 class_scores=-2,-1,0,1,3 rank_col=class_3_4_prob fallback_min_positions=3 max_positions=4 xgb_n_estimators=10 xgb_max_depth=3 stop_loss=0.05 close-trigger only when benchmark below SMA, benchmark_regime entry filter disabled

## Metrics

- cagr: 0.3194312068739391
- max_drawdown: 0.23725025386906537
- sharpe: 1.4751711969591064
- sortino: 1.7238330246362827
- win_rate: 0.5833333333333334

## Log Tail

```text
output_dir=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/179_098base_rank_probsum_max4_sl05_benchmark_filtered_close_no_regime
config=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/179_098base_rank_probsum_max4_sl05_benchmark_filtered_close_no_regime/trial_config.yaml
params: run_id=autoresearch_179_098base_rank_probsum_max4_sl05_benchmark_filtered_close_no_regime fwd=12 cw=120 threshold_scope=classification_window balanced_sample_weight=False tw=630 rb=8 max_pos=4 threshold=0.45 min_predicted_class=None stop_loss=0.05 stop_loss_min_hold=1 stop_loss_benchmark_filter=below_sma take_profit=None trailing_stop=None benchmark_regime_enabled=False
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
selected_rows                501.000000
selected_dates               167.000000
top_mean_forward_return        0.025961
class_accuracy                 0.356175
backtest_symbols            75
order_dates                203
first_order_date    2021-01-05
last_order_date     2026-05-21
wrote outputs to /home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/179_098base_rank_probsum_max4_sl05_benchmark_filtered_close_no_regime
```
