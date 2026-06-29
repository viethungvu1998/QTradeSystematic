# Trial 119_098base_threshold45_rebalance5_xgb10_max3_no_sl

- Status: discard
- Stop reason: continue
- Objective: sharpe = 0.5290361002569622
- Best value: 1.9132302197463205
- Criteria hit: false
- Description: base 098 probability_threshold=0.45 class_quantiles=0.15,0.35,0.55,0.70 class_scores=-2,-1,0,1,3 rebalance_frequency=5 fallback_min_positions=3 max_positions=3 xgb_n_estimators=10 xgb_max_depth=3 no SL/TP

## Metrics

- cagr: 0.10826406550104739
- max_drawdown: 0.6613121231974483
- sharpe: 0.5290361002569622
- sortino: 0.5703501375594995
- win_rate: 0.5870646766169154

## Log Tail

```text
output_dir=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/119_098base_threshold45_rebalance5_xgb10_max3_no_sl
config=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/119_098base_threshold45_rebalance5_xgb10_max3_no_sl/trial_config.yaml
params: run_id=autoresearch_119_098base_threshold45_rebalance5_xgb10_max3_no_sl fwd=12 cw=120 threshold_scope=classification_window balanced_sample_weight=False tw=630 rb=5 max_pos=3 threshold=0.45 min_predicted_class=None stop_loss=None take_profit=None
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
selected_rows                798.000000
selected_dates               266.000000
top_mean_forward_return        0.028989
class_accuracy                 0.350464
backtest_symbols            83
order_dates                238
first_order_date    2021-01-08
last_order_date     2026-05-25
wrote outputs to /home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/119_098base_threshold45_rebalance5_xgb10_max3_no_sl
```
