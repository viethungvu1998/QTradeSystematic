# Trial 118_098base_threshold45_rebalance10_xgb10_max3_no_sl

- Status: discard
- Stop reason: continue
- Objective: sharpe = 0.9347313768787463
- Best value: 1.9132302197463205
- Criteria hit: false
- Description: base 098 probability_threshold=0.45 class_quantiles=0.15,0.35,0.55,0.70 class_scores=-2,-1,0,1,3 rebalance_frequency=10 fallback_min_positions=3 max_positions=3 xgb_n_estimators=10 xgb_max_depth=3 no SL/TP

## Metrics

- cagr: 0.22728887873534065
- max_drawdown: 0.3602968389151254
- sharpe: 0.9347313768787463
- sortino: 1.042344992254274
- win_rate: 0.5920398009950248

## Log Tail

```text
output_dir=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/118_098base_threshold45_rebalance10_xgb10_max3_no_sl
config=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/118_098base_threshold45_rebalance10_xgb10_max3_no_sl/trial_config.yaml
params: run_id=autoresearch_118_098base_threshold45_rebalance10_xgb10_max3_no_sl fwd=12 cw=120 threshold_scope=classification_window balanced_sample_weight=False tw=630 rb=10 max_pos=3 threshold=0.45 min_predicted_class=None stop_loss=None take_profit=None
loaded raw: 202669 rows, 101 symbols
rows                   200574
dates                    2096
symbols                   100
feature_ready_rows     177774
training_ready_rows    176574
training rebalance 1/134: 2021-01-14
training rebalance 25/134: 2021-12-30
training rebalance 50/134: 2022-12-30
training rebalance 75/134: 2024-01-02
training rebalance 100/134: 2025-01-02
training rebalance 125/134: 2026-01-06
prediction_rows            12991.000000
prediction_dates             133.000000
selected_rows                399.000000
selected_dates               133.000000
top_mean_forward_return        0.029953
class_accuracy                 0.351089
backtest_symbols            75
order_dates                127
first_order_date    2021-01-15
last_order_date     2026-05-25
wrote outputs to /home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/118_098base_threshold45_rebalance10_xgb10_max3_no_sl
```
