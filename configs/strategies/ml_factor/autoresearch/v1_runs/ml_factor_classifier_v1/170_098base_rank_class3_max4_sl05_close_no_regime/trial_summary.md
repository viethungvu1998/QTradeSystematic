# Trial 170_098base_rank_class3_max4_sl05_close_no_regime

- Status: discard
- Stop reason: continue
- Objective: sharpe = 0.5542876585718703
- Best value: 1.9132302197463205
- Criteria hit: false
- Description: base 098 probability_threshold=0.45 class_quantiles=0.15,0.35,0.55,0.70 class_scores=-2,-1,0,1,3 rank_col=class_3_prob fallback_min_positions=3 max_positions=4 xgb_n_estimators=10 xgb_max_depth=3 stop_loss=0.05 close-trigger SL/TP rule with benchmark_regime disabled

## Metrics

- cagr: 0.08151249842992159
- max_drawdown: 0.3068322812437431
- sharpe: 0.5542876585718703
- sortino: 0.5622911375638925
- win_rate: 0.5059523809523809

## Log Tail

```text
output_dir=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/170_098base_rank_class3_max4_sl05_close_no_regime
config=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/170_098base_rank_class3_max4_sl05_close_no_regime/trial_config.yaml
params: run_id=autoresearch_170_098base_rank_class3_max4_sl05_close_no_regime fwd=12 cw=120 threshold_scope=classification_window balanced_sample_weight=False tw=630 rb=8 max_pos=4 threshold=0.45 min_predicted_class=None stop_loss=0.05 take_profit=None trailing_stop=None benchmark_regime_enabled=False
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
top_mean_forward_return        0.007900
class_accuracy                 0.356175
backtest_symbols            84
order_dates                254
first_order_date    2021-01-05
last_order_date     2026-05-28
wrote outputs to /home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/170_098base_rank_class3_max4_sl05_close_no_regime
```
