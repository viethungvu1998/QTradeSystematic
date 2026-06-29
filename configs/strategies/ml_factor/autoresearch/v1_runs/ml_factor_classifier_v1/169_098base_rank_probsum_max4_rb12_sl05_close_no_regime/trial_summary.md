# Trial 169_098base_rank_probsum_max4_rb12_sl05_close_no_regime

- Status: discard
- Stop reason: continue
- Objective: sharpe = 1.0991385154782858
- Best value: 1.9132302197463205
- Criteria hit: false max_drawdown
- Description: base 098 probability_threshold=0.45 class_quantiles=0.15,0.35,0.55,0.70 class_scores=-2,-1,0,1,3 rank_col=class_3_4_prob fallback_min_positions=3 max_positions=4 rebalance_frequency=12 xgb_n_estimators=10 xgb_max_depth=3 stop_loss=0.05 close-trigger SL/TP rule with benchmark_regime disabled

## Metrics

- cagr: 0.21377990285308268
- max_drawdown: 0.20596174227573746
- sharpe: 1.0991385154782858
- sortino: 1.1907174234577689
- win_rate: 0.5654761904761905

## Log Tail

```text
output_dir=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/169_098base_rank_probsum_max4_rb12_sl05_close_no_regime
config=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/169_098base_rank_probsum_max4_rb12_sl05_close_no_regime/trial_config.yaml
params: run_id=autoresearch_169_098base_rank_probsum_max4_rb12_sl05_close_no_regime fwd=12 cw=120 threshold_scope=classification_window balanced_sample_weight=False tw=630 rb=12 max_pos=4 threshold=0.45 min_predicted_class=None stop_loss=0.05 take_profit=None trailing_stop=None benchmark_regime_enabled=False
loaded raw: 202669 rows, 101 symbols
rows                   200574
dates                    2096
symbols                   100
feature_ready_rows     177774
training_ready_rows    176574
training rebalance 1/112: 2021-01-08
training rebalance 25/112: 2022-03-10
training rebalance 50/112: 2023-05-24
training rebalance 75/112: 2024-08-02
training rebalance 100/112: 2025-10-16
prediction_rows            10846.000000
prediction_dates             111.000000
selected_rows                333.000000
selected_dates               111.000000
top_mean_forward_return        0.025674
class_accuracy                 0.345104
backtest_symbols            73
order_dates                189
first_order_date    2021-01-11
last_order_date     2026-05-28
wrote outputs to /home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/169_098base_rank_probsum_max4_rb12_sl05_close_no_regime
```
