# Trial 174_098base_rank_probsum_max4_rb6_sl05_close_no_regime

- Status: discard
- Stop reason: continue
- Objective: sharpe = 0.9853473152882921
- Best value: 1.9132302197463205
- Criteria hit: false max_drawdown
- Description: base 098 probability_threshold=0.45 class_quantiles=0.15,0.35,0.55,0.70 class_scores=-2,-1,0,1,3 rank_col=class_3_4_prob fallback_min_positions=3 max_positions=4 rebalance_frequency=6 xgb_n_estimators=10 xgb_max_depth=3 stop_loss=0.05 close-trigger SL/TP rule with benchmark_regime disabled

## Metrics

- cagr: 0.19565177608128081
- max_drawdown: 0.24413048524079117
- sharpe: 0.9853473152882921
- sortino: 1.0734283979106585
- win_rate: 0.5833333333333334

## Log Tail

```text
output_dir=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/174_098base_rank_probsum_max4_rb6_sl05_close_no_regime
config=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/174_098base_rank_probsum_max4_rb6_sl05_close_no_regime/trial_config.yaml
params: run_id=autoresearch_174_098base_rank_probsum_max4_rb6_sl05_close_no_regime fwd=12 cw=120 threshold_scope=classification_window balanced_sample_weight=False tw=630 rb=6 max_pos=4 threshold=0.45 min_predicted_class=None stop_loss=0.05 stop_loss_min_hold=1 take_profit=None trailing_stop=None benchmark_regime_enabled=False
loaded raw: 202669 rows, 101 symbols
rows                   200574
dates                    2096
symbols                   100
feature_ready_rows     177774
training_ready_rows    176574
training rebalance 1/224: 2021-01-08
training rebalance 25/224: 2021-08-10
training rebalance 50/224: 2022-03-18
training rebalance 75/224: 2022-10-21
training rebalance 100/224: 2023-06-01
training rebalance 125/224: 2024-01-02
training rebalance 150/224: 2024-08-12
training rebalance 175/224: 2025-03-20
training rebalance 200/224: 2025-10-24
prediction_rows            21691.000000
prediction_dates             222.000000
selected_rows                666.000000
selected_dates               222.000000
top_mean_forward_return        0.029283
class_accuracy                 0.350514
backtest_symbols            81
order_dates                287
first_order_date    2021-01-11
last_order_date     2026-05-29
wrote outputs to /home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/174_098base_rank_probsum_max4_rb6_sl05_close_no_regime
```
