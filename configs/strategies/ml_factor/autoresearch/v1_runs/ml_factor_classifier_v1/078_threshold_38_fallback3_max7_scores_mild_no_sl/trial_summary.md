# Trial 078_threshold_38_fallback3_max7_scores_mild_no_sl

- Status: discard
- Stop reason: continue
- Objective: sharpe = 1.7166478015623365
- Best value: 1.8280890880755754
- Criteria hit: false max_drawdown
- Description: probability_threshold=0.38 rank_col=score fallback_min_positions=3 max_positions=7 class_scores=-1,-0.5,0,1,3 no SL/TP

## Metrics

- cagr: 0.27349331364025287
- max_drawdown: 0.20403140368021466
- sharpe: 1.7166478015623365
- sortino: 1.9730886649065675
- win_rate: 0.3654580152671756

## Log Tail

```text
output_dir=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/078_threshold_38_fallback3_max7_scores_mild_no_sl
config=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/078_threshold_38_fallback3_max7_scores_mild_no_sl/trial_config.yaml
params: run_id=autoresearch_078_threshold_38_fallback3_max7_scores_mild_no_sl fwd=12 cw=120 threshold_scope=classification_window balanced_sample_weight=False tw=630 rb=8 max_pos=7 threshold=0.38 min_predicted_class=None stop_loss=None take_profit=None
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
top_mean_forward_return        0.036626
class_accuracy                 0.326877
backtest_symbols            78
order_dates                158
first_order_date    2021-01-05
last_order_date     2026-05-21
wrote outputs to /home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/078_threshold_38_fallback3_max7_scores_mild_no_sl
```
