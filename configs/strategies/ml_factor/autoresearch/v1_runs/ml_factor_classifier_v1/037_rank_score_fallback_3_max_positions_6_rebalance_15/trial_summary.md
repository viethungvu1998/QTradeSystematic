# Trial 037_rank_score_fallback_3_max_positions_6_rebalance_15

- Status: discard
- Stop reason: continue
- Objective: sharpe = 1.3102949873354137
- Best value: 1.8075242427328562
- Criteria hit: false
- Description: rank_col=score fallback_min_positions=3 max_positions=6 rebalance_frequency=15

## Metrics

- cagr: 0.25407030151157617
- max_drawdown: 0.2908036205090187
- sharpe: 1.3102949873354137
- sortino: 1.3700626308584287
- win_rate: 0.3511450381679389

## Log Tail

```text
output_dir=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/037_rank_score_fallback_3_max_positions_6_rebalance_15
config=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/037_rank_score_fallback_3_max_positions_6_rebalance_15/trial_config.yaml
params: run_id=autoresearch_037_rank_score_fallback_3_max_positions_6_rebalance_15 fwd=12 cw=120 tw=630 rb=15 max_pos=6 threshold=0.3 stop_loss=None take_profit=None
loaded raw: 202669 rows, 101 symbols
rows                   200574
dates                    2096
symbols                   100
feature_ready_rows     177774
training_ready_rows    176574
training rebalance 1/89: 2021-01-21
training rebalance 25/89: 2022-07-06
training rebalance 50/89: 2024-01-02
training rebalance 75/89: 2025-07-09
prediction_rows            8593.000000
prediction_dates             88.000000
selected_rows               307.000000
selected_dates               88.000000
top_mean_forward_return       0.040002
class_accuracy                0.330967
backtest_symbols            70
order_dates                 89
first_order_date    2021-01-22
last_order_date     2026-05-18
wrote outputs to /home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/037_rank_score_fallback_3_max_positions_6_rebalance_15
```
