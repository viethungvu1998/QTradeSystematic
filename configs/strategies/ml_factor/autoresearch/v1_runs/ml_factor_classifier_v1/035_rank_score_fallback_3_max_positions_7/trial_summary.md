# Trial 035_rank_score_fallback_3_max_positions_7

- Status: discard
- Stop reason: continue
- Objective: sharpe = 1.649991358248622
- Best value: 1.8075242427328562
- Criteria hit: false
- Description: rank_col=score fallback_min_positions=3 max_positions=7

## Metrics

- cagr: 0.3108435153920437
- max_drawdown: 0.2899535654793481
- sharpe: 1.649991358248622
- sortino: 1.7684279076570237
- win_rate: 0.36593511450381677

## Log Tail

```text
output_dir=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/035_rank_score_fallback_3_max_positions_7
config=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/035_rank_score_fallback_3_max_positions_7/trial_config.yaml
params: run_id=autoresearch_035_rank_score_fallback_3_max_positions_7 fwd=12 cw=120 tw=630 rb=8 max_pos=7 threshold=0.3 stop_loss=None take_profit=None
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
selected_rows                597.000000
selected_dates               167.000000
top_mean_forward_return        0.038885
class_accuracy                 0.326877
backtest_symbols            80
order_dates                157
first_order_date    2021-01-05
last_order_date     2026-05-21
wrote outputs to /home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/035_rank_score_fallback_3_max_positions_7
```
