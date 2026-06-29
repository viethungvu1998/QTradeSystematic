# Trial 019_rank_score_take_profit_30

- Status: discard
- Stop reason: continue
- Objective: sharpe = 1.3692956166786312
- Best value: 1.4502032808497105
- Criteria hit: false max_drawdown
- Description: rank_col=score take_profit=0.30

## Metrics

- cagr: 0.22623214896017796
- max_drawdown: 0.240272802752034
- sharpe: 1.3692956166786312
- sortino: 1.3449189253129907
- win_rate: 0.3077290076335878

## Log Tail

```text
output_dir=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/019_rank_score_take_profit_30
config=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/019_rank_score_take_profit_30/trial_config.yaml
params: run_id=autoresearch_019_rank_score_take_profit_30 fwd=12 cw=120 tw=630 rb=8 max_pos=5 threshold=0.3 stop_loss=None take_profit=0.3
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
selected_rows                261.000000
selected_dates                94.000000
top_mean_forward_return        0.054175
class_accuracy                 0.326877
backtest_symbols            63
order_dates                109
first_order_date    2021-01-05
last_order_date     2026-05-11
wrote outputs to /home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/019_rank_score_take_profit_30
```
