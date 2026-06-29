# Trial 013_rank_score_xgb_lr_005

- Status: discard
- Stop reason: continue
- Objective: sharpe = 0.33867873503709744
- Best value: 1.4502032808497105
- Criteria hit: false
- Description: rank_col=score xgb_learning_rate=0.05

## Metrics

- cagr: 0.030620351670392454
- max_drawdown: 0.29946703840670064
- sharpe: 0.33867873503709744
- sortino: 0.3165539309018852
- win_rate: 0.22948473282442747

## Log Tail

```text
output_dir=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/013_rank_score_xgb_lr_005
config=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/013_rank_score_xgb_lr_005/trial_config.yaml
params: run_id=autoresearch_013_rank_score_xgb_lr_005 fwd=12 cw=120 tw=630 rb=8 max_pos=5 threshold=0.3
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
selected_rows                 16.000000
selected_dates                 9.000000
top_mean_forward_return        0.197781
class_accuracy                 0.307631
backtest_symbols             9
order_dates                  6
first_order_date    2022-05-17
last_order_date     2022-12-07
wrote outputs to /home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/013_rank_score_xgb_lr_005
```
