# Trial 041_rank_score_fallback_3_max_positions_6_stop_loss_20_exit_only

- Status: keep
- Stop reason: continue
- Objective: sharpe = 1.8273926444640762
- Best value: 1.8273926444640762
- Criteria hit: false sharpe
- Description: rank_col=score fallback_min_positions=3 max_positions=6 exit-only stop_loss=0.20

## Metrics

- cagr: 0.38172327492591096
- max_drawdown: 0.26038185640051315
- sharpe: 1.8273926444640762
- sortino: 2.0525727083361436
- win_rate: 0.3640267175572519

## Log Tail

```text
output_dir=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/041_rank_score_fallback_3_max_positions_6_stop_loss_20_exit_only
config=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/041_rank_score_fallback_3_max_positions_6_stop_loss_20_exit_only/trial_config.yaml
params: run_id=autoresearch_041_rank_score_fallback_3_max_positions_6_stop_loss_20_exit_only fwd=12 cw=120 tw=630 rb=8 max_pos=6 threshold=0.3 stop_loss=0.2 take_profit=None
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
selected_rows                581.000000
selected_dates               167.000000
top_mean_forward_return        0.039511
class_accuracy                 0.326877
backtest_symbols            80
order_dates                165
first_order_date    2021-01-05
last_order_date     2026-05-21
wrote outputs to /home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/041_rank_score_fallback_3_max_positions_6_stop_loss_20_exit_only
```
