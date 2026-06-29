# Trial 175_098base_rank_probsum_fallback0_max4_sl05_close_no_regime

- Status: crash
- Stop reason: continue
- Objective: sharpe = None
- Best value: 1.9132302197463205
- Criteria hit: false
- Description: base 098 probability_threshold=0.45 class_quantiles=0.15,0.35,0.55,0.70 class_scores=-2,-1,0,1,3 rank_col=class_3_4_prob fallback_min_positions=0 max_positions=4 xgb_n_estimators=10 xgb_max_depth=3 stop_loss=0.05 close-trigger SL/TP rule with benchmark_regime disabled

## Log Tail

```text
output_dir=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/175_098base_rank_probsum_fallback0_max4_sl05_close_no_regime
config=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/175_098base_rank_probsum_fallback0_max4_sl05_close_no_regime/trial_config.yaml
params: run_id=autoresearch_175_098base_rank_probsum_fallback0_max4_sl05_close_no_regime fwd=12 cw=120 threshold_scope=classification_window balanced_sample_weight=False tw=630 rb=8 max_pos=4 threshold=0.45 min_predicted_class=None stop_loss=0.05 stop_loss_min_hold=1 take_profit=None trailing_stop=None benchmark_regime_enabled=False
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
Traceback (most recent call last):
  File "/home/hungvu/code/quant/QTradeSystematic/notebooks/ml_factor_classifier.py", line 1203, in <module>
    main()
    ~~~~^^
  File "/home/hungvu/code/quant/QTradeSystematic/notebooks/ml_factor_classifier.py", line 1144, in main
    predictions, selected_predictions, selected_eval, run_eval = run_predictions(
                                                                 ~~~~~~~~~~~~~~~^
        model_frame,
        ^^^^^^^^^^^^
        args.output_dir,
        ^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/hungvu/code/quant/QTradeSystematic/notebooks/ml_factor_classifier.py", line 660, in run_predictions
    raise RuntimeError(
    ...<4 lines>...
    )
RuntimeError: No predictions passed signal filters: class_3_4_prob>0.45, min_predicted_class=None, fallback_min_positions=0.
```
