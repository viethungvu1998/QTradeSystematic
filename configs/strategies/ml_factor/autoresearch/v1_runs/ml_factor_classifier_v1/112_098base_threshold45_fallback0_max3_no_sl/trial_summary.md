# Trial 112_098base_threshold45_fallback0_max3_no_sl

- Status: crash
- Stop reason: continue
- Objective: sharpe = None
- Best value: 1.9132302197463205
- Criteria hit: false
- Description: base 098 probability_threshold=0.45 class_quantiles=0.15,0.35,0.55,0.70 class_scores=-2,-1,0,1,3 fallback_min_positions=0 max_positions=3 xgb_n_estimators=10 xgb_max_depth=3 no SL/TP

## Log Tail

```text
output_dir=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/112_098base_threshold45_fallback0_max3_no_sl
config=/home/hungvu/code/quant/QTradeSystematic/configs/strategies/ml_factor/autoresearch/v1_runs/ml_factor_classifier_v1/112_098base_threshold45_fallback0_max3_no_sl/trial_config.yaml
params: run_id=autoresearch_112_098base_threshold45_fallback0_max3_no_sl fwd=12 cw=120 threshold_scope=classification_window balanced_sample_weight=False tw=630 rb=8 max_pos=3 threshold=0.45 min_predicted_class=None stop_loss=None take_profit=None
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
  File "/home/hungvu/code/quant/QTradeSystematic/notebooks/ml_factor_classifier.py", line 1129, in <module>
    main()
    ~~~~^^
  File "/home/hungvu/code/quant/QTradeSystematic/notebooks/ml_factor_classifier.py", line 1070, in main
    predictions, selected_predictions, selected_eval, run_eval = run_predictions(
                                                                 ~~~~~~~~~~~~~~~^
        model_frame,
        ^^^^^^^^^^^^
        args.output_dir,
        ^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/hungvu/code/quant/QTradeSystematic/notebooks/ml_factor_classifier.py", line 640, in run_predictions
    raise RuntimeError(
    ...<4 lines>...
    )
RuntimeError: No predictions passed signal filters: class_3_4_prob>0.45, min_predicted_class=None, fallback_min_positions=0.
```
