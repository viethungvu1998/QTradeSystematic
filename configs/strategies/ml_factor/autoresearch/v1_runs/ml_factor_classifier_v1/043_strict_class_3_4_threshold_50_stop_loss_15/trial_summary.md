# Trial 043_strict_class_3_4_threshold_50_stop_loss_15

- Status: crash
- Stop reason: continue
- Objective: sharpe = None
- Best value: 1.8280890880755754
- Criteria hit: false
- Description: strict predicted_class>=3 and class_3_4_prob>0.50 with no fallback

## Log Tail

```text
first_order_date    NaT
last_order_date     NaT
Traceback (most recent call last):
  File "/home/hungvu/code/quant/QTradeSystematic/notebooks/ml_factor_classifier.py", line 1049, in <module>
    main()
    ~~~~^^
  File "/home/hungvu/code/quant/QTradeSystematic/notebooks/ml_factor_classifier.py", line 1006, in main
    portfolio, equity, returns, qts_metrics, portfolio_stats = run_portfolio(close_bt, weights_bt)
                                                               ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
  File "/home/hungvu/code/quant/QTradeSystematic/notebooks/ml_factor_classifier.py", line 718, in run_portfolio
    portfolio = vbt.Portfolio.from_orders(
        close=close_bt,
    ...<6 lines>...
        freq=pd.tseries.frequencies.to_offset(close_bt.index[1] - close_bt.index[0]),
    )
  File "/home/hungvu/code/quant/QTradeSystematic/.venv/lib/python3.13/site-packages/vectorbtpro/portfolio/base.py", line 2883, in from_orders
    prep_result = preparer.result
                  ^^^^^^^^^^^^^^^
  File "/home/hungvu/miniconda3/lib/python3.13/functools.py", line 1026, in __get__
    val = self.func(instance)
  File "/home/hungvu/code/quant/QTradeSystematic/.venv/lib/python3.13/site-packages/vectorbtpro/portfolio/preparing.py", line 561, in result
    return PFPrepResult(target_func=self.target_func, target_args=self.target_args, pf_args=self.pf_args)
                                                                  ^^^^^^^^^^^^^^^^
  File "/home/hungvu/miniconda3/lib/python3.13/functools.py", line 1026, in __get__
    val = self.func(instance)
  File "/home/hungvu/code/quant/QTradeSystematic/.venv/lib/python3.13/site-packages/vectorbtpro/base/preparing.py", line 614, in target_args
    if arg_attr is not None and hasattr(self, arg_attr):
                                ~~~~~~~^^^^^^^^^^^^^^^^
  File "/home/hungvu/miniconda3/lib/python3.13/functools.py", line 1026, in __get__
    val = self.func(instance)
  File "/home/hungvu/code/quant/QTradeSystematic/.venv/lib/python3.13/site-packages/vectorbtpro/portfolio/preparing.py", line 838, in max_order_records
    max_order_records = int(np.max(np.sum(~np.isnan(_size), axis=0)))
                            ~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/hungvu/code/quant/QTradeSystematic/.venv/lib/python3.13/site-packages/numpy/_core/fromnumeric.py", line 3123, in max
    return _wrapreduction(a, np.maximum, 'max', axis, None, out,
                          keepdims=keepdims, initial=initial, where=where)
  File "/home/hungvu/code/quant/QTradeSystematic/.venv/lib/python3.13/site-packages/numpy/_core/fromnumeric.py", line 83, in _wrapreduction
    return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
           ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: zero-size array to reduction operation maximum which has no identity
```
