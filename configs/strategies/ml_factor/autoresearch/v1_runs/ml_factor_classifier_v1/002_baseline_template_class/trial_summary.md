# Trial 002_baseline_template_class

- Status: crash
- Stop reason: continue
- Objective: sharpe = None
- Best value: None
- Criteria hit: false
- Description: baseline after plotly template class compatibility fix

## Log Tail

```text
Traceback (most recent call last):
  File "/home/hungvu/code/quant/QTradeSystematic/notebooks/ml_factor_classifier.py", line 990, in <module>
    main()
    ~~~~^^
  File "/home/hungvu/code/quant/QTradeSystematic/notebooks/ml_factor_classifier.py", line 903, in main
    load_config(args.config, repo_root)
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/hungvu/code/quant/QTradeSystematic/notebooks/ml_factor_classifier.py", line 156, in load_config
    payload = merge(read_yaml(Path(payload["default_params_path"])), payload)
                    ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/hungvu/code/quant/QTradeSystematic/notebooks/ml_factor_classifier.py", line 126, in read_yaml
    raise FileNotFoundError(f"YAML config not found: {path}")
FileNotFoundError: YAML config not found: /Users/s2997726/Desktop/code/quant/QS/QTradeSystematic/configs/notebooks/stock_ml_e2e_classification/default_params.yaml
```
