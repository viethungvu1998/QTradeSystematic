# Manifest Example

Place the manifest under the algorithm artifact folder:

```yaml
version: 1
run_tag: vn100_ml_factor_20260518
strategy: ml_factor
algorithm: vn100_quantamental

quota:
  max_trials: 50

objective:
  metric: sharpe
  direction: maximize
  min_delta: 0.0

criteria:
  mode: any
  targets:
    - metric: win_rate
      op: ">="
      value: 0.75
    - metric: sharpe
      op: ">"
      value: 2.0
    - metric: max_drawdown
      op: "<"
      value: 0.2

research_limits:
  allowed_actions:
    - id: parameter_tuning
      description: Tweak existing strategy and configuration parameters only.
      aliases: [parameters, tweak_parameters, model_params]
    - id: hyperparameter_optimization
      description: Run bounded hyperparameter optimization using existing sweep/config seams.
      aliases: [hyperparams, hyperparam_opt, hyperopt]
    - id: feature_engineering
      description: Create new non-leaking feature engineering processes.
      aliases: [features, non_leaking_features]
    - id: strategy_model_update
      description: Update the strategy model or trainer implementation inside allowed roots.
      aliases: [strategy_model, model_update, trainer]

paths:
  artifact_root: qts/research/vn100_quantamental/autoresearch
  run_root: .qts_notebook_runtime/vn100_quantamental_mlflow/runs
  allowed_edit_roots:
    - qts/research/strategies/ml_factor
    - configs/strategies/ml_factor

command:
  argv:
    - .venv/bin/python
    - notebooks/vn100_quantamental_mlflow_classification.py
    - --strategy-config
    - configs/strategies/ml_factor/base.yaml
    - --output-dir
    - "{run_dir}"
    - --run-name
    - "{run_id}"
```

Supported placeholders in `command.argv`:

- `{run_dir}`: the trial artifact directory.
- `{run_id}`: the trial id passed to `run`.
- `{artifact_root}`: the manifest artifact root.
- `{run_root}`: the root directory used for per-trial run outputs.
- `{repo_root}`: the QTradeSystematic project root.
- `{manifest_dir}`: the directory containing `autoresearch.yaml`.

Default artifact layout:

```text
qts/research/<algorithm>/autoresearch/
  autoresearch.yaml
  results.tsv
  events.jsonl
  summary.md

.qts_notebook_runtime/vn100_quantamental_mlflow/runs/
  <run_tag>/
    000_baseline/
      run.log
      metrics.json
      trial_summary.md
```
