# Classifier and Ranker Correct-Version Audit

## Confirmed high-version source reports

| Component | Run ID | Sharpe | Max DD | CAGR | Export/Data Root |
|---|---|---:|---:|---:|---|
| Classifier ensemble | vn_ml_classifier_weight_ensemble_20260625_014751 | 1.5722 | 0.2009 | 0.3208 | /Users/s2997726/.qts |
| Ranker ensemble | vn_ml_ranker_weight_ensemble_20260625_013317 | 1.3824 | 0.2053 | 0.2395 | .qts_notebooke_runtime/isolated_qts_root |

## Single-root hybrid rerun evidence

| Rerun | Family | Run ID | Sharpe | Max DD | CAGR |
|---|---|---|---:|---:|---:|
| isolated root rerun | classifier | vn_ml_classifier_weight_ensemble_20260625_022626 | 1.1251 | 0.2416 | 0.2029 |
| isolated root rerun | ranker | vn_ml_ranker_weight_ensemble_20260625_022626 | 1.3824 | 0.2053 | 0.2395 |
| isolated root rerun | hybrid | vn_ml_classifier_gated_ranker_hybrid_20260625_022626 | 1.0182 | 0.1036 | 0.0706 |
| default root rerun | classifier | vn_ml_classifier_weight_ensemble_20260625_024029 | 1.5722 | 0.2009 | 0.3208 |
| default root rerun | ranker | vn_ml_ranker_weight_ensemble_20260625_024029 | 1.0438 | 0.2294 | 0.1734 |
| default root rerun | hybrid | vn_ml_classifier_gated_ranker_hybrid_20260625_024029 | 1.0421 | 0.2025 | 0.1256 |

## Conclusion

The high classifier and high ranker reports are both valid reports, but they are not from the same QTS data root.

Classifier high source uses /Users/s2997726/.qts. Ranker high source uses .qts_notebooke_runtime/isolated_qts_root.

Therefore a single honest hybrid backtest cannot claim both high component Sharpe values unless the data roots are synchronized and both components are rebuilt on one root.

The default-root hybrid reproduces the high classifier but not the high ranker. The isolated-root hybrid reproduces the high ranker but not the high classifier.

