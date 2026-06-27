# Ablation ranking

Latest run per experiment from `ablation_results.tsv` for the active config:

| Item | Value |
| --- | --- |
| forward_period | 21 |
| train_window | 120 |
| rebalance | 12 |
| top_positions | 5 |
| xgb_n_estimators | 5 |
| xgb_learning_rate | 0.100000 |
| xgb_max_depth | 3 |

| Final Rank | Experiment | Feature Groups | Full Sharpe | Full MDD | Active Sharpe | Active MDD | Top Mean Fwd Return | Decision | Run ID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | roc | baseline,roc | 0.690511 | 0.311256 | 1.088664 | 0.311256 | 0.029310 | drop | vn_reference_xgb_ranker_roc_top5_fwd21_rb12_20260623_204153 |
| 2 | volume_ratio | baseline,volume_ratio | 0.662462 | 0.249993 | 1.044324 | 0.249993 | 0.031385 | keep | vn_reference_xgb_ranker_volume_ratio_top5_fwd21_rb12_20260623_203852 |
| 3 | ma_5 | baseline,ma_5 | 0.628319 | 0.255903 | 0.990372 | 0.255903 | 0.029047 | keep | vn_reference_xgb_ranker_ma_5_top5_fwd21_rb12_20260623_203250 |
| 4 | ma_10 | baseline,ma_10 | 0.591631 | 0.280059 | 0.932420 | 0.280059 | 0.022493 | keep | vn_reference_xgb_ranker_ma_10_top5_fwd21_rb12_20260623_203348 |
| 5 | zscore_21 | baseline,zscore_21 | 0.559170 | 0.276642 | 0.881164 | 0.276642 | 0.027886 | keep | vn_reference_xgb_ranker_zscore_21_top5_fwd21_rb12_20260623_203652 |
| 6 | adx_14 | baseline,adx_14 | 0.548333 | 0.244086 | 0.864057 | 0.244086 | 0.024732 | keep | vn_reference_xgb_ranker_adx_14_top5_fwd21_rb12_20260623_203055 |
| 7 | zscore_10 | baseline,zscore_10 | 0.532381 | 0.249993 | 0.838876 | 0.249993 | 0.030644 | keep | vn_reference_xgb_ranker_zscore_10_top5_fwd21_rb12_20260623_203550 |
| 8 | adx_21 | baseline,adx_21 | 0.531724 | 0.297235 | 0.837840 | 0.297235 | 0.028000 | drop | vn_reference_xgb_ranker_adx_21_top5_fwd21_rb12_20260623_203152 |
| 9 | baseline | baseline | 0.491709 | 0.264189 | 0.774695 | 0.264189 | 0.029865 | baseline | vn_reference_xgb_ranker_baseline_top5_fwd21_rb12_20260623_202958 |
| 10 | volume_ratio+ma_5+ma_10+zscore_21+adx_14+zscore_10 | baseline,volume_ratio,ma_5,ma_10,zscore_21,adx_14,zscore_10 | 0.435000 | 0.227908 | 0.685244 | 0.227908 | 0.024065 | drop | vn_reference_xgb_ranker_volume_ratio_ma_5_ma_10_zscore_21_adx_14_zscore_10_top5_fwd21_rb12_20260623_204330 |
| 11 | ma_20 | baseline,ma_20 | 0.406575 | 0.327538 | 0.640422 | 0.327538 | 0.019278 | drop | vn_reference_xgb_ranker_ma_20_top5_fwd21_rb12_20260623_203450 |
| 12 | rsi | baseline,rsi | 0.369965 | 0.294798 | 0.582705 | 0.294798 | 0.026678 | drop | vn_reference_xgb_ranker_rsi_top5_fwd21_rb12_20260623_204053 |
| 13 | hist_vol | baseline,hist_vol | 0.341592 | 0.348074 | 0.537985 | 0.348074 | 0.024430 | drop | vn_reference_xgb_ranker_hist_vol_top5_fwd21_rb12_20260623_203951 |
| 14 | zscore_42 | baseline,zscore_42 | 0.320033 | 0.294436 | 0.504011 | 0.294436 | 0.021969 | drop | vn_reference_xgb_ranker_zscore_42_top5_fwd21_rb12_20260623_203753 |

Decision rule: keep only if a group beats baseline on at least one Sharpe metric without increasing full or active max drawdown by more than 0.03; drawdown above 0.35 is disqualified.