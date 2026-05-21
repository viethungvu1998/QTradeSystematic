"""ML training wrappers for factor strategies.

All public functions accept polars DataFrames or pandas DataFrames and return
numpy arrays of predictions aligned to the predict_data index.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd



def _to_pandas(frame: object, *, label: str = "frame") -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame
    if hasattr(frame, "to_pandas"):
        result = frame.to_pandas()
        if isinstance(result, pd.DataFrame):
            return result
    raise TypeError(f"{label} must be a pandas or polars DataFrame; got {type(frame)!r}")


def train_and_predict_xgb_regressor(
    train_data: object,
    predict_data: object,
    predictor_cols: list[str],
    target_col: str,
    model_params: dict,
) -> np.ndarray:
    """Fit an XGBoost regressor and return predictions for predict_data."""
    from qts.research.strategies.ml_factor.models import fit_xgb_regressor

    train = _to_pandas(train_data, label="train_data")
    predict = _to_pandas(predict_data, label="predict_data")

    X_train = train[predictor_cols]
    y_train = train[target_col]
    X_predict = predict[predictor_cols]

    model = fit_xgb_regressor(X_train, y_train, model_params)
    return model.predict(X_predict)


def train_and_predict_xgb_ranker(
    train_data: object,
    predict_data: object,
    predictor_cols: list[str],
    target_col: str,
    model_params: dict,
) -> np.ndarray:
    """Fit an XGBoost ranker (per-date groups) and return predicted scores."""
    from qts.research.strategies.ml_factor.models import fit_xgb_ranker

    train = _to_pandas(train_data, label="train_data").sort_values("date").reset_index(drop=True)
    predict = _to_pandas(predict_data, label="predict_data")

    train["_group_id"] = pd.factorize(train["date"])[0]
    qid = train["_group_id"].values

    model = fit_xgb_ranker(train[predictor_cols], train[target_col], qid, model_params)
    return model.predict(predict[predictor_cols])


def train_and_predict_linear_regression(
    data: object,
    predictor_cols: Sequence[str],
    target_col: str = "forward_return",
    model_params: dict | None = None,
) -> pd.Series:
    """Fit a sklearn linear model on the cross-section and return in-sample predictions.

    model_params keys: model ("linear"|"ridge"|"lasso"|"elasticnet"), alpha, l1_ratio,
    standardize, add_intercept.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    cfg = model_params or {}
    df = _to_pandas(data, label="data")
    features = [c for c in predictor_cols if c in df.columns]
    if not features or target_col not in df.columns:
        return pd.Series(dtype=float)

    X_raw = df[features]
    y_raw = df[target_col]
    mask = X_raw.notna().all(axis=1) & y_raw.notna()
    if not mask.any():
        return pd.Series(dtype=float)

    X, y = X_raw.loc[mask].copy(), y_raw.loc[mask].copy()
    # Drop zero-variance features
    features = [c for c in X.columns if pd.to_numeric(X[c], errors="coerce").std() > 0]
    if not features:
        return pd.Series(dtype=float)
    X = X[features]

    model_name = str(cfg.get("model", "linear")).lower()
    alpha = float(cfg.get("alpha", 1.0))
    l1_ratio = float(cfg.get("l1_ratio", 0.5))
    add_intercept = bool(cfg.get("add_intercept", True))
    standardize = bool(cfg.get("standardize", True))

    model_cls = {
        "linear": LinearRegression(fit_intercept=add_intercept),
        "ridge": Ridge(alpha=alpha, fit_intercept=add_intercept),
        "lasso": Lasso(alpha=alpha, fit_intercept=add_intercept),
        "elasticnet": ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=add_intercept),
    }.get(model_name)
    if model_cls is None:
        return pd.Series(dtype=float)

    steps: list = [("impute", SimpleImputer(strategy="median"))]
    if standardize:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model_cls))
    pipe = Pipeline(steps)
    pipe.fit(X, y)

    X_pred = df.loc[mask, features]
    preds = pipe.predict(X_pred)
    return pd.Series(preds, index=df.index[mask], name="y_pred")


def train_and_predict_ic_composite(
    train_data: object,
    predict_data: object,
    factor_cols: Sequence[str],
    future_return_cols: dict[int, str],
    composite_params: dict | None = None,
    factor_horizons: dict[str, int] | None = None,
    default_horizon: int | None = None,
) -> np.ndarray:
    """IC-weighted composite factor scoring.

    Computes per-date Spearman IC for each factor against its forward return,
    applies EWMA weighting with sign-aware guardrails, then scores predict_data.

    Parameters
    ----------
    train_data:
        Panel data with columns [date, symbol] + factor_cols + forward return cols.
    predict_data:
        Cross-section for the prediction date containing factor_cols.
    factor_cols:
        Factor columns to combine.
    future_return_cols:
        {horizon_bars: column_name} mapping of available forward returns in train_data.
    composite_params:
        Optional config: method ("ewma_ic"|"ewma_ir"), half_life, winsorize_q,
        min_abs_ic, min_pos_frac, min_dates, train_sessions, default_sign.
    factor_horizons:
        Per-factor preferred horizon. Falls back to default_horizon.
    default_horizon:
        Horizon to use when a factor has no explicit entry. Defaults to max key in
        future_return_cols.
    """
    from scipy.stats import spearmanr

    train = _to_pandas(train_data, label="train_data")
    predict = _to_pandas(predict_data, label="predict_data")

    cfg = (composite_params or {}).copy()
    factor_cols = [c for c in factor_cols if c in train.columns and c in predict.columns]
    if not factor_cols:
        raise RuntimeError("no_factors")

    method = str(cfg.get("method", "ewma_ic")).lower()
    train_sessions = int(cfg.get("train_sessions", 252))
    wins_q = float(cfg.get("winsorize_q", 0.01))
    hl_days = float(cfg.get("half_life", 63))
    min_dates = int(cfg.get("min_dates", 60))
    min_abs_ic = float(cfg.get("min_abs_ic", 0.02))
    min_pos_frac = float(cfg.get("min_pos_frac", 0.55))
    default_sign = int(np.sign(cfg.get("default_sign", 1))) or 1

    future_return_cols = {int(k): v for k, v in future_return_cols.items() if v in train.columns}
    if not future_return_cols:
        raise RuntimeError("future_returns_unavailable")
    if default_horizon is None:
        default_horizon = max(future_return_cols.keys())

    horizon_map: dict[str, int] = {}
    for c in factor_cols:
        h = (factor_horizons or {}).get(c)
        try:
            h = int(h) if h is not None else None
        except Exception:
            h = None
        if h not in future_return_cols:
            h = default_horizon
        if h not in future_return_cols:
            h = next(iter(future_return_cols))
        horizon_map[c] = h

    # Drop rows missing required forward returns
    req_cols = {future_return_cols[horizon_map[c]] for c in factor_cols}
    train = train.dropna(subset=list(req_cols))
    if train.empty:
        raise RuntimeError("no_train_rows")

    # Trim to train_sessions most recent dates
    train_dates = pd.DatetimeIndex(pd.to_datetime(train["date"]).unique()).sort_values()
    if len(train_dates) > train_sessions:
        keep = set(train_dates[-train_sessions:])
        train = train[pd.to_datetime(train["date"]).isin(keep)]
    if train["date"].nunique() < min_dates:
        raise RuntimeError("too_few_dates")

    def _winsorize(g: pd.DataFrame) -> pd.DataFrame:
        for c in factor_cols:
            if c in g:
                lo, hi = g[c].quantile([wins_q, 1 - wins_q])
                g[c] = g[c].clip(lo, hi)
        return g

    def _zscore_cross(g: pd.DataFrame) -> pd.DataFrame:
        for c in factor_cols:
            s = g[c]
            std = s.std(ddof=0)
            g[c] = (s - s.mean()) / std if std and np.isfinite(std) else 0.0
        return g

    proc = train.sort_values(["symbol", "date"]).copy()
    if 0.0 < wins_q < 0.5:
        _dates = proc["date"].values
        proc = proc.groupby("date", group_keys=False).apply(_winsorize)
        if "date" not in proc.columns:  # pandas 3.0 drops groupby key column
            proc["date"] = _dates
    _dates = proc["date"].values
    proc = proc.groupby("date", group_keys=False).apply(_zscore_cross)
    if "date" not in proc.columns:  # pandas 3.0 drops groupby key column
        proc["date"] = _dates

    # Compute daily IC per factor
    def _daily_ic(col: str, ret_col: str) -> pd.Series:
        def _ic(g: pd.DataFrame) -> float:
            gg = g[[col, ret_col]].dropna()
            if len(gg) < 3:
                return np.nan
            rho, _ = spearmanr(gg[col], gg[ret_col], nan_policy="omit")
            return float(rho) if np.isfinite(rho) else np.nan
        return proc.groupby("date").apply(_ic).astype(float)

    ic_df = pd.DataFrame({c: _daily_ic(c, future_return_cols[horizon_map[c]]) for c in factor_cols}).sort_index()

    def _ewma_last(s: pd.Series, hl: float) -> float:
        s = s.dropna()
        return float(s.ewm(halflife=hl, adjust=False, min_periods=1).mean().iloc[-1]) if not s.empty else np.nan

    def _ewma_pos_frac(s: pd.Series, hl: float) -> float:
        s = (s > 0).astype(float)
        return float(s.ewm(halflife=hl, adjust=False, min_periods=1).mean().iloc[-1]) if not s.dropna().empty else np.nan

    ic_ewma = ic_df.apply(lambda s: _ewma_last(s, hl_days))
    ic_ewma_std = ic_df.apply(lambda s: _ewma_last((s - s.ewm(halflife=hl_days, adjust=False).mean()).abs(), hl_days))
    pos_frac = ic_df.apply(lambda s: _ewma_pos_frac(s, hl_days))

    raw_w = ic_ewma / ic_ewma_std.replace(0, np.nan) if method in {"ewma_ir", "ir_weighted"} else ic_ewma.copy()
    raw_w = raw_w.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    adj_w = pd.Series(0.0, index=raw_w.index, dtype=float)
    for c in raw_w.index:
        ic_now = float(ic_ewma.get(c, 0.0))
        pos_now = float(pos_frac.get(c, np.nan))
        guardrail_ok = (
            abs(ic_now) >= min_abs_ic
            and ((ic_now > 0 and pos_now >= min_pos_frac) or (ic_now < 0 and (1 - pos_now) >= min_pos_frac))
        )
        signed = np.sign(ic_now) * abs(float(raw_w.get(c, 0.0))) if guardrail_ok else default_sign * abs(float(raw_w.get(c, 0.0)))
        adj_w[c] = signed

    l1 = adj_w.abs().sum()
    weights = adj_w / l1 if l1 > 0 else pd.Series(1.0 / len(factor_cols), index=factor_cols)

    # Score predict_data
    preds_proc = predict.copy()
    if 0.0 < wins_q < 0.5:
        for c in factor_cols:
            lo, hi = preds_proc[c].quantile([wins_q, 1 - wins_q])
            preds_proc[c] = preds_proc[c].clip(lo, hi)
    for c in factor_cols:
        s = preds_proc[c]
        std = s.std(ddof=0)
        preds_proc[c] = (s - s.mean()) / std if std and np.isfinite(std) else 0.0

    return preds_proc[factor_cols].mul(weights, axis=1).sum(axis=1).values
