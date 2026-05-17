"""XGBoost-based feature importance ranking."""

from __future__ import annotations

import polars as pl


def get_feature_importance(
    df: pl.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
    model_params: dict | None = None,
) -> pl.DataFrame:
    """Rank features by XGBoost gain importance against target_col.

    Parameters
    ----------
    df:
        DataFrame containing features and target_col. Rows with nulls in any
        feature or target are dropped before fitting.
    target_col:
        The column to predict (e.g. a forward return).
    feature_cols:
        Columns to use as features. Defaults to all non-OHLCV, non-signal columns
        except target_col.
    model_params:
        XGBoost parameters passed to XGBRegressor. Defaults: n_estimators=100,
        max_depth=4, learning_rate=0.1, random_state=42.

    Returns
    -------
    pl.DataFrame with columns [feature, importance] sorted descending by importance.
    Raises ImportError if xgboost is not installed.
    """
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("xgboost is not installed: pip install xgboost") from exc

    _exclude = {"date", "symbol", "open", "high", "low", "close", "volume", "signal", "weight"}
    if feature_cols is None:
        feature_cols = [
            c for c in df.columns
            if c not in _exclude and c != target_col and not c.startswith("forward_return_")
        ]

    needed = feature_cols + [target_col]
    clean = df.select([c for c in needed if c in df.columns]).drop_nulls()
    if clean.is_empty() or not feature_cols:
        return pl.DataFrame(schema={"feature": pl.String, "importance": pl.Float64})

    params = {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1, "random_state": 42}
    params.update(model_params or {})

    X = clean.select(feature_cols).to_pandas()
    y = clean[target_col].to_pandas()

    model = xgb.XGBRegressor(**params)
    model.fit(X, y)

    scores = model.get_booster().get_score(importance_type="gain")
    return (
        pl.DataFrame({"feature": list(scores.keys()), "importance": list(scores.values())})
        .sort("importance", descending=True)
    )
