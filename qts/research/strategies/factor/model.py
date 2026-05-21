"""Factor model strategy."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial

import numpy as np
import pandas as pd
import polars as pl

from qts.core.registry import Registry

from .base import BaseFactorStrategy


def _resolve_named_section(raw: object, section_name: str) -> dict:
    """Normalise a YAML sub-section into {'name': str, 'params': dict}."""
    if isinstance(raw, str):
        return {"name": raw, "params": {}}
    if isinstance(raw, dict):
        return {"name": str(raw["name"]), "params": dict(raw.get("params", {}))}
    raise ValueError(f"Cannot resolve {section_name!r} section from {raw!r}")


@Registry.register_strategy("factor")
class FactorStrategy(BaseFactorStrategy):
    """Simple factor ranking strategy."""

    def __init__(
        self,
        long_quantile: float = 0.7,
        short_quantile: float = 0.3,
        predictor_cols: list[str] | None = None,
        algorithm_func: Callable | None = None,
        portfolio_func: Callable | None = None,
    ) -> None:
        self.long_quantile = long_quantile
        self.short_quantile = short_quantile
        self.predictor_cols = predictor_cols or []
        self.algorithm_func = algorithm_func
        self.portfolio_func = portfolio_func

    @classmethod
    def from_config_params(
        cls,
        params: Mapping[str, object],
        *,
        portfolio_func: Callable | None = None,
    ) -> FactorStrategy:
        payload = dict(params)

        predictor_cols = [str(column) for column in payload.pop("predictor_cols", [])]

        algorithm_raw = payload.pop("algorithm", {"name": "cross_sectional_rank"})
        algorithm_cfg = _resolve_named_section(algorithm_raw, "algorithm")
        algorithm_fn = Registry.get_signal_algorithm(algorithm_cfg["name"])
        algorithm_func = partial(
            algorithm_fn,
            predictor_cols=predictor_cols,
            **algorithm_cfg["params"],
        )

        if portfolio_func is None and "portfolio" in payload:
            portfolio_raw = _resolve_named_section(payload.pop("portfolio"), "portfolio")
            port_fn = Registry.get_portfolio_constructor(portfolio_raw["name"])
            portfolio_func = partial(port_fn, **portfolio_raw["params"])
        else:
            payload.pop("portfolio", None)

        return cls(
            predictor_cols=predictor_cols,
            algorithm_func=algorithm_func,
            portfolio_func=portfolio_func,
            **payload,
        )

    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        if data.is_empty():
            return self.empty_signal_frame()

        if self.algorithm_func is not None and self.portfolio_func is not None:
            last_date = data["date"].max()
            scores: pd.Series = self.algorithm_func(data)
            weights: pd.Series = self.portfolio_func(scores)
            return self.signal_frame_from_weights(last_date, weights)

        feature_columns = self.feature_columns(data)
        if not feature_columns:
            frame = data.select("date", "symbol").with_columns(
                pl.lit(0).alias("signal"),
                pl.lit(0.0).alias("weight"),
            )
            return self.validate_signal_frame(frame)

        zscore_columns = [f"_{column}_zscore" for column in feature_columns]
        scored = data.with_columns(
            [
                (
                    (pl.col(column) - pl.col(column).mean().over("date"))
                    / (pl.col(column).std().over("date") + 1e-8)
                )
                .fill_null(0.0)
                .alias(alias)
                for column, alias in zip(feature_columns, zscore_columns, strict=True)
            ]
        ).with_columns(
            (sum(pl.col(column) for column in zscore_columns) / len(zscore_columns)).alias("factor_score")
        )
        rows: list[dict[str, object]] = []
        for item in scored.partition_by("date", as_dict=False):
            scores = item["factor_score"].fill_null(0).to_numpy()
            long_cutoff = float(np.quantile(scores, self.long_quantile))
            short_cutoff = float(np.quantile(scores, self.short_quantile))
            max_abs_score = float(np.abs(scores).max()) if scores.size else 0.0
            for record in item.iter_rows(named=True):
                score = float(record["factor_score"] or 0)
                signal = 1 if score >= long_cutoff else -1 if score <= short_cutoff else 0
                weight = min(1.0, abs(score) / (max_abs_score + 1e-8)) if signal else 0.0
                rows.append(
                    {
                        "date": record["date"],
                        "symbol": record["symbol"],
                        "signal": signal,
                        "weight": float(weight),
                    }
                )
        if not rows:
            return self.empty_signal_frame()
        return self.validate_signal_frame(pl.DataFrame(rows))
