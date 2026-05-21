"""Family base for stat-arb strategies."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
import polars as pl

from qts.research.strategies.base import BaseStrategy

from .core import (
    clip_hedge_ratio,
    compute_rolling_ols_spread,
    compute_spread,
    compute_zscore,
    ensure_pair_list,
    estimate_hedge_ratio,
    find_cointegrated_pairs,
    generate_zscore_signals,
    preselect_pairs_by_correlation,
)


class BaseStatArbStrategy(BaseStrategy):
    """Shared universe-level stat-arb processing."""

    def __init__(
        self,
        *,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.0,
        zscore_window: int = 22,
        hedge_method: str = "ols",
        stop_z: float | None = None,
        max_holding_bars: int | None = None,
        side: str = "long_short",
        pairs: Sequence[Sequence[str]] | None = None,
        pvalue_threshold: float = 0.05,
        max_pairs: int = 20,
        min_obs: int = 120,
        fallback_pairs: int = 3,
        prefilter_correlation: bool = True,
        prefilter_min_corr: float = 0.6,
        pairs_per_symbol: int | None = 5,
        hedge_window: int | None = None,
        hedge_ratio_min: float | None = None,
        hedge_ratio_max: float | None = None,
        hedge_ratio_ewm_span: int | None = None,
        allow_negative_hedge_ratio: bool = True,
        spread_fn: Callable | None = None,
        signal_fn: Callable | None = None,
        portfolio_func: Callable | None = None,
    ) -> None:
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.zscore_window = zscore_window
        self.hedge_method = hedge_method
        self.stop_z = stop_z
        self.max_holding_bars = max_holding_bars
        self.side = side
        self.pairs = ensure_pair_list(pairs) if pairs is not None else None
        self.pvalue_threshold = pvalue_threshold
        self.max_pairs = max_pairs
        self.min_obs = min_obs
        self.fallback_pairs = fallback_pairs
        self.prefilter_correlation = prefilter_correlation
        self.prefilter_min_corr = prefilter_min_corr
        self.pairs_per_symbol = pairs_per_symbol
        self.hedge_window = hedge_window
        self.hedge_ratio_min = hedge_ratio_min
        self.hedge_ratio_max = hedge_ratio_max
        self.hedge_ratio_ewm_span = hedge_ratio_ewm_span
        self.allow_negative_hedge_ratio = allow_negative_hedge_ratio
        self.spread_fn = spread_fn
        self.signal_fn = signal_fn
        self.portfolio_func = portfolio_func

    def universe_close_matrix(self, data: pl.DataFrame) -> pd.DataFrame:
        close_wide = (
            data.select(["date", "symbol", "close"])
            .to_pandas()
            .pivot(index="date", columns="symbol", values="close")
            .sort_index()
        )
        if close_wide.empty:
            return close_wide
        close_wide.index = pd.to_datetime(close_wide.index)
        return close_wide

    def select_candidate_pairs(self, prices: pd.DataFrame) -> list[tuple[str, str]]:
        symbols = list(prices.columns)
        if len(symbols) < 2:
            return []
        if self.pairs is not None:
            return [(sym_a, sym_b) for sym_a, sym_b in self.pairs if sym_a in symbols and sym_b in symbols]

        candidate_pairs = None
        if self.prefilter_correlation:
            candidate_pairs = preselect_pairs_by_correlation(
                prices,
                min_corr=self.prefilter_min_corr,
                pairs_per_symbol=self.pairs_per_symbol,
                max_pairs=self.max_pairs,
                use_returns=True,
                method="pearson",
            )
            if not candidate_pairs:
                candidate_pairs = None

        candidates = find_cointegrated_pairs(
            prices,
            candidate_pairs=candidate_pairs,
            max_pairs=self.max_pairs,
            pvalue_threshold=self.pvalue_threshold,
            min_obs=self.min_obs,
        )
        selected = [(candidate.symbol_a, candidate.symbol_b) for candidate in candidates]
        if selected or self.fallback_pairs <= 0:
            return selected

        fallback = find_cointegrated_pairs(
            prices,
            candidate_pairs=candidate_pairs,
            max_pairs=self.fallback_pairs,
            pvalue_threshold=1.0,
            min_obs=self.min_obs,
        )
        return [(candidate.symbol_a, candidate.symbol_b) for candidate in fallback]

    def compute_pair_positions(
        self,
        prices: pd.DataFrame,
        symbol_a: str,
        symbol_b: str,
    ) -> pl.DataFrame | None:
        pair_prices = prices[[symbol_a, symbol_b]].dropna()
        if len(pair_prices) < max(self.zscore_window + 1, 2):
            return None

        if self.spread_fn is not None and self.signal_fn is not None:
            spread, hedge_series = self.spread_fn(
                pair_prices,
                symbol_a,
                symbol_b,
                hedge_method=self.hedge_method,
                window=self.hedge_window or self.min_obs,
                lookback_days=self.min_obs,
                ewm_span=self.hedge_ratio_ewm_span,
                ratio_min=self.hedge_ratio_min,
                ratio_max=self.hedge_ratio_max,
                allow_negative=self.allow_negative_hedge_ratio,
            )
            if spread.empty:
                return None
            signal_state = self.signal_fn(
                spread,
                side=self.side,
                stop_z=self.stop_z,
                max_holding_bars=self.max_holding_bars,
            )
        else:
            if self.hedge_method == "rolling_ols":
                spread, hedge_series = compute_rolling_ols_spread(
                    pair_prices,
                    symbol_a,
                    symbol_b,
                    self.hedge_window,
                    self.min_obs,
                    self.hedge_ratio_ewm_span,
                    self.hedge_ratio_min,
                    self.hedge_ratio_max,
                    self.allow_negative_hedge_ratio,
                )
            else:
                hedge_ratio = estimate_hedge_ratio(pair_prices[symbol_a], pair_prices[symbol_b], method=self.hedge_method)
                if not np.isfinite(hedge_ratio):
                    return None
                hedge_ratio = clip_hedge_ratio(hedge_ratio, self.hedge_ratio_min, self.hedge_ratio_max)
                if not self.allow_negative_hedge_ratio:
                    hedge_ratio = abs(hedge_ratio)
                hedge_series = pd.Series(float(hedge_ratio), index=pair_prices.index)
                spread = compute_spread(pair_prices[symbol_a], pair_prices[symbol_b], hedge_ratio)

            zscore = compute_zscore(spread, self.zscore_window).replace([np.inf, -np.inf], np.nan).dropna()
            if zscore.empty:
                return None

            signal_state = generate_zscore_signals(
                zscore,
                entry_z=self.entry_zscore,
                exit_z=self.exit_zscore,
                side=self.side,
                stop_z=self.stop_z,
                max_holding_bars=self.max_holding_bars,
            )

        long_entries = signal_state["long_entries"]
        long_exits = signal_state["long_exits"]
        short_entries = signal_state["short_entries"]
        short_exits = signal_state["short_exits"]
        if long_entries.empty:
            return None

        hedge_series = hedge_series.loc[long_entries.index].replace([np.inf, -np.inf], np.nan).ffill().bfill()
        if hedge_series.empty:
            return None

        rows: list[dict[str, object]] = []
        spread_state = 0
        for timestamp in long_entries.index:
            next_state = spread_state
            if bool(long_exits.loc[timestamp]) or bool(short_exits.loc[timestamp]):
                next_state = 0
            if bool(long_entries.loc[timestamp]):
                next_state = 1
            elif bool(short_entries.loc[timestamp]):
                next_state = -1

            if next_state == spread_state:
                continue

            spread_state = next_state

            ratio = float(hedge_series.loc[timestamp]) if timestamp in hedge_series.index else 1.0
            ratio_sign = 1.0 if ratio >= 0 or not np.isfinite(ratio) else -1.0
            abs_ratio = abs(ratio) if np.isfinite(ratio) else 1.0
            denom = 1.0 + abs_ratio
            weight_a = 1.0 / denom
            weight_b = abs_ratio / denom

            signed_weight_a = float(spread_state) * weight_a
            signed_weight_b = float(-spread_state) * ratio_sign * weight_b
            rows.extend(
                [
                    {
                        "date": pd.Timestamp(timestamp).date(),
                        "symbol": symbol_a,
                        "signed_weight": signed_weight_a,
                    },
                    {
                        "date": pd.Timestamp(timestamp).date(),
                        "symbol": symbol_b,
                        "signed_weight": signed_weight_b,
                    },
                ]
            )

        if not rows:
            return None
        return pl.DataFrame(rows).with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("symbol").cast(pl.String),
            pl.col("signed_weight").cast(pl.Float64),
        )

    def aggregate_pair_positions(
        self,
        pair_positions: list[pl.DataFrame],
    ) -> pl.DataFrame:
        if not pair_positions:
            return self.empty_signal_frame()

        exposures = (
            pl.concat(pair_positions, how="vertical")
            .group_by(["date", "symbol"])
            .agg(pl.col("signed_weight").sum().alias("signed_weight"))
        )
        aggregated = (
            exposures.with_columns(pl.col("signed_weight").abs().sum().over("date").alias("gross_exposure"))
            .with_columns(
                pl.when(pl.col("gross_exposure") > 1.0)
                .then(pl.col("signed_weight") / pl.col("gross_exposure"))
                .otherwise(pl.col("signed_weight"))
                .alias("signed_weight")
            )
            .with_columns(
                pl.when(pl.col("signed_weight") > 0)
                .then(pl.lit(1))
                .when(pl.col("signed_weight") < 0)
                .then(pl.lit(-1))
                .otherwise(pl.lit(0))
                .cast(pl.Int32)
                .alias("signal"),
                pl.col("signed_weight").abs().cast(pl.Float64).alias("weight"),
            )
            .select("date", "symbol", "signal", "weight")
            .sort(["date", "symbol"])
        )
        return self.validate_signal_frame(aggregated)

    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        if data.is_empty():
            return self.empty_signal_frame()

        prices = self.universe_close_matrix(data)
        if prices.empty or prices.shape[1] < 2:
            return self.empty_signal_frame()

        selected_pairs = self.select_candidate_pairs(prices)
        if not selected_pairs:
            return self.empty_signal_frame()

        pair_positions = []
        for symbol_a, symbol_b in selected_pairs:
            if symbol_a not in prices.columns or symbol_b not in prices.columns:
                continue
            positions = self.compute_pair_positions(prices, symbol_a, symbol_b)
            if positions is not None and not positions.is_empty():
                pair_positions.append(positions)

        if not pair_positions:
            return self.empty_signal_frame()
        return self.aggregate_pair_positions(pair_positions)
