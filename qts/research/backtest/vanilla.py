"""Transparent vanilla backtest helpers for signal-level verification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC
from decimal import Decimal
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.backends.backend_pdf import PdfPages

from qts.research.backtest.base import (
    PORTFOLIO_SNAPSHOTS_SCHEMA,
    BacktestConfig,
    BacktestResult,
    empty_trade_log_frame,
)
from qts.research.backtest.engines._targets import build_target_schedule
from qts.research.backtest.metrics import build_metrics
from qts.research.backtest.tearsheet import shade_test_intervals_on_axis
from qts.utils.date_intervals import DateInterval, format_date_intervals


@dataclass(frozen=True, slots=True)
class VanillaBacktestArtifacts:
    """Backtest result plus human-readable rebalance audit logs."""

    result: BacktestResult
    portfolio_log: pd.DataFrame
    entry_log: pd.DataFrame
    fills: pd.DataFrame


def run_vanilla_backtest_from_signals(
    *,
    data: pl.DataFrame,
    signals: pl.DataFrame,
    config: BacktestConfig,
) -> VanillaBacktestArtifacts:
    """Run a simple target-percent simulator from standard QTS signals.

    The simulator mirrors the ML notebook vectorbt timing: signal dates are
    rebalance dates, fills occur at the same-session close, and the portfolio is
    marked to close each session. Commission and slippage are charged on each
    fill using the same config defaults as the vectorbt adapter.
    """

    close = _pivot_close(data)
    if close.empty or close.shape[1] == 0:
        return _empty_artifacts(signals)

    fee_rate = _fee_rate(config)
    slippage_rate = _slippage_rate(config)
    initial_cash = float(config.initial_capital or Decimal("100000"))
    target_schedule = build_target_schedule(
        signals,
        close.index,
        close.columns.astype(str).tolist(),
        shift_by_one_bar=False,
    )

    cash = initial_cash
    quantities = pd.Series(0.0, index=close.columns, dtype=float)
    last_prices = pd.Series(np.nan, index=close.columns, dtype=float)

    equity_rows: list[dict[str, object]] = []
    portfolio_rows: list[dict[str, object]] = []
    entry_rows: list[dict[str, object]] = []
    fill_rows: list[dict[str, object]] = []
    snapshot_rows: list[dict[str, object]] = []

    for timestamp, price_row in close.iterrows():
        current_prices = price_row.combine_first(last_prices).astype(float)
        last_prices = current_prices.combine_first(last_prices)
        equity_before = _equity(cash, quantities, current_prices)
        event_targets = target_schedule.events.loc[timestamp].dropna()

        if not event_targets.empty:
            (
                cash,
                quantities,
                portfolio_row,
                entry_row,
                fills_for_date,
            ) = _rebalance(
                timestamp=timestamp,
                cash=cash,
                quantities=quantities,
                prices=current_prices,
                targets=event_targets.astype(float),
                target_weights=target_schedule.targets.loc[timestamp].astype(float),
                equity_before=equity_before,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
            )
            portfolio_rows.append(portfolio_row)
            entry_rows.append(entry_row)
            fill_rows.extend(fills_for_date)

        equity_after = _equity(cash, quantities, current_prices)
        equity_rows.append({"date": timestamp.date(), "equity": equity_after})
        snapshot_rows.append(
            {
                "timestamp": timestamp.to_pydatetime().replace(tzinfo=UTC),
                "tokens": _snapshot_tokens(quantities, current_prices),
                "equity": equity_after,
            }
        )

    equity = pd.DataFrame(equity_rows)
    returns = equity["equity"].pct_change()
    if not returns.empty:
        returns.iloc[0] = (
            0.0 if initial_cash == 0.0 else float(equity["equity"].iloc[0]) / initial_cash - 1.0
        )
    returns = returns.fillna(0.0)
    metrics = build_metrics(returns.tolist(), equity["equity"].tolist())
    returns_frame = pl.from_pandas(
        pd.DataFrame(
            {
                "date": equity["date"],
                "portfolio_return": returns,
            }
        )
    ).with_columns(pl.col("date").cast(pl.Date))
    equity_frame = pl.from_pandas(equity).with_columns(pl.col("date").cast(pl.Date))
    snapshots = (
        pl.DataFrame(snapshot_rows, schema=PORTFOLIO_SNAPSHOTS_SCHEMA)
        if snapshot_rows
        else pl.DataFrame(schema=PORTFOLIO_SNAPSHOTS_SCHEMA)
    )

    result = BacktestResult(
        engine_name="vanilla",
        metrics=metrics,
        returns=returns_frame,
        equity_curve=equity_frame,
        signals=signals,
        trade_log=empty_trade_log_frame(),
        portfolio_snapshots=snapshots,
    )
    return VanillaBacktestArtifacts(
        result=result,
        portfolio_log=_portfolio_log_frame(portfolio_rows),
        entry_log=_entry_log_frame(entry_rows),
        fills=_fills_frame(fill_rows),
    )


def export_vanilla_backtest_artifacts(
    artifacts: VanillaBacktestArtifacts,
    *,
    out_dir: Path,
    run_id: str,
) -> dict[str, Path]:
    """Write vanilla outputs, including rebalance-grain portfolio and entry CSVs."""

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "returns": out_dir / f"{run_id}_vanilla_returns.csv",
        "equity_curve": out_dir / f"{run_id}_vanilla_equity_curve.csv",
        "portfolio_log": out_dir / f"{run_id}_vanilla_portfolio_rebalance_log.csv",
        "entry_log": out_dir / f"{run_id}_vanilla_entry_rebalance_log.csv",
        "fills": out_dir / f"{run_id}_vanilla_fills.csv",
    }
    artifacts.result.returns.write_csv(paths["returns"])
    artifacts.result.equity_curve.write_csv(paths["equity_curve"])
    artifacts.portfolio_log.to_csv(paths["portfolio_log"], index=False)
    artifacts.entry_log.to_csv(paths["entry_log"], index=False)
    artifacts.fills.to_csv(paths["fills"], index=False)
    return paths


def save_vanilla_comparison_report_pdf(
    *,
    vectorbt_result: BacktestResult,
    vanilla_artifacts: VanillaBacktestArtifacts,
    run_id: str,
    output_dir: Path,
    vectorbt_report_path: Path | None = None,
    exported_paths: dict[str, Path] | None = None,
    filename: str | None = None,
    test_intervals: list[DateInterval] | None = None,
) -> Path:
    """Save a compact PDF comparing the vanilla simulation to vectorbt."""

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / (filename or f"{run_id}_vanilla_comparison_report.pdf")
    comparison = _comparison_frame(vectorbt_result, vanilla_artifacts.result)
    aligned = _aligned_results(vectorbt_result, vanilla_artifacts.result)
    exported_paths = exported_paths or {}

    plt.close("all")
    with PdfPages(report_path) as pdf:
        _add_summary_page(
            pdf,
            run_id=run_id,
            comparison=comparison,
            vanilla_artifacts=vanilla_artifacts,
            vectorbt_report_path=vectorbt_report_path,
            exported_paths=exported_paths,
            test_intervals=test_intervals,
        )
        _add_equity_page(pdf, aligned, test_intervals=test_intervals)
        _add_return_difference_page(pdf, aligned)
        _add_rebalance_page(pdf, vanilla_artifacts)
        pdf.infodict().update(
            {
                "Title": f"QTS vanilla comparison - {run_id}",
                "Author": "QTradeSystematic",
            }
        )
    plt.close("all")
    return report_path


def _rebalance(
    *,
    timestamp: pd.Timestamp,
    cash: float,
    quantities: pd.Series,
    prices: pd.Series,
    targets: pd.Series,
    target_weights: pd.Series,
    equity_before: float,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[float, pd.Series, dict[str, object], dict[str, object], list[dict[str, object]]]:
    cash_before = cash
    quantities_before = quantities.copy()
    quantities = quantities.copy()
    valid_targets = {
        symbol: target
        for symbol, target in targets.items()
        if symbol in prices.index and _is_executable_price(float(prices[symbol]))
    }
    skipped = sorted(set(map(str, targets.index)) - set(map(str, valid_targets)))
    planned: list[dict[str, float | str]] = []
    for symbol, target in valid_targets.items():
        price = float(prices[symbol])
        current_value = float(quantities[symbol]) * price
        target_value = float(target) * equity_before
        value_delta = target_value - current_value
        if np.isclose(value_delta, 0.0, atol=1e-10, rtol=0.0):
            continue
        qty_delta = value_delta / price
        planned.append(
            {
                "symbol": str(symbol),
                "target_weight": float(target),
                "price": price,
                "quantity_delta": float(qty_delta),
            }
        )

    sells = [item for item in planned if float(item["quantity_delta"]) < 0.0]
    buys = [item for item in planned if float(item["quantity_delta"]) > 0.0]
    fills: list[dict[str, object]] = []
    date_value = timestamp.date().isoformat()
    cash, quantities = _execute_planned_trades(
        date_value=date_value,
        cash=cash,
        quantities=quantities,
        planned=sells,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        fills=fills,
    )

    buy_cash_need = sum(
        _cash_needed_for_trade(
            float(item["quantity_delta"]),
            float(item["price"]),
            fee_rate,
            slippage_rate,
        )
        for item in buys
    )
    scale = 1.0
    if buy_cash_need > cash and buy_cash_need > 0.0:
        scale = max(cash, 0.0) / buy_cash_need
        buys = [
            {
                **item,
                "quantity_delta": float(item["quantity_delta"]) * scale,
            }
            for item in buys
        ]

    cash, quantities = _execute_planned_trades(
        date_value=date_value,
        cash=cash,
        quantities=quantities,
        planned=buys,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        fills=fills,
    )

    fee = sum(float(item["fee"]) for item in fills)
    slippage_cost = sum(float(item["slippage_cost"]) for item in fills)
    buy_notional = sum(
        float(item["notional"]) for item in fills if float(item["quantity_delta"]) > 0.0
    )
    sell_notional = sum(
        float(item["notional"]) for item in fills if float(item["quantity_delta"]) < 0.0
    )
    equity_after = _equity(cash, quantities, prices)
    holdings = _holdings_payload(quantities, prices, equity_after)
    target_payload = _weights_payload(target_weights)

    entries = [
        item["symbol"]
        for item in fills
        if float(item["quantity_before"]) <= 0.0 < float(item["quantity_after"])
    ]
    exits = [
        item["symbol"]
        for item in fills
        if float(item["quantity_before"]) > 0.0 and float(item["quantity_after"]) <= 0.0
    ]
    adjusted = [
        item["symbol"]
        for item in fills
        if item["symbol"] not in entries and item["symbol"] not in exits
    ]

    turnover = (
        0.0
        if equity_before == 0.0
        else sum(float(item["notional"]) for item in fills) / equity_before
    )
    portfolio_row = {
        "date": date_value,
        "equity_before_rebalance": equity_before,
        "cash_before_rebalance": cash_before,
        "gross_exposure_before": _gross_exposure(quantities_before, prices, equity_before),
        "equity_after_rebalance": equity_after,
        "portfolio_value": equity_after,
        "cash_after_rebalance": cash,
        "gross_exposure_after": _gross_exposure(quantities, prices, equity_after),
        "turnover": turnover,
        "fee": fee,
        "slippage_cost": slippage_cost,
        "buy_notional": buy_notional,
        "sell_notional": sell_notional,
        "trade_count": len(fills),
        "cash_scaled_buy_orders": scale < 1.0,
        "skipped_symbols": _json_list(skipped),
        "target_weights": _json_mapping(target_payload),
        "holdings": _json_mapping(holdings),
    }
    entry_row = {
        "date": date_value,
        "entries": _json_list(entries),
        "exits": _json_list(exits),
        "rebalanced": _json_list(adjusted),
        "trade_count": len(fills),
        "buy_notional": buy_notional,
        "sell_notional": sell_notional,
        "fee": fee,
        "slippage_cost": slippage_cost,
        "cash_scaled_buy_orders": scale < 1.0,
        "skipped_symbols": _json_list(skipped),
        "fills": _json_list(
            [
                {
                    "symbol": item["symbol"],
                    "quantity_delta": item["quantity_delta"],
                    "price": item["price"],
                    "fill_price": item["fill_price"],
                    "notional": item["notional"],
                }
                for item in fills
            ]
        ),
    }
    return cash, quantities, portfolio_row, entry_row, fills


def _execute_planned_trades(
    *,
    date_value: str,
    cash: float,
    quantities: pd.Series,
    planned: list[dict[str, float | str]],
    fee_rate: float,
    slippage_rate: float,
    fills: list[dict[str, object]],
) -> tuple[float, pd.Series]:
    for item in planned:
        symbol = str(item["symbol"])
        price = float(item["price"])
        quantity_delta = float(item["quantity_delta"])
        if np.isclose(quantity_delta, 0.0, atol=1e-12, rtol=0.0):
            continue
        side = "buy" if quantity_delta > 0.0 else "sell"
        fill_price = _fill_price(price, quantity_delta, slippage_rate)
        fill_notional = abs(quantity_delta) * fill_price
        reference_notional = abs(quantity_delta) * price
        fee = fill_notional * fee_rate
        slippage_cost = abs(fill_notional - reference_notional)
        quantity_before = float(quantities[symbol])
        quantities[symbol] = quantity_before + quantity_delta
        cash -= quantity_delta * fill_price
        cash -= fee
        fills.append(
            {
                "date": date_value,
                "symbol": symbol,
                "side": side,
                "quantity_before": quantity_before,
                "quantity_delta": quantity_delta,
                "quantity_after": float(quantities[symbol]),
                "price": price,
                "fill_price": fill_price,
                "notional": fill_notional,
                "fee": fee,
                "slippage_cost": slippage_cost,
                "target_weight": float(item["target_weight"]),
            }
        )
    return cash, quantities


def _cash_needed_for_trade(
    quantity_delta: float,
    price: float,
    fee_rate: float,
    slippage_rate: float,
) -> float:
    if quantity_delta <= 0.0:
        return 0.0
    fill_price = _fill_price(price, quantity_delta, slippage_rate)
    return quantity_delta * fill_price * (1.0 + fee_rate)


def _fill_price(price: float, quantity_delta: float, slippage_rate: float) -> float:
    if quantity_delta > 0.0:
        return price * (1.0 + slippage_rate)
    return price * (1.0 - slippage_rate)


def _fee_rate(config: BacktestConfig) -> float:
    if config.commission is None:
        return 0.001
    return float(config.commission.rate)


def _slippage_rate(config: BacktestConfig) -> float:
    if config.slippage_model is None:
        return 0.0
    return {"fixed": 0.0005, "volatility_scaled": 0.001}.get(config.slippage_model, 0.0)


def _pivot_close(data: pl.DataFrame) -> pd.DataFrame:
    wide = (
        data.select(["date", "symbol", "close"])
        .to_pandas()
        .pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
        .rename_axis(index=None, columns=None)
        .sort_index()
        .sort_index(axis=1)
    )
    wide.index = pd.to_datetime(wide.index)
    wide.columns = wide.columns.astype(str)
    return wide


def _equity(cash: float, quantities: pd.Series, prices: pd.Series) -> float:
    return float(cash + quantities.fillna(0.0).mul(prices.fillna(0.0)).sum())


def _gross_exposure(quantities: pd.Series, prices: pd.Series, equity: float) -> float:
    if equity == 0.0:
        return 0.0
    return float(quantities.fillna(0.0).mul(prices.fillna(0.0)).abs().sum() / equity)


def _is_executable_price(price: float) -> bool:
    return np.isfinite(price) and price > 0.0


def _snapshot_tokens(quantities: pd.Series, prices: pd.Series) -> list[dict[str, float | str]]:
    tokens = []
    for symbol, quantity in quantities.items():
        qty = float(quantity)
        price = float(prices.get(symbol, np.nan))
        if np.isclose(qty, 0.0, atol=1e-12, rtol=0.0) or not np.isfinite(price):
            continue
        tokens.append(
            {
                "token": str(symbol),
                "quantity": qty,
                "avg_buy_price": price,
                "current_price": price,
            }
        )
    return tokens


def _holdings_payload(quantities: pd.Series, prices: pd.Series, equity: float) -> dict[str, Any]:
    holdings: dict[str, Any] = {}
    for symbol, quantity in quantities.items():
        qty = float(quantity)
        price = float(prices.get(symbol, np.nan))
        if np.isclose(qty, 0.0, atol=1e-12, rtol=0.0) or not np.isfinite(price):
            continue
        market_value = qty * price
        holdings[str(symbol)] = {
            "quantity": qty,
            "price": price,
            "market_value": market_value,
            "weight": 0.0 if equity == 0.0 else market_value / equity,
        }
    return holdings


def _weights_payload(weights: pd.Series) -> dict[str, float]:
    return {
        str(symbol): float(weight)
        for symbol, weight in weights.items()
        if not np.isclose(float(weight), 0.0, atol=1e-12, rtol=0.0)
    }


def _json_mapping(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json_list(value: list[Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _portfolio_log_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    columns = [
        "date",
        "equity_before_rebalance",
        "cash_before_rebalance",
        "gross_exposure_before",
        "equity_after_rebalance",
        "portfolio_value",
        "cash_after_rebalance",
        "gross_exposure_after",
        "turnover",
        "fee",
        "slippage_cost",
        "buy_notional",
        "sell_notional",
        "trade_count",
        "cash_scaled_buy_orders",
        "skipped_symbols",
        "target_weights",
        "holdings",
    ]
    return pd.DataFrame(rows, columns=columns)


def _entry_log_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    columns = [
        "date",
        "entries",
        "exits",
        "rebalanced",
        "trade_count",
        "buy_notional",
        "sell_notional",
        "fee",
        "slippage_cost",
        "cash_scaled_buy_orders",
        "skipped_symbols",
        "fills",
    ]
    return pd.DataFrame(rows, columns=columns)


def _fills_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    columns = [
        "date",
        "symbol",
        "side",
        "quantity_before",
        "quantity_delta",
        "quantity_after",
        "price",
        "fill_price",
        "notional",
        "fee",
        "slippage_cost",
        "target_weight",
    ]
    return pd.DataFrame(rows, columns=columns)


def _empty_artifacts(signals: pl.DataFrame) -> VanillaBacktestArtifacts:
    result = BacktestResult(
        engine_name="vanilla",
        metrics=build_metrics([], []),
        returns=pl.DataFrame(schema={"date": pl.Date, "portfolio_return": pl.Float64}),
        equity_curve=pl.DataFrame(schema={"date": pl.Date, "equity": pl.Float64}),
        signals=signals,
        trade_log=empty_trade_log_frame(),
        portfolio_snapshots=pl.DataFrame(schema=PORTFOLIO_SNAPSHOTS_SCHEMA),
    )
    return VanillaBacktestArtifacts(
        result=result,
        portfolio_log=_portfolio_log_frame([]),
        entry_log=_entry_log_frame([]),
        fills=_fills_frame([]),
    )


def _comparison_frame(
    vectorbt_result: BacktestResult,
    vanilla_result: BacktestResult,
) -> pd.DataFrame:
    metric_names = sorted(set(vectorbt_result.metrics) | set(vanilla_result.metrics))
    rows = []
    for metric in metric_names:
        vbt_value = float(vectorbt_result.metrics.get(metric, np.nan))
        vanilla_value = float(vanilla_result.metrics.get(metric, np.nan))
        rows.append(
            {
                "metric": metric,
                "vectorbt": vbt_value,
                "vanilla": vanilla_value,
                "diff": vanilla_value - vbt_value,
            }
        )

    vbt_final = _final_equity(vectorbt_result)
    vanilla_final = _final_equity(vanilla_result)
    rows.append(
        {
            "metric": "final_equity",
            "vectorbt": vbt_final,
            "vanilla": vanilla_final,
            "diff": vanilla_final - vbt_final,
        }
    )
    return pd.DataFrame(rows)


def _final_equity(result: BacktestResult) -> float:
    if result.equity_curve.is_empty():
        return np.nan
    return float(result.equity_curve.select("equity").tail(1).item())


def _aligned_results(
    vectorbt_result: BacktestResult,
    vanilla_result: BacktestResult,
) -> pd.DataFrame:
    vbt_equity = vectorbt_result.equity_curve.to_pandas().rename(
        columns={"equity": "vectorbt_equity"}
    )
    vanilla_equity = vanilla_result.equity_curve.to_pandas().rename(
        columns={"equity": "vanilla_equity"}
    )
    vbt_returns = vectorbt_result.returns.to_pandas().rename(
        columns={"portfolio_return": "vectorbt_return"}
    )
    vanilla_returns = vanilla_result.returns.to_pandas().rename(
        columns={"portfolio_return": "vanilla_return"}
    )
    for frame in [vbt_equity, vanilla_equity, vbt_returns, vanilla_returns]:
        if not frame.empty:
            frame["date"] = pd.to_datetime(frame["date"])
    aligned = (
        vbt_equity.merge(vanilla_equity, on="date", how="outer")
        .merge(vbt_returns, on="date", how="outer")
        .merge(vanilla_returns, on="date", how="outer")
        .sort_values("date")
    )
    aligned["equity_diff"] = aligned["vanilla_equity"] - aligned["vectorbt_equity"]
    aligned["return_diff"] = aligned["vanilla_return"] - aligned["vectorbt_return"]
    return aligned


def _add_summary_page(
    pdf: PdfPages,
    *,
    run_id: str,
    comparison: pd.DataFrame,
    vanilla_artifacts: VanillaBacktestArtifacts,
    vectorbt_report_path: Path | None,
    exported_paths: dict[str, Path],
    test_intervals: list[DateInterval] | None,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title("Vanilla Backtest vs VectorBT", fontsize=16, loc="left", pad=16)
    text_lines = [
        f"Run ID: {run_id}",
        "Assumptions: target-percent rebalance at signal-date close; configured "
        "commission and slippage charged per fill.",
        f"VectorBT report: {vectorbt_report_path or 'not generated'}",
        f"Test intervals shaded grey: {format_date_intervals(test_intervals or [])}",
        f"Portfolio log rows: {len(vanilla_artifacts.portfolio_log)}",
        f"Entry log rows: {len(vanilla_artifacts.entry_log)}",
    ]
    if exported_paths:
        text_lines.append("CSV outputs:")
        for name, path in exported_paths.items():
            text_lines.append(f"  {name}: {path}")
    ax.text(0.02, 0.95, "\n".join(text_lines), va="top", fontsize=9, family="monospace")

    table = comparison.copy()
    for column in ["vectorbt", "vanilla", "diff"]:
        table[column] = table[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    mpl_table = ax.table(
        cellText=table[["metric", "vectorbt", "vanilla", "diff"]].values,
        colLabels=["Metric", "VectorBT", "Vanilla", "Diff"],
        cellLoc="right",
        colLoc="right",
        bbox=[0.02, 0.08, 0.82, 0.55],
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(8)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _add_equity_page(
    pdf: PdfPages,
    aligned: pd.DataFrame,
    *,
    test_intervals: list[DateInterval] | None,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
    axes[0].set_title("Equity Curve")
    if not aligned.empty:
        axes[0].plot(aligned["date"], aligned["vectorbt_equity"], label="VectorBT", linewidth=1.2)
        axes[0].plot(aligned["date"], aligned["vanilla_equity"], label="Vanilla", linewidth=1.2)
        axes[1].plot(
            aligned["date"],
            aligned["equity_diff"],
            label="Vanilla - VectorBT",
            color="tab:red",
        )
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].set_title("Equity Difference")
    axes[1].grid(alpha=0.25)
    for ax in axes:
        shade_test_intervals_on_axis(ax, test_intervals)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _add_return_difference_page(pdf: PdfPages, aligned: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
    axes[0].set_title("Daily Return Scatter")
    axes[1].set_title("Daily Return Difference")
    if not aligned.empty:
        valid = aligned.dropna(subset=["vectorbt_return", "vanilla_return"])
        axes[0].scatter(valid["vectorbt_return"], valid["vanilla_return"], s=8, alpha=0.65)
        if not valid.empty:
            lo = float(valid[["vectorbt_return", "vanilla_return"]].min().min())
            hi = float(valid[["vectorbt_return", "vanilla_return"]].max().max())
            axes[0].plot([lo, hi], [lo, hi], color="black", linewidth=0.8)
            axes[1].hist(valid["return_diff"].dropna(), bins=40, color="tab:blue", alpha=0.75)
    axes[0].set_xlabel("VectorBT")
    axes[0].set_ylabel("Vanilla")
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel("Vanilla - VectorBT")
    axes[1].grid(alpha=0.25)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _add_rebalance_page(pdf: PdfPages, vanilla_artifacts: VanillaBacktestArtifacts) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title("Recent Vanilla Rebalances", fontsize=16, loc="left", pad=16)
    recent = vanilla_artifacts.portfolio_log.tail(12).copy()
    if recent.empty:
        ax.text(0.02, 0.9, "No rebalance rows.", fontsize=10)
    else:
        table = recent[
            [
                "date",
                "portfolio_value",
                "cash_after_rebalance",
                "gross_exposure_after",
                "turnover",
                "fee",
                "trade_count",
            ]
        ].copy()
        for column in [
            "portfolio_value",
            "cash_after_rebalance",
            "gross_exposure_after",
            "turnover",
            "fee",
        ]:
            table[column] = table[column].map(lambda value: f"{float(value):.6f}")
        mpl_table = ax.table(
            cellText=table.values,
            colLabels=list(table.columns),
            cellLoc="right",
            colLoc="right",
            bbox=[0.02, 0.2, 0.96, 0.62],
        )
        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(8)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
