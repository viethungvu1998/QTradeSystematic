"""Backtest metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def sharpe_ratio(returns: list[float], periods_per_year: int = 252) -> float:
    if not returns:
        return 0.0
    std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    if math.isclose(std, 0.0):
        return 0.0
    return float(np.mean(returns) / std * math.sqrt(periods_per_year))


def sortino_ratio(returns: list[float], periods_per_year: int = 252) -> float:
    downside = [value for value in returns if value < 0]
    if not downside:
        return 0.0
    std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    if math.isclose(std, 0.0):
        return 0.0
    return float(np.mean(returns) / std * math.sqrt(periods_per_year))


def cagr(equity_curve: list[float], periods_per_year: int = 252) -> float:
    if len(equity_curve) < 2 or equity_curve[0] <= 0:
        return 0.0
    if equity_curve[-1] <= 0:
        return -1.0
    years = max((len(equity_curve) - 1) / periods_per_year, 1 / periods_per_year)
    return float((equity_curve[-1] / equity_curve[0]) ** (1 / years) - 1)


def max_drawdown(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    drawdown = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        drawdown = min(drawdown, (value - peak) / peak)
    return float(abs(drawdown))


def win_rate(returns: list[float]) -> float:
    if not returns:
        return 0.0
    wins = sum(1 for value in returns if value > 0)
    return float(wins / len(returns))


def _safe_metric(value: float) -> float:
    return 0.0 if not math.isfinite(value) else float(value)


def build_metrics(
    returns: Sequence[float],
    equity_curve: Sequence[float],
) -> dict[str, float]:
    """Build the canonical metrics payload for every engine."""

    returns_list = [float(value) for value in returns]
    equity_list = [float(value) for value in equity_curve]
    return {
        "sharpe": _safe_metric(sharpe_ratio(returns_list)),
        "sortino": _safe_metric(sortino_ratio(returns_list)),
        "cagr": _safe_metric(cagr(equity_list)),
        "max_drawdown": _safe_metric(max_drawdown(equity_list)),
        "win_rate": _safe_metric(win_rate(returns_list)),
    }
