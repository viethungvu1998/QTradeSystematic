"""QTradeSystematic package."""

from qts.core import events, instrument, order, portfolio, registry  # noqa: F401
from qts.data.sources import binance, fmp, yahoo  # noqa: F401
from qts.data.storage import duckdb, parquet  # noqa: F401
from qts.execution.brokers import binance as execution_binance  # noqa: F401
from qts.execution.brokers import moomoo  # noqa: F401
from qts.research.backtest.engines import vectorbtpro_engine, zipline_engine  # noqa: F401
from qts.research.backtest.simulation import calendar, commission, fills, slippage  # noqa: F401
from qts.research.features import fundamentals, onchain, technical  # noqa: F401
from qts.research.strategies.factor import model as factor_model  # noqa: F401
from qts.research.strategies.stat_arb import model as stat_arb_model  # noqa: F401

__all__ = [
    "events",
    "instrument",
    "order",
    "portfolio",
    "registry",
]
