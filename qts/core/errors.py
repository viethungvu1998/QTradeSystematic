"""Project-specific exceptions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qts.core.order import Order


class QTSError(Exception):
    """Base exception for the project."""


class RegistryError(QTSError):
    """Raised when a registry key cannot be resolved."""


class ConfigError(QTSError):
    """Raised when configuration parsing or validation fails."""


@dataclass(slots=True)
class DataSourceError(QTSError):
    """Raised when a data source cannot serve a request."""

    message: str
    symbol: str
    date_range: tuple[date, date] | None = None

    def __str__(self) -> str:
        if self.date_range is None:
            return f"{self.message} [{self.symbol}]"
        start, end = self.date_range
        return f"{self.message} [{self.symbol}] ({start} -> {end})"


@dataclass(slots=True)
class BrokerError(QTSError):
    """Raised when broker execution fails."""

    message: str
    order: Order | None = None

    def __str__(self) -> str:
        if self.order is None:
            return self.message
        return f"{self.message} [{self.order.instrument.symbol}]"
