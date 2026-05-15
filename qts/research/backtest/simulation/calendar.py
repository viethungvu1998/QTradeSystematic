"""Trading calendars."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from qts.core.registry import Registry


@dataclass(slots=True)
class BaseCalendar:
    """Calendar contract."""

    holidays: set[date]

    def is_session(self, value: date) -> bool:
        raise NotImplementedError

    def sessions_between(self, start: date, end: date) -> list[date]:
        current = start
        sessions = []
        while current <= end:
            if self.is_session(current):
                sessions.append(current)
            current += timedelta(days=1)
        return sessions


@Registry.register_calendar("nyse")
class NYSECalendar(BaseCalendar):
    """Weekday trading calendar with a minimal holiday set."""

    def __init__(self) -> None:
        super().__init__(holidays={date(2024, 1, 1), date(2024, 7, 4), date(2024, 12, 25)})

    def is_session(self, value: date) -> bool:
        return value.weekday() < 5 and value not in self.holidays


@Registry.register_calendar("hkex")
class HKEXCalendar(BaseCalendar):
    """HKEX weekday calendar."""

    def __init__(self) -> None:
        super().__init__(holidays={date(2024, 1, 1), date(2024, 2, 12), date(2024, 10, 1)})

    def is_session(self, value: date) -> bool:
        return value.weekday() < 5 and value not in self.holidays


@Registry.register_calendar("crypto")
class CryptoCalendar(BaseCalendar):
    """24/7 calendar."""

    def __init__(self) -> None:
        super().__init__(holidays=set())

    def is_session(self, value: date) -> bool:
        return True
