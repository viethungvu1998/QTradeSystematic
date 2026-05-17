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


@Registry.register_calendar("hose")
class HOSECalendar(BaseCalendar):
    """HOSE/HNX/UPCOM weekday calendar with Vietnamese national holidays.

    Fixed holidays: New Year (Jan 1), Reunification Day (Apr 30),
    International Workers Day (May 1), National Day (Sep 2).
    Variable lunar holidays (Tet, Hung Kings) are listed for 2024-2026.
    """

    def __init__(self) -> None:
        super().__init__(
            holidays={
                # 2024
                date(2024, 1, 1),   # New Year
                date(2024, 2, 8), date(2024, 2, 9), date(2024, 2, 12),
                date(2024, 2, 13), date(2024, 2, 14),                   # Tet
                date(2024, 4, 18),  # Hung Kings (10th/3rd lunar)
                date(2024, 4, 30),  # Reunification Day
                date(2024, 5, 1),   # Labour Day
                date(2024, 9, 2),   # National Day
                # 2025
                date(2025, 1, 1),   # New Year
                date(2025, 1, 27), date(2025, 1, 28), date(2025, 1, 29),
                date(2025, 1, 30), date(2025, 1, 31),                   # Tet
                date(2025, 4, 7),   # Hung Kings
                date(2025, 4, 30),  # Reunification Day
                date(2025, 5, 1),   # Labour Day
                date(2025, 9, 1), date(2025, 9, 2),                     # National Day + substitute
                # 2026
                date(2026, 1, 1),   # New Year
                date(2026, 2, 16), date(2026, 2, 17), date(2026, 2, 18),
                date(2026, 2, 19), date(2026, 2, 20),                   # Tet
                date(2026, 3, 31),  # Hung Kings
                date(2026, 4, 30),  # Reunification Day
                date(2026, 5, 1),   # Labour Day
                date(2026, 9, 2),   # National Day
            }
        )

    def is_session(self, value: date) -> bool:
        return value.weekday() < 5 and value not in self.holidays


@Registry.register_calendar("crypto")
class CryptoCalendar(BaseCalendar):
    """24/7 calendar."""

    def __init__(self) -> None:
        super().__init__(holidays=set())

    def is_session(self, value: date) -> bool:
        return True
