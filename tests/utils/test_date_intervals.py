from __future__ import annotations

from datetime import date, timedelta

import pytest

from qts.utils.date_intervals import (
    date_in_intervals,
    filter_dates_in_intervals,
    normalize_date_intervals,
    select_training_dates,
)


def test_date_interval_membership_is_inclusive():
    intervals = normalize_date_intervals([(date(2021, 1, 1), date(2023, 1, 1))])

    assert date_in_intervals(date(2021, 1, 1), intervals)
    assert date_in_intervals(date(2023, 1, 1), intervals)
    assert not date_in_intervals(date(2023, 1, 2), intervals)


def test_filter_dates_in_intervals_keeps_only_test_dates():
    dates = [date(2021, 1, 1) + timedelta(days=index) for index in range(6)]
    intervals = normalize_date_intervals(
        [
            (date(2021, 1, 2), date(2021, 1, 3)),
            (date(2021, 1, 5), date(2021, 1, 5)),
        ]
    )

    assert filter_dates_in_intervals(dates, intervals) == [
        date(2021, 1, 2),
        date(2021, 1, 3),
        date(2021, 1, 5),
    ]


def test_select_training_dates_excludes_all_test_intervals_and_keeps_gap():
    dates = [date(2021, 1, 1) + timedelta(days=index) for index in range(12)]
    intervals = normalize_date_intervals(
        [
            (date(2021, 1, 5), date(2021, 1, 6)),
            (date(2021, 1, 10), date(2021, 1, 11)),
        ]
    )

    assert select_training_dates(
        dates,
        before=date(2021, 1, 10),
        train_window=3,
        test_intervals=intervals,
    ) == [
        date(2021, 1, 7),
        date(2021, 1, 8),
        date(2021, 1, 9),
    ]


def test_normalize_date_intervals_rejects_overlaps():
    with pytest.raises(ValueError, match="overlap"):
        normalize_date_intervals(
            [
                (date(2021, 1, 1), date(2021, 1, 3)),
                (date(2021, 1, 3), date(2021, 1, 5)),
            ]
        )
