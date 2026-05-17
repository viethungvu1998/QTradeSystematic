# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Default data types for internal use."""

from vectorbtpro._settings import settings

int_ = settings["numpy"]["int_"]
"""Default integer data type."""

float_ = settings["numpy"]["float_"]
"""Default floating data type."""

__all__ = [
    "int_",
    "float_",
]
