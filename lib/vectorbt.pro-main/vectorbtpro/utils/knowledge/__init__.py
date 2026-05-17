# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Utilities for working with knowledge.

Run for the examples:

```pycon
>>> dataset = [
...     {"s": "ABC", "b": True, "d2": {"c": "red", "l": [1, 2]}},
...     {"s": "BCD", "b": True, "d2": {"c": "blue", "l": [3, 4]}},
...     {"s": "CDE", "b": False, "d2": {"c": "green", "l": [5, 6]}},
...     {"s": "DEF", "b": False, "d2": {"c": "yellow", "l": [7, 8]}},
...     {"s": "EFG", "b": False, "d2": {"c": "black", "l": [9, 10]}, "xyz": 123}
... ]
>>> asset = vbt.KnowledgeAsset(dataset)
```
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorbtpro.utils.knowledge.asset_pipelines import *
    from vectorbtpro.utils.knowledge.base_asset_funcs import *
    from vectorbtpro.utils.knowledge.base_assets import *
    from vectorbtpro.utils.knowledge.custom_asset_funcs import *
    from vectorbtpro.utils.knowledge.custom_assets import *
