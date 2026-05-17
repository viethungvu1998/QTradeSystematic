# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Pandas accessors for Plotly Express.

!!! note
    Accessors do not utilize caching."""

from vectorbtpro.utils.module_ import assert_can_import

assert_can_import("plotly")

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.accessors import BaseAccessor, BaseDFAccessor, BaseSRAccessor
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.accessors import register_vbt_accessor, register_df_vbt_accessor, register_sr_vbt_accessor
from vectorbtpro.px.decorators import attach_px_methods

__all__ = [
    "PXAccessor",
    "PXSRAccessor",
    "PXDFAccessor",
]


@register_vbt_accessor("px")
@attach_px_methods
class PXAccessor(BaseAccessor):
    """Accessor for running Plotly Express functions.

    Accessible via `pd.Series.vbt.px` and `pd.DataFrame.vbt.px`.

    Usage:
        ```pycon
        >>> from vectorbtpro import *

        >>> pd.Series([1, 2, 3]).vbt.px.bar().show()
        ```

        ![](/assets/images/api/px_bar.light.svg#only-light){: .iimg loading=lazy }
        ![](/assets/images/api/px_bar.dark.svg#only-dark){: .iimg loading=lazy }
    """

    def __init__(
        self,
        wrapper: tp.Union[ArrayWrapper, tp.ArrayLike],
        obj: tp.Optional[tp.ArrayLike] = None,
        **kwargs,
    ) -> None:
        BaseAccessor.__init__(self, wrapper, obj=obj, **kwargs)


@register_sr_vbt_accessor("px")
class PXSRAccessor(PXAccessor, BaseSRAccessor):
    """Accessor for running Plotly Express functions. For Series only.

    Accessible via `pd.Series.vbt.px`."""

    def __init__(
        self,
        wrapper: tp.Union[ArrayWrapper, tp.ArrayLike],
        obj: tp.Optional[tp.ArrayLike] = None,
        _full_init: bool = True,
        **kwargs,
    ) -> None:
        BaseSRAccessor.__init__(self, wrapper, obj=obj, _full_init=False, **kwargs)

        if _full_init:
            PXAccessor.__init__(self, wrapper, obj=obj, **kwargs)


@register_df_vbt_accessor("px")
class PXDFAccessor(PXAccessor, BaseDFAccessor):
    """Accessor for running Plotly Express functions. For DataFrames only.

    Accessible via `pd.DataFrame.vbt.px`."""

    def __init__(
        self,
        wrapper: tp.Union[ArrayWrapper, tp.ArrayLike],
        obj: tp.Optional[tp.ArrayLike] = None,
        _full_init: bool = True,
        **kwargs,
    ) -> None:
        BaseDFAccessor.__init__(self, wrapper, obj=obj, _full_init=False, **kwargs)

        if _full_init:
            PXAccessor.__init__(self, wrapper, obj=obj, **kwargs)
