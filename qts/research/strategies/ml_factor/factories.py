"""Registered ML factor strategy factories."""

from __future__ import annotations

from qts.core.registry import Registry

from .models import ic_composite as _ic_composite_models  # noqa: F401
from .models import linear as _linear_models  # noqa: F401
from .models import xgb as _xgb_models  # noqa: F401
from .models.xgb import train_and_predict_xgb_classifier

Registry.register_factor_trainer("xgb_classifier")(train_and_predict_xgb_classifier)

__all__: list[str] = []
