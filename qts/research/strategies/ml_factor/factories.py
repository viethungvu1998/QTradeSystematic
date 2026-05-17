"""Registered ML factor strategy factories."""

from __future__ import annotations

from qts.core.registry import Registry

from .model import train_and_predict_xgb_classifier

Registry.register_factor_trainer("xgb_classifier")(train_and_predict_xgb_classifier)
