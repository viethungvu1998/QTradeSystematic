from __future__ import annotations

from datetime import date

import polars as pl

from qts.core.registry import Registry
from qts.research.features.transforms.targets import ml_factor_target_class_transform


def _target_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [date(2024, 1, day) for day in range(1, 7)],
            "symbol": ["AAA"] * 6,
            "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "forward_return_1": [-0.10, -0.02, 0.00, 0.03, 0.08, None],
        }
    )


def test_ml_factor_target_class_transform_appends_five_class_labels():
    result = ml_factor_target_class_transform(
        _target_frame(),
        target_column="forward_return_1",
    )

    assert "forward_return_1_class" in result.columns
    assert result["forward_return_1_class"].drop_nulls().to_list() == [0, 1, 2, 3, 4]
    assert result["forward_return_1_class"][-1] is None


def test_ml_factor_target_class_transform_derives_missing_forward_return_column():
    frame = _target_frame().drop("forward_return_1")

    result = ml_factor_target_class_transform(
        frame,
        target_column="forward_return_1",
        output_column="label",
    )

    assert "forward_return_1" in result.columns
    assert result["label"].drop_nulls().to_list() == [4, 3, 2, 1, 0]
    assert result["label"][-1] is None


def test_ml_factor_target_class_transform_registers_with_registry_after_module_import():
    assert Registry.get_transform("ml_factor_target_class") is ml_factor_target_class_transform
