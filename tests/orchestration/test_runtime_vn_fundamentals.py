from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from qts.core.instrument import AssetType
from qts.data._schemas import DataType
from qts.data.manager import DataManager
from qts.data.sources.vnstock import VnstockDataSource
from qts.data.storage.duckdb import DuckDBStorage
from qts.data.storage.parquet import ParquetStorage
from qts.data.vn100_quantamental import make_vn_manager
from qts.orchestration.runtime import build_data_manager


def test_build_data_manager_uses_separate_vci_source_for_vn_fundamentals(tmp_path):
    resolved = SimpleNamespace(
        raw=SimpleNamespace(asset_types=["vn_stock"]),
        stock_source=None,
        vn_stock_source=VnstockDataSource.from_env(),
        vn_warrant_source=None,
        vn_futures_source=None,
        crypto_source=None,
        crypto_futures_source=None,
        storage=DuckDBStorage(),
        cache=ParquetStorage(tmp_path / "cache"),
        bundle_adapter=None,
    )

    manager = build_data_manager(resolved)

    price_source = manager._capability_map[(AssetType.VN_STOCK, DataType.OHLCV)]
    fundamentals_source = manager._capability_map[(AssetType.VN_STOCK, DataType.FUNDAMENTALS)]

    assert isinstance(price_source, VnstockDataSource)
    assert isinstance(fundamentals_source, VnstockDataSource)
    assert price_source is resolved.vn_stock_source
    assert fundamentals_source is not price_source
    assert price_source.fundamentals_source == "kbs"
    assert fundamentals_source.fundamentals_source == "vci"


def test_make_vn_manager_defaults_vn_fundamentals_to_vci(tmp_path):
    manager = make_vn_manager(Path(tmp_path))

    price_source = manager._capability_map[(AssetType.VN_STOCK, DataType.OHLCV)]
    fundamentals_source = manager._capability_map[(AssetType.VN_STOCK, DataType.FUNDAMENTALS)]

    assert isinstance(price_source, VnstockDataSource)
    assert isinstance(fundamentals_source, VnstockDataSource)
    assert price_source is not fundamentals_source
    assert price_source.fundamentals_source == "kbs"
    assert fundamentals_source.fundamentals_source == "vci"
