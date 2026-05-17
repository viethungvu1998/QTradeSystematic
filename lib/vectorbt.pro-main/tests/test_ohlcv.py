import os

import pytest

import vectorbtpro as vbt
from tests.utils import *

ohlcv_df = pd.DataFrame(
    {
        "open": [1, 2, 3, 4, 5],
        "high": [2.5, 3.5, 4.5, 5.5, 6.5],
        "low": [0.5, 1.5, 2.5, 3.5, 4.5],
        "close": [2, 3, 4, 5, 6],
        "volume": [1, 2, 3, 2, 1],
    },
    index=pd.date_range("2020", periods=5),
)


# ############# Global ############# #


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True
    vbt.settings.chunking["n_chunks"] = 2


def teardown_module():
    vbt.settings.reset()


# ############# accessors ############# #


class TestAccessors:
    @pytest.mark.parametrize("test_freq", ["1h", "10h", "3d"])
    def test_resample(self, test_freq):
        assert_frame_equal(
            ohlcv_df.vbt.ohlcv.resample(test_freq).obj,
            ohlcv_df.resample(test_freq).agg(
                {
                    "open": lambda x: float(x[0] if len(x) > 0 else np.nan),
                    "high": lambda x: float(x.max() if len(x) > 0 else np.nan),
                    "low": lambda x: float(x.min() if len(x) > 0 else np.nan),
                    "close": lambda x: float(x[-1] if len(x) > 0 else np.nan),
                    "volume": lambda x: float(x.sum() if len(x) > 0 else np.nan),
                }
            ),
        )
