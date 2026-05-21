"""Generate and save a pyfolio tearsheet PDF from a BacktestResult."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from qts.research.backtest.base import BacktestResult
from qts.research.backtest.pyfolio_adapter import (
    positions_frame,
    returns_series,
    transactions_frame,
)

logger = logging.getLogger(__name__)


def save_tearsheet(
    result: BacktestResult,
    out_dir: Path,
    run_id: str,
    benchmark_rets: pd.Series | None = None,
) -> Path | None:
    """Generate a pyfolio tearsheet and save it as a multi-page PDF."""

    try:
        import pyfolio as pf
    except ImportError:
        logger.warning(
            "pyfolio-reloaded is not installed; skipping tearsheet. "
            "Install with: pip install 'qtradesystematic[reporting]'"
        )
        return None

    returns = returns_series(result)
    positions = positions_frame(result)
    transactions = transactions_frame(result)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{run_id}_tearsheet.pdf"

    plt.close("all")
    try:
        pf.create_full_tear_sheet(
            returns=returns,
            positions=positions,
            transactions=transactions,
            benchmark_rets=benchmark_rets,
            set_context=False,
        )
    except Exception:
        logger.exception(
            "pyfolio tearsheet generation failed; attempting returns-only tearsheet"
        )
        plt.close("all")
        try:
            pf.create_returns_tear_sheet(
                returns=returns,
                benchmark_rets=benchmark_rets,
                set_context=False,
            )
        except Exception:
            logger.exception("pyfolio returns-only tearsheet also failed; skipping")
            plt.close("all")
            return None

    with PdfPages(pdf_path) as pdf:
        for fig_num in plt.get_fignums():
            pdf.savefig(plt.figure(fig_num), bbox_inches="tight")
        pdf.infodict().update(
            {
                "Title": f"QTS Tearsheet - {run_id}",
                "Author": "QTradeSystematic",
            }
        )

    plt.close("all")
    logger.info("Tearsheet saved: %s", pdf_path)
    return pdf_path
