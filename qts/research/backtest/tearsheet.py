"""Generate and save a pyfolio tearsheet PDF from a BacktestResult."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from qts.research.backtest.base import BacktestResult
from qts.research.backtest.pyfolio_adapter import (
    positions_frame,
    returns_series,
    transactions_frame,
)
from qts.utils.date_intervals import DateInterval, format_date_intervals

logger = logging.getLogger(__name__)
_TEST_INTERVAL_COLOR = "#d9d9d9"
_TEST_INTERVAL_ALPHA = 0.35


def save_tearsheet(
    result: BacktestResult,
    out_dir: Path,
    run_id: str,
    benchmark_rets: pd.Series | None = None,
    test_intervals: Sequence[DateInterval] | None = None,
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
        if test_intervals:
            _add_interval_overview_page(
                pdf,
                result=result,
                run_id=run_id,
                benchmark_rets=benchmark_rets,
                test_intervals=test_intervals,
            )
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            shade_test_intervals_on_figure(fig, test_intervals)
            pdf.savefig(fig, bbox_inches="tight")
        pdf.infodict().update(
            {
                "Title": f"QTS Tearsheet - {run_id}",
                "Author": "QTradeSystematic",
            }
        )

    plt.close("all")
    logger.info("Tearsheet saved: %s", pdf_path)
    return pdf_path


def shade_test_intervals_on_figure(
    fig: plt.Figure,
    test_intervals: Sequence[DateInterval] | None,
) -> None:
    """Shade configured test intervals on every date-like x-axis in a figure."""

    if not test_intervals:
        return
    for ax in fig.axes:
        shade_test_intervals_on_axis(ax, test_intervals)


def shade_test_intervals_on_axis(
    ax: plt.Axes,
    test_intervals: Sequence[DateInterval] | None,
) -> None:
    """Shade configured test intervals when the axis uses matplotlib date numbers."""

    if not test_intervals:
        return
    x_min, x_max = ax.get_xlim()
    if max(abs(x_min), abs(x_max)) < 1_000:
        return

    lower = min(x_min, x_max)
    upper = max(x_min, x_max)
    label_used = any(
        getattr(collection, "get_label", lambda: "")() == "Test interval"
        for collection in ax.collections
    )
    for start, end in test_intervals:
        start_num = mdates.date2num(pd.Timestamp(start).to_pydatetime())
        if end is None:
            end_num = upper
        else:
            end_num = mdates.date2num((pd.Timestamp(end) + timedelta(days=1)).to_pydatetime())
        if end_num < lower or start_num > upper:
            continue
        label = "_nolegend_" if label_used else "Test interval"
        ax.axvspan(
            max(start_num, lower),
            min(end_num, upper),
            color=_TEST_INTERVAL_COLOR,
            alpha=_TEST_INTERVAL_ALPHA,
            linewidth=0,
            zorder=0,
            label=label,
        )
        label_used = True


def _add_interval_overview_page(
    pdf: PdfPages,
    *,
    result: BacktestResult,
    run_id: str,
    benchmark_rets: pd.Series | None,
    test_intervals: Sequence[DateInterval],
) -> None:
    equity = result.equity_curve.to_pandas()
    returns = result.returns.to_pandas()
    if equity.empty and returns.empty:
        return

    for frame in (equity, returns):
        if not frame.empty:
            frame["date"] = pd.to_datetime(frame["date"])

    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    fig.suptitle("Full-Period Report with Test Intervals", fontsize=15, x=0.02, ha="left")
    fig.text(0.02, 0.94, f"Run ID: {run_id}", fontsize=8, family="monospace")
    fig.text(
        0.02,
        0.915,
        f"Test intervals shaded grey: {format_date_intervals(test_intervals)}",
        fontsize=8,
        family="monospace",
    )

    if not equity.empty:
        axes[0].plot(equity["date"], equity["equity"], color="tab:blue", linewidth=1.2)
    axes[0].set_title("Equity Curve")
    axes[0].grid(alpha=0.25)

    if not returns.empty:
        cumulative = (1.0 + returns["portfolio_return"].astype(float)).cumprod()
        axes[1].plot(returns["date"], cumulative, label="Strategy", linewidth=1.2)
        if benchmark_rets is not None and not benchmark_rets.empty:
            benchmark = benchmark_rets.copy()
            benchmark.index = pd.to_datetime(benchmark.index).tz_localize(None).normalize()
            benchmark = benchmark.sort_index()
            benchmark_cumulative = (1.0 + benchmark.astype(float)).cumprod()
            axes[1].plot(
                benchmark_cumulative.index,
                benchmark_cumulative.to_numpy(),
                label=getattr(benchmark_rets, "name", None) or "Benchmark",
                linewidth=1.0,
            )
            axes[1].legend(loc="best")
        axes[2].bar(
            returns["date"],
            returns["portfolio_return"].astype(float),
            width=1.0,
            color="tab:green",
            alpha=0.65,
        )
    axes[1].set_title("Cumulative Return")
    axes[1].grid(alpha=0.25)
    axes[2].set_title("Daily Portfolio Return")
    axes[2].grid(alpha=0.25)

    for ax in axes:
        shade_test_intervals_on_axis(ax, test_intervals)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
