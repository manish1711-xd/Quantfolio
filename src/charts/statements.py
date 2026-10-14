"""Financial-statement figures: the balance sheet, drawn as the identity it is.

One builder, and it exists because a balance sheet is the rare financial
statement with a *shape*: what a company owns must equal what funds it. A table
of numbers makes a reader verify that by arithmetic; two bars side by side make
it visible, and make the gap between them — the minority interest the provider
excludes from both of its figures — impossible to miss rather than a footnote.

Reading conventions this module commits to:

* **Assets on the left, funding on the right, per period.** The two bars are
  the two sides of the identity, so an eye can check it without reading a
  single figure.
* **The residual is drawn, not hidden.** When assets exceed liabilities plus
  equity, the difference is stacked on the funding bar and labelled, because a
  gap silently omitted is a gap the reader will assume does not exist.
* **Liabilities take the loss hue and equity the gain hue** — not decoration:
  the app reserves those two colours for "owed" and "owned" everywhere else,
  and the funding side of a balance sheet is exactly that distinction.

Nothing here imports Streamlit; the figure never raises, so a company whose
provider extract is missing a line degrades to a titled placeholder.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.charts import theme

__all__ = ["balance_sheet_chart"]

#: Prefix of the last-resort placeholder. Any figure carrying it is a bug.
_FALLBACK_PREFIX = "Chart unavailable: "

_HEIGHT = 420

#: Bar width, as a fraction of the category slot. Two bars share each period,
#: so anything above 0.4 makes adjacent periods touch.
_BAR_WIDTH = 0.38


def _never_raises(func: Callable[..., go.Figure]) -> Callable[..., go.Figure]:
    """Guarantee the chart contract: a Figure comes back, always."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> go.Figure:
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - defensive only  # pragma: no cover
            return theme.empty_figure(
                f"{_FALLBACK_PREFIX}{type(exc).__name__} while drawing this figure."
            )

    return wrapper


def _rgba(color: str, alpha: float) -> str:
    """Re-express a theme colour at a given opacity, so tints stay themed."""
    text = str(color).strip()
    if text.startswith("#"):
        digits = text[1:]
        if len(digits) == 3:
            digits = "".join(ch * 2 for ch in digits)
        if len(digits) >= 6:
            red, green, blue = (int(digits[i : i + 2], 16) for i in (0, 2, 4))
            return f"rgba({red}, {green}, {blue}, {alpha:.3f})"
    return text


@_never_raises
def balance_sheet_chart(
    check_rows: pd.DataFrame,
    currency_symbol: str = "",
    height: int | None = None,
) -> go.Figure:
    """Assets against liabilities plus equity, one pair of bars per period.

    Parameters
    ----------
    check_rows:
        The ``rows`` frame from
        :func:`src.analytics.fundamentals.balance_sheet_check` — one row per
        period with ``period, total_assets, total_liabilities, equity,
        residual``.
    currency_symbol:
        Prefix for the value axis and hover text.

    The left bar of each pair is total assets. The right bar stacks total
    liabilities, total equity, and — when it is positive — the residual, which
    is the minority interest the provider's two figures both exclude. Two bars
    of equal height mean the extract balances; a visible cap on the right-hand
    bar is exactly the amount by which it does not, and it is labelled rather
    than absorbed into one of the other two segments.

    Oldest period on the left, because a balance sheet is read as a trend and
    time runs forwards.
    """
    if not isinstance(check_rows, pd.DataFrame) or check_rows.empty:
        return theme.empty_figure(
            "No period reported total assets, liabilities and equity together."
        )
    needed = {"period", "total_assets", "total_liabilities", "equity"}
    if not needed.issubset(check_rows.columns):
        missing = ", ".join(sorted(needed - set(check_rows.columns)))
        return theme.empty_figure(f"The balance-sheet check is missing: {missing}.")

    frame = check_rows.copy()
    for column in ("total_assets", "total_liabilities", "equity", "residual"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["total_assets", "total_liabilities", "equity"])
    if frame.empty:
        return theme.empty_figure("Every period is missing one of the three totals.")

    # Oldest first: the table reads newest-first for comparison, a trend reads
    # forwards. The two orders are deliberate and each suits its own medium.
    frame = frame.iloc[::-1].reset_index(drop=True)

    periods = frame["period"].astype(str).tolist()
    assets = frame["total_assets"].to_numpy(dtype="float64")
    liabilities = frame["total_liabilities"].to_numpy(dtype="float64")
    equity = frame["equity"].to_numpy(dtype="float64")
    residual = (
        frame["residual"].to_numpy(dtype="float64")
        if "residual" in frame.columns
        else assets - (liabilities + equity)
    )
    gap = np.where(np.isfinite(residual) & (residual > 0), residual, 0.0)

    palette = theme.palette()
    symbol = currency_symbol or ""
    money = f"{symbol}%{{y:,.0f}}"
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Total assets",
            x=periods,
            y=assets,
            offsetgroup="assets",
            width=_BAR_WIDTH,
            marker={"color": _rgba(palette["accent"], 0.85)},
            hovertemplate=f"%{{x}}<br>Total assets  {money}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Liabilities",
            x=periods,
            y=liabilities,
            offsetgroup="funding",
            width=_BAR_WIDTH,
            marker={"color": _rgba(palette["loss"], 0.80)},
            hovertemplate=f"%{{x}}<br>Liabilities  {money}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Equity",
            x=periods,
            y=equity,
            offsetgroup="funding",
            width=_BAR_WIDTH,
            marker={"color": _rgba(palette["gain"], 0.80)},
            hovertemplate=f"%{{x}}<br>Equity  {money}<extra></extra>",
        )
    )
    if float(np.nansum(gap)) > 0.0:
        fig.add_trace(
            go.Bar(
                name="Minority interest / unexplained",
                x=periods,
                y=gap,
                offsetgroup="funding",
                width=_BAR_WIDTH,
                marker={
                    "color": _rgba(palette["warning"], 0.55),
                    "pattern": {"shape": "/", "size": 5, "solidity": 0.25},
                },
                hovertemplate=(
                    f"%{{x}}<br>Excluded from both provider figures  {money}<extra></extra>"
                ),
            )
        )

    theme.apply(
        fig,
        title="Balance sheet: what it owns against what funds it",
        subtitle="Left bar is total assets; right bar stacks the funding side",
        height=height or _HEIGHT,
        barmode="stack",
        xaxis_title="Fiscal period end",
        yaxis_title=f"Amount ({symbol})" if symbol else "Amount",
    )
    theme.money_axis(fig, currency_symbol)
    theme.caption(
        fig,
        "Equal bars mean the extract satisfies assets = liabilities + equity. The hatched "
        "cap is the difference, which is normally minority interest — excluded from the "
        "provider's liabilities AND its equity, so it falls out of the middle.",
    )
    fig.update_layout(margin={"b": 96})
    return fig
