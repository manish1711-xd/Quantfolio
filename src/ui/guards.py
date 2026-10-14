"""Page preconditions.

Sixteen of the seventeen pages are meaningless without a completed analysis, and
several need history, a benchmark or multiple holdings on top of that. Rather
than repeating the same defensive ``if`` blocks everywhere, each page opens
with one guard call that either returns what it needs or renders an honest
explanation of what is missing and why.

The guards return ``None`` on failure so a page can simply ``return``.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.config import MIN_OBSERVATIONS
from src.ui import components as ui
from src.ui import session


def require_result() -> Any | None:
    """The completed :class:`AnalysisResult`, or an empty state prompting a run."""
    result = session.get_result()
    if result is not None:
        return result
    ui.empty_state(
        "No analysis yet",
        "Enter holdings on the Portfolio › Build page (or click Load sample in the "
        "sidebar), then press Analyze portfolio.",
        icon="📊",
        hint="Everything on this page is computed from your holdings and live market data.",
    )
    if st.button("Analyze now", type="primary", key="guard_analyze"):
        session.request_run()
        st.rerun()
    return None


def require_history(result: Any, what: str = "This analysis") -> bool:
    """True when there is a usable portfolio value series."""
    if result.has_history:
        return True
    ui.empty_state(
        "Not enough price history",
        f"{what} needs at least two overlapping trading days across your holdings. "
        "A recently listed name shortens the window for the whole portfolio — try a "
        "longer history window, or remove the newest holding.",
        icon="🕰",
    )
    return False


def require_observations(
    result: Any, minimum: int = MIN_OBSERVATIONS, what: str = "This analysis"
) -> bool:
    """True when the window is long enough for the statistic to mean anything."""
    if not require_history(result, what):
        return False
    n = result.n_days
    if n >= minimum:
        return True
    ui.empty_state(
        "Window too short to be meaningful",
        f"{what} needs at least {minimum} overlapping trading days; this window has {n}. "
        "The numbers would be noise dressed up as analysis.",
        icon="📉",
        hint="Increase the history window in the sidebar.",
    )
    return False


def require_benchmark(result: Any) -> dict | None:
    """The benchmark comparison block, or an explanation of why it is missing."""
    bench = result.benchmark or {}
    if bench.get("ok"):
        return bench
    reason = bench.get("reason") or "the benchmark series could not be fetched"
    ui.empty_state(
        "Benchmark comparison unavailable",
        f"Could not compare against the selected index: {reason}.",
        icon="🆚",
        hint="Pick a different benchmark in the sidebar, or widen the history window.",
    )
    return None


def require_multiple_assets(result: Any, what: str = "This analysis", minimum: int = 2) -> bool:
    """True when the portfolio has enough priced holdings with history."""
    frame = result.asset_returns
    n = 0 if frame is None else int(getattr(frame, "shape", (0, 0))[1])
    if n >= minimum:
        return True
    ui.empty_state(
        f"Needs at least {minimum} holdings",
        f"{what} compares holdings against each other, and this portfolio has "
        f"{n} with usable price history.",
        icon="🔗",
        hint="Add more holdings on the Portfolio page.",
    )
    return False


def require_ledger(result: Any) -> dict | None:
    """The FIFO ledger block, or a prompt to supply transactions."""
    ledger = result.ledger or {}
    if ledger.get("transactions"):
        return ledger
    ui.empty_state(
        "No trade ledger supplied",
        "Realised P&L, FIFO tax lots, holding periods and money-weighted return "
        "(XIRR) all need the actual trades — when you bought, how much, and at what "
        "price. Average cost alone cannot produce them.",
        icon="🧾",
        hint="Add trades on the Portfolio page, or load the sample portfolio, "
        "which ships with an 18-trade ledger.",
    )
    return None
