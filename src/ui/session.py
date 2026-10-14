"""Session-state schema and helpers.

Streamlit reruns the entire script on every interaction, so ``st.session_state``
is the application's only memory. Centralising the keys here (rather than
sprinkling ``st.session_state.get("something")`` across seventeen pages) means
the shape of that memory is declared in one readable place — and that a typo in
a key name is a missing attribute, not a silently-empty widget.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.config import (
    DEFAULT_BASE_CURRENCY,
    DEFAULT_BENCHMARK,
    DEFAULT_PERIOD,
    DEFAULT_RISK_FREE,
    AppConfig,
)

# ---------------------------------------------------------------------------
# key names (single source of truth)
# ---------------------------------------------------------------------------
HOLDINGS = "holdings"
TRANSACTIONS = "transactions"
EDITOR_VERSION = "editor_version"
TXN_EDITOR_VERSION = "txn_editor_version"
RUN = "run"
RESULT = "result"
CONFIG = "config"
WARNINGS = "warnings"
SELECTED_TICKER = "selected_ticker"
TARGET_WEIGHTS = "target_weights"
UPLOAD_ID = "upload_id"
TXN_UPLOAD_ID = "txn_upload_id"
LAST_RUN_KEY = "last_run_key"
WATCHLIST = "watchlist"
WATCHLIST_LOADED = "watchlist_loaded"


def default_holdings() -> pd.DataFrame:
    return pd.DataFrame({"ticker": ["RELIANCE.NS"], "quantity": [10.0], "buy_price": [1300.0]})


def empty_transactions() -> pd.DataFrame:
    return pd.DataFrame(columns=["ticker", "trade_date", "action", "quantity", "price", "fees"])


def default_config() -> AppConfig:
    return AppConfig(
        period=DEFAULT_PERIOD,
        benchmark_name=DEFAULT_BENCHMARK,
        base_currency=DEFAULT_BASE_CURRENCY,
        risk_free_rate=DEFAULT_RISK_FREE[DEFAULT_BASE_CURRENCY],
    )


_DEFAULTS: dict[str, Any] = {
    EDITOR_VERSION: 0,
    TXN_EDITOR_VERSION: 0,
    RUN: False,
    RESULT: None,
    WARNINGS: [],
    SELECTED_TICKER: None,
    TARGET_WEIGHTS: {},
    UPLOAD_ID: None,
    TXN_UPLOAD_ID: None,
    LAST_RUN_KEY: None,
    # Watchlist items live as a tuple of src.watchlist.WatchItem. Not a
    # DataFrame: it is a small ordered collection edited one entry at a time,
    # and the immutable-tuple helpers in src.watchlist make every mutation a
    # replacement rather than an in-place edit Streamlit could half-apply.
    WATCHLIST: (),
    WATCHLIST_LOADED: False,
}


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------
def init() -> None:
    """Populate any missing session keys. Safe to call on every rerun."""
    if HOLDINGS not in st.session_state:
        st.session_state[HOLDINGS] = default_holdings()
    if TRANSACTIONS not in st.session_state:
        st.session_state[TRANSACTIONS] = empty_transactions()
    if CONFIG not in st.session_state:
        st.session_state[CONFIG] = default_config()
    for key, value in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def replace_holdings(df: pd.DataFrame) -> None:
    """Swap the holdings table AND bump the editor key.

    ``st.data_editor`` keeps widget state per key. Changing the input frame
    while reusing the key leaves the editor showing the user's stale edits, so
    versioning the key forces a clean re-render whenever the data *source*
    changes (sample loaded, CSV uploaded, portfolio opened, table cleared).
    """
    st.session_state[HOLDINGS] = df.reset_index(drop=True)
    st.session_state[EDITOR_VERSION] += 1


def replace_transactions(df: pd.DataFrame) -> None:
    st.session_state[TRANSACTIONS] = df.reset_index(drop=True)
    st.session_state[TXN_EDITOR_VERSION] += 1


def request_run() -> None:
    st.session_state[RUN] = True
    st.session_state[LAST_RUN_KEY] = None  # force recompute even if inputs match


# ---------------------------------------------------------------------------
# accessors
# ---------------------------------------------------------------------------
def get_config() -> AppConfig:
    return st.session_state.get(CONFIG) or default_config()


def set_config(cfg: AppConfig) -> None:
    st.session_state[CONFIG] = cfg


def get_result() -> Any:
    """The most recent :class:`src.models.AnalysisResult`, or ``None``."""
    return st.session_state.get(RESULT)


def set_result(result: Any) -> None:
    st.session_state[RESULT] = result


def has_result() -> bool:
    return st.session_state.get(RESULT) is not None


def get_holdings() -> pd.DataFrame:
    return st.session_state.get(HOLDINGS, default_holdings())


def get_transactions() -> pd.DataFrame:
    return st.session_state.get(TRANSACTIONS, empty_transactions())


def has_transactions() -> bool:
    txns = st.session_state.get(TRANSACTIONS)
    return txns is not None and not txns.empty


def get_watchlist() -> tuple:
    """The current watchlist as a tuple of :class:`src.watchlist.WatchItem`."""
    return tuple(st.session_state.get(WATCHLIST) or ())


def set_watchlist(items: Any) -> None:
    st.session_state[WATCHLIST] = tuple(items or ())


def selected_ticker(default: str | None = None) -> str | None:
    return st.session_state.get(SELECTED_TICKER) or default


def select_ticker(ticker: str | None) -> None:
    st.session_state[SELECTED_TICKER] = ticker


def run_key(cfg: AppConfig, holdings: pd.DataFrame) -> str:
    """Fingerprint of everything that would change the analysis.

    Lets the pipeline skip recomputation when a page switch (which reruns the
    whole script) did not actually change any input.
    """
    try:
        rows = "|".join(
            f"{r.ticker}:{r.quantity}:{r.buy_price}" for r in holdings.itertuples(index=False)
        )
    except Exception:  # noqa: BLE001 - a malformed grid must not crash the app
        rows = str(len(holdings))
    return "|".join(
        [
            rows,
            cfg.period,
            cfg.benchmark_name,
            cfg.base_currency,
            f"{cfg.risk_free_rate:.6f}",
            f"{cfg.var_confidence:.4f}",
            str(cfg.convert_fx),
            str(cfg.rolling_window),
        ]
    )
