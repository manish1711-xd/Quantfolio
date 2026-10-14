"""Position-level P&L, FX conversion and portfolio aggregation.

Pure pandas/numpy -- no network, no Streamlit -- so every formula in this
module is directly unit-testable against hand-computed fixtures.

Core formulas
-------------
Investment          = quantity x average buy price x fx_rate     (base currency)
Current value       = quantity x latest market price x fx_rate   (base currency)
Profit/Loss         = current value - investment                 (base currency)
Profit/Loss %       = P&L / investment x 100          (guarded: investment > 0)
Portfolio weight %  = current value / total value x 100   (guarded: total > 0)
HHI                 = sum(w_i^2) with w as fractions
Effective holdings  = 1 / HHI

Currency convention (the subtle bit)
------------------------------------
``buy_price`` and ``current_price`` stay in the instrument's NATIVE quote
currency: a US name inside a rupee-based portfolio is still shown at its
dollar price, because that is the number the holder recognises. Every MONEY
column (``investment``, ``current_value``, ``pnl``) is converted into the
portfolio's base currency by multiplying with ``fx_rate``, where ``fx_rate``
is the value of one unit of the instrument's currency expressed in the base
currency. Because both legs of the P&L are converted at the SAME rate,
``pnl_pct`` is FX-neutral and equals ``(current_price / buy_price - 1) * 100``.

When ``fx_rates`` is ``None`` every rate is 1.0, which reproduces the legacy
single-currency behaviour of ``src/portfolio_calculator.py`` exactly.

Nothing here fabricates a number: undefined statistics are ``float("nan")``
and unusable inputs are reported rather than silently coerced to zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DEFAULT_BASE_CURRENCY
from src.models import Quote, SecurityProfile

# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------
#: Column names (after normalisation) that a holdings table must contain.
REQUIRED_COLUMNS: tuple[str, ...] = ("ticker", "quantity", "buy_price")

#: Columns that are carried through when the user supplies them.
OPTIONAL_COLUMNS: tuple[str, ...] = ("buy_date", "sector", "target_weight_pct")

#: Exact output schema of :func:`build_positions`, in order.
POSITION_COLUMNS: tuple[str, ...] = (
    "ticker",
    "name",
    "sector",
    "industry",
    "quantity",
    "buy_price",
    "current_price",
    "currency",
    "fx_rate",
    "day_change_pct",
    "investment",
    "current_value",
    "pnl",
    "pnl_pct",
    "weight_pct",
)

#: Holdings columns passed through onto the positions table when the caller
#: supplied them. They are appended AFTER ``POSITION_COLUMNS``, never inserted
#: into it, so a plain holdings frame yields exactly the documented schema.
_PASSTHROUGH_COLUMNS: tuple[str, ...] = ("buy_date", "target_weight_pct")

#: Keys ``allocation_by`` knows how to group on.
ALLOCATION_KEYS: tuple[str, ...] = ("sector", "industry", "currency", "ticker")

_UNCLASSIFIED = "Unclassified"

#: Placeholder ticker strings produced by empty grid rows / CSV padding.
_BLANK_TICKERS: frozenset[str] = frozenset({"", "NAN", "NONE"})

_BAD_ROW_MESSAGE = "{ticker}: quantity and buy price must be positive numbers - row skipped."


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _nan() -> float:
    return float("nan")


def _as_float(value: object) -> float:
    """Best-effort float conversion; anything unparseable becomes NaN."""
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return _nan()
    return out if np.isfinite(out) else _nan()


def _clean_text(value: object) -> str | None:
    """Strip a possibly-missing string; blanks and NaN become ``None``."""
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    try:
        if bool(pd.isna(value)):  # type: ignore[arg-type]
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.upper() in {"NAN", "NONE", "NAT", "NULL"}:
        return None
    return text


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case / snake-case the columns of a copy, dropping duplicates.

    ``"Buy Price"`` -> ``"buy_price"``. If two source columns normalise to the
    same name (``"Ticker"`` and ``"ticker"``) the first one wins, which keeps
    ``df["ticker"]`` a Series rather than a DataFrame.
    """
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()]
    return out


def _first_text(values: pd.Series) -> str | None:
    """First non-blank string in a group (used when merging duplicate rows)."""
    for value in values:
        text = _clean_text(value)
        if text is not None:
            return text
    return None


def _weight_fractions(positions: pd.DataFrame | None) -> np.ndarray:
    """Finite portfolio weights as fractions summing to 1 (empty if unknown).

    ``weight_pct`` is stored in percent; it is divided by 100 and renormalised
    by its own sum so that a *subset* of positions (say the top five) still
    yields a self-consistent concentration measure.
    """
    if positions is None or len(positions) == 0:
        return np.empty(0, dtype=float)
    if "weight_pct" not in positions.columns:
        return np.empty(0, dtype=float)
    w = pd.to_numeric(positions["weight_pct"], errors="coerce").to_numpy(dtype=float)
    w = w[np.isfinite(w)] / 100.0
    total = float(np.sum(w))
    if w.size == 0 or not np.isfinite(total) or total <= 0:
        return np.empty(0, dtype=float)
    return w / total


def _fx_rate_for(currency: str | None, fx_rates: dict[str, float] | None) -> float:
    """Rate converting one unit of ``currency`` into the base currency.

    Returns NaN when the rate is absent or unusable (non-finite, <= 0) and
    lets the caller decide how to degrade. Lookup is tried verbatim and then
    upper-cased, so ``{"USD": 83.0}`` still matches a quote reporting ``usd``.
    """
    if not fx_rates:
        return _nan()
    key = _clean_text(currency)
    if key is None:
        return _nan()
    for candidate in (key, key.upper()):
        if candidate in fx_rates:
            rate = _as_float(fx_rates[candidate])
            return rate if np.isfinite(rate) and rate > 0 else _nan()
    return _nan()


# ---------------------------------------------------------------------------
# holdings validation
# ---------------------------------------------------------------------------
def clean_holdings(raw: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str]]:
    """Validate and normalise raw user input (editor grid or uploaded CSV).

    Rules applied, each reported at most once per ticker in ``problems``:
      * column names are lower-cased / snake-cased ("Buy Price" -> "buy_price")
      * tickers are stripped and upper-cased; blank/placeholder tickers are
        dropped silently (they are just empty grid rows)
      * quantity and buy price must parse as numbers and be strictly positive
      * duplicate tickers are merged: quantity is summed and the buy price
        becomes the quantity-weighted average

            p_avg = sum(q_i * p_i) / sum(q_i)

        e.g. 10 @ 100 plus 30 @ 200 -> 40 @ (1000 + 6000) / 40 = 175.

    Optional columns are preserved when present: ``buy_date`` (coerced with
    ``pd.to_datetime``; a merged row keeps the EARLIEST date), ``sector`` (the
    first non-blank label) and ``target_weight_pct`` (summed across the merged
    rows, since two lots of one name share a single target).

    Returns ``(clean_df, problems)`` and never raises on bad input.
    """
    empty = pd.DataFrame(columns=list(REQUIRED_COLUMNS))
    problems: dict[str, str] = {}

    if raw is None:
        return empty, ["Portfolio is empty - add at least one holding."]
    if not isinstance(raw, pd.DataFrame):
        try:
            raw = pd.DataFrame(raw)
        except (TypeError, ValueError):
            return empty, ["Holdings input could not be read as a table."]
    if raw.empty:
        return empty, ["Portfolio is empty - add at least one holding."]

    df = _normalise_columns(raw)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return empty, [f"Missing column(s): {', '.join(missing)}."]

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["buy_price"] = pd.to_numeric(df["buy_price"], errors="coerce")

    df = df[~df["ticker"].isin(_BLANK_TICKERS)]

    bad = (
        df["quantity"].isna()
        | df["buy_price"].isna()
        | (df["quantity"] <= 0)
        | (df["buy_price"] <= 0)
    )
    for ticker in df.loc[bad, "ticker"].tolist():
        problems.setdefault(str(ticker), _BAD_ROW_MESSAGE.format(ticker=ticker))
    df = df.loc[~bad]

    if df.empty:
        return empty, [*list(problems.values()), "No valid holdings after validation."]

    optional = [c for c in OPTIONAL_COLUMNS if c in df.columns]
    if "buy_date" in optional:
        df["buy_date"] = pd.to_datetime(df["buy_date"], errors="coerce")
    if "target_weight_pct" in optional:
        df["target_weight_pct"] = pd.to_numeric(df["target_weight_pct"], errors="coerce")

    # merge duplicate tickers with a quantity-weighted average buy price
    df = df.assign(_cost=df["quantity"] * df["buy_price"])
    aggregations: dict[str, tuple] = {
        "quantity": ("quantity", "sum"),
        "_cost": ("_cost", "sum"),
    }
    if "buy_date" in optional:
        aggregations["buy_date"] = ("buy_date", "min")
    if "sector" in optional:
        aggregations["sector"] = ("sector", _first_text)
    if "target_weight_pct" in optional:
        aggregations["target_weight_pct"] = (
            "target_weight_pct",
            lambda s: s.sum(min_count=1),
        )

    merged = df.groupby("ticker", as_index=False).agg(**aggregations)
    merged["buy_price"] = np.where(
        merged["quantity"] > 0, merged["_cost"] / merged["quantity"], np.nan
    )

    return merged[list(REQUIRED_COLUMNS) + optional], list(problems.values())


# ---------------------------------------------------------------------------
# positions table
# ---------------------------------------------------------------------------
def build_positions(
    holdings: pd.DataFrame | None,
    quotes: dict[str, Quote] | None,
    profiles: dict[str, SecurityProfile] | None = None,
    fx_rates: dict[str, float] | None = None,
    base_currency: str = DEFAULT_BASE_CURRENCY,
) -> tuple[pd.DataFrame, list[str]]:
    """Combine validated holdings, quotes, reference data and FX into positions.

    Columns, in order: ``ticker, name, sector, industry, quantity, buy_price,
    current_price, currency, fx_rate, day_change_pct, investment,
    current_value, pnl, pnl_pct, weight_pct`` -- with ``buy_date`` and
    ``target_weight_pct`` appended after those when the holdings frame carries
    them.

    Money maths (base currency)::

        investment    = quantity * buy_price     * fx_rate
        current_value = quantity * current_price * fx_rate
        pnl           = current_value - investment
        pnl_pct       = pnl / investment * 100              (investment > 0)
        weight_pct    = current_value / total_value * 100   (total > 0)

    Prices stay in the instrument's native currency; ``fx_rate`` is
    ``fx_rates.get(currency, 1.0)`` and is 1.0 for every row when ``fx_rates``
    is ``None``. A rate that is present but unusable (non-finite or <= 0) also
    degrades to 1.0 rather than poisoning the totals with NaN; every currency
    left unconverted that way is listed in
    ``positions.attrs["fx_unconverted"]`` so the UI can say so out loud.
    ``positions.attrs["base_currency"]`` records the currency the money
    columns are expressed in.

    ``name``/``sector``/``industry`` come from ``profiles``, falling back to
    the ticker and to ``"Unclassified"``; a ``sector`` column on the holdings
    frame is used when the profile supplies none.

    Tickers with no usable quote (missing, non-finite or <= 0 price) are
    skipped and returned in the second element, so one dead symbol never
    breaks the whole analysis.
    """
    base = _clean_text(base_currency) or DEFAULT_BASE_CURRENCY
    passthrough: list[str] = []
    if isinstance(holdings, pd.DataFrame):
        passthrough = [c for c in _PASSTHROUGH_COLUMNS if c in holdings.columns]
    columns = list(POSITION_COLUMNS) + passthrough

    empty = pd.DataFrame(columns=columns)
    empty.attrs["base_currency"] = base
    empty.attrs["fx_unconverted"] = []
    if holdings is None or not isinstance(holdings, pd.DataFrame) or holdings.empty:
        return empty, []

    quotes = quotes or {}
    profiles = profiles or {}
    has_sector_column = "sector" in holdings.columns

    rows: list[dict] = []
    skipped: list[str] = []
    unconverted: set[str] = set()

    for _, holding in holdings.iterrows():
        raw_ticker = _clean_text(holding.get("ticker"))
        if raw_ticker is None:
            continue
        ticker = raw_ticker.upper()

        quote = quotes.get(raw_ticker)
        if quote is None:
            quote = quotes.get(ticker)

        price = _as_float(getattr(quote, "price", None))
        quantity = _as_float(holding.get("quantity"))
        buy_price = _as_float(holding.get("buy_price"))
        if not np.isfinite(price) or price <= 0:
            skipped.append(ticker)
            continue
        if not np.isfinite(quantity) or not np.isfinite(buy_price):
            skipped.append(ticker)
            continue

        currency = _clean_text(getattr(quote, "currency", None)) or base
        rate = _fx_rate_for(currency, fx_rates)
        if not np.isfinite(rate):
            rate = 1.0
            if fx_rates and currency.upper() != base.upper():
                unconverted.add(currency)

        profile = profiles.get(ticker)
        if profile is None:
            profile = profiles.get(raw_ticker)
        name = _clean_text(getattr(profile, "display_name", None)) or ticker
        sector = _clean_text(getattr(profile, "sector", None))
        if sector is None and has_sector_column:
            sector = _clean_text(holding.get("sector"))
        if sector is None:
            sector = _clean_text(getattr(profile, "sector_label", None)) or _UNCLASSIFIED
        industry = _clean_text(getattr(profile, "industry", None)) or _UNCLASSIFIED

        investment = quantity * buy_price * rate
        current_value = quantity * price * rate
        pnl = current_value - investment
        pnl_pct = (pnl / investment * 100.0) if investment > 0 else _nan()

        row = {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "industry": industry,
            "quantity": quantity,
            "buy_price": buy_price,
            "current_price": price,
            "currency": currency,
            "fx_rate": float(rate),
            "day_change_pct": _as_float(getattr(quote, "day_change_pct", None)),
            "investment": float(investment),
            "current_value": float(current_value),
            "pnl": float(pnl),
            "pnl_pct": float(pnl_pct),
        }
        for column in passthrough:
            row[column] = holding.get(column)
        rows.append(row)

    if not rows:
        return empty, skipped

    positions = pd.DataFrame(rows)
    total_value = float(positions["current_value"].sum())
    positions["weight_pct"] = np.where(
        total_value > 0, positions["current_value"] / total_value * 100.0, np.nan
    )

    positions = (
        positions.reindex(columns=columns)
        .sort_values("current_value", ascending=False)
        .reset_index(drop=True)
    )
    positions.attrs["base_currency"] = base
    positions.attrs["fx_unconverted"] = sorted(unconverted)
    return positions, skipped


# ---------------------------------------------------------------------------
# portfolio aggregation
# ---------------------------------------------------------------------------
def _day_change_totals(positions: pd.DataFrame) -> tuple[float, float]:
    """Value-weighted one-day move of the portfolio, in base currency.

    For a position quoted at ``P`` with a day change of ``d`` (decimal) the
    previous close is ``P / (1 + d)``, so::

        previous_value = current_value / (1 + d)
        day_change     = current_value - previous_value

    Summing over the positions that HAVE a previous close (the others are
    excluded, never treated as flat) gives::

        day_change_pct = sum(day_change) / sum(previous_value) * 100

    Returns ``(nan, nan)`` when no position reports a previous close.
    """
    if "day_change_pct" not in positions.columns:
        return _nan(), _nan()

    pct = pd.to_numeric(positions["day_change_pct"], errors="coerce")
    d = pct.to_numpy(dtype=float) / 100.0
    value = pd.to_numeric(positions["current_value"], errors="coerce").to_numpy(float)
    usable = np.isfinite(d) & np.isfinite(value) & (d > -1.0)
    if not usable.any():
        return _nan(), _nan()

    previous = value[usable] / (1.0 + d[usable])
    change = float(np.sum(value[usable] - previous))
    base_value = float(np.sum(previous))
    return change, (change / base_value * 100.0) if base_value > 0 else _nan()


def summarise(positions: pd.DataFrame | None) -> dict:
    """Portfolio-level totals, concentration and best/worst performers.

    Keys
    ----
    ``n_holdings, total_investment, current_value, total_pnl,
    total_return_pct, currencies, best, worst, largest_holding`` (unchanged
    from v1) plus ``day_change`` and ``day_change_pct`` (value-weighted;
    positions without a previous close are excluded, not zeroed), ``hhi``
    (``sum(w^2)``), ``effective_holdings`` (``1 / HHI``), ``n_gainers``,
    ``n_losers``, ``top5_weight_pct``, ``sectors`` (sector -> weight %) and
    ``base_currency`` when the positions frame records one.

    All money figures are in the portfolio's base currency. An empty or
    missing frame returns ``{"n_holdings": 0}`` -- the v1 behaviour -- so
    callers must read the rest with ``.get``.
    """
    if positions is None or len(positions) == 0:
        return {"n_holdings": 0}

    total_investment = float(pd.to_numeric(positions["investment"], errors="coerce").sum())
    current_value = float(pd.to_numeric(positions["current_value"], errors="coerce").sum())
    total_pnl = current_value - total_investment

    summary: dict = {
        "n_holdings": int(len(positions)),
        "total_investment": total_investment,
        "current_value": current_value,
        "total_pnl": total_pnl,
        "total_return_pct": (total_pnl / total_investment * 100.0)
        if total_investment > 0
        else _nan(),
        "currencies": sorted(positions["currency"].dropna().unique().tolist())
        if "currency" in positions.columns
        else [],
    }

    # POSITIONAL lookup, not .loc[idxmax()]. idxmax returns an index *label*,
    # and on a non-unique index -- which any caller who concatenates or slices
    # positions without reset_index will hand us -- .loc[label] yields a
    # DataFrame rather than a Series, and float() on it raises. Every other
    # entry point in this module degrades rather than raises; this one now does
    # too.
    ranked = positions.dropna(subset=["pnl_pct"])
    if not ranked.empty:
        best = ranked.iloc[int(ranked["pnl_pct"].to_numpy().argmax())]
        worst = ranked.iloc[int(ranked["pnl_pct"].to_numpy().argmin())]
        summary["best"] = (str(best["ticker"]), float(best["pnl_pct"]))
        summary["worst"] = (str(worst["ticker"]), float(worst["pnl_pct"]))

    if "weight_pct" in positions.columns and positions["weight_pct"].notna().any():
        weights = positions["weight_pct"].to_numpy(dtype="float64")
        top = positions.iloc[int(np.nanargmax(weights))]
        summary["largest_holding"] = (str(top["ticker"]), float(top["weight_pct"]))

    day_change, day_change_pct = _day_change_totals(positions)
    summary["day_change"] = day_change
    summary["day_change_pct"] = day_change_pct

    weights = _weight_fractions(positions)
    summary["hhi"] = float(np.sum(weights**2)) if weights.size else _nan()
    summary["effective_holdings"] = effective_holdings(positions)

    pnl = pd.to_numeric(positions["pnl"], errors="coerce")
    summary["n_gainers"] = int((pnl > 0).sum())
    summary["n_losers"] = int((pnl < 0).sum())

    if "weight_pct" in positions.columns:
        weight_pct = pd.to_numeric(positions["weight_pct"], errors="coerce").dropna()
        summary["top5_weight_pct"] = (
            float(weight_pct.nlargest(5).sum()) if not weight_pct.empty else _nan()
        )
    else:
        summary["top5_weight_pct"] = _nan()

    if "sector" in positions.columns and "weight_pct" in positions.columns:
        by_sector = (
            pd.DataFrame(
                {
                    "sector": positions["sector"].map(lambda v: _clean_text(v) or _UNCLASSIFIED),
                    "weight": pd.to_numeric(positions["weight_pct"], errors="coerce"),
                }
            )
            .groupby("sector")["weight"]
            .sum()
            .sort_values(ascending=False)
        )
        summary["sectors"] = {str(k): float(v) for k, v in by_sector.items()}
    else:
        summary["sectors"] = {}

    base = positions.attrs.get("base_currency")
    if base is None:
        base = getattr(positions, "_base_currency", None)
    base = _clean_text(base)
    if base is not None:
        summary["base_currency"] = base

    return summary


def allocation_by(positions: pd.DataFrame | None, key: str) -> pd.DataFrame:
    """Aggregate value and weight by ``sector``/``industry``/``currency``/``ticker``.

    Returns ``[key, current_value, weight_pct, n]`` sorted by value descending,
    where ``weight_pct`` is the bucket's share of total current value -- so the
    column sums to 100 whenever the portfolio has a positive value. Missing
    labels are bucketed as ``"Unclassified"``. An unknown ``key``, a missing
    column or an empty portfolio yields an empty frame rather than raising.
    """
    columns = [key, "current_value", "weight_pct", "n"]

    if key not in ALLOCATION_KEYS:
        return pd.DataFrame(columns=["key", "current_value", "weight_pct", "n"])

    empty = pd.DataFrame(columns=columns)
    if positions is None or len(positions) == 0:
        return empty
    if key not in positions.columns or "current_value" not in positions.columns:
        return empty

    frame = pd.DataFrame(
        {
            key: positions[key].map(lambda v: _clean_text(v) or _UNCLASSIFIED),
            "current_value": pd.to_numeric(positions["current_value"], errors="coerce"),
        }
    )
    grouped = frame.groupby(key, as_index=False).agg(
        current_value=("current_value", "sum"), n=("current_value", "size")
    )

    total = float(grouped["current_value"].sum())
    grouped["weight_pct"] = np.where(total > 0, grouped["current_value"] / total * 100.0, np.nan)
    grouped["n"] = grouped["n"].astype(int)

    return grouped[columns].sort_values("current_value", ascending=False).reset_index(drop=True)


def effective_holdings(positions: pd.DataFrame | None) -> float:
    """Effective number of holdings, ``1 / sum(w_i^2)`` with w as fractions.

    Also known as the effective number of bets: an equal-weight portfolio of N
    names scores exactly N, while one name at 90% of the book scores close to
    1 however many tickers are listed. Weights are renormalised internally so
    that a subset of the positions frame still gives a coherent answer.
    Returns NaN for an empty frame or when no finite weight is available.
    """
    weights = _weight_fractions(positions)
    if weights.size == 0:
        return _nan()
    hhi = float(np.sum(weights**2))
    if not np.isfinite(hhi) or hhi <= 0:
        return _nan()
    return float(1.0 / hhi)
