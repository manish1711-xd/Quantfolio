"""Allocation and exposure charts: where the money sits, and how it is doing.

Six views, each returning a ready-to-render ``go.Figure``:

* :func:`allocation_donut`      share of market value by ticker/sector/industry/currency
* :func:`allocation_treemap`    sector -> ticker map; area = value, colour = unrealised return
* :func:`allocation_sunburst`   the same hierarchy read radially, coloured by sector
* :func:`weight_bar`            ranked weights against the equal-weight line
* :func:`weight_vs_target_bar`  current versus target weights ahead of a rebalance
* :func:`sector_bar`            value per group, from ``analytics.positions.allocation_by``

Conventions this module commits to
----------------------------------
* **Area means base-currency market value.** ``current_value`` is assumed to be
  already FX-converted into the reporting currency by
  ``analytics.positions.build_positions``; nothing here re-converts anything.
* **A holding with no usable market value has no area**, so it cannot be drawn.
  Those rows are excluded and the count is stated in the figure caption rather
  than silently dropped or coerced to zero.
* **Weights** use the supplied ``weight_pct`` where it is finite and fall back to
  a recomputation over the rows actually drawn, so the shares always sum to 100.
* **Colour carries meaning.** Unrealised return uses the diverging ramp centred
  on zero (red = loss, blue = gain); structural groupings use the colourblind-safe
  qualitative ramp. Gain/loss hues are never used merely to tell two holdings
  apart. Sector colours are assigned in descending-value order in every chart
  here, so a sector keeps the same colour in the sunburst and the sector bar.
* **Nothing raises.** Degenerate input returns ``theme.empty_figure`` with the
  actual reason, so the dashboard never shows a blank rectangle.

No Streamlit import: these builders work in a notebook or the HTML report
writer exactly as they do in the app.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.charts import theme

__all__ = [
    "allocation_donut",
    "allocation_treemap",
    "allocation_sunburst",
    "weight_bar",
    "weight_vs_target_bar",
    "sector_bar",
]

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
_VALUE = "current_value"
_MISSING = "n/a"
_ROOT_LABEL = "Portfolio"
_UNCLASSIFIED = "Unclassified"
_ALLOC_META = ("current_value", "weight_pct", "n")

#: Strings that mean "the provider gave us nothing" once stringified.
_BLANKS = frozenset({"", "nan", "none", "null", "<na>", "nat"})

#: Minimum share a wedge needs before its label is drawn inside it.
_LABEL_SHARE = 0.045

#: Half-width of the portfolio map's unrealised-return colour scale, in percent.
#: The ramp runs a full -30% .. +30% and SATURATES beyond it, so a tile's colour
#: carries the same meaning on every portfolio and in every session rather than
#: being rescaled by whatever the current holdings happen to have done. Public
#: because it is the legend's promise, and a test pins it.
RETURN_COLOR_LIMIT_PCT: float = 30.0


# ---------------------------------------------------------------------------
# safety net
# ---------------------------------------------------------------------------
_F = TypeVar("_F", bound=Callable[..., go.Figure])


def _never_raises(fn: _F) -> _F:
    """Enforce the chart contract: a builder always yields a Figure.

    The guard clauses below cover every degenerate shape we can name. This is
    the backstop for the ones we cannot, because a dashboard that raises inside
    a render loop takes the whole page down with it.
    """

    @functools.wraps(fn)
    def _wrapped(*args: Any, **kwargs: Any) -> go.Figure:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - defensive  # pragma: no cover
            return theme.empty_figure(
                f"This chart could not be built from the data supplied ({type(exc).__name__})."
            )

    return cast(_F, _wrapped)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _is_frame(obj: Any) -> bool:
    return isinstance(obj, pd.DataFrame)


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    """Coerce a column to float64, or an all-NaN column when it is absent."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").astype("float64")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _labels(df: pd.DataFrame, col: str, fallback: str) -> pd.Series:
    """Stringify a label column, mapping blanks and NaN to ``fallback``."""
    if col not in df.columns:
        return pd.Series(fallback, index=df.index, dtype="object")
    raw = df[col].where(df[col].notna(), fallback).astype(str).str.strip()
    return raw.mask(raw.str.lower().isin(_BLANKS), fallback).astype(str)


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _floats(values: Any) -> list[float]:
    """Plain Python floats - numpy scalars are a classic ``to_json`` trap."""
    return [float(v) for v in np.asarray(values, dtype="float64")]


def _strings(values: Any) -> list[str]:
    return [str(v) for v in values]


def _money(value: Any, symbol: str = "") -> str:
    """Money as a hover-ready string; ``n/a`` when the number is undefined."""
    if not _finite(value):
        return _MISSING
    v = float(value)
    return f"{symbol}{v:,.2f}" if abs(v) < 100 else f"{symbol}{v:,.0f}"


def _pct(value: Any, decimals: int = 2, signed: bool = True) -> str:
    """Percent as a hover-ready string; ``n/a`` when the number is undefined."""
    if not _finite(value):
        return _MISSING
    spec = ("+" if signed else "") + f".{decimals}f"
    return format(float(value), spec) + "%"


def _compact(value: Any) -> str:
    """SI-style compaction for in-figure labels (1.24M, 45.3K)."""
    if not _finite(value):
        return _MISSING
    v = float(value)
    sign = "-" if v < 0 else ""
    size = abs(v)
    for cut, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if size >= cut:
            scaled = size / cut
            fmt = ",.2f" if scaled < 10 else ",.1f"
            return f"{sign}{format(scaled, fmt)}{suffix}"
    return f"{sign}{size:,.0f}"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _tint(color: str, alpha: float) -> str:
    """Same hue, lower opacity - used for child rings under a parent colour."""
    raw = str(color).lstrip("#")
    if len(raw) != 6:
        return str(color)
    try:
        r, g, b = (int(raw[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return str(color)
    return f"rgba({r}, {g}, {b}, {alpha:.2f})"


def _category_colors(n: int) -> list[str]:
    ramp = theme.CATEGORICAL
    return [ramp[i % len(ramp)] for i in range(max(int(n), 0))]


def _bar_height(
    n_rows: int, per_row: int = 26, base: int = 150, lo: int = 300, hi: int = 520
) -> int:
    """Row-count driven height so bar labels never collide."""
    return int(min(hi, max(lo, base + per_row * max(int(n_rows), 1))))


def _sum_or_nan(values: Any) -> float:
    """Sum that refuses to pretend a missing input was zero."""
    arr = np.asarray(values, dtype="float64")
    if arr.size == 0 or not bool(np.isfinite(arr).all()):
        return float("nan")
    return float(arr.sum())


def _weighted_return(frame: pd.DataFrame) -> float:
    """Aggregate unrealised return (%) for a set of holdings.

    Preferred definition is ``sum(pnl) / sum(investment) * 100`` - a true
    cost-weighted return. When the cost basis is unavailable it falls back to a
    current-value-weighted mean of ``pnl_pct``, which is an approximation.
    Returns NaN when any constituent is missing, rather than a partial total.
    """
    pnl = frame["_pnl"].to_numpy(dtype="float64")
    inv = frame["_investment"].to_numpy(dtype="float64")
    if pnl.size and bool(np.isfinite(pnl).all()) and bool(np.isfinite(inv).all()) and inv.sum() > 0:
        return float(pnl.sum() / inv.sum() * 100.0)
    ret = frame["_pnl_pct"].to_numpy(dtype="float64")
    val = frame[_VALUE].to_numpy(dtype="float64")
    if ret.size and bool(np.isfinite(ret).all()) and val.sum() > 0:
        return float((ret * val).sum() / val.sum())
    return float("nan")


# ---------------------------------------------------------------------------
# input preparation
# ---------------------------------------------------------------------------
def _clean_positions(positions: Any) -> tuple[pd.DataFrame | None, int, str]:
    """Validate a positions frame and derive the private ``_`` columns.

    Returns ``(frame, n_excluded, reason)``. ``frame`` is None when no honest
    chart can be drawn and ``reason`` says precisely why. The input is never
    mutated - the first thing this does is copy.
    """
    if not _is_frame(positions):
        return None, 0, "No positions table was supplied to this chart."
    if positions.empty:
        return None, 0, "The portfolio is empty - add holdings to see its allocation."
    df = positions.copy()
    if _VALUE not in df.columns:
        return None, 0, "The positions table has no 'current_value' column to size by."

    value = _numeric(df, _VALUE)
    keep = np.isfinite(value) & (value > 0)
    excluded = int((~keep).sum())
    df = df.loc[keep].copy()
    if df.empty:
        return (
            None,
            excluded,
            "Every holding has a missing, zero or negative market value.",
        )

    df[_VALUE] = value.loc[keep].astype("float64")
    df["_ticker"] = _labels(df, "ticker", "?")
    df["_name"] = _labels(df, "name", "")
    blank_name = df["_name"].str.len() == 0
    df.loc[blank_name, "_name"] = df.loc[blank_name, "_ticker"]
    df["_sector"] = _labels(df, "sector", _UNCLASSIFIED)
    df["_pnl"] = _numeric(df, "pnl")
    df["_pnl_pct"] = _numeric(df, "pnl_pct")
    df["_investment"] = _numeric(df, "investment")

    total = float(df[_VALUE].sum())
    recomputed = df[_VALUE] / total * 100.0
    supplied = _numeric(df, "weight_pct")
    df["_weight"] = supplied.where(np.isfinite(supplied), recomputed).astype("float64")

    df = df.sort_values(_VALUE, ascending=False).reset_index(drop=True)
    return df, excluded, ""


def _clean_drift(drift: Any) -> tuple[pd.DataFrame | None, str]:
    """Validate a rebalance drift frame and normalise its numeric columns."""
    if not _is_frame(drift):
        return None, "No rebalancing table was supplied to this chart."
    if drift.empty:
        return None, "The rebalancing table is empty - set target weights first."
    df = drift.copy()
    if "current_weight_pct" not in df.columns and "target_weight_pct" not in df.columns:
        return None, ("The drift table needs 'current_weight_pct' and 'target_weight_pct'.")

    cur = _numeric(df, "current_weight_pct")
    tgt = _numeric(df, "target_weight_pct")
    keep = np.isfinite(cur) | np.isfinite(tgt)
    df = df.loc[keep].copy()
    if df.empty:
        return None, "Neither current nor target weights are available for any name."

    cur = cur.loc[keep].fillna(0.0)
    tgt = tgt.loc[keep].fillna(0.0)
    if float(cur.abs().sum() + tgt.abs().sum()) <= 0:
        return None, "Current and target weights are all zero - nothing to compare."

    df["_ticker"] = _labels(df, "ticker", "?")
    df["_current"] = cur.astype("float64")
    df["_target"] = tgt.astype("float64")
    supplied_drift = _numeric(df, "drift_pct")
    df["_drift"] = supplied_drift.where(
        np.isfinite(supplied_drift), df["_current"] - df["_target"]
    ).astype("float64")
    df["_gap"] = _numeric(df, "value_gap")
    if "in_band" in df.columns:
        df["_in_band"] = df["in_band"].fillna(True).astype(bool)
        df["_has_band"] = True
    else:
        df["_in_band"] = True
        df["_has_band"] = False

    df = df.sort_values("_drift", key=lambda s: s.abs(), ascending=False)
    return df.reset_index(drop=True), ""


def _pool_tail(
    frame: pd.DataFrame, top_n: int | None, value_col: str
) -> tuple[pd.DataFrame, int, float]:
    """Split a value-sorted frame into a head and a pooled tail.

    Returns ``(head, n_pooled, pooled_value)``. Pooling keeps a long tail of
    sub-1% names from turning the axis labels into mush.
    """
    if not isinstance(top_n, (int, np.integer)) or top_n <= 0 or top_n >= len(frame):
        return frame, 0, 0.0
    head = frame.iloc[: int(top_n)].copy()
    tail = frame.iloc[int(top_n) :]
    return head, int(len(tail)), float(tail[value_col].sum())


# ---------------------------------------------------------------------------
# hierarchy assembly (shared by treemap and sunburst)
# ---------------------------------------------------------------------------
def _hierarchy(df: pd.DataFrame, symbol: str) -> dict[str, Any]:
    """Build the sector -> ticker node arrays both hierarchy charts need.

    The sector level is *collapsed* when the portfolio resolves to a single
    sector - the common case being a portfolio the provider classified nothing
    in, where every row reads "Unclassified". A one-box wrapper around every
    holding is pure noise, so in that case holdings hang straight off the root.
    """
    sector_totals = df.groupby("_sector")[_VALUE].sum().sort_values(ascending=False)
    sectors = [str(s) for s in sector_totals.index]
    collapsed = len(sectors) <= 1
    total = float(df[_VALUE].sum())
    sector_color = dict(zip(sectors, _category_colors(len(sectors))))

    ids: list[str] = []
    labels: list[str] = []
    parents: list[str] = []
    values: list[float] = []
    returns: list[float] = []
    text: list[str] = []
    custom: list[list[Any]] = []
    hues: list[str] = []
    depth: list[int] = []

    root_return = _weighted_return(df)
    ids.append("root")
    labels.append(_ROOT_LABEL)
    parents.append("")
    values.append(total)
    returns.append(root_return)
    text.append(f"<b>{_ROOT_LABEL}</b>")
    hues.append(_tint(theme.palette()["accent"], 0.35))
    depth.append(0)
    root_note = (
        _plural(len(df), "holding")
        if collapsed
        else f"{_plural(len(df), 'holding')} in {_plural(len(sectors), 'sector')}"
    )
    custom.append(
        [
            100.0,
            _money(_sum_or_nan(df["_pnl"]), symbol),
            _pct(root_return),
            root_note,
            "Whole portfolio",
        ]
    )

    for i, sector in enumerate(sectors):
        sub = df.loc[df["_sector"] == sector]
        sector_value = float(sub[_VALUE].sum())
        sector_id = f"S{i}"
        if not collapsed:
            sector_return = _weighted_return(sub)
            ids.append(sector_id)
            labels.append(sector)
            parents.append("root")
            values.append(sector_value)
            returns.append(sector_return)
            text.append(f"<b>{sector}</b>")
            hues.append(sector_color[sector])
            depth.append(1)
            custom.append(
                [
                    sector_value / total * 100.0,
                    _money(_sum_or_nan(sub["_pnl"]), symbol),
                    _pct(sector_return),
                    _plural(len(sub), "holding"),
                    "Sector total",
                ]
            )
        for _, row in sub.iterrows():
            ticker = str(row["_ticker"])
            leaf_return = float(row["_pnl_pct"])
            ids.append(f"L{len(ids)}")
            labels.append(ticker)
            parents.append("root" if collapsed else sector_id)
            values.append(float(row[_VALUE]))
            returns.append(leaf_return)
            text.append(f"<b>{ticker}</b><br>{_pct(leaf_return, decimals=1)}")
            hues.append(_tint(sector_color[sector], 0.62))
            depth.append(1 if collapsed else 2)
            custom.append(
                [
                    float(row["_weight"]),
                    _money(row["_pnl"], symbol),
                    _pct(leaf_return),
                    str(row["_name"]),
                    f"Sector: {sector}",
                ]
            )

    return {
        "ids": ids,
        "labels": labels,
        "parents": parents,
        "values": _floats(values),
        "returns": _floats(returns),
        "text": text,
        "custom": custom,
        "hues": hues,
        "depth": depth,
        "collapsed": collapsed,
        "sectors": sectors,
        "sector_color": sector_color,
        "n_holdings": int(len(df)),
        "total": total,
    }


def _hierarchy_hover(symbol: str) -> str:
    """Hand-written hover shared by the treemap and the sunburst."""
    return (
        "<b>%{label}</b><br>"
        "%{customdata[4]}<br>"
        "Value: " + symbol + "%{value:,.0f}"
        "  (%{customdata[0]:.2f}% of portfolio)<br>"
        "Unrealised P/L: %{customdata[1]}  (%{customdata[2]})<br>"
        "%{customdata[3]}"
        "<extra></extra>"
    )


def _hierarchy_subtitle(node: dict[str, Any], extra: str) -> str:
    """Name the grouping actually drawn, including the collapsed fallback."""
    if node["collapsed"]:
        only = node["sectors"][0] if node["sectors"] else _UNCLASSIFIED
        if only == _UNCLASSIFIED:
            head = "No sector data from the provider - holdings shown flat"
        else:
            head = f"Every holding sits in {only} - shown flat"
    else:
        head = (
            f"{_plural(len(node['sectors']), 'sector')} / {_plural(node['n_holdings'], 'holding')}"
        )
    return f"{head}; {extra}"


# ---------------------------------------------------------------------------
# donut
# ---------------------------------------------------------------------------
@_never_raises
def allocation_donut(
    positions: pd.DataFrame,
    by: str = "ticker",
    *,
    currency_symbol: str = "",
    max_slices: int = 12,
    height: int | None = None,
) -> go.Figure:
    """Share of portfolio market value as a donut, grouped by ``by``.

    Shows how today's market value splits across ``"ticker"``, ``"sector"``,
    ``"industry"``, ``"currency"`` or any other column of the positions frame.
    Assumes ``current_value`` is already in the reporting currency. Slices
    beyond ``max_slices`` are pooled into a muted "Other" wedge so the labels
    stay legible, and only wedges worth more than ~4.5% of the portfolio are
    labelled in place - the rest are read from the legend or the hover. The
    centre states the total the ring represents.
    """
    df, excluded, reason = _clean_positions(positions)
    if df is None:
        return theme.empty_figure(reason)

    key = str(by or "ticker").strip()
    lowered = key.lower()
    if lowered == "ticker":
        keys = df["_ticker"]
    elif lowered == "sector":
        keys = df["_sector"]
    elif key in df.columns:
        keys = _labels(df, key, _UNCLASSIFIED)
    else:
        return theme.empty_figure(
            f"The positions table has no '{key}' column to group the allocation by."
        )

    grouped = (
        df.assign(_key=keys)
        .groupby("_key", sort=False, dropna=False)
        .agg(value=(_VALUE, "sum"), n=("_ticker", "size"))
        .reset_index()
        .sort_values("value", ascending=False)
        .reset_index(drop=True)
    )

    pooled = 0
    if isinstance(max_slices, (int, np.integer)) and 1 < max_slices < len(grouped):
        cut = int(max_slices) - 1
        head, pooled, pooled_value = _pool_tail(grouped, cut, "value")
        pooled_n = int(grouped.iloc[cut:]["n"].sum())
        other = pd.DataFrame(
            {
                "_key": [f"Other ({_plural(pooled, 'name')})"],
                "value": [pooled_value],
                "n": [pooled_n],
            }
        )
        grouped = pd.concat([head, other], ignore_index=True)

    total = float(grouped["value"].sum())
    shares = grouped["value"] / total
    p = theme.palette()
    colors = _category_colors(len(grouped))
    if pooled:
        colors[-1] = p["muted"]

    label_text = [
        f"{name}<br>{share * 100:.1f}%" if share >= _LABEL_SHARE else ""
        for name, share in zip(grouped["_key"], shares)
    ]

    fig = go.Figure(
        go.Pie(
            labels=_strings(grouped["_key"]),
            values=_floats(grouped["value"]),
            customdata=[[int(v)] for v in grouped["n"]],
            hole=0.58,
            sort=False,
            direction="clockwise",
            marker={"colors": colors, "line": {"color": p["grid"], "width": 1}},
            text=label_text,
            textinfo="text",
            textposition="inside",
            insidetextfont={"size": 11},
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Value: " + currency_symbol + "%{value:,.0f}<br>"
                "Share: %{percent:.1%}<br>"
                "Holdings: %{customdata[0]:,d}"
                "<extra></extra>"
            ),
        )
    )
    fig.add_annotation(
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="center",
        text=(
            f"<span style='font-size:20px;color:{p['text']}'>"
            f"{currency_symbol}{_compact(total)}</span><br>"
            f"<span style='font-size:10.5px;color:{p['subtext']}'>Total value</span>"
        ),
    )

    word = {"ticker": "holding", "sector": "sector"}.get(lowered, lowered.replace("_", " "))
    subtitle = f"Current market value, {_plural(len(df), 'holding')}"
    if pooled:
        subtitle += f"; smallest {pooled} pooled as Other"
    theme.apply(
        fig,
        f"Allocation by {word}",
        subtitle=subtitle,
        height=height or 380,
        legend=True,
        uniformtext={"minsize": 10, "mode": "hide"},
    )
    if excluded:
        theme.caption(fig, f"{_plural(excluded, 'holding')} excluded: no usable market value.")
    return fig


# ---------------------------------------------------------------------------
# treemap (the showpiece)
# ---------------------------------------------------------------------------
@_never_raises
def allocation_treemap(
    positions: pd.DataFrame,
    *,
    currency_symbol: str = "",
    height: int | None = None,
) -> go.Figure:
    """Portfolio map: tile area = market value, tile colour = unrealised return.

    Two dimensions in one glance - *how much* is in a name (area) and *how it is
    doing* (colour) - grouped sector -> ticker so concentration by theme is as
    obvious as concentration by name. Colour uses the diverging ramp centred
    exactly on zero, so a tile is only red or blue to the extent the position is
    genuinely down or up.

    Assumptions: ``current_value`` is in the reporting currency; the colour
    scale is symmetric and FIXED at +/-30% (:data:`RETURN_COLOR_LIMIT_PCT`), so
    a tile's colour means the same thing on every portfolio and in every
    session and anything beyond that saturates rather than rescaling the ramp;
    sector aggregates are cost-weighted (``sum(pnl) / sum(investment)``). Holdings whose P/L is unknown are drawn at
    the neutral midpoint and labelled ``n/a`` rather than shown as flat. When the
    portfolio has no sector data at all, the sector level is dropped instead of
    wrapping everything in one "Unclassified" box.
    """
    df, excluded, reason = _clean_positions(positions)
    if df is None:
        return theme.empty_figure(reason)

    node = _hierarchy(df, currency_symbol)
    p = theme.palette()
    by_return = bool(np.isfinite(df["_pnl_pct"].to_numpy(dtype="float64")).any())
    limit = RETURN_COLOR_LIMIT_PCT

    if by_return:
        marker: dict[str, Any] = {
            # Non-finite returns sit at the neutral midpoint; the hover says n/a.
            "colors": [v if np.isfinite(v) else 0.0 for v in node["returns"]],
            "colorscale": theme.DIVERGING,
            "cmin": -limit,
            "cmid": 0.0,
            "cmax": limit,
            "showscale": True,
            "colorbar": {
                "title": {
                    "text": "Unrealised<br>return",
                    "side": "right",
                    "font": {"size": 10.5, "color": p["subtext"]},
                },
                "ticksuffix": "%",
                "tickformat": "+.0f",
                # The bound is fixed, so the ticks can be too: five
                # labels at -30/-15/0/+15/+30 rather than whatever
                # Plotly picks for an axis that used to move.
                "dtick": RETURN_COLOR_LIMIT_PCT / 2.0,
                "thickness": 12,
                "len": 0.68,
                "outlinewidth": 0,
                "tickfont": {"size": 10, "color": p["subtext"]},
            },
            "line": {"color": p["grid"], "width": 1},
        }
        colour_note = (
            f"colour = unrealised return on a fixed +/-{limit:.0f}% scale (beyond that, saturated)"
        )
    else:
        marker = {"colors": node["hues"], "line": {"color": p["grid"], "width": 1}}
        colour_note = "colour = sector (no P/L data available)"

    fig = go.Figure(
        go.Treemap(
            ids=node["ids"],
            labels=node["labels"],
            parents=node["parents"],
            values=node["values"],
            branchvalues="total",
            customdata=node["custom"],
            text=node["text"],
            texttemplate="%{text}",
            textposition="middle center",
            marker=marker,
            tiling={"pad": 2},
            pathbar={"visible": True, "thickness": 18},
            sort=True,
            insidetextfont={"size": 12},
            hovertemplate=_hierarchy_hover(currency_symbol),
        )
    )
    theme.apply(
        fig,
        "Portfolio map: size by value, colour by return",
        subtitle=_hierarchy_subtitle(node, f"area = market value, {colour_note}"),
        height=height or 460,
        legend=False,
        uniformtext={"minsize": 9, "mode": "hide"},
    )
    footnotes = [
        f"Colour saturates at +/-{RETURN_COLOR_LIMIT_PCT:.0f}% so maps stay comparable; "
        "the hover carries the exact return. Sector returns are cost-weighted "
        "(sum of P/L over sum of cost); holdings with no P/L are drawn neutral "
        "and labelled n/a."
    ]
    if excluded:
        footnotes.append(f"{_plural(excluded, 'holding')} excluded: no market value.")
    theme.caption(fig, " ".join(footnotes))
    return fig


# ---------------------------------------------------------------------------
# sunburst
# ---------------------------------------------------------------------------
@_never_raises
def allocation_sunburst(
    positions: pd.DataFrame,
    *,
    currency_symbol: str = "",
    height: int | None = None,
) -> go.Figure:
    """Sector -> holding hierarchy read radially, each ring summing to the whole.

    The radial cousin of :func:`allocation_treemap`: the same hierarchy and the
    same market-value areas, but coloured by *structure* rather than performance
    - one qualitative colour per sector, with holdings drawn as a lighter tint of
    their parent. That deliberate split means the two charts answer different
    questions instead of competing: the treemap for "how is it doing", the
    sunburst for "how is it built". Click a sector to zoom into it.

    Assumes ``current_value`` is in the reporting currency; the sector ring is
    dropped when the provider classified nothing.
    """
    df, excluded, reason = _clean_positions(positions)
    if df is None:
        return theme.empty_figure(reason)

    node = _hierarchy(df, currency_symbol)
    p = theme.palette()
    ring_text = [
        label if d == 0 else f"{label}<br>{fields[0]:.1f}%"
        for label, d, fields in zip(node["labels"], node["depth"], node["custom"])
    ]

    fig = go.Figure(
        go.Sunburst(
            ids=node["ids"],
            labels=node["labels"],
            parents=node["parents"],
            values=node["values"],
            branchvalues="total",
            customdata=node["custom"],
            text=ring_text,
            texttemplate="%{text}",
            insidetextorientation="radial",
            marker={"colors": node["hues"], "line": {"color": p["grid"], "width": 1}},
            sort=True,
            rotation=90,
            hovertemplate=_hierarchy_hover(currency_symbol),
        )
    )
    theme.apply(
        fig,
        "Allocation hierarchy",
        subtitle=_hierarchy_subtitle(node, "arc width = share of market value"),
        height=height or 440,
        legend=False,
        uniformtext={"minsize": 9, "mode": "hide"},
    )
    if excluded:
        theme.caption(fig, f"{_plural(excluded, 'holding')} excluded: no usable market value.")
    return fig


# ---------------------------------------------------------------------------
# weight bars
# ---------------------------------------------------------------------------
@_never_raises
def weight_bar(
    positions: pd.DataFrame,
    top_n: int | None = None,
    *,
    currency_symbol: str = "",
    height: int | None = None,
) -> go.Figure:
    """Ranked portfolio weights, largest first, against the equal-weight line.

    Horizontal bars because tickers and fund names are long; ranked by weight
    because alphabetical order hides concentration. The dotted reference line is
    ``100 / n`` - the weight every name would carry if the portfolio were equally
    weighted - which turns "is this concentrated?" into a one-glance read.

    Weights are shares of *current* market value, so they move with price and not
    only with what was originally invested. Passing ``top_n`` pools the remaining
    names into a single muted bar rather than dropping them.
    """
    df, excluded, reason = _clean_positions(positions)
    if df is None:
        return theme.empty_figure(reason)

    total_names = len(df)
    shown, pooled, pooled_value = _pool_tail(df, top_n, _VALUE)
    p = theme.palette()

    labels = _strings(shown["_ticker"])
    weights = _floats(shown["_weight"])
    custom: list[list[Any]] = [
        [
            str(row["_name"]),
            float(row[_VALUE]),
            _money(row["_pnl"], currency_symbol),
            _pct(row["_pnl_pct"]),
        ]
        for _, row in shown.iterrows()
    ]
    colors = [p["accent"]] * len(labels)

    if pooled:
        tail = df.iloc[len(shown) :]
        labels.append(f"Other ({_plural(pooled, 'name')})")
        weights.append(float(tail["_weight"].sum()))
        custom.append(
            [
                f"{_plural(pooled, 'smaller holding')} pooled",
                pooled_value,
                _money(_sum_or_nan(tail["_pnl"]), currency_symbol),
                _pct(_weighted_return(tail)),
            ]
        )
        colors.append(p["muted"])

    # Plotly draws the first category at the bottom: reverse for largest-on-top.
    labels, weights, custom, colors = (
        labels[::-1],
        weights[::-1],
        custom[::-1],
        colors[::-1],
    )

    fig = go.Figure(
        go.Bar(
            x=weights,
            y=labels,
            orientation="h",
            marker={"color": colors, "line": {"color": p["grid"], "width": 0.5}},
            customdata=custom,
            texttemplate="%{x:.1f}%",
            textposition="outside",
            textfont={"size": 10, "color": p["subtext"]},
            cliponaxis=False,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "%{customdata[0]}<br>"
                "Weight: %{x:.2f}% of portfolio<br>"
                "Value: " + currency_symbol + "%{customdata[1]:,.0f}<br>"
                "Unrealised P/L: %{customdata[2]}  (%{customdata[3]})"
                "<extra></extra>"
            ),
        )
    )

    widest = max(weights) if weights else 1.0
    fig.update_xaxes(title="Weight (% of portfolio market value)", range=[0, widest * 1.18])
    fig.update_yaxes(title="Holding")
    theme.pct_axis(fig, axis="x")

    subtitle = f"Share of {currency_symbol}{_compact(df[_VALUE].sum())} market value"
    if pooled:
        subtitle += f"; top {len(shown)} of {total_names} shown"
    theme.apply(
        fig,
        "Position weights",
        subtitle=subtitle,
        height=height or _bar_height(len(labels)),
        legend=False,
        bargap=0.28,
    )
    if total_names > 1:
        equal = 100.0 / total_names
        theme.add_reference_line(
            fig,
            equal,
            label=f"Equal weight {equal:.1f}%",
            axis="x",
            color=p["benchmark"],
        )
    if excluded:
        theme.caption(fig, f"{_plural(excluded, 'holding')} excluded: no usable market value.")
    return fig


@_never_raises
def weight_vs_target_bar(
    drift: pd.DataFrame,
    *,
    band_pct: float | None = None,
    top_n: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Current weight against target weight, ranked by how far each name drifted.

    Reads as a rebalancing worksheet: the solid bar is where the money is now,
    the outlined bar behind it is where the policy says it should be, and the
    label at the end is the gap in percentage points. Names outside the
    rebalancing band are drawn in amber because those are the actionable ones.

    Over- and underweight are deliberately *not* coloured with the gain/loss
    hues: being 2pp heavy in a name is neither a profit nor a loss, and those two
    colours are reserved throughout the app for money made and lost.

    Assumes ``drift`` is the frame from ``analytics.rebalance.drift_table``, with
    targets already renormalised to 100%. Names in the target policy that are not
    currently held show a zero-length current bar.
    """
    df, reason = _clean_drift(drift)
    if df is None:
        return theme.empty_figure(reason)

    if isinstance(top_n, (int, np.integer)) and 0 < top_n < len(df):
        df = df.iloc[: int(top_n)].copy()

    p = theme.palette()
    # Reverse so the largest drift sits at the top of a horizontal bar chart.
    df = df.iloc[::-1].reset_index(drop=True)
    has_band = bool(df["_has_band"].iloc[0])
    out_of_band = int((~df["_in_band"]).sum()) if has_band else 0

    tickers = _strings(df["_ticker"])
    current = _floats(df["_current"])
    target = _floats(df["_target"])
    drift_vals = _floats(df["_drift"])
    bar_colors = [p["warning"] if (has_band and not ok) else p["accent"] for ok in df["_in_band"]]
    status = [
        ("Within band" if ok else "Outside rebalancing band") if has_band else "No band supplied"
        for ok in df["_in_band"]
    ]
    custom = [[t, d, _money(g), s] for t, d, g, s in zip(target, drift_vals, df["_gap"], status)]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=target,
            y=tickers,
            orientation="h",
            name="Target",
            # Outlined "ghost" bar: the policy weight, drawn behind the money.
            marker={
                "color": p["surface"],
                "line": {"color": p["benchmark"], "width": 1.4},
            },
            hovertemplate="<b>%{y}</b><br>Target weight: %{x:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=current,
            y=tickers,
            orientation="h",
            name="Current",
            marker={"color": bar_colors},
            customdata=custom,
            text=[f"{d:+.1f} pp" for d in drift_vals],
            textposition="outside",
            textfont={"size": 10, "color": p["subtext"]},
            cliponaxis=False,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Current: %{x:.2f}%<br>"
                "Target: %{customdata[0]:.2f}%<br>"
                "Drift: %{customdata[1]:+.2f} pp<br>"
                "Value gap: %{customdata[2]}<br>"
                "%{customdata[3]}"
                "<extra></extra>"
            ),
        )
    )

    widest = max(current + target + [1.0])
    fig.update_xaxes(title="Weight (% of portfolio value)", range=[0, widest * 1.22])
    fig.update_yaxes(title="Holding")
    theme.pct_axis(fig, axis="x")

    subtitle = "Ranked by absolute drift"
    if band_pct is not None and _finite(band_pct):
        subtitle += f"; rebalancing band +/-{float(band_pct):.1f} pp"
    theme.apply(
        fig,
        "Current weight vs target",
        subtitle=subtitle,
        # Grouped pairs need more vertical room per name than a single bar.
        height=height or _bar_height(len(tickers), per_row=34),
        legend=True,
        barmode="group",
        bargap=0.3,
        bargroupgap=0.05,
    )
    note = "Drift is current minus target, in percentage points."
    if out_of_band:
        note += f" Amber marks the {out_of_band} name(s) outside the band."
    theme.caption(fig, note)
    return fig


# ---------------------------------------------------------------------------
# grouped allocation bars
# ---------------------------------------------------------------------------
@_never_raises
def sector_bar(
    allocation: pd.DataFrame,
    *,
    currency_symbol: str = "",
    top_n: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Market value per group, from ``analytics.positions.allocation_by``.

    Takes the ``[<key>, current_value, weight_pct, n]`` frame that
    ``allocation_by`` emits - the key column is whichever one is not part of that
    metadata trio - and draws it as ranked horizontal bars: horizontal because
    "Consumer Discretionary" does not fit under a vertical bar, ranked because
    that is the order the question gets asked in.

    Bars are sized in reporting currency and labelled with the weight, so
    absolute and relative exposure are both readable. Group colours follow the
    same descending-value assignment as the sunburst, so a sector keeps its
    colour across the page.
    """
    if not _is_frame(allocation):
        return theme.empty_figure("No allocation table was supplied to this chart.")
    if allocation.empty:
        return theme.empty_figure("The allocation table is empty - nothing to group.")

    df = allocation.copy()
    key_cols = [c for c in df.columns if c not in _ALLOC_META]
    if not key_cols:
        return theme.empty_figure(
            "The allocation table has no category column (expected sector, "
            "industry, currency or ticker)."
        )
    key = str(key_cols[0])
    if _VALUE not in df.columns:
        return theme.empty_figure(
            "The allocation table has no 'current_value' column to size the bars by."
        )

    value = _numeric(df, _VALUE)
    keep = np.isfinite(value) & (value > 0)
    df = df.loc[keep].copy()
    if df.empty:
        return theme.empty_figure(
            f"Every {key} group has a missing, zero or negative market value."
        )

    df[_VALUE] = value.loc[keep].astype("float64")
    df["_label"] = _labels(df, key, _UNCLASSIFIED)
    total = float(df[_VALUE].sum())
    supplied = _numeric(df, "weight_pct")
    df["_weight"] = supplied.where(np.isfinite(supplied), df[_VALUE] / total * 100.0).astype(
        "float64"
    )
    counts = _numeric(df, "n")
    df["_n"] = counts.where(np.isfinite(counts), 1.0).astype("float64")
    df = df.sort_values(_VALUE, ascending=False).reset_index(drop=True)

    shown, pooled, pooled_value = _pool_tail(df, top_n, _VALUE)
    p = theme.palette()
    labels = _strings(shown["_label"])
    values = _floats(shown[_VALUE])
    custom: list[list[Any]] = [[float(w), int(n)] for w, n in zip(shown["_weight"], shown["_n"])]
    colors = _category_colors(len(labels))

    if pooled:
        tail = df.iloc[len(shown) :]
        labels.append(f"Other ({_plural(pooled, 'group')})")
        values.append(pooled_value)
        custom.append([float(tail["_weight"].sum()), int(tail["_n"].sum())])
        colors.append(p["muted"])

    labels, values, custom, colors = (
        labels[::-1],
        values[::-1],
        custom[::-1],
        colors[::-1],
    )

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={"color": colors, "line": {"color": p["grid"], "width": 0.5}},
            customdata=custom,
            text=[f"{fields[0]:.1f}%" for fields in custom],
            textposition="outside",
            textfont={"size": 10, "color": p["subtext"]},
            cliponaxis=False,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Value: " + currency_symbol + "%{x:,.0f}<br>"
                "Weight: %{customdata[0]:.2f}% of portfolio<br>"
                "Holdings: %{customdata[1]:,d}"
                "<extra></extra>"
            ),
        )
    )

    widest = max(values) if values else 1.0
    word = key.replace("_", " ")
    unit = f" ({currency_symbol})" if currency_symbol else ""
    fig.update_xaxes(title=f"Current market value{unit}", range=[0, widest * 1.18])
    fig.update_yaxes(title=word.title())
    theme.money_axis(fig, currency_symbol, axis="x")

    theme.apply(
        fig,
        f"Exposure by {word}",
        subtitle=(
            f"{_plural(len(df), 'group')} covering "
            f"{currency_symbol}{_compact(total)} of market value"
        ),
        height=height or _bar_height(len(labels)),
        legend=False,
        bargap=0.3,
    )
    if _UNCLASSIFIED in set(labels):
        theme.caption(
            fig,
            "Unclassified covers instruments the data provider reports no "
            f"{word} for - typically ETFs and index funds.",
        )
    return fig
