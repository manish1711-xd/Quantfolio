"""Price-alert configuration: the rules, the delivery settings, and the clock.

Mounted by the Watchlist page, which already owns the items these rules attach
to.

Three things this panel exists to make unambiguous, because each of them is a
way an alert feature can quietly lie to its user:

* **Direction.** A bare target cannot say whether you want to hear about the
  price rising to it or falling to it, and a rule with no direction cannot
  fire. So the direction is a required, visible choice, and a target without
  one is shown as *inactive* rather than as an armed alert that will never go
  off.
* **The clock is outside the app.** As with scheduled tearsheets, Streamlit has
  no background worker — the same ``python -m src.reporting.cli`` command run
  by cron or Task Scheduler checks the rules. Nothing is ever sent until that
  is set up, and this panel says so above the settings rather than below them.
* **What "triggered" means.** Prices here are delayed, and the runner sees
  samples rather than a continuous price, so an alert reports that a level
  *had been* reached as of a timestamp. The panel repeats that where the rules
  are configured, not only in the e-mail.

No credential is collected here either: alert e-mail uses the same SMTP
settings as the tearsheets, read from ``st.secrets`` or the environment at
send time.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime

import pandas as pd
import streamlit as st

from src import alerts as al
from src import watchlist as wl
from src.data.store import PortfolioStore
from src.reporting.schedule import clean_recipients
from src.ui import components as ui
from src.ui import formatters as fmt

__all__ = ["render", "rule_editor", "direction_picker"]

#: Session keys.
_LOADED = "alert_settings_loaded"
_SETTINGS = "alert_settings"

_DIRECTION_HELP = (
    "Which way the price has to move for you to be told. This is the one thing "
    "a bare target cannot say, and without it the rule cannot fire."
)


@st.cache_resource(show_spinner=False)
def _store() -> PortfolioStore:
    """One long-lived store per server process (shared with the sidebar)."""
    return PortfolioStore()


def _load_once() -> None:
    if st.session_state.get(_LOADED):
        return
    try:
        record = _store().load_alert_settings()
    except Exception as exc:  # noqa: BLE001 - a missing DB must not break the page
        st.warning(f"Alert settings could not be read: {exc}")
        record = {}
    st.session_state[_SETTINGS] = al.AlertSettings.parse(
        enabled=record.get("enabled", False),
        recipients=record.get("recipients"),
        rearm_pct=record.get("rearm_pct"),
        cooldown_hours=record.get("cooldown_hours"),
        use_session_range=record.get("use_session_range", True),
        timezone=record.get("timezone"),
    )
    st.session_state[_LOADED] = True


def settings() -> al.AlertSettings:
    _load_once()
    return st.session_state.get(_SETTINGS) or al.AlertSettings()


def _persist(config: al.AlertSettings) -> None:
    st.session_state[_SETTINGS] = config
    try:
        _store().save_alert_settings(
            {
                "enabled": config.enabled,
                "recipients": ", ".join(config.recipients),
                "rearm_pct": config.rearm_pct,
                "cooldown_hours": config.cooldown_hours,
                "use_session_range": config.use_session_range,
                "timezone": config.timezone,
            }
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Alert settings could not be saved: {exc}")


# ---------------------------------------------------------------------------
# reusable widgets
# ---------------------------------------------------------------------------
def direction_picker(
    key: str,
    level: float | None = None,
    price: float | None = None,
    current: str = "",
) -> str:
    """A radio for the alert direction, pre-selected from the current price.

    Pre-selection is a convenience, never a decision: if the target sits above
    today's price then "rises to" is almost certainly what was meant, so that
    is offered first — but the user owns the choice, and it is recorded
    explicitly rather than re-inferred later from a price that will have moved.
    """
    suggested = al.normalise_direction(current) or al.infer_direction(level, price)
    options = list(al.DIRECTIONS)
    index = options.index(suggested) if suggested in options else 0
    return st.radio(
        "Tell me when the price",
        options=options,
        index=index,
        format_func=lambda d: al.DIRECTION_LABELS[d],
        horizontal=True,
        key=key,
        help=_DIRECTION_HELP,
    )


def rule_editor(
    item: wl.WatchItem,
    price: float | None,
    key_prefix: str,
) -> tuple[bool, str] | None:
    """Alert controls for one item. ``(alert_enabled, direction)``.

    Returns ``None`` when the item has no target, because an alert needs a
    level and offering the controls anyway would imply otherwise.
    """
    if not item.has_target:
        st.caption("Set a target price first — an alert needs a level to watch.")
        return None

    enabled = st.checkbox(
        "E-mail me when this level is reached",
        value=bool(item.alert_enabled),
        key=f"{key_prefix}_alert_on",
        help=("Off by default. Setting a target never starts sending e-mail on its own."),
    )
    direction = direction_picker(
        f"{key_prefix}_alert_dir",
        level=item.target_price,
        price=price,
        current=item.direction,
    )
    # Described from a preview copy, so the caption reflects what the widgets
    # currently say rather than what was last saved.
    preview = replace(item, direction=direction, alert_enabled=enabled)
    st.caption(al.describe_rule(preview))
    return enabled, direction


# ---------------------------------------------------------------------------
# the runner command
# ---------------------------------------------------------------------------
def _runner_note() -> None:
    st.markdown(
        "**Alerts are checked by a command, not by this page.** Streamlit has "
        "no background worker — it only runs while a browser is connected — so "
        "nothing is checked and nothing is sent until you schedule the command "
        "below. It is the same command that sends the scheduled tearsheets, so "
        "one cron entry covers both."
    )
    st.code(
        "# every 5 minutes: alerts want a short interval, and re-checking is free\n"
        "*/5 * * * * cd /srv/quantfolio && .venv/bin/python -m src.reporting.cli "
        ">> /var/log/quantfolio-mail.log 2>&1",
        language="bash",
    )
    st.caption(
        "Run it as often as you like: each rule e-mails once when it is reached "
        "and stays quiet until the price retreats past the level again. "
        "`--alerts` checks only alerts; `--no-alerts` only tearsheets; "
        "`--list` shows every rule and whether it can fire."
    )


# ---------------------------------------------------------------------------
# the rules overview
# ---------------------------------------------------------------------------
def _rules_overview(items, table: pd.DataFrame | None = None) -> None:
    state = al.alert_state_table(items)
    active = state[state["alert"]]

    if active.empty:
        st.info(
            "No instrument has an alert switched on. Open **Edit or remove** "
            "below, pick one with a target, and tick *E-mail me when this "
            "level is reached*."
        )
    else:
        display = active.copy()
        display["direction"] = display["direction"].map(
            lambda d: al.DIRECTION_LABELS.get(d, "— not set —")
        )
        display["state"] = display["state"].map(
            {al.ARMED: "waiting", al.TRIGGERED: "already through"}
        )
        display["last_fired"] = display["last_fired"].map(
            lambda v: v.strftime("%d %b %H:%M") if isinstance(v, datetime) else "never"
        )
        ui.dataframe(
            display.rename(
                columns={
                    "ticker": "Instrument",
                    "direction": "Tell me when the price",
                    "level": "Level",
                    "state": "Status",
                    "last_fired": "Last e-mail",
                    "rule": "What happens",
                }
            ).drop(columns=["alert"])
        )

    # A target with alerting on but no direction is the one configuration that
    # looks armed and can never fire, so it is called out rather than listed.
    broken = [
        item.ticker
        for item in items
        if item.alert_enabled
        and item.has_target
        and not (
            al.normalise_direction(item.direction)
            or al.infer_direction(item.target_price, item.reference_price)
        )
    ]
    if broken:
        st.warning(
            "These have alerting switched on but **no direction**, so they "
            "cannot fire: "
            + ", ".join(f"`{t}`" for t in broken)
            + ". Set *rises to* or *falls to* on each."
        )

    if table is not None and not table.empty:
        _preview(items, table)


def _preview(items, table: pd.DataFrame) -> None:
    """What the runner would find right now, from prices already on this page.

    A dry preview, computed from the quotes the page has already fetched — it
    sends nothing and records nothing, so it cannot consume an alert. It exists
    because "is my rule set up correctly?" is otherwise unanswerable until the
    next cron run.
    """
    prices = dict(zip(table["ticker"], table["price"], strict=False))
    reached: list[str] = []
    for item in items:
        if not (item.alert_enabled and item.has_target):
            continue
        rule = al.normalise_direction(item.direction) or al.infer_direction(
            item.target_price, item.reference_price
        )
        if not rule:
            continue
        price = prices.get(item.ticker)
        if price is None or (isinstance(price, float) and math.isnan(price)):
            continue
        hit, evidence = al.crossing(rule, item.target_price, price)
        if hit:
            reached.append(
                f"`{item.ticker}` {al.DIRECTION_LABELS[rule]} "
                f"{fmt.fmt_num(item.target_price, 2)} ({evidence} "
                f"{fmt.fmt_num(price, 2)})"
            )
    if reached:
        st.info(
            "**At the prices on this page, these are through their level:** "
            + " · ".join(reached)
            + ". Whether an e-mail goes out depends on whether each one had "
            "already been reported — this preview sends nothing."
        )


# ---------------------------------------------------------------------------
# delivery settings
# ---------------------------------------------------------------------------
def _delivery_form(config: al.AlertSettings) -> None:
    with st.form("alert_settings_form"):
        st.markdown("**Delivery**")
        enabled = st.checkbox(
            "Send alert e-mails",
            value=config.enabled,
            help="The master switch. Off means the command checks nothing.",
        )
        recipients = st.text_input(
            "Send to",
            value=", ".join(config.recipients),
            placeholder="you@example.com, someone.else@example.com",
            help="Comma-separated. Invalid addresses are reported, not silently dropped.",
        )

        st.markdown("**How hard a rule is to re-trigger**")
        left, right = st.columns(2)
        rearm = left.number_input(
            "Re-arm band (% past the level)",
            min_value=0.0,
            max_value=25.0,
            value=float(config.rearm_pct),
            step=0.25,
            format="%.2f",
            help=(
                "After an alert fires, the price must retreat this far back "
                "past your level before the same rule can fire again. Without "
                "a band, a price wobbling on a round number produces a burst "
                "of e-mails."
            ),
        )
        cooldown = right.number_input(
            "Minimum hours between alerts for one instrument",
            min_value=0.0,
            max_value=720.0,
            value=float(config.cooldown_hours),
            step=1.0,
            help=(
                "A second guard behind the band, for a price volatile enough to clear even that."
            ),
        )
        use_range = st.checkbox(
            "Also use the session high/low, not just the last price",
            value=config.use_session_range,
            help=(
                "Catches a level that was touched between two checks and then "
                "moved away. Every alert says which observation saw it, so a "
                "session-high trigger is never presented as the current price."
            ),
        )

        saved = st.form_submit_button("Save alert settings", use_container_width=True)

    if not saved:
        return

    accepted, rejected = clean_recipients(recipients)
    for message in rejected:
        st.warning(message)

    updated = al.AlertSettings(
        enabled=enabled,
        recipients=accepted,
        rearm_pct=rearm,
        cooldown_hours=cooldown,
        use_session_range=use_range,
        timezone=config.timezone,
    )
    if updated.enabled and not updated.recipients:
        st.error(
            "Alerts are switched on but there is no valid recipient, so nothing "
            "could be delivered. Add an address, or switch them off."
        )
        return
    _persist(updated)
    st.success("Alert settings saved.")
    st.rerun()


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def render(items, table: pd.DataFrame | None = None) -> None:
    """Draw the alert panel. Safe on every rerun; sends and checks nothing."""
    config = settings()

    if config.enabled and config.recipients:
        ui.badges(
            [
                ("alerts on", "gain"),
                (f"to {', '.join(config.recipients)}", "neutral"),
                (f"re-arm {config.rearm_pct:g}%", "neutral"),
                (f"cooldown {config.cooldown_hours:g}h", "neutral"),
            ]
        )
    else:
        ui.badges([("alerts off", "warn")])

    _rules_overview(items, table)

    with st.expander("Alert delivery and sensitivity", expanded=not config.enabled):
        _runner_note()
        st.divider()
        _delivery_form(config)
        st.caption(
            "Alert e-mail uses the same SMTP settings as the scheduled "
            "tearsheets — `.streamlit/secrets.toml` or the `SMTP_*` "
            "environment variables. No password is ever stored by this "
            "application."
        )

    ui.methodology(
        "What a price alert can and cannot tell you",
        """
**It fires once per crossing, not once per check.** Each rule is a two-state
machine: *waiting* until the level is reached, then *already through* and
silent. It returns to *waiting* only after the price retreats past your level
by the re-arm band. That is what stops a price sitting on a round number from
e-mailing you every five minutes.

**Prices are delayed.** Typically ~15 minutes for NSE/BSE, and outside market
hours the latest available close. An alert therefore says your level *had been*
reached as of the timestamp in the e-mail — never that it is being reached now.

**Between two checks, only the session range is evidence.** The command sees
samples, not a continuous price. A move that crossed your level and came back
between two checks is visible only if the session high or low caught it, and
every alert names which observation saw the crossing so a touched-then-reversed
level is never presented as the current price. A crossing entirely inside an
earlier, already-closed session cannot be recovered from a quote at all.

**A rule that cannot be checked is reported, not ignored.** No quote, no
direction — those travel in the e-mail and in the command's output, because
silence about a rule looks exactly like "your level was not reached", and that
is the one failure an alert must never have.

**The level is a number you typed.** Nothing here suggests a target, ranks one
instrument above another, or treats a level being reached as a reason to trade.
""",
    )
