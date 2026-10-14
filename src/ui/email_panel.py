"""Scheduled e-mail tearsheets: the configuration surface, not the scheduler.

Mounted by the Export page.

The honesty problem this panel exists to solve
---------------------------------------------
Streamlit runs your script when a browser asks it to, and not otherwise. It
has no background worker, no timer and no guarantee that any process is alive
at 08:00 on a Monday. A "scheduler" built from a thread inside the app would
therefore:

* not fire at all when nobody has the app open,
* fire two, five or ten times when several browser tabs are open, and
* stop dead on the next restart, silently.

So this panel does not claim to schedule anything. It saves the schedule, and
it prints the exact command line the user must hand to cron or Windows Task
Scheduler, along with a plain statement that **no e-mail is sent until that is
done**. The alternative -- a toggle labelled "Enabled" that quietly does
nothing -- is the single most dishonest thing this application could ship.

Two further rules the UI obeys:

* **No credential is ever collected here.** There is no password field,
  because a password typed into a Streamlit input would end up in the
  application's SQLite file inside ``data/``. The panel shows the
  ``secrets.toml`` block to write instead, and reports whether the settings
  were found -- never their values.
* **Nothing outward-facing fires on a page render.** "Send a test now" is
  behind a form plus an explicit confirmation checkbox, because a Streamlit
  script re-runs on every widget interaction anywhere on the page, and an
  e-mail is not undoable.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pandas as pd
import streamlit as st

from src.config import DEFAULT_BENCHMARKS, PERIOD_LABELS
from src.data.store import PortfolioStore
from src.reporting import email_report as mail
from src.reporting import schedule as sched
from src.ui import components as ui
from src.ui import theme as ui_theme

__all__ = ["render"]

#: Session keys.
_LOADED = "email_schedules_loaded"
_SCHEDULES = "email_schedules"
_LAST_RESULT = "email_last_delivery"

#: Currencies the reporting layer formats. Kept short on purpose -- the point
#: of the field is which currency the totals are stated in, not a full list.
_CURRENCIES = ("INR", "USD", "EUR", "GBP")


@st.cache_resource(show_spinner=False)
def _store() -> PortfolioStore:
    """One long-lived store per server process (shared with the sidebar)."""
    return PortfolioStore()


def _load_once() -> None:
    """Hydrate schedules from SQLite on the first visit of a session.

    Flag-guarded rather than emptiness-guarded: without the flag, deleting
    your last schedule would be undone by the next rerun.
    """
    if st.session_state.get(_LOADED):
        return
    try:
        records = _store().load_schedules()
    except Exception as exc:  # noqa: BLE001 - a missing DB must not break the page
        st.warning(f"Saved schedules could not be read: {exc}")
        records = []
    st.session_state[_SCHEDULES] = [sched.from_record(r) for r in records]
    st.session_state[_LOADED] = True


def _schedules() -> list[sched.Schedule]:
    _load_once()
    return list(st.session_state.get(_SCHEDULES, []))


def _persist(schedules: list[sched.Schedule]) -> None:
    st.session_state[_SCHEDULES] = schedules
    try:
        _store().save_schedules([sched.to_record(s) for s in schedules])
    except Exception as exc:  # noqa: BLE001
        st.error(f"The schedule could not be saved: {exc}")


def _portfolio_names() -> list[str]:
    try:
        return [str(row.get("name")) for row in _store().list_portfolios()]
    except Exception:  # noqa: BLE001
        return []


# ---------------------------------------------------------------------------
# the delivery command
# ---------------------------------------------------------------------------
def _runner_command() -> None:
    """Show what actually has to be scheduled, and say so plainly."""
    st.markdown(
        "**Nothing is sent until you schedule the command below.** Streamlit "
        "has no background scheduler — it only runs while a browser is "
        "connected — so the clock has to live outside the app. This page "
        "stores *what* to send and *when*; your operating system decides when "
        "to ask."
    )
    st.caption(
        "Run it as often as you like. Each schedule is sent at most once per "
        "occurrence, so a five-minute cron still delivers an 08:00 report "
        "exactly once — and a slot missed while the machine was asleep goes "
        "out once when it wakes."
    )

    linux, windows, manual = st.tabs(["cron (Linux/macOS)", "Windows", "Run it by hand"])
    with linux:
        st.code(
            "# every 15 minutes; logs to a file you can actually read\n"
            "*/15 * * * * cd /srv/quantfolio && "
            ".venv/bin/python -m src.reporting.cli >> "
            "/var/log/quantfolio-mail.log 2>&1",
            language="bash",
        )
        st.caption(
            "`crontab -e` to edit. The SMTP_* variables must be visible to "
            "cron — cron does not read your shell profile, so set them in the "
            "crontab itself or point the command at a wrapper script."
        )
    with windows:
        st.code(
            "schtasks /Create /TN QuantFolioMail /SC MINUTE /MO 15 ^\n"
            '  /TR "cmd /c cd /d C:\\path\\to\\portfolio_analyzer && '
            '.venv\\Scripts\\python.exe -m src.reporting.cli"',
            language="powershell",
        )
        st.caption(
            "Or Task Scheduler → Create Task → Triggers → Repeat every 15 "
            "minutes. Set the SMTP_* variables as machine environment "
            "variables so the task account can see them."
        )
    with manual:
        st.code(
            "python -m src.reporting.cli --list            # what is configured, and what is due\n"
            "python -m src.reporting.cli --dry-run         # build and validate, send nothing\n"
            "python -m src.reporting.cli --force --only 'Weekly review'\n"
            "python -m src.reporting.cli --now 2026-09-07T08:00 --list   # check the timing",
            language="bash",
        )
        st.caption(
            "Exit codes: 0 sent or nothing due · 1 a schedule failed · "
            "2 could not start (no SMTP settings, unreadable database)."
        )


# ---------------------------------------------------------------------------
# SMTP status
# ---------------------------------------------------------------------------
def _smtp_from_secrets() -> tuple[mail.SmtpConfig | None, tuple[str, ...]]:
    """Read SMTP settings from ``st.secrets``, falling back to the environment.

    Wrapped because ``st.secrets`` raises when no secrets file exists at all,
    which is the normal state of a fresh checkout, not an error.
    """
    try:
        source = dict(st.secrets)
    except Exception:  # noqa: BLE001 - no secrets.toml present
        source = {}
    if not source:
        return mail.load_smtp_config()
    config, problems = mail.load_smtp_config(source)
    if config is None:
        # Nothing in secrets.toml; the environment may still carry it.
        return mail.load_smtp_config()
    return config, problems


def _smtp_status() -> mail.SmtpConfig | None:
    """Report whether mail is configured. Never displays a secret value."""
    config, problems = _smtp_from_secrets()

    if config is None:
        st.info(
            "**Mail is not configured yet.** Add this to "
            "`.streamlit/secrets.toml` (already git-ignored), or set the same "
            "values as environment variables for the scheduled command."
        )
        st.code(mail.secrets_template(), language="toml")
        st.caption(
            "This panel never asks for your password in a text box: anything "
            "typed into the app would be saved into `data/quantfolio.db`, "
            "which is a plaintext secret sitting in your project folder. "
            "Gmail and Outlook both need an app-specific password here, not "
            "your account password."
        )
        return None

    if problems:
        st.error("Mail is configured but unusable:")
        for issue in problems:
            st.markdown(f"- {issue}")
        return None

    transport = (
        "implicit SSL"
        if config.use_ssl
        else ("STARTTLS" if config.use_starttls else "NO ENCRYPTION")
    )
    ui.badges(
        [
            (f"{config.host}:{config.port}", "neutral"),
            (transport, "gain" if config.encrypted else "warn"),
            (f"from {config.sender}", "neutral"),
            ("password loaded" if config.password else "no password", "neutral"),
        ]
    )
    st.caption(
        "Read from your secrets file or environment at send time — the "
        "password is never stored by this application and never shown here."
    )
    return config


# ---------------------------------------------------------------------------
# the schedule list
# ---------------------------------------------------------------------------
def _schedule_table(schedules: list[sched.Schedule]) -> None:
    now = datetime.now()
    rows = [
        {
            "Schedule": s.name,
            "Portfolio": s.portfolio or "—",
            "Recipients": ", ".join(s.recipients) or "—",
            "Cadence": sched.describe(s, now),
            "Last sent": (s.last_sent.strftime("%d %b %Y %H:%M") if s.last_sent else "never"),
            "Due now": "yes" if sched.is_due(s, now) else "no",
        }
        for s in schedules
    ]
    ui.dataframe(pd.DataFrame(rows))

    broken = [(s.name, s.problems) for s in schedules if s.problems]
    if broken:
        st.warning("These schedules cannot run as configured:")
        for name, issues in broken:
            for issue in issues:
                st.markdown(f"- **{name}** — {issue}")


def _add_form(portfolios: list[str]) -> None:
    with st.form("email_schedule_add", clear_on_submit=False):
        st.markdown("**Add or update a schedule**")
        name_col, portfolio_col = st.columns(2)
        name = name_col.text_input(
            "Name",
            value="",
            placeholder="Weekly review",
            help="A schedule with this name is replaced if it already exists.",
        )
        portfolio = portfolio_col.selectbox(
            "Saved portfolio",
            options=portfolios or ["(save a portfolio first)"],
            help=(
                "A scheduled run has no browser session to read from, so it "
                "rebuilds the analysis from a portfolio saved in the sidebar."
            ),
            disabled=not portfolios,
        )
        recipients = st.text_input(
            "Send to",
            placeholder="you@example.com, someone.else@example.com",
            help="Comma-separated. Invalid addresses are reported, not silently dropped.",
        )

        freq_col, time_col, when_col = st.columns(3)
        frequency = freq_col.selectbox("Frequency", options=sched.FREQUENCIES, index=1)
        clock = time_col.time_input("At", value=datetime(2000, 1, 1, 8, 0))
        weekday = 0
        day_of_month = 1
        if frequency == sched.WEEKLY:
            weekday = when_col.selectbox(
                "On",
                options=range(7),
                format_func=lambda i: sched.WEEKDAY_NAMES[i],
            )
        elif frequency == sched.MONTHLY:
            day_of_month = when_col.number_input(
                "Day of month",
                min_value=1,
                max_value=31,
                value=1,
                help=(
                    "A day past the end of a short month is clamped to the "
                    "last day, so 31 still fires every month."
                ),
            )
        else:
            when_col.caption("Daily — no day to pick.")

        tz_col, period_col = st.columns(2)
        timezone = tz_col.text_input(
            "Timezone",
            value=sched.DEFAULT_TIMEZONE,
            help="IANA name, e.g. Asia/Kolkata, America/New_York, Europe/London.",
        )
        period = period_col.selectbox(
            "History window",
            options=list(PERIOD_LABELS),
            index=list(PERIOD_LABELS).index("1y"),
            format_func=lambda p: PERIOD_LABELS[p],
        )

        bench_col, ccy_col = st.columns(2)
        benchmark = bench_col.selectbox("Benchmark", options=list(DEFAULT_BENCHMARKS))
        base_currency = ccy_col.selectbox("Reporting currency", options=_CURRENCIES)

        submitted = st.form_submit_button("Save schedule", use_container_width=True)

    if not submitted:
        return

    accepted, rejected = sched.clean_recipients(recipients)
    for message in rejected:
        st.warning(message)

    candidate = sched.Schedule(
        name=name,
        portfolio=portfolio if portfolios else "",
        recipients=accepted,
        frequency=frequency,
        hour=clock.hour,
        minute=clock.minute,
        weekday=int(weekday),
        day_of_month=int(day_of_month),
        timezone=timezone,
        period=period,
        benchmark=benchmark,
        base_currency=base_currency,
    )
    if candidate.problems:
        for issue in candidate.problems:
            st.error(issue)
        return

    zone_warning = sched.resolve_timezone(candidate.timezone)[1]
    if zone_warning:
        st.warning(zone_warning)

    kept = [s for s in _schedules() if s.name.lower() != candidate.name.lower()]
    previous = next((s for s in _schedules() if s.name.lower() == candidate.name.lower()), None)
    if previous is not None and previous.last_sent is not None:
        # An edited schedule inherits its last_sent, so changing the recipient
        # list does not resend a slot that already went out.
        candidate = candidate.mark_sent(previous.last_sent)
    else:
        # A new one starts from its next slot rather than firing immediately
        # for a slot that passed before it existed.
        candidate = sched.starting_next_slot(candidate)
    _persist([*kept, candidate])
    st.success(f"Saved {candidate.name!r}: {sched.describe(candidate)}")
    st.rerun()


def _manage(schedules: list[sched.Schedule]) -> None:
    names = [s.name for s in schedules]
    pick_col, pause_col, delete_col = st.columns([2, 1, 1])
    target = pick_col.selectbox("Schedule", options=names, key="email_manage_pick")
    chosen = next((s for s in schedules if s.name == target), None)
    if chosen is None:
        return

    label = "Resume" if not chosen.enabled else "Pause"
    if pause_col.button(label, use_container_width=True, key="email_pause"):
        updated = [replace(s, enabled=not s.enabled) if s.name == target else s for s in schedules]
        _persist(updated)
        st.rerun()

    if delete_col.button("Delete", use_container_width=True, key="email_delete"):
        _persist([s for s in schedules if s.name != target])
        st.rerun()


# ---------------------------------------------------------------------------
# the test send
# ---------------------------------------------------------------------------
def _test_send(schedules: list[sched.Schedule], smtp: mail.SmtpConfig | None) -> None:
    """One e-mail, on an explicit double confirmation.

    Behind a form AND a checkbox on purpose. Streamlit re-runs the whole script
    on every widget interaction anywhere on the page, so a bare button would
    be one stray rerun away from mailing someone. Sending is not undoable.
    """
    st.markdown("**Send one now, to check it works**")
    st.caption(
        "Builds the report exactly as the scheduled command would and sends it "
        "immediately. This does not change the schedule and does not count as "
        "that occurrence — the next scheduled report still goes out."
    )

    if smtp is None:
        st.caption("Configure SMTP above first.")
        return
    if not schedules:
        return

    with st.form("email_test_send"):
        target = st.selectbox("Schedule", options=[s.name for s in schedules])
        confirm = st.checkbox("Yes, send a real e-mail now", value=False, key="email_test_confirm")
        go = st.form_submit_button("Send test", use_container_width=True)

    if not go:
        return
    if not confirm:
        st.info("Tick the confirmation box to send.")
        return

    chosen = next((s for s in schedules if s.name == target), None)
    if chosen is None:
        return

    # Imported here, not at module scope: the runner defers Streamlit's heavy
    # pipeline import, and a page render should not pay for it.
    from src.reporting import cli

    with st.spinner(f"Building and sending {chosen.name!r}…"):
        outcome = cli.run_one(chosen, _store(), smtp, dry_run=False, record=False)
    st.session_state[_LAST_RESULT] = outcome

    if outcome.sent:
        st.success(outcome.summary)
    else:
        st.error(outcome.summary)
    for note in outcome.notes:
        st.warning(note)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def render() -> None:
    """Draw the whole panel. Safe to call on every rerun; sends nothing."""
    schedules = _schedules()
    portfolios = _portfolio_names()

    _runner_command()

    ui_theme.spacer(0.3)
    st.markdown("**Mail settings**")
    smtp = _smtp_status()

    ui_theme.spacer(0.3)
    st.markdown("**Schedules**")
    if not portfolios:
        st.info(
            "Save a portfolio in the sidebar first. A scheduled report rebuilds "
            "its analysis from saved holdings, because a cron job has no "
            "browser session to read them from."
        )
    if schedules:
        _schedule_table(schedules)
        _manage(schedules)
    else:
        ui.empty_state(
            "No scheduled reports",
            "Add one below. It is stored locally in the same SQLite file as "
            "your portfolios — no credential is stored with it.",
        )

    ui_theme.spacer(0.3)
    _add_form(portfolios)

    ui_theme.spacer(0.3)
    _test_send(schedules, smtp)

    ui_theme.spacer(0.3)
    ui.methodology(
        "What the e-mail contains, and what it cannot",
        """
**In the body:** value, invested, unrealised P&L, today's change, annualised
return and volatility, Sharpe, Sortino, max drawdown, daily VaR, XIRR where a
trade ledger exists, and the benchmark comparison — as plain text, so it reads
on a phone with attachments blocked.

**Attached:** the same self-contained HTML tearsheet the Export page produces,
with the full statistic tables, the insight log and the holdings table.

**Not attached: the charts.** They are rendered by the app's chart layer from a
live session, and a cron job has none. Building a second set for e-mail would
mean two renderings of the same numbers that could disagree, so the attachment
carries the figures as tables instead. Open the app for the charts.

**Prices are as of when the command ran**, not when you read the mail. The
report stamps its own generation time, and any data problem the pipeline hit
travels with it in a "data notes" section rather than being quietly dropped.

**Reliability is your scheduler's, not this application's.** If the machine is
off at 08:00 the report is sent once when the scheduler next runs, however much
later. If the scheduler is never set up, nothing is ever sent — and this panel
will still happily list your schedules, which is why the command is at the top
of the page rather than the bottom.
""",
    )
