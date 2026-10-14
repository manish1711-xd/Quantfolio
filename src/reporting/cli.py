"""Command-line runner for scheduled tearsheets: ``python -m src.reporting.cli``.

This module is the honest half of "scheduled email tearsheets" and of
"watchlist price alerts". Streamlit cannot schedule anything -- it is a
request-response server that runs your script when a browser asks it to -- so
the clock lives outside the application. The app stores the schedules and the
alert rules; this command asks what is due, builds it and sends it; and a real
scheduler runs this command.

It does two independent jobs, and the idempotence key differs between them:

* **Tearsheets are time-triggered.** One e-mail per scheduled *occurrence*,
  however often this command runs.
* **Alerts are event-triggered.** One e-mail per ``armed -> triggered``
  *transition* of a rule. There is no cadence to configure: the rules are
  checked on every invocation, so the cron interval simply is the check
  interval, and :mod:`src.alerts` decides what is new.

Alerts therefore want a short interval (five to fifteen minutes) and
tearsheets do not care, which is why one command does both and running it
often is safe.

    # cron -- every 15 minutes, which is fine: is_due() is idempotent, so an
    # 08:00 schedule still goes out exactly once.
    */15 * * * * cd /srv/quantfolio && .venv/bin/python -m src.reporting.cli >> \\
        /var/log/quantfolio-mail.log 2>&1

    # Windows Task Scheduler, same cadence
    schtasks /Create /TN QuantFolioMail /SC MINUTE /MO 15 ^
      /TR "cmd /c cd /d E:\\quantfolio && .venv\\Scripts\\python.exe -m src.reporting.cli"

Design commitments
------------------
* **The same pipeline as the dashboard.** ``pipeline.run(cfg, holdings, txns)``
  is called with inputs loaded from SQLite instead of from a session. There is
  no second, simplified computation path that could drift out of agreement
  with what the app shows on screen.
* **Nothing is faked when data is missing.** A portfolio that no longer exists,
  a symbol that will not price, an empty history -- each is reported on stderr
  and reflected in the exit code. The report is not sent with zeros in it.
* **No credential is ever printed**, including in ``--verbose`` and including
  in the error paths.
* **``last_sent`` is written only after a successful send.** A crash between
  sending and recording costs one duplicate at worst; the reverse order would
  lose a report silently, which is much worse. Alert state is committed the
  same way, for the same reason.
* **A rule that could not be checked is reported, never treated as quiet.**
  An alert with no quote is silence that looks exactly like "your level was
  not reached", so it travels in the e-mail and in the exit code.

Exit codes, because a scheduler only ever reads those:

===  ============================================================
  0  everything due was sent (or nothing was due)
  1  at least one schedule failed -- built, or sent, unsuccessfully
  2  the runner could not start: no SMTP settings, unreadable store
===  ============================================================
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src import alerts as al
from src import watchlist as wl
from src.config import DEFAULT_BENCHMARK, DEFAULT_RISK_FREE, AppConfig
from src.data.store import DEFAULT_DB_PATH, PortfolioStore
from src.reporting import email_report as mail
from src.reporting import html_report as report
from src.reporting import schedule as sched

logger = logging.getLogger("quantfolio.mail")

__all__ = [
    "build_config",
    "build_result",
    "run_one",
    "run_due",
    "load_alert_settings",
    "check_alerts",
    "main",
]

EXIT_OK = 0
EXIT_PARTIAL_FAILURE = 1
EXIT_CANNOT_START = 2


def _utf8_streams() -> None:
    """Make stdout/stderr survive any character a report can contain.

    On Windows, ``print`` encodes with the console or locale codepage, and a
    rupee sign in a pipeline warning is not representable in cp1252 -- so the
    runner would die with a ``UnicodeEncodeError`` while reporting a problem,
    which is a spectacularly unhelpful way to fail. UTF-8 with ``replace``
    guarantees the log line always gets written.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        with contextlib.suppress(OSError, ValueError):  # exotic or closed stream
            reconfigure(encoding="utf-8", errors="replace")


def _quiet_streamlit() -> None:
    """Silence Streamlit's "no runtime" chatter, before it starts.

    ``src.ui.pipeline`` imports Streamlit for its caching decorators. They
    work perfectly well headlessly, but constructing each one logs "No runtime
    found, using MemoryCacheStorageManager" -- twenty-odd lines emitted at
    IMPORT time, burying the one line a scheduler operator needs to read.

    Because the noise happens during the import, this has to run first, and it
    has to use Streamlit's own setter: Streamlit keeps a private registry of
    its loggers and re-levels all of them from its configuration, so a plain
    ``logging.getLogger(...).setLevel`` set beforehand does not survive. The
    environment variable is set too, with ``setdefault``, so an operator who
    has deliberately asked for debug logging still gets it.

    Importing ``streamlit.logger`` pulls in the Streamlit package, which is
    why this is called from :func:`build_result` rather than from
    :func:`main`: ``--list`` never needs Streamlit and should not pay several
    seconds to find out that nothing is due. Only Streamlit's own loggers are
    affected; nothing else is suppressed.
    """
    os.environ.setdefault("STREAMLIT_LOGGER_LEVEL", "error")
    try:
        from streamlit import logger as streamlit_logger

        streamlit_logger.set_log_level("error")
    except Exception:
        logger.debug("Streamlit log level could not be lowered.", exc_info=True)
    for name in (
        "streamlit",
        "streamlit.runtime.caching.cache_data_api",
        "streamlit.runtime.state.session_state_proxy",
        "streamlit.runtime.scriptrunner_utils.script_run_context",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)


def build_config(schedule: sched.Schedule) -> AppConfig:
    """The :class:`AppConfig` a schedule's analysis should run under.

    Deliberately reads its window, benchmark and currency from the schedule
    rather than from a default, so a weekly India report and a monthly US one
    can coexist. The risk-free rate follows the reporting currency, matching
    what the sidebar does when the user changes it there.
    """
    currency = str(schedule.base_currency or "INR").upper()
    benchmark = str(schedule.benchmark or DEFAULT_BENCHMARK)
    return AppConfig(
        period=str(schedule.period or "1y"),
        benchmark_name=benchmark,
        base_currency=currency,
        risk_free_rate=DEFAULT_RISK_FREE.get(currency, DEFAULT_RISK_FREE["INR"]),
    )


def build_result(schedule: sched.Schedule, store: PortfolioStore) -> tuple[Any | None, str]:
    """Rebuild one schedule's analysis from persisted data.

    ``(result, "")`` on success, ``(None, reason)`` on failure. The import of
    :mod:`src.ui.pipeline` is deferred to here so that ``--list`` costs
    nothing: importing the pipeline pulls in Streamlit, pandas, scipy and
    yfinance, which is several seconds a scheduler should not pay just to
    discover that nothing is due.
    """
    # Before the import, not after: the cache decorators log their "no
    # runtime" warning as they are constructed, which happens on import.
    _quiet_streamlit()

    from src.ui import pipeline

    holdings, transactions = store.load_portfolio(schedule.portfolio)
    if holdings.empty:
        # load_portfolio returns empty frames for an unknown name, so the two
        # cases are separated here: a deleted portfolio and an empty one are
        # different mistakes and deserve different messages.
        known = {str(row.get("name")) for row in store.list_portfolios()}
        if schedule.portfolio not in known:
            return None, (
                f"No saved portfolio named {schedule.portfolio!r}. "
                f"Saved portfolios: {', '.join(sorted(known)) or 'none'}."
            )
        return None, f"Portfolio {schedule.portfolio!r} has no holdings to report on."

    try:
        result = pipeline.run(build_config(schedule), holdings, transactions)
    except Exception as exc:
        logger.exception("Analysis failed for schedule %r", schedule.name)
        return None, f"The analysis could not be completed: {exc}"
    return result, ""


def run_one(
    schedule: sched.Schedule,
    store: PortfolioStore,
    smtp: mail.SmtpConfig,
    *,
    now: datetime | None = None,
    dry_run: bool = False,
    record: bool = True,
) -> mail.DeliveryResult:
    """Build and send one schedule's tearsheet, then record the send.

    ``record=False`` sends without stamping ``last_sent``. That is what the
    app's "send a test now" button uses: a test proves the plumbing works and
    must not consume the occurrence, or the user would confirm delivery and
    then never receive the report they actually configured.

    Figures are NOT embedded. They are authored by the Streamlit chart layer
    from a live session; a cron job has none, and inventing a second set for
    e-mail would mean two renderings of the same numbers that could disagree.
    The attachment therefore carries the full statistics, insight list and
    holdings table -- everything except the charts -- which also keeps it in
    the tens of kilobytes rather than the megabytes a Plotly bundle costs.
    """
    moment = now if now is not None else sched.now_in(schedule.timezone)
    occurrence = sched.previous_occurrence(schedule, moment)

    result, problem = build_result(schedule, store)
    if result is None:
        return mail.DeliveryResult(
            sent=False,
            recipients=schedule.recipients,
            subject=schedule.name,
            error=problem,
        )

    html = report.build_html_report(
        result, config=build_config(schedule), portfolio_name=schedule.portfolio
    )
    attachment = mail.Attachment(
        filename=report.report_filename(schedule.portfolio, moment),
        payload=html.encode("utf-8"),
        maintype="text",
        subtype="html",
    )

    summary = dict(getattr(result, "summary", {}) or {})
    currency = str(summary.get("base_currency") or schedule.base_currency)
    change = summary.get("total_return_pct")
    change_text = f" ({float(change):+.2f}%)" if isinstance(change, (int, float)) else ""
    subject = (
        f"{schedule.portfolio} tearsheet — {moment:%d %b %Y} — "
        f"{currency} {float(summary.get('current_value') or 0):,.0f}{change_text}"
    )

    body = mail.text_summary(
        result,
        portfolio_name=schedule.portfolio,
        schedule_note=(f"Scheduled report {schedule.name!r}: {sched.describe(schedule, moment)}"),
    )

    message, compose_notes = mail.compose_tearsheet_email(
        schedule.recipients,
        subject,
        body,
        [attachment],
        config=smtp,
    )
    if message is None:
        return mail.DeliveryResult(
            sent=False,
            recipients=schedule.recipients,
            subject=subject,
            error=" ".join(compose_notes) or "The message could not be composed.",
        )

    outcome = mail.send_message(smtp, message, dry_run=dry_run)
    outcome = mail.DeliveryResult(
        sent=outcome.sent,
        recipients=outcome.recipients,
        subject=outcome.subject,
        size_bytes=outcome.size_bytes,
        dry_run=outcome.dry_run,
        error=outcome.error,
        notes=(*compose_notes, *outcome.notes),
    )

    if outcome.sent and record and occurrence is not None:
        # Stamped with the OCCURRENCE, not with the send time. Stamping "now"
        # would be wrong for a late run: a 08:00 slot sent at 11:20 and
        # recorded as 11:20 would suppress the next day's 08:00 report if the
        # runner happened to fire before 11:20.
        store.mark_schedule_sent(schedule.name, occurrence)

    return outcome


def run_due(
    store: PortfolioStore,
    smtp: mail.SmtpConfig,
    *,
    now: datetime | None = None,
    dry_run: bool = False,
    only: str = "",
    force: bool = False,
) -> list[tuple[sched.Schedule, mail.DeliveryResult]]:
    """Process every due schedule and report what happened to each.

    One schedule's failure never stops the others: each is built and sent
    independently, and the caller decides the exit code from the collection.
    """
    schedules = [sched.from_record(record) for record in store.load_schedules()]
    if only:
        schedules = [s for s in schedules if s.name.lower() == only.strip().lower()]

    if force:
        # --force ignores the timing but never the validity: an unusable
        # schedule still cannot be sent, it just says so immediately.
        pending = [s for s in schedules if s.ok]
    else:
        pending = sched.due_schedules(schedules, now)

    outcomes: list[tuple[sched.Schedule, mail.DeliveryResult]] = []
    for schedule in pending:
        logger.info("Processing %r (%s)", schedule.name, schedule.portfolio)
        outcomes.append((schedule, run_one(schedule, store, smtp, now=now, dry_run=dry_run)))
    return outcomes


def load_alert_settings(store: PortfolioStore) -> al.AlertSettings:
    """The stored alert settings as an :class:`~src.alerts.AlertSettings`."""
    record = store.load_alert_settings()
    return al.AlertSettings.parse(
        enabled=record.get("enabled"),
        recipients=record.get("recipients"),
        rearm_pct=record.get("rearm_pct"),
        cooldown_hours=record.get("cooldown_hours"),
        use_session_range=record.get("use_session_range", True),
        timezone=record.get("timezone") or sched.DEFAULT_TIMEZONE,
    )


def check_alerts(
    store: PortfolioStore,
    smtp: mail.SmtpConfig | None,
    *,
    now: datetime | None = None,
    dry_run: bool = False,
    record: bool = True,
) -> tuple[mail.DeliveryResult | None, al.AlertOutcome]:
    """Evaluate every watchlist alert rule and e-mail whatever is new.

    Returns ``(delivery, outcome)``. ``delivery`` is ``None`` when there was
    nothing to send -- no rule crossed its level, which is the normal result
    of almost every run and must not read as a failure.

    Only instruments with an active rule are priced. A fifty-name watchlist
    with three alerts costs three quote requests, not fifty: this runs every
    few minutes against a free endpoint, and fetching prices nothing is
    waiting on would be the fastest way to get throttled.

    State is committed with :meth:`PortfolioStore.update_alert_state`, one row
    at a time, *after* a successful send -- so a delivery failure leaves the
    rule armed and it is retried on the next run rather than silently
    swallowed.
    """
    settings = load_alert_settings(store)
    items = wl.from_frame(store.load_watchlist())
    moment = now if now is not None else datetime.now(timezone.utc).replace(tzinfo=None)

    if not settings.enabled:
        # Switched off means the feature is off: no quotes fetched, nothing
        # evaluated, and NOT reported as a failure. A cron entry running every
        # five minutes must not log an error because the user turned alerts
        # off deliberately. `enabled` with no recipient is different -- that is
        # an intent that cannot be honoured, and it is reported below.
        return None, al.AlertOutcome(items=items)

    active = tuple(item for item in items if item.alert_enabled and item.has_target)
    if not active:
        return None, al.AlertOutcome(items=items)

    # Deferred, and quieted first: importing the data layer pulls in Streamlit
    # for its cache decorators, which log a warning per decorator at import.
    _quiet_streamlit()
    from src.ui import data_access

    tickers = tuple(dict.fromkeys(item.ticker for item in active))
    try:
        quotes, quote_errors = data_access.load_quotes(tickers)
    except Exception as exc:
        logger.exception("Quotes could not be fetched for alert checking")
        return (
            mail.DeliveryResult(
                sent=False,
                recipients=settings.recipients,
                subject="price alerts",
                error=f"Quotes could not be fetched, so no alert was checked: {exc}",
            ),
            al.AlertOutcome(items=items),
        )

    outcome = al.evaluate(items, quotes, settings, now=moment)
    # A provider error is a reason a rule could not be checked, and the user
    # must be told: silence about a rule looks exactly like "not triggered".
    skipped = tuple(outcome.skipped) + tuple(
        (ticker, f"the price could not be fetched ({message})")
        for ticker, message in quote_errors.items()
        if not any(ticker == existing for existing, _ in outcome.skipped)
    )

    # Re-arming is a state change with nothing to announce, so it is committed
    # whether or not an e-mail goes out.
    if record:
        for ticker in outcome.rearmed:
            store.update_alert_state(ticker, al.ARMED, "")

    if not outcome.events:
        return None, al.AlertOutcome(
            events=(),
            items=outcome.items,
            skipped=skipped,
            rearmed=outcome.rearmed,
            notices=outcome.notices,
        )

    if smtp is None or not settings.ok:
        reasons = list(settings.problems) or ["No SMTP configuration is available."]
        return (
            mail.DeliveryResult(
                sent=False,
                recipients=settings.recipients,
                subject=mail.alert_subject(outcome.events),
                error=" ".join(reasons),
            ),
            al.AlertOutcome(
                events=outcome.events,
                items=outcome.items,
                skipped=skipped,
                rearmed=outcome.rearmed,
                notices=outcome.notices,
            ),
        )

    message, compose_notes = mail.compose_tearsheet_email(
        settings.recipients,
        mail.alert_subject(outcome.events),
        mail.alert_summary(
            outcome.events,
            skipped=skipped,
            notices=outcome.notices,
            checked_at=moment,
            n_rules=len(active),
        ),
        (),  # an alert needs no attachment; the whole message is four lines
        config=smtp,
    )
    if message is None:
        return (
            mail.DeliveryResult(
                sent=False,
                recipients=settings.recipients,
                subject=mail.alert_subject(outcome.events),
                error=" ".join(compose_notes) or "The alert e-mail could not be composed.",
            ),
            al.AlertOutcome(
                events=outcome.events,
                items=outcome.items,
                skipped=skipped,
                rearmed=outcome.rearmed,
                notices=outcome.notices,
            ),
        )

    delivery = mail.send_message(smtp, message, dry_run=dry_run)
    delivery = mail.DeliveryResult(
        sent=delivery.sent,
        recipients=delivery.recipients,
        subject=delivery.subject,
        size_bytes=delivery.size_bytes,
        dry_run=delivery.dry_run,
        error=delivery.error,
        notes=(*compose_notes, *delivery.notes),
    )

    if delivery.sent and record:
        fired = {event.ticker for event in outcome.events}
        for item in outcome.items:
            if item.ticker in fired:
                store.update_alert_state(
                    item.ticker, item.alert_state, item.last_fired, item.direction
                )

    return delivery, al.AlertOutcome(
        events=outcome.events,
        items=outcome.items,
        skipped=skipped,
        rearmed=outcome.rearmed,
        notices=outcome.notices,
    )


def _print_alerts(store: PortfolioStore) -> None:
    """Show every alert rule and whether it can actually fire."""
    settings = load_alert_settings(store)
    items = wl.from_frame(store.load_watchlist())
    rules = [item for item in items if item.alert_enabled and item.has_target]

    print("\nPrice alerts:")
    if settings.enabled and settings.recipients:
        print(f"  delivery   ON -> {', '.join(settings.recipients)}")
    else:
        for issue in settings.problems:
            print(f"  ! {issue}")
    print(
        f"  re-arm     {settings.rearm_pct:g}% past the level"
        f"   cooldown {settings.cooldown_hours:g}h"
        f"   session range {'yes' if settings.use_session_range else 'no'}"
    )

    if not rules:
        print("  No instrument has an alert switched on. Set one on the Watchlist page.")
        return
    for item in rules:
        state = "WAITING" if item.alert_state == al.ARMED else "already through"
        print(f"\n  {item.ticker}  [{state}]")
        print(f"    {al.describe_rule(item)}")
        if item.last_fired:
            print(f"    last alert {item.last_fired:%d %b %Y %H:%M}")


def _print_list(store: PortfolioStore, now: datetime | None) -> int:
    records = store.load_schedules()
    if not records:
        print("No tearsheet schedules are configured. Add one on the Export page.")
        _print_alerts(store)
        return EXIT_OK

    print(f"{len(records)} schedule(s):")
    for record in records:
        schedule = sched.from_record(record)
        moment = now if now is not None else sched.now_in(schedule.timezone)
        state = "DUE NOW" if sched.is_due(schedule, moment) else "waiting"
        last = schedule.last_sent.strftime("%d %b %Y %H:%M") if schedule.last_sent else "never"
        print(f"\n  {schedule.name}  [{state}]")
        print(f"    portfolio  {schedule.portfolio or '(none)'}")
        print(f"    to         {', '.join(schedule.recipients) or '(nobody)'}")
        print(f"    cadence    {sched.describe(schedule, moment)}")
        print(f"    last sent  {last}")
        for issue in schedule.problems:
            print(f"    ! {issue}")
    _print_alerts(store)
    return EXIT_OK


def _parse_now(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not an ISO-8601 datetime (try 2026-09-02T08:00)"
        ) from None
    return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.reporting.cli",
        description=(
            "Send any scheduled portfolio tearsheets that are due. Intended to "
            "be run by cron or Windows Task Scheduler; safe to run as often as "
            "every few minutes, because each schedule is sent at most once per "
            "occurrence."
        ),
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="SQLite store to read schedules and portfolios from.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Show every schedule and whether it is due, then exit. Sends nothing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build the reports and validate the SMTP settings but open no "
            "connection. Reports the message size that would have been sent."
        ),
    )
    parser.add_argument(
        "--only",
        default="",
        metavar="NAME",
        help="Restrict to the tearsheet schedule with this name.",
    )
    parser.add_argument(
        "--alerts",
        action="store_true",
        help=(
            "Check watchlist price alerts only, skipping tearsheets. Alerts are "
            "checked on every run by default; this is for a second, more "
            "frequent cron entry."
        ),
    )
    parser.add_argument(
        "--no-alerts",
        action="store_true",
        help="Skip the price-alert check and process tearsheet schedules only.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Send regardless of whether it is due (still skips invalid "
            "schedules). Use with --only to send one report on demand."
        ),
    )
    parser.add_argument(
        "--now",
        type=_parse_now,
        default=None,
        metavar="ISO",
        help="Pretend it is this local time. For verifying a schedule's timing.",
    )
    parser.add_argument(
        "--allow-insecure",
        action="store_true",
        help=(
            "Permit an unencrypted SMTP session. Only for a trusted local "
            "relay; your password would otherwise travel in clear text."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log each step. Never prints credentials.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns an exit code and never raises."""
    _utf8_streams()
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not Path(args.db).is_file():
        print(
            f"No portfolio database at {args.db}. Save a portfolio in the app first, or pass --db.",
            file=sys.stderr,
        )
        return EXIT_CANNOT_START

    try:
        store = PortfolioStore(args.db)
    except Exception as exc:  # noqa: BLE001 - report, do not traceback into a cron log
        print(f"The portfolio database at {args.db} could not be opened: {exc}", file=sys.stderr)
        return EXIT_CANNOT_START

    try:
        if args.list:
            return _print_list(store, args.now)

        smtp, problems = mail.load_smtp_config(allow_insecure=args.allow_insecure)
        if smtp is None:
            # Distinguished from a misconfiguration: nothing is set up at all.
            print("\n".join(problems), file=sys.stderr)
            print("\n" + mail.secrets_template(), file=sys.stderr)
            return EXIT_CANNOT_START
        if problems:
            print("SMTP configuration problems:", file=sys.stderr)
            for issue in problems:
                print(f"  - {issue}", file=sys.stderr)
            return EXIT_CANNOT_START

        outcomes = (
            []
            if args.alerts
            else run_due(
                store,
                smtp,
                now=args.now,
                dry_run=args.dry_run,
                only=args.only,
                force=args.force,
            )
        )

        alert_delivery = None
        alert_outcome = None
        if not args.no_alerts and not args.only:
            alert_delivery, alert_outcome = check_alerts(
                store, smtp, now=args.now, dry_run=args.dry_run
            )
    finally:
        store.close()

    failures = 0
    for schedule, outcome in outcomes:
        prefix = "OK  " if (outcome.sent or outcome.dry_run) else "FAIL"
        print(f"{prefix} {schedule.name}: {outcome.summary}")
        for note in outcome.notes:
            print(f"     note: {note}")
        if not (outcome.sent or outcome.dry_run):
            failures += 1

    if alert_delivery is not None:
        prefix = "OK  " if (alert_delivery.sent or alert_delivery.dry_run) else "FAIL"
        print(f"{prefix} price alerts: {alert_delivery.summary}")
        for note in alert_delivery.notes:
            print(f"     note: {note}")
        if not (alert_delivery.sent or alert_delivery.dry_run):
            failures += 1

    if alert_outcome is not None:
        # Printed even on a clean run, and even when nothing was e-mailed: a
        # rule that could not be checked is silence that looks exactly like
        # "your level was not reached".
        for ticker, reason in alert_outcome.skipped:
            print(f"WARN {ticker}: {reason}")
        if args.verbose:
            for ticker in alert_outcome.rearmed:
                print(f"     {ticker} re-armed.")
            for notice in alert_outcome.notices:
                print(f"     note: {notice}")

    if not outcomes and alert_delivery is None:
        if args.verbose:
            print("Nothing is due and no alert levels were reached.")
        return EXIT_PARTIAL_FAILURE if failures else EXIT_OK

    return EXIT_PARTIAL_FAILURE if failures else EXIT_OK


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
