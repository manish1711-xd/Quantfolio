"""Tests for the scheduled-report runner.

The runner is the part that makes "scheduled" true rather than decorative, so
these tests exercise it as a scheduler would: through ``main(argv)``, reading
the exit code, with the analysis and the SMTP transport both replaced by fakes.
Nothing here touches the network or a real mail server, and every store is a
throwaway SQLite file in ``tmp_path``.

The behaviours under test, in order of how much they would hurt if wrong:

1. ``last_sent`` is stamped with the OCCURRENCE and only after a successful
   send -- the commit point of the idempotence guarantee.
2. A failure is reported with a non-zero exit code, never a traceback.
3. One schedule's failure does not stop the others.
4. The exit codes a scheduler actually branches on.
"""

from __future__ import annotations

import smtplib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import pytest

from src import alerts as al
from src import watchlist as wl
from src.data.store import PortfolioStore
from src.models import Quote
from src.reporting import cli
from src.reporting import email_report as mail
from src.reporting import schedule as sch

WED = datetime(2026, 9, 2, 9, 30)  # a Wednesday, 09:30

SMTP_ENV = {
    "SMTP_HOST": "smtp.example.com",
    "SMTP_PORT": "587",
    "SMTP_USERNAME": "bot@example.com",
    "SMTP_PASSWORD": "app-specific-secret",
    "SMTP_SENDER": "bot@example.com",
}


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------
class FakeResult:
    """The subset of ``AnalysisResult`` the report and the body read."""

    def __init__(self) -> None:
        self.positions = pd.DataFrame(
            {
                "ticker": ["AAA.NS", "BBB.NS"],
                "quantity": [10.0, 5.0],
                "avg_cost": [100.0, 200.0],
                "current_price": [120.0, 190.0],
                "invested": [1000.0, 1000.0],
                "current_value": [1200.0, 950.0],
                "pnl": [200.0, -50.0],
                "weight_pct": [55.8, 44.2],
                "currency": ["INR", "INR"],
            }
        )
        self.summary = {
            "current_value": 2150.0,
            "total_investment": 2000.0,
            "total_pnl": 150.0,
            "total_return_pct": 7.5,
            "n_holdings": 2,
            "base_currency": "INR",
        }
        self.risk = {"ann_return": 0.12, "ann_vol": 0.18, "sharpe": 0.7, "mdd": -0.15}
        self.benchmark: dict[str, Any] = {}
        self.attribution: dict[str, Any] = {}
        self.return_stats: dict[str, Any] = {}
        self.score: dict[str, Any] = {}
        self.ledger: dict[str, Any] = {}
        self.insights: list[Any] = []
        self.quotes: dict[str, Any] = {}
        self.profiles: dict[str, Any] = {}
        self.drawdowns: list[Any] = []
        self.fx: dict[str, Any] = {}
        self.warnings: list[str] = []
        self.config = None
        self.generated_at = datetime(2026, 9, 2, 8, 0)
        self.prices = None
        self.values = None
        self.returns = None


class Sent:
    """Records the messages a fake SMTP session accepted."""

    messages: ClassVar[list[Any]] = []
    fail: ClassVar[Exception | None] = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if type(self).fail is not None:
            raise type(self).fail

    def __enter__(self) -> Sent:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def ehlo(self) -> None:
        return None

    def starttls(self, context: Any = None) -> None:
        return None

    def login(self, username: str, password: str) -> None:
        return None

    def send_message(self, message: Any) -> dict[str, Any]:
        type(self).messages.append(message)
        return {}


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clean SMTP environment, a working fake transport, no real analysis."""
    for key in [*SMTP_ENV, "SMTP_SSL", "SMTP_STARTTLS", "SMTP_SENDER_NAME"]:
        monkeypatch.delenv(key, raising=False)
    for key, value in SMTP_ENV.items():
        monkeypatch.setenv(key, value)

    Sent.messages = []
    Sent.fail = None
    monkeypatch.setattr(smtplib, "SMTP", Sent)
    monkeypatch.setattr(smtplib, "SMTP_SSL", Sent)

    from src.ui import pipeline

    monkeypatch.setattr(pipeline, "run", lambda *_a, **_k: FakeResult())


@pytest.fixture
def db(tmp_path: Path) -> Path:
    """A store holding one saved portfolio named ``Core``."""
    path = tmp_path / "quantfolio.db"
    with PortfolioStore(path) as store:
        store.save_portfolio(
            "Core",
            pd.DataFrame(
                {
                    "ticker": ["AAA.NS", "BBB.NS"],
                    "quantity": [10.0, 5.0],
                    "buy_price": [100.0, 200.0],
                }
            ),
            None,
            "INR",
        )
    return path


def add(db: Path, *schedules: sch.Schedule) -> None:
    with PortfolioStore(db) as store:
        store.save_schedules([sch.to_record(s) for s in schedules])


def stored(db: Path) -> list[sch.Schedule]:
    with PortfolioStore(db) as store:
        return [sch.from_record(r) for r in store.load_schedules()]


def weekly(**overrides: Any) -> sch.Schedule:
    base: dict[str, Any] = {
        "name": "Weekly review",
        "portfolio": "Core",
        "recipients": ("owner@example.com",),
        "frequency": sch.WEEKLY,
        "hour": 8,
        "weekday": 0,
    }
    base.update(overrides)
    return sch.Schedule(**base)


# ---------------------------------------------------------------------------
# build_config
# ---------------------------------------------------------------------------
def test_the_analysis_config_comes_from_the_schedule() -> None:
    """So a weekly India report and a monthly US one can coexist."""
    cfg = cli.build_config(weekly(period="5y", benchmark="S&P 500", base_currency="USD"))

    assert cfg.period == "5y"
    assert cfg.benchmark_name == "S&P 500"
    assert cfg.base_currency == "USD"


def test_the_risk_free_rate_follows_the_reporting_currency() -> None:
    inr = cli.build_config(weekly(base_currency="INR")).risk_free_rate
    usd = cli.build_config(weekly(base_currency="USD")).risk_free_rate

    assert inr != usd


def test_an_unknown_currency_falls_back_rather_than_raising() -> None:
    assert cli.build_config(weekly(base_currency="XYZ")).risk_free_rate > 0


# ---------------------------------------------------------------------------
# --list
# ---------------------------------------------------------------------------
def test_listing_an_empty_store_says_so(db: Path, capsys: pytest.CaptureFixture) -> None:
    code = cli.main(["--db", str(db), "--list"])
    out = capsys.readouterr().out

    assert code == cli.EXIT_OK
    # Named as *tearsheet* schedules: the same command also checks price
    # alerts, and "no schedules" would leave a user unsure which was meant.
    assert "No tearsheet schedules are configured" in out
    assert "Price alerts:" in out


def test_listing_shows_the_cadence_recipients_and_due_state(
    db: Path, capsys: pytest.CaptureFixture
) -> None:
    add(db, weekly())

    cli.main(["--db", str(db), "--list", "--now", "2026-09-02T09:30"])
    out = capsys.readouterr().out

    assert "Weekly review" in out
    assert "owner@example.com" in out
    assert "every Monday at 08:00" in out
    assert "DUE NOW" in out
    assert "never" in out


def test_listing_reports_a_broken_schedule(db: Path, capsys: pytest.CaptureFixture) -> None:
    add(db, sch.Schedule(name="Broken", portfolio="Core", recipients=()))

    cli.main(["--db", str(db), "--list"])

    assert "No valid recipient" in capsys.readouterr().out


def test_listing_sends_nothing(db: Path) -> None:
    add(db, weekly())

    cli.main(["--db", str(db), "--list", "--now", "2026-09-02T09:30"])

    assert Sent.messages == []


def test_listing_needs_no_smtp_configuration(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Inspecting the configuration must work before mail is set up."""
    for key in SMTP_ENV:
        monkeypatch.delenv(key, raising=False)
    add(db, weekly())

    assert cli.main(["--db", str(db), "--list"]) == cli.EXIT_OK


# ---------------------------------------------------------------------------
# the happy path
# ---------------------------------------------------------------------------
def test_a_due_schedule_is_sent(db: Path) -> None:
    add(db, weekly())

    code = cli.main(["--db", str(db), "--now", "2026-09-02T09:30"])

    assert code == cli.EXIT_OK
    assert len(Sent.messages) == 1


def test_the_sent_message_is_addressed_to_the_recipients(db: Path) -> None:
    add(db, weekly(recipients=("a@x.com", "b@x.com")))

    cli.main(["--db", str(db), "--now", "2026-09-02T09:30"])

    assert Sent.messages[0]["To"] == "a@x.com, b@x.com"


def test_the_subject_carries_the_portfolio_value_and_return(db: Path) -> None:
    """A scheduled report that says only "your report" makes the reader open
    an attachment to learn nothing changed."""
    add(db, weekly())

    cli.main(["--db", str(db), "--now", "2026-09-02T09:30"])
    subject = Sent.messages[0]["Subject"]

    assert "Core" in subject
    assert "2,150" in subject
    assert "+7.50%" in subject


def test_the_tearsheet_is_attached_as_html(db: Path) -> None:
    add(db, weekly())

    cli.main(["--db", str(db), "--now", "2026-09-02T09:30"])
    attachments = list(Sent.messages[0].iter_attachments())

    assert len(attachments) == 1
    assert attachments[0].get_content_type() == "text/html"
    assert attachments[0].get_filename().endswith(".html")
    assert b"<html" in attachments[0].get_payload(decode=True).lower()


def test_the_body_is_readable_text_not_just_an_attachment(db: Path) -> None:
    """Attachments get stripped by gateways and ignored on a phone, so the
    numbers a reader wants at a glance have to be in the text."""
    add(db, weekly())

    cli.main(["--db", str(db), "--now", "2026-09-02T09:30"])
    body = Sent.messages[0].get_body(("plain",)).get_content()

    assert "POSITION" in body
    assert "INR 2,150" in body
    assert "not investment advice" in body


def test_the_body_names_the_schedule_that_produced_it(db: Path) -> None:
    add(db, weekly())

    cli.main(["--db", str(db), "--now", "2026-09-02T09:30"])
    body = Sent.messages[0].get_body(("plain",)).get_content()

    assert "Weekly review" in body
    assert "every Monday at 08:00" in body


def test_no_credential_appears_anywhere_in_the_message(db: Path) -> None:
    add(db, weekly())

    cli.main(["--db", str(db), "--now", "2026-09-02T09:30"])

    assert SMTP_ENV["SMTP_PASSWORD"] not in Sent.messages[0].as_string()


# ---------------------------------------------------------------------------
# idempotence -- the commit point
# ---------------------------------------------------------------------------
def test_a_successful_send_is_recorded_against_the_occurrence(db: Path) -> None:
    """Not against the send time. Stamping "now" for a late run would make
    tomorrow's slot look already-delivered."""
    add(db, weekly())

    cli.main(["--db", str(db), "--now", "2026-09-02T15:00"])

    assert stored(db)[0].last_sent == datetime(2026, 8, 31, 8, 0)


def test_running_the_command_again_sends_nothing(db: Path) -> None:
    """The guarantee a cron entry depends on."""
    add(db, weekly())

    first = cli.main(["--db", str(db), "--now", "2026-09-02T09:30"])
    second = cli.main(["--db", str(db), "--now", "2026-09-02T09:35"])

    assert (first, second) == (cli.EXIT_OK, cli.EXIT_OK)
    assert len(Sent.messages) == 1


def test_a_quarter_hourly_scheduler_sends_one_email_per_day(db: Path) -> None:
    """Ninety-six invocations of the actual command, one e-mail."""
    add(db, sch.starting_next_slot(weekly(frequency=sch.DAILY, hour=8), datetime(2026, 9, 2, 0, 0)))

    for tick in range(96):
        moment = datetime(2026, 9, 2, 0, 0) + timedelta(minutes=15 * tick)
        cli.main(["--db", str(db), "--now", moment.isoformat()])

    assert len(Sent.messages) == 1


def test_a_failed_send_is_not_recorded_so_it_is_retried(db: Path) -> None:
    """The ordering that matters: record after sending, never before. A crash
    between the two costs one duplicate; the reverse loses the report."""
    add(db, weekly())
    Sent.fail = OSError("network down")

    code = cli.main(["--db", str(db), "--now", "2026-09-02T09:30"])

    assert code == cli.EXIT_PARTIAL_FAILURE
    assert stored(db)[0].last_sent is None

    Sent.fail = None
    assert cli.main(["--db", str(db), "--now", "2026-09-02T09:30"]) == cli.EXIT_OK
    assert len(Sent.messages) == 1


def test_a_failed_analysis_is_not_recorded_either(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    add(db, weekly())
    from src.ui import pipeline

    def explode(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("no market data")

    monkeypatch.setattr(pipeline, "run", explode)

    code = cli.main(["--db", str(db), "--now", "2026-09-02T09:30"])

    assert code == cli.EXIT_PARTIAL_FAILURE
    assert stored(db)[0].last_sent is None
    assert Sent.messages == []


def test_run_one_can_send_without_recording(db: Path) -> None:
    """What the app's "send a test now" button uses: proving the plumbing
    works must not consume the occurrence the user configured."""
    schedule = weekly()
    add(db, schedule)
    smtp, _problems = mail.load_smtp_config()

    with PortfolioStore(db) as store:
        outcome = cli.run_one(schedule, store, smtp, now=WED, record=False)

    assert outcome.sent
    assert stored(db)[0].last_sent is None


def test_recording_touches_only_the_named_schedule(db: Path) -> None:
    """A targeted UPDATE, so the runner cannot clobber a schedule the user
    edited while the report was being built."""
    add(db, weekly(name="A", frequency=sch.DAILY, hour=8), weekly(name="B", enabled=False))

    cli.main(["--db", str(db), "--now", "2026-09-02T09:30"])
    by_name = {s.name: s for s in stored(db)}

    assert by_name["A"].last_sent is not None
    assert by_name["B"].last_sent is None
    assert by_name["B"].enabled is False


# ---------------------------------------------------------------------------
# selection: due, --only, --force
# ---------------------------------------------------------------------------
def test_a_schedule_that_is_not_due_is_skipped(db: Path) -> None:
    add(db, weekly().mark_sent(datetime(2026, 8, 31, 8, 0)))

    code = cli.main(["--db", str(db), "--now", "2026-09-02T09:30"])

    assert code == cli.EXIT_OK
    assert Sent.messages == []


def test_a_paused_schedule_is_skipped(db: Path) -> None:
    add(db, weekly(enabled=False))

    cli.main(["--db", str(db), "--now", "2026-09-02T09:30"])

    assert Sent.messages == []


def test_force_sends_regardless_of_timing(db: Path) -> None:
    add(db, weekly().mark_sent(datetime(2026, 8, 31, 8, 0)))

    code = cli.main(["--db", str(db), "--force", "--now", "2026-09-02T09:30"])

    assert code == cli.EXIT_OK
    assert len(Sent.messages) == 1


def test_force_still_refuses_an_unusable_schedule(db: Path) -> None:
    """--force ignores the clock, never the validity."""
    add(db, sch.Schedule(name="Broken", portfolio="Core", recipients=()))

    code = cli.main(["--db", str(db), "--force", "--now", "2026-09-02T09:30"])

    assert code == cli.EXIT_OK
    assert Sent.messages == []


def test_only_restricts_to_one_schedule(db: Path) -> None:
    add(db, weekly(name="A"), weekly(name="B"))

    cli.main(["--db", str(db), "--only", "B", "--now", "2026-09-02T09:30"])

    assert len(Sent.messages) == 1
    assert stored(db)[1].last_sent is not None
    assert stored(db)[0].last_sent is None


def test_only_matches_a_name_case_insensitively(db: Path) -> None:
    add(db, weekly(name="Weekly Review"))

    cli.main(["--db", str(db), "--only", "weekly review", "--now", "2026-09-02T09:30"])

    assert len(Sent.messages) == 1


def test_only_with_an_unknown_name_sends_nothing(db: Path) -> None:
    add(db, weekly())

    code = cli.main(["--db", str(db), "--only", "nope", "--now", "2026-09-02T09:30"])

    assert code == cli.EXIT_OK
    assert Sent.messages == []


# ---------------------------------------------------------------------------
# --dry-run
# ---------------------------------------------------------------------------
def test_a_dry_run_builds_the_report_but_sends_nothing(
    db: Path, capsys: pytest.CaptureFixture
) -> None:
    add(db, weekly())

    code = cli.main(["--db", str(db), "--dry-run", "--now", "2026-09-02T09:30"])

    assert code == cli.EXIT_OK
    assert Sent.messages == []
    assert "Dry run" in capsys.readouterr().out


def test_a_dry_run_does_not_consume_the_occurrence(db: Path) -> None:
    add(db, weekly())

    cli.main(["--db", str(db), "--dry-run", "--now", "2026-09-02T09:30"])

    assert stored(db)[0].last_sent is None


def test_a_dry_run_reports_the_message_size(db: Path, capsys: pytest.CaptureFixture) -> None:
    add(db, weekly())

    cli.main(["--db", str(db), "--dry-run", "--now", "2026-09-02T09:30"])

    assert "KB" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# failures are reported, one bad schedule never stops the rest
# ---------------------------------------------------------------------------
def test_a_deleted_portfolio_is_named_and_the_saved_ones_listed(
    db: Path, capsys: pytest.CaptureFixture
) -> None:
    """The two cases are different mistakes: a portfolio that never existed
    and one that exists but is empty."""
    add(db, weekly(name="Ghost", portfolio="Deleted"))

    code = cli.main(["--db", str(db), "--now", "2026-09-02T09:30"])
    out = capsys.readouterr().out

    assert code == cli.EXIT_PARTIAL_FAILURE
    assert "No saved portfolio named 'Deleted'" in out
    assert "Core" in out


def test_one_failing_schedule_does_not_stop_the_others(db: Path) -> None:
    add(db, weekly(name="Ghost", portfolio="Deleted"), weekly(name="Good"))

    code = cli.main(["--db", str(db), "--now", "2026-09-02T09:30"])

    assert code == cli.EXIT_PARTIAL_FAILURE
    assert len(Sent.messages) == 1
    assert {s.name: s.last_sent is not None for s in stored(db)} == {
        "Ghost": False,
        "Good": True,
    }


def test_a_delivery_failure_is_reported_with_its_cause(
    db: Path, capsys: pytest.CaptureFixture
) -> None:
    add(db, weekly())
    Sent.fail = smtplib.SMTPAuthenticationError(535, b"denied")

    code = cli.main(["--db", str(db), "--now", "2026-09-02T09:30"])
    out = capsys.readouterr().out

    assert code == cli.EXIT_PARTIAL_FAILURE
    assert "FAIL" in out
    assert "app-specific password" in out


def test_a_delivery_failure_never_prints_the_password(
    db: Path, capsys: pytest.CaptureFixture
) -> None:
    add(db, weekly())
    Sent.fail = smtplib.SMTPAuthenticationError(535, b"denied")

    cli.main(["--db", str(db), "--now", "2026-09-02T09:30"])
    captured = capsys.readouterr()

    assert SMTP_ENV["SMTP_PASSWORD"] not in captured.out
    assert SMTP_ENV["SMTP_PASSWORD"] not in captured.err


def test_a_warning_containing_a_rupee_sign_does_not_kill_the_runner(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On Windows ``print`` encodes with the locale codepage, and the runner
    dying with a UnicodeEncodeError while reporting a problem would be a
    spectacularly unhelpful way to fail."""
    add(db, weekly())
    from src.ui import pipeline

    def with_rupees(*_a: Any, **_k: Any) -> FakeResult:
        result = FakeResult()
        result.warnings = ["Skipping AAA.NS: quote of ₹1,25,000 looks stale — em dash ok"]
        return result

    monkeypatch.setattr(pipeline, "run", with_rupees)

    assert cli.main(["--db", str(db), "--now", "2026-09-02T09:30"]) == cli.EXIT_OK
    body = Sent.messages[0].get_body(("plain",)).get_content()
    assert "₹1,25,000" in body


# ---------------------------------------------------------------------------
# exit codes a scheduler branches on
# ---------------------------------------------------------------------------
def test_nothing_due_exits_zero(db: Path) -> None:
    assert cli.main(["--db", str(db), "--now", "2026-09-02T09:30"]) == cli.EXIT_OK


def test_a_missing_database_cannot_start(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    code = cli.main(["--db", str(tmp_path / "nope.db")])

    assert code == cli.EXIT_CANNOT_START
    assert "No portfolio database" in capsys.readouterr().err


def test_no_smtp_settings_cannot_start_and_shows_the_template(
    db: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    for key in SMTP_ENV:
        monkeypatch.delenv(key, raising=False)
    add(db, weekly())

    code = cli.main(["--db", str(db), "--now", "2026-09-02T09:30"])
    err = capsys.readouterr().err

    assert code == cli.EXIT_CANNOT_START
    assert "No SMTP settings found" in err
    assert "[smtp]" in err


def test_a_broken_smtp_configuration_cannot_start(
    db: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Distinct from "not configured": here the user tried and got it wrong,
    so the message names what is wrong rather than showing the template."""
    monkeypatch.setenv("SMTP_SENDER", "not-an-address")
    add(db, weekly())

    code = cli.main(["--db", str(db), "--now", "2026-09-02T09:30"])
    err = capsys.readouterr().err

    assert code == cli.EXIT_CANNOT_START
    assert "does not look like an address" in err


def test_sending_a_password_unencrypted_cannot_start(
    db: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setenv("SMTP_PORT", "1025")
    monkeypatch.setenv("SMTP_STARTTLS", "false")
    add(db, weekly())

    code = cli.main(["--db", str(db), "--now", "2026-09-02T09:30"])

    assert code == cli.EXIT_CANNOT_START
    assert "unencrypted" in capsys.readouterr().err


def test_the_insecure_flag_permits_a_plaintext_relay(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SMTP_PORT", "1025")
    monkeypatch.setenv("SMTP_STARTTLS", "false")
    add(db, weekly())

    code = cli.main(["--db", str(db), "--allow-insecure", "--now", "2026-09-02T09:30"])

    assert code == cli.EXIT_OK
    assert len(Sent.messages) == 1


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------
def test_an_unparseable_now_is_rejected_with_a_hint(db: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--db", str(db), "--now", "yesterday"])

    assert exc.value.code == 2


def test_now_accepts_an_iso_datetime_with_a_zone(db: Path) -> None:
    add(db, weekly())

    code = cli.main(["--db", str(db), "--now", "2026-09-02T09:30+05:30"])

    assert code == cli.EXIT_OK
    assert len(Sent.messages) == 1


def test_verbose_logs_without_printing_credentials(db: Path, capsys: pytest.CaptureFixture) -> None:
    add(db, weekly())

    cli.main(["--db", str(db), "-v", "--now", "2026-09-02T09:30"])
    captured = capsys.readouterr()

    assert SMTP_ENV["SMTP_PASSWORD"] not in captured.out + captured.err


# ---------------------------------------------------------------------------
# price alerts through the runner
# ---------------------------------------------------------------------------
def alert_rule(**overrides: Any) -> wl.WatchItem:
    base: dict[str, Any] = {
        "ticker": "AAA.NS",
        "target_price": 110.0,
        "direction": al.ABOVE,
        "alert_enabled": True,
        "reference_price": 100.0,
    }
    base.update(overrides)
    return wl.WatchItem(**base)


def seed_alerts(
    db: Path,
    *items: wl.WatchItem,
    enabled: bool = True,
    recipients: str = "owner@example.com",
    rearm_pct: float = 1.0,
    cooldown_hours: float = 12.0,
    use_session_range: bool = True,
) -> None:
    with PortfolioStore(db) as store:
        store.save_watchlist(wl.to_frame(items))
        store.save_alert_settings(
            {
                "enabled": enabled,
                "recipients": recipients,
                "rearm_pct": rearm_pct,
                "cooldown_hours": cooldown_hours,
                "use_session_range": use_session_range,
            }
        )


def stub_quotes(monkeypatch: pytest.MonkeyPatch, prices: dict[str, Any], errors=None):
    """Replace the cached quote loader with fixed prices. No network."""
    from src.ui import data_access

    def loader(tickers):
        quotes = {t: prices[t] for t in tickers if t in prices}
        return quotes, dict(errors or {})

    monkeypatch.setattr(data_access, "load_quotes", loader)


def price(
    ticker: str = "AAA.NS",
    value: float = 111.0,
    high: float | None = None,
    low: float | None = None,
) -> Quote:
    stamp = datetime(2026, 9, 3, 10, 15)
    return Quote(ticker, value, "INR", stamp, stamp, True, day_high=high, day_low=low)


def watch_state(db: Path) -> dict[str, wl.WatchItem]:
    with PortfolioStore(db) as store:
        return {i.ticker: i for i in wl.from_frame(store.load_watchlist())}


# -- the happy path ---------------------------------------------------------
def test_a_crossed_level_is_emailed(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seed_alerts(db, alert_rule())
    stub_quotes(monkeypatch, {"AAA.NS": price()})

    code = cli.main(["--db", str(db), "--alerts"])

    assert code == cli.EXIT_OK
    assert len(Sent.messages) == 1
    assert "AAA.NS" in Sent.messages[0]["Subject"]


def test_the_alert_email_goes_to_the_configured_recipients(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_alerts(db, alert_rule(), recipients="a@x.com, b@x.com")
    stub_quotes(monkeypatch, {"AAA.NS": price()})

    cli.main(["--db", str(db), "--alerts"])

    assert Sent.messages[0]["To"] == "a@x.com, b@x.com"


def test_the_alert_body_carries_the_level_and_the_caveats(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_alerts(db, alert_rule(note="breakout watch"))
    stub_quotes(monkeypatch, {"AAA.NS": price()})

    cli.main(["--db", str(db), "--alerts"])
    body = Sent.messages[0].get_body(("plain",)).get_content()

    assert "rose to or above INR 110.00" in body
    assert "breakout watch" in body
    assert "delayed" in body
    assert "not investment advice" in body


def test_an_uncrossed_level_sends_nothing_and_succeeds(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The normal outcome of almost every run, and it must not read as a
    failure."""
    seed_alerts(db, alert_rule())
    stub_quotes(monkeypatch, {"AAA.NS": price(value=105.0)})

    code = cli.main(["--db", str(db), "--alerts"])

    assert code == cli.EXIT_OK
    assert Sent.messages == []


# -- fire once --------------------------------------------------------------
def test_a_crossed_level_is_recorded_so_it_is_not_resent(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_alerts(db, alert_rule())
    stub_quotes(monkeypatch, {"AAA.NS": price()})

    cli.main(["--db", str(db), "--alerts"])

    assert watch_state(db)["AAA.NS"].alert_state == al.TRIGGERED
    assert watch_state(db)["AAA.NS"].last_fired is not None


def test_running_the_alert_check_again_sends_nothing(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_alerts(db, alert_rule())
    stub_quotes(monkeypatch, {"AAA.NS": price()})

    for _ in range(5):
        cli.main(["--db", str(db), "--alerts"])

    assert len(Sent.messages) == 1


def test_a_five_minute_cron_sends_one_email_for_a_parked_price(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The headline guarantee, through the actual command: 96 invocations with
    the price sitting above the level produce one e-mail."""
    seed_alerts(db, alert_rule())
    stub_quotes(monkeypatch, {"AAA.NS": price()})

    for _ in range(96):
        cli.main(["--db", str(db), "--alerts"])

    assert len(Sent.messages) == 1


def test_a_failed_send_leaves_the_rule_armed_for_a_retry(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Record after sending, never before: a delivery failure must not swallow
    the alert."""
    seed_alerts(db, alert_rule())
    stub_quotes(monkeypatch, {"AAA.NS": price()})
    Sent.fail = OSError("network down")

    code = cli.main(["--db", str(db), "--alerts"])

    assert code == cli.EXIT_PARTIAL_FAILURE
    assert watch_state(db)["AAA.NS"].alert_state == al.ARMED

    Sent.fail = None
    assert cli.main(["--db", str(db), "--alerts"]) == cli.EXIT_OK
    assert len(Sent.messages) == 1


def test_a_retreat_rearms_the_rule_without_sending(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_alerts(db, alert_rule())
    stub_quotes(monkeypatch, {"AAA.NS": price()})
    cli.main(["--db", str(db), "--alerts"])

    stub_quotes(monkeypatch, {"AAA.NS": price(value=105.0)})
    code = cli.main(["--db", str(db), "--alerts"])

    assert code == cli.EXIT_OK
    assert len(Sent.messages) == 1
    assert watch_state(db)["AAA.NS"].alert_state == al.ARMED


def test_a_rearmed_rule_fires_again_on_the_next_crossing(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_alerts(db, alert_rule(), cooldown_hours=0.0)
    stub_quotes(monkeypatch, {"AAA.NS": price()})
    cli.main(["--db", str(db), "--alerts"])
    stub_quotes(monkeypatch, {"AAA.NS": price(value=105.0)})
    cli.main(["--db", str(db), "--alerts"])
    stub_quotes(monkeypatch, {"AAA.NS": price(value=112.0)})
    cli.main(["--db", str(db), "--alerts"])

    assert len(Sent.messages) == 2


def test_recording_an_alert_does_not_disturb_the_users_note(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A targeted UPDATE, so the runner cannot undo an edit made in the
    browser while it was fetching quotes."""
    seed_alerts(db, alert_rule(note="do not lose me"), alert_rule(ticker="BBB.NS"))
    stub_quotes(monkeypatch, {"AAA.NS": price(), "BBB.NS": price("BBB.NS", 105.0)})

    cli.main(["--db", str(db), "--alerts"])
    state = watch_state(db)

    assert state["AAA.NS"].note == "do not lose me"
    assert state["AAA.NS"].target_price == 110.0
    assert state["BBB.NS"].alert_state == al.ARMED


# -- what it refuses to do silently ----------------------------------------
def test_a_rule_without_a_quote_is_reported_on_stdout(
    db: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Printed even on a clean run: silence about a rule looks exactly like
    "your level was not reached"."""
    seed_alerts(db, alert_rule())
    stub_quotes(monkeypatch, {})

    cli.main(["--db", str(db), "--alerts"])
    out = capsys.readouterr().out

    assert "WARN AAA.NS" in out
    assert "no quote" in out


def test_a_provider_error_is_reported_as_a_reason(
    db: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    seed_alerts(db, alert_rule())
    stub_quotes(monkeypatch, {}, errors={"AAA.NS": "delisted"})

    cli.main(["--db", str(db), "--alerts"])
    out = capsys.readouterr().out

    assert "AAA.NS" in out
    assert "no quote" in out or "could not be fetched" in out


def test_a_rule_without_a_direction_is_reported(
    db: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    seed_alerts(db, alert_rule(direction="", reference_price=None))
    stub_quotes(monkeypatch, {"AAA.NS": price()})

    cli.main(["--db", str(db), "--alerts"])
    out = capsys.readouterr().out

    assert "no direction" in out
    assert Sent.messages == []


def test_an_unchecked_rule_travels_in_the_email_too(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_alerts(db, alert_rule(), alert_rule(ticker="BBB.NS", target_price=50.0))
    stub_quotes(monkeypatch, {"AAA.NS": price()})  # BBB.NS has no quote

    cli.main(["--db", str(db), "--alerts"])
    body = Sent.messages[0].get_body(("plain",)).get_content()

    assert "RULES THAT COULD NOT BE CHECKED" in body
    assert "BBB.NS" in body


def test_a_quote_fetch_that_explodes_is_reported_not_raised(
    db: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    seed_alerts(db, alert_rule())
    from src.ui import data_access

    def explode(_tickers):
        raise RuntimeError("provider is down")

    monkeypatch.setattr(data_access, "load_quotes", explode)

    code = cli.main(["--db", str(db), "--alerts"])

    assert code == cli.EXIT_PARTIAL_FAILURE
    assert "could not be fetched" in capsys.readouterr().out


# -- switches and scoping ---------------------------------------------------
def test_alerting_switched_off_checks_nothing_and_is_not_a_failure(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Off means off. A cron entry running every five minutes must not log an
    error because the user deliberately switched alerts off."""
    seed_alerts(db, alert_rule(), enabled=False)
    asked: list[tuple[str, ...]] = []
    from src.ui import data_access

    monkeypatch.setattr(data_access, "load_quotes", lambda t: (asked.append(tuple(t)), ({}, {}))[1])

    code = cli.main(["--db", str(db), "--alerts"])

    assert code == cli.EXIT_OK
    assert Sent.messages == []
    assert asked == []  # not even a quote request


def test_alerting_switched_off_leaves_the_rule_state_alone(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """So switching alerts back on later does not find every rule already
    triggered and stay permanently silent."""
    seed_alerts(db, alert_rule(), enabled=False)
    stub_quotes(monkeypatch, {"AAA.NS": price()})

    cli.main(["--db", str(db), "--alerts"])

    assert watch_state(db)["AAA.NS"].alert_state == al.ARMED


def test_alerts_with_no_recipient_report_rather_than_sending(
    db: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    seed_alerts(db, alert_rule(), recipients="")
    stub_quotes(monkeypatch, {"AAA.NS": price()})

    code = cli.main(["--db", str(db), "--alerts"])

    assert code == cli.EXIT_PARTIAL_FAILURE
    assert "recipient" in capsys.readouterr().out
    assert Sent.messages == []


def test_an_undeliverable_alert_is_not_recorded_as_sent(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise switching alerts on later would find every rule already
    triggered and stay permanently silent."""
    seed_alerts(db, alert_rule(), recipients="")
    stub_quotes(monkeypatch, {"AAA.NS": price()})

    cli.main(["--db", str(db), "--alerts"])

    assert watch_state(db)["AAA.NS"].alert_state == al.ARMED


def test_no_alert_rules_at_all_costs_no_quote_request(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fifty-name watchlist with no alerts must not price fifty instruments
    every five minutes against a free endpoint."""
    seed_alerts(db, wl.WatchItem("AAA.NS", target_price=110.0))  # alerting off
    asked: list[tuple[str, ...]] = []
    from src.ui import data_access

    def loader(tickers):
        asked.append(tuple(tickers))
        return {}, {}

    monkeypatch.setattr(data_access, "load_quotes", loader)

    cli.main(["--db", str(db), "--alerts"])

    assert asked == []


def test_only_instruments_with_a_rule_are_priced(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seed_alerts(
        db,
        alert_rule(ticker="AAA.NS"),
        wl.WatchItem("BBB.NS", target_price=50.0),  # no alert
        wl.WatchItem("CCC.NS"),  # no target
    )
    asked: list[tuple[str, ...]] = []
    from src.ui import data_access

    def loader(tickers):
        asked.append(tuple(tickers))
        return {}, {}

    monkeypatch.setattr(data_access, "load_quotes", loader)

    cli.main(["--db", str(db), "--alerts"])

    assert asked == [("AAA.NS",)]


def test_no_alerts_flag_skips_the_check(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seed_alerts(db, alert_rule())
    stub_quotes(monkeypatch, {"AAA.NS": price()})

    code = cli.main(["--db", str(db), "--no-alerts"])

    assert code == cli.EXIT_OK
    assert Sent.messages == []


def test_alerts_are_checked_alongside_tearsheets_by_default(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One cron entry covers both, which is why running it often is safe."""
    add(db, weekly())
    seed_alerts(db, alert_rule())
    stub_quotes(monkeypatch, {"AAA.NS": price()})

    code = cli.main(["--db", str(db), "--now", "2026-09-02T09:30"])

    assert code == cli.EXIT_OK
    assert len(Sent.messages) == 2
    subjects = " ".join(str(m["Subject"]) for m in Sent.messages)
    assert "tearsheet" in subjects
    assert "price alert" in subjects


def test_the_alerts_flag_skips_tearsheets(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    add(db, weekly())
    seed_alerts(db, alert_rule())
    stub_quotes(monkeypatch, {"AAA.NS": price()})

    cli.main(["--db", str(db), "--alerts", "--now", "2026-09-02T09:30"])

    assert len(Sent.messages) == 1
    assert "price alert" in str(Sent.messages[0]["Subject"])


def test_a_dry_run_checks_the_rules_but_sends_nothing(
    db: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    seed_alerts(db, alert_rule())
    stub_quotes(monkeypatch, {"AAA.NS": price()})

    code = cli.main(["--db", str(db), "--alerts", "--dry-run"])

    assert code == cli.EXIT_OK
    assert Sent.messages == []
    assert "Dry run" in capsys.readouterr().out


def test_a_dry_run_does_not_consume_the_alert(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seed_alerts(db, alert_rule())
    stub_quotes(monkeypatch, {"AAA.NS": price()})

    cli.main(["--db", str(db), "--alerts", "--dry-run"])

    assert watch_state(db)["AAA.NS"].alert_state == al.ARMED


def test_check_alerts_can_run_without_recording(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seed_alerts(db, alert_rule())
    stub_quotes(monkeypatch, {"AAA.NS": price()})
    smtp, _problems = mail.load_smtp_config()

    with PortfolioStore(db) as store:
        delivery, outcome = cli.check_alerts(store, smtp, record=False)

    assert delivery is not None and delivery.sent
    assert len(outcome.events) == 1
    assert watch_state(db)["AAA.NS"].alert_state == al.ARMED


# -- --list -----------------------------------------------------------------
def test_listing_shows_every_alert_rule(
    db: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    seed_alerts(db, alert_rule())

    cli.main(["--db", str(db), "--list"])
    out = capsys.readouterr().out

    assert "Price alerts:" in out
    assert "AAA.NS" in out
    assert "rises to or above 110.00" in out
    assert "owner@example.com" in out


def test_listing_names_a_rule_that_cannot_fire(db: Path, capsys: pytest.CaptureFixture) -> None:
    """A target with alerting on but no direction looks armed and never
    fires, so it must not be listed as merely waiting."""
    seed_alerts(db, alert_rule(direction="", reference_price=None))

    cli.main(["--db", str(db), "--list"])

    assert "no direction set" in capsys.readouterr().out


def test_listing_says_when_no_alert_is_configured(db: Path, capsys: pytest.CaptureFixture) -> None:
    cli.main(["--db", str(db), "--list"])

    assert "No instrument has an alert switched on" in capsys.readouterr().out


def test_listing_reports_the_sensitivity_settings(db: Path, capsys: pytest.CaptureFixture) -> None:
    seed_alerts(db, alert_rule(), rearm_pct=2.5, cooldown_hours=6.0)

    cli.main(["--db", str(db), "--list"])
    out = capsys.readouterr().out

    assert "re-arm     2.5%" in out
    assert "cooldown 6h" in out


def test_listing_alerts_needs_no_smtp_configuration(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in SMTP_ENV:
        monkeypatch.delenv(key, raising=False)
    seed_alerts(db, alert_rule())

    assert cli.main(["--db", str(db), "--list"]) == cli.EXIT_OK


def test_no_credential_appears_in_an_alert_email(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seed_alerts(db, alert_rule())
    stub_quotes(monkeypatch, {"AAA.NS": price()})

    cli.main(["--db", str(db), "--alerts"])

    assert SMTP_ENV["SMTP_PASSWORD"] not in Sent.messages[0].as_string()
