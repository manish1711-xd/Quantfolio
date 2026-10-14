"""Tests for SMTP composition and delivery.

Nothing here opens a socket. ``smtplib.SMTP`` and ``SMTP_SSL`` are replaced by
a recording fake, which is how the transport decisions (STARTTLS versus
implicit SSL, verified context, login only when credentials exist) can be
asserted at all -- they are invisible from the outside.

Two themes run through the file:

* **No credential ever escapes.** Asserted on ``repr``, ``str``, the error
  paths and the persisted record.
* **Every failure is reported, never raised.** A scheduled job that dies with
  a traceback into a log nobody reads is indistinguishable from one that never
  ran, so ``send_message`` returns a result for every SMTP failure mode.
"""

from __future__ import annotations

import smtplib
import ssl
from datetime import datetime, timezone
from typing import Any, ClassVar

import pytest

from src.reporting import email_report as mail

SECRET = "correct-horse-battery-staple"


def config(**overrides: Any) -> mail.SmtpConfig:
    base: dict[str, Any] = {
        "host": "smtp.example.com",
        "port": 587,
        "username": "bot@example.com",
        "password": SECRET,
        "sender": "bot@example.com",
    }
    base.update(overrides)
    return mail.SmtpConfig(**base)


class FakeSMTP:
    """Records what a real ``smtplib.SMTP`` would have been asked to do."""

    instances: ClassVar[list[FakeSMTP]] = []
    refuse: ClassVar[dict[str, tuple[int, bytes]]] = {}
    raise_on_login: ClassVar[Exception | None] = None
    raise_on_send: ClassVar[Exception | None] = None
    raise_on_connect: ClassVar[Exception | None] = None

    def __init__(self, host: str, port: int, timeout: float = 0.0, context: Any = None) -> None:
        if type(self).raise_on_connect is not None:
            raise type(self).raise_on_connect
        self.host = host
        self.port = port
        self.timeout = timeout
        self.context = context
        self.started_tls = False
        self.tls_context: Any = None
        self.logged_in_as: str | None = None
        self.password_seen: str | None = None
        self.sent: list[Any] = []
        self.ehlo_count = 0
        type(self).instances.append(self)

    # -- context manager --------------------------------------------------
    def __enter__(self) -> FakeSMTP:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    # -- the SMTP surface used by the module ------------------------------
    def ehlo(self) -> None:
        self.ehlo_count += 1

    def starttls(self, context: Any = None) -> None:
        self.started_tls = True
        self.tls_context = context

    def login(self, username: str, password: str) -> None:
        if type(self).raise_on_login is not None:
            raise type(self).raise_on_login
        self.logged_in_as = username
        self.password_seen = password

    def send_message(self, message: Any) -> dict[str, tuple[int, bytes]]:
        if type(self).raise_on_send is not None:
            raise type(self).raise_on_send
        self.sent.append(message)
        return dict(type(self).refuse)


@pytest.fixture(autouse=True)
def fake_smtp(monkeypatch: pytest.MonkeyPatch) -> type[FakeSMTP]:
    """Install the fake for both transports and reset its class state."""
    FakeSMTP.instances = []
    FakeSMTP.refuse = {}
    FakeSMTP.raise_on_login = None
    FakeSMTP.raise_on_send = None
    FakeSMTP.raise_on_connect = None
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(smtplib, "SMTP_SSL", FakeSMTP)
    return FakeSMTP


def message(recipients: Any = "to@example.com", **kwargs: Any) -> Any:
    built, _notes = mail.compose_tearsheet_email(
        recipients,
        kwargs.pop("subject", "Subject"),
        kwargs.pop("body", "Body"),
        kwargs.pop("attachments", ()),
        config=kwargs.pop("config", config()),
        **kwargs,
    )
    return built


# ---------------------------------------------------------------------------
# secrets never leak
# ---------------------------------------------------------------------------
def test_the_password_is_absent_from_repr_and_str() -> None:
    """A traceback prints repr. If the password is in it, the password is in
    every crash log the application ever produces."""
    cfg = config()

    assert SECRET not in repr(cfg)
    assert SECRET not in str(cfg)
    assert SECRET not in f"{cfg}"


def test_repr_says_a_password_is_set_without_hinting_at_it() -> None:
    text = repr(config())

    assert "<set, 28 chars>" in text
    assert SECRET[:3] not in text


def test_redact_never_reveals_a_prefix() -> None:
    """Even two leading characters narrow a brute-force search."""
    redacted = mail.redact(SECRET)

    assert SECRET not in redacted
    assert SECRET[:2] not in redacted
    assert "28" in redacted


def test_redact_distinguishes_unset_from_empty() -> None:
    assert mail.redact("") == "<not set>"
    assert mail.redact(None) == "<not set>"


def test_the_password_is_absent_from_the_composed_message() -> None:
    built = message()

    assert SECRET not in built.as_string()


def test_the_secrets_template_is_a_placeholder_not_a_credential() -> None:
    template = mail.secrets_template()

    assert "[smtp]" in template
    assert "app-specific" in template
    assert "never commit" in template


# ---------------------------------------------------------------------------
# loading configuration
# ---------------------------------------------------------------------------
def test_flat_environment_style_keys_are_read() -> None:
    cfg, problems = mail.load_smtp_config(
        {
            "SMTP_HOST": "smtp.gmail.com",
            "SMTP_PORT": "587",
            "SMTP_USERNAME": "me@example.com",
            "SMTP_PASSWORD": "pw",
        }
    )

    assert problems == ()
    assert cfg is not None
    assert (cfg.host, cfg.port, cfg.username) == ("smtp.gmail.com", 587, "me@example.com")


def test_a_nested_smtp_table_is_unwrapped() -> None:
    """``st.secrets`` gives a ``[smtp]`` table; the environment gives flat
    keys. One loader must serve both without the caller reshaping anything."""
    cfg, problems = mail.load_smtp_config(
        {"smtp": {"host": "smtp.example.com", "port": 465, "sender": "a@b.com"}}
    )

    assert problems == ()
    assert cfg is not None
    assert (cfg.host, cfg.port) == ("smtp.example.com", 465)


def test_keys_are_matched_case_insensitively() -> None:
    cfg, _problems = mail.load_smtp_config({"Smtp_Host": "h.example.com", "SENDER": "a@b.com"})

    assert cfg is not None
    assert cfg.host == "h.example.com"


def test_nothing_configured_is_reported_as_nothing_not_as_broken() -> None:
    """A fresh checkout has no mail set up. That is the normal state and must
    not be presented as a misconfiguration."""
    cfg, problems = mail.load_smtp_config({})

    assert cfg is None
    assert problems and "No SMTP settings found" in problems[0]


def test_the_sender_defaults_to_the_username() -> None:
    cfg, _problems = mail.load_smtp_config({"host": "h.example.com", "username": "me@x.com"})

    assert cfg is not None
    assert cfg.sender == "me@x.com"


def test_port_465_is_inferred_as_implicit_ssl() -> None:
    """The port is the one setting users reliably copy from a help page, so it
    is the right thing to infer the transport from."""
    cfg, _problems = mail.load_smtp_config(
        {"host": "h.example.com", "sender": "a@b.com", "port": 465}
    )

    assert cfg is not None
    assert cfg.use_ssl
    assert not cfg.use_starttls


def test_port_587_is_inferred_as_starttls() -> None:
    cfg, _problems = mail.load_smtp_config(
        {"host": "h.example.com", "sender": "a@b.com", "port": 587}
    )

    assert cfg is not None
    assert cfg.use_starttls
    assert not cfg.use_ssl


def test_an_explicit_transport_setting_overrides_the_inference() -> None:
    cfg, _problems = mail.load_smtp_config(
        {
            "host": "h.example.com",
            "sender": "a@b.com",
            "port": 465,
            "starttls": "true",
            "ssl": "false",
        }
    )

    assert cfg is not None
    assert cfg.use_starttls
    assert not cfg.use_ssl


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("true", True), ("YES", True), ("1", True), ("off", False), ("0", False), ("", None)],
)
def test_boolean_settings_accept_the_spellings_people_actually_write(
    raw: str, expected: bool | None
) -> None:
    cfg, _problems = mail.load_smtp_config(
        {"host": "h.example.com", "sender": "a@b.com", "port": 25, "ssl": raw}
    )

    assert cfg is not None
    # "" is absent as far as _lookup is concerned, so the port inference wins.
    assert cfg.use_ssl is (False if expected is None else expected)


def test_an_unparseable_port_falls_back_to_587() -> None:
    cfg, _problems = mail.load_smtp_config(
        {"host": "h.example.com", "sender": "a@b.com", "port": "five eighty seven"}
    )

    assert cfg is not None
    assert cfg.port == 587


def test_the_environment_is_the_default_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMTP_HOST", "env.example.com")
    monkeypatch.setenv("SMTP_SENDER", "env@example.com")

    cfg, problems = mail.load_smtp_config()

    assert problems == ()
    assert cfg is not None
    assert cfg.host == "env.example.com"


def test_a_source_that_raises_on_lookup_is_survived() -> None:
    """``st.secrets`` raises rather than returning None for a missing key."""

    class Hostile(dict):
        def get(self, key: object, default: object = None) -> object:
            raise KeyError(key)

    source = Hostile({"host": "h.example.com", "sender": "a@b.com"})
    cfg, problems = mail.load_smtp_config(source)

    assert problems == ()
    assert cfg is not None
    assert cfg.host == "h.example.com"


# ---------------------------------------------------------------------------
# configuration validity
# ---------------------------------------------------------------------------
def test_a_complete_configuration_is_ok() -> None:
    assert config().ok


def test_a_missing_host_is_a_problem() -> None:
    assert any("host" in p for p in config(host="").problems)


def test_a_missing_sender_is_a_problem() -> None:
    assert any("sender" in p for p in config(sender="", username="").problems)


def test_a_sender_without_an_at_sign_is_a_problem() -> None:
    assert any("does not look like" in p for p in config(sender="not-an-address").problems)


@pytest.mark.parametrize("port", [0, -1, 70000])
def test_an_impossible_port_is_a_problem(port: int) -> None:
    assert any("TCP port" in p for p in config(port=port).problems)


def test_asking_for_both_ssl_and_starttls_is_a_problem() -> None:
    """465 is implicit SSL, 587 upgrades. Doing both is a misconfiguration
    that produces a confusing handshake error rather than an obvious one."""
    problems = config(use_ssl=True, use_starttls=True).problems

    assert any("choose one" in p for p in problems)


def test_a_password_over_an_unencrypted_connection_is_refused() -> None:
    """The security control that matters most here. "It didn't work so I
    turned off TLS" is how mail passwords end up on the wire in clear text."""
    problems = config(use_ssl=False, use_starttls=False).problems

    assert any("unencrypted" in p for p in problems)


def test_the_insecure_escape_hatch_must_be_asked_for_explicitly() -> None:
    cfg = config(use_ssl=False, use_starttls=False, allow_insecure=True)

    assert cfg.ok


def test_no_password_means_no_encryption_complaint() -> None:
    """An unauthenticated local relay has no credential to expose."""
    cfg = config(password="", use_ssl=False, use_starttls=False)

    assert cfg.ok


def test_a_display_name_produces_a_proper_from_header() -> None:
    assert config(sender_name="QuantFolio").from_header == "QuantFolio <bot@example.com>"


def test_no_display_name_leaves_the_bare_address() -> None:
    assert config().from_header == "bot@example.com"


# ---------------------------------------------------------------------------
# the text body
# ---------------------------------------------------------------------------
class Result:
    """The subset of ``AnalysisResult`` the summary reads."""

    def __init__(self, **fields: Any) -> None:
        self.summary = fields.pop("summary", {})
        self.risk = fields.pop("risk", {})
        self.benchmark = fields.pop("benchmark", {})
        self.ledger = fields.pop("ledger", {})
        self.warnings = fields.pop("warnings", [])
        self.generated_at = fields.pop("generated_at", datetime(2026, 9, 2, 8, 0))


def test_the_body_carries_the_headline_numbers() -> None:
    body = mail.text_summary(
        Result(
            summary={
                "current_value": 1_250_000,
                "total_investment": 1_000_000,
                "total_pnl": 250_000,
                "total_return_pct": 25.0,
                "n_holdings": 7,
                "base_currency": "INR",
            },
            risk={"ann_return": 0.184, "ann_vol": 0.152, "sharpe": 1.24, "mdd": -0.22},
        ),
        "Core Equity",
    )

    assert "Core Equity" in body
    assert "INR 1,250,000" in body
    assert "+25.00%" in body
    assert "1.24" in body
    assert "+18.40%" in body  # a ratio rendered as a percentage


def test_an_uncomputed_statistic_prints_na_not_zero() -> None:
    """The rule the whole project runs on: a missing number is never a 0.00
    that reads as a real measurement."""
    body = mail.text_summary(Result(), "Empty")

    assert "n/a" in body
    assert "0.00%" not in body


def test_the_body_states_it_is_not_advice() -> None:
    body = mail.text_summary(Result(), "Core")

    assert "not investment advice" in body
    assert "educational" in body


def test_the_body_names_its_data_as_delayed() -> None:
    assert "delayed" in mail.text_summary(Result(), "Core")


def test_data_warnings_travel_with_the_report() -> None:
    body = mail.text_summary(Result(warnings=["Skipping XYZ: no usable quote."]), "Core")

    assert "DATA NOTES" in body
    assert "Skipping XYZ" in body


def test_a_long_warning_list_is_capped_but_the_count_is_stated() -> None:
    """Capped, not silently truncated: a reader must know to open the
    attachment rather than assuming there were five."""
    body = mail.text_summary(Result(warnings=[f"note {i}" for i in range(12)]), "Core")

    assert "and 7 more" in body


def test_the_benchmark_block_appears_only_when_there_is_one() -> None:
    with_bench = mail.text_summary(
        Result(benchmark={"name": "NIFTY 50", "beta": 0.94, "alpha": 0.021}), "Core"
    )
    without = mail.text_summary(Result(), "Core")

    assert "VERSUS NIFTY 50" in with_bench
    assert "0.94" in with_bench
    assert "VERSUS" not in without


def test_xirr_appears_only_when_a_ledger_exists() -> None:
    assert "XIRR" in mail.text_summary(Result(ledger={"xirr": 0.171}), "Core")
    assert "XIRR" not in mail.text_summary(Result(), "Core")


def test_a_schedule_note_is_included_when_given() -> None:
    body = mail.text_summary(Result(), "Core", schedule_note="Scheduled 'Weekly'")

    assert "Scheduled 'Weekly'" in body


def test_a_result_with_no_timestamp_falls_back_to_now() -> None:
    body = mail.text_summary(Result(generated_at=None), "Core")

    assert "Generated" in body
    assert "UTC" in body


def test_an_aware_timestamp_is_rendered() -> None:
    body = mail.text_summary(
        Result(generated_at=datetime(2026, 9, 2, 8, 0, tzinfo=timezone.utc)), "Core"
    )

    assert "02 Sep 2026" in body


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------
def test_a_composed_message_carries_the_expected_headers() -> None:
    built = message(config=config(sender_name="QuantFolio"))

    assert built["From"] == "QuantFolio <bot@example.com>"
    assert built["To"] == "to@example.com"
    assert built["Subject"] == "Subject"
    assert built["Date"]
    assert built["Message-ID"]
    assert built["Auto-Submitted"] == "auto-generated"


def test_auto_submitted_marks_it_as_machine_generated() -> None:
    """Stops a vacation autoresponder from replying to a robot forever."""
    assert message()["Auto-Submitted"] == "auto-generated"


def test_multiple_recipients_are_joined_into_one_header() -> None:
    built = message(["a@x.com", "b@x.com"])

    assert built["To"] == "a@x.com, b@x.com"


def test_an_invalid_recipient_is_reported_and_the_rest_still_go() -> None:
    built, notes = mail.compose_tearsheet_email(["good@x.com", "nope"], "S", "B", config=config())

    assert built is not None
    assert built["To"] == "good@x.com"
    assert notes and "nope" in notes[0]


def test_no_valid_recipient_composes_nothing_and_says_why() -> None:
    built, notes = mail.compose_tearsheet_email(["nope"], "S", "B", config=config())

    assert built is None
    assert any("No valid recipient" in n for n in notes)


def test_no_sender_composes_nothing_and_says_why() -> None:
    built, notes = mail.compose_tearsheet_email(["a@x.com"], "S", "B")

    assert built is None
    assert any("No sender" in n for n in notes)


def test_a_reply_to_is_set_when_asked_for() -> None:
    built = message(reply_to="human@example.com")

    assert built["Reply-To"] == "human@example.com"


def test_the_attachment_arrives_with_its_filename_and_type() -> None:
    built = message(attachments=[mail.Attachment("tearsheet.html", b"<html></html>")])
    parts = list(built.iter_attachments())

    assert len(parts) == 1
    assert parts[0].get_filename() == "tearsheet.html"
    assert parts[0].get_content_type() == "text/html"


def test_attachment_content_survives_intact() -> None:
    payload = "<html><body>₹1,25,000 — done</body></html>".encode()
    built = message(attachments=[mail.Attachment("r.html", payload)])
    part = next(iter(built.iter_attachments()))

    assert part.get_payload(decode=True) == payload


def test_an_oversized_attachment_is_dropped_but_the_email_still_goes() -> None:
    """A report that silently bounces at the recipient's gateway is worse than
    a short mail that explains itself."""
    huge = mail.Attachment("huge.html", b"x" * (mail.MAX_ATTACHMENT_BYTES + 1))
    built, notes = mail.compose_tearsheet_email(["a@x.com"], "S", "Body", [huge], config=config())

    assert built is not None
    assert list(built.iter_attachments()) == []
    assert any("attachment omitted" in n for n in notes)


def test_the_recipient_is_told_in_the_body_that_the_attachment_was_dropped() -> None:
    huge = mail.Attachment("huge.html", b"x" * (mail.MAX_ATTACHMENT_BYTES + 1))
    built, _notes = mail.compose_tearsheet_email(["a@x.com"], "S", "Body", [huge], config=config())

    assert "was not attached" in built.get_body(("plain",)).get_content()


def test_an_attachment_exactly_at_the_limit_is_kept() -> None:
    at_limit = mail.Attachment("ok.html", b"x" * mail.MAX_ATTACHMENT_BYTES)
    built, _notes = mail.compose_tearsheet_email(
        ["a@x.com"], "S", "Body", [at_limit], config=config()
    )

    assert len(list(built.iter_attachments())) == 1


def test_the_body_is_plain_text_utf8() -> None:
    built = message(body="₹1,25,000 — fine")
    part = built.get_body(("plain",))

    assert part.get_content_type() == "text/plain"
    assert "₹1,25,000" in part.get_content()


# ---------------------------------------------------------------------------
# delivery: the happy path and the transport decisions
# ---------------------------------------------------------------------------
def test_a_message_is_delivered_and_reported_as_sent() -> None:
    outcome = mail.send_message(config(), message())

    assert outcome.sent
    assert outcome.recipients == ("to@example.com",)
    assert outcome.error == ""
    assert "Sent" in outcome.summary


def test_starttls_is_negotiated_on_port_587(fake_smtp: type[FakeSMTP]) -> None:
    mail.send_message(config(port=587, use_starttls=True), message())
    client = fake_smtp.instances[-1]

    assert client.started_tls
    assert client.ehlo_count == 2  # before and after the upgrade, as required


def test_the_tls_context_verifies_certificates(fake_smtp: type[FakeSMTP]) -> None:
    """An unverified TLS session protects against a passive listener but not
    against the active attacker who can actually read the password."""
    mail.send_message(config(), message())
    context = fake_smtp.instances[-1].tls_context

    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname


def test_implicit_ssl_skips_starttls_and_still_verifies(fake_smtp: type[FakeSMTP]) -> None:
    mail.send_message(config(port=465, use_ssl=True, use_starttls=False), message())
    client = fake_smtp.instances[-1]

    assert not client.started_tls
    assert isinstance(client.context, ssl.SSLContext)
    assert client.context.verify_mode == ssl.CERT_REQUIRED


def test_login_happens_only_when_both_username_and_password_exist(
    fake_smtp: type[FakeSMTP],
) -> None:
    mail.send_message(config(password="", use_starttls=True), message())

    assert fake_smtp.instances[-1].logged_in_as is None


def test_the_configured_timeout_reaches_the_client(fake_smtp: type[FakeSMTP]) -> None:
    """Without a timeout a cron job can hang forever on an unresponsive host,
    and the next run stacks up behind it."""
    mail.send_message(config(timeout=12.0), message())

    assert fake_smtp.instances[-1].timeout == 12.0


# ---------------------------------------------------------------------------
# delivery: dry run
# ---------------------------------------------------------------------------
def test_a_dry_run_opens_no_connection(fake_smtp: type[FakeSMTP]) -> None:
    """The default for anything a page render could trigger: sending mail is
    outward-facing and irreversible."""
    outcome = mail.send_message(config(), message(), dry_run=True)

    assert fake_smtp.instances == []
    assert outcome.dry_run
    assert not outcome.sent
    assert "Dry run" in outcome.summary


def test_a_dry_run_still_reports_the_size() -> None:
    outcome = mail.send_message(
        config(), message(attachments=[mail.Attachment("r.html", b"x" * 5000)]), dry_run=True
    )

    assert outcome.size_bytes > 5000


def test_a_dry_run_still_validates_the_configuration(fake_smtp: type[FakeSMTP]) -> None:
    outcome = mail.send_message(config(host=""), message(), dry_run=True)

    assert not outcome.dry_run
    assert "host" in outcome.error
    assert fake_smtp.instances == []


# ---------------------------------------------------------------------------
# delivery: every failure is reported, never raised
# ---------------------------------------------------------------------------
def test_an_invalid_configuration_is_refused_before_connecting(
    fake_smtp: type[FakeSMTP],
) -> None:
    outcome = mail.send_message(config(host=""), message())

    assert not outcome.sent
    assert fake_smtp.instances == []


def test_authentication_failure_names_the_app_password_cause(
    fake_smtp: type[FakeSMTP],
) -> None:
    """Gmail and Outlook both reject an account password here. Saying so is
    the difference between a five-minute fix and an hour of guessing."""
    fake_smtp.raise_on_login = smtplib.SMTPAuthenticationError(535, b"nope")

    outcome = mail.send_message(config(), message())

    assert not outcome.sent
    assert "refused the login" in outcome.error
    assert "app-specific password" in outcome.error


def test_authentication_failure_does_not_echo_the_password(
    fake_smtp: type[FakeSMTP],
) -> None:
    fake_smtp.raise_on_login = smtplib.SMTPAuthenticationError(535, b"nope")

    outcome = mail.send_message(config(), message())

    assert SECRET not in outcome.error
    assert SECRET not in outcome.summary


def test_all_recipients_refused_is_reported(fake_smtp: type[FakeSMTP]) -> None:
    fake_smtp.raise_on_send = smtplib.SMTPRecipientsRefused({"to@example.com": (550, b"no")})

    outcome = mail.send_message(config(), message())

    assert not outcome.sent
    assert "refused" in outcome.error


def test_a_tls_handshake_failure_suggests_the_port(fake_smtp: type[FakeSMTP]) -> None:
    fake_smtp.raise_on_connect = ssl.SSLError("wrong version number")

    outcome = mail.send_message(config(), message())

    assert not outcome.sent
    assert "TLS handshake" in outcome.error
    assert "465" in outcome.error


def test_an_unreachable_host_is_reported(fake_smtp: type[FakeSMTP]) -> None:
    fake_smtp.raise_on_connect = OSError("no route to host")

    outcome = mail.send_message(config(), message())

    assert not outcome.sent
    assert "failed" in outcome.error
    assert "no route to host" in outcome.error


def test_a_generic_smtp_exception_is_reported(fake_smtp: type[FakeSMTP]) -> None:
    fake_smtp.raise_on_send = smtplib.SMTPDataError(451, b"try later")

    outcome = mail.send_message(config(), message())

    assert not outcome.sent
    assert outcome.error


def test_a_partial_refusal_is_not_reported_as_a_clean_send(
    fake_smtp: type[FakeSMTP],
) -> None:
    """Saying "sent" would hide the address that was not -- exactly the
    failure a scheduled report has to surface."""
    fake_smtp.refuse = {"b@x.com": (550, b"unknown user")}

    outcome = mail.send_message(config(), message(["a@x.com", "b@x.com"]))

    assert outcome.sent  # a@x.com did go
    assert outcome.recipients == ("a@x.com",)
    assert any("refused" in n for n in outcome.notes)


def test_an_unencrypted_send_is_flagged_in_the_result(fake_smtp: type[FakeSMTP]) -> None:
    """Using the escape hatch is recorded so it cannot be quietly forgotten."""
    cfg = config(use_ssl=False, use_starttls=False, allow_insecure=True)

    outcome = mail.send_message(cfg, message(config=cfg))

    assert outcome.sent
    assert any("UNENCRYPTED" in n for n in outcome.notes)


def test_send_message_never_raises(fake_smtp: type[FakeSMTP]) -> None:
    """The blanket property. Anything that escapes here dies in a cron log
    nobody reads, which is indistinguishable from never having run."""
    for failure in (
        smtplib.SMTPAuthenticationError(535, b"x"),
        smtplib.SMTPServerDisconnected("gone"),
        smtplib.SMTPConnectError(421, b"busy"),
        OSError("broken pipe"),
        ssl.SSLError("bad cert"),
    ):
        fake_smtp.raise_on_connect = failure
        outcome = mail.send_message(config(), message())
        assert not outcome.sent
        assert outcome.error


# ---------------------------------------------------------------------------
# alert subject and body
# ---------------------------------------------------------------------------
def event(**overrides: Any) -> Any:
    from src import alerts as al

    base: dict[str, Any] = {
        "ticker": "RELIANCE.NS",
        "direction": al.ABOVE,
        "level": 1350.0,
        "price": 1355.0,
        "currency": "INR",
        "through_by_pct": 0.37,
        "evidence": al.EVIDENCE_LAST,
        "as_of": datetime(2026, 9, 3, 10, 15),
        "is_intraday": True,
        "note": "",
    }
    base.update(overrides)
    return al.AlertEvent(**base)


def test_the_subject_names_the_instrument_and_the_level() -> None:
    """An alert read on a phone lock screen should already say what happened,
    so the subject carries the facts rather than "you have an alert"."""
    subject = mail.alert_subject([event()])

    assert "RELIANCE.NS" in subject
    assert ">= 1,350.00" in subject
    assert "1 price alert" in subject


def test_the_subject_uses_the_right_comparison_for_a_downward_rule() -> None:
    subject = mail.alert_subject([event(direction="below", level=1200.0)])

    assert "<= 1,200.00" in subject


def test_the_subject_pluralises() -> None:
    subject = mail.alert_subject([event(), event(ticker="TCS.NS")])

    assert subject.startswith("2 price alerts")


def test_a_long_batch_is_truncated_with_a_count() -> None:
    """A subject line that runs to 200 characters is shown as its first forty."""
    subject = mail.alert_subject([event(ticker=f"T{i}.NS") for i in range(7)])

    assert "and 4 more" in subject
    assert subject.count("·") == 3


def test_an_empty_batch_has_a_subject_anyway() -> None:
    assert mail.alert_subject([]) == "No price alerts"


def test_an_event_with_no_level_still_produces_a_subject() -> None:
    subject = mail.alert_subject([event(level=float("nan"))])

    assert "your level" in subject


# -- the body ---------------------------------------------------------------
def test_the_alert_body_states_what_happened() -> None:
    body = mail.alert_summary([event()], checked_at=datetime(2026, 9, 3, 10, 30))

    assert "1 watchlist level reached" in body
    assert "RELIANCE.NS rose to or above INR 1,350.00" in body
    assert "INR 1,355.00" in body
    assert "+0.37% past your level" in body


def test_a_downward_alert_reads_naturally() -> None:
    body = mail.alert_summary([event(direction="below", level=1200.0, price=1195.0)])

    assert "fell to or below" in body


def test_the_body_carries_the_price_timestamp() -> None:
    """Against a ~15-minute-delayed feed the timestamp is the difference
    between "had been reached" and "is being reached"."""
    body = mail.alert_summary([event(as_of=datetime(2026, 9, 3, 10, 15))])

    assert "price from 03 Sep 2026 10:15" in body


def test_a_session_high_trigger_warns_that_the_price_may_have_moved_back() -> None:
    """The distinction that decides whether the alert is still actionable."""
    from src import alerts as al

    body = mail.alert_summary([event(evidence=al.EVIDENCE_HIGH, price=1360.0)])

    assert "session high" in body
    assert "may have since moved back" in body


def test_a_last_price_trigger_carries_no_such_warning() -> None:
    body = mail.alert_summary([event()])

    assert "may have since moved back" not in body


def test_the_users_own_note_travels_with_the_alert() -> None:
    body = mail.alert_summary([event(note="breakout watch")])

    assert "your note breakout watch" in body


def test_an_unchecked_rule_is_reported_as_prominently_as_an_alert() -> None:
    """The failure an alert system must never have: silence about a rule looks
    exactly like "your level was not reached"."""
    body = mail.alert_summary([event()], skipped=[("INFY.NS", "no quote was available")])

    assert "RULES THAT COULD NOT BE CHECKED" in body
    assert "INFY.NS: no quote was available" in body
    assert "does not mean your" in body


def test_a_suppressed_repeat_is_mentioned() -> None:
    body = mail.alert_summary([event()], notices=["TCS.NS reached your level again"])

    assert "ALSO" in body
    assert "TCS.NS reached your level again" in body


def test_the_body_states_the_feed_is_delayed() -> None:
    body = mail.alert_summary([event()])

    assert "delayed" in body
    assert "~15 minutes" in body


def test_the_alert_body_states_it_is_not_advice() -> None:
    body = mail.alert_summary([event()])

    assert "not investment advice" in body
    assert "numbers you typed" in body


def test_the_body_states_the_between_checks_gap() -> None:
    """The limitation a user would otherwise discover by missing a spike."""
    body = mail.alert_summary([event()])

    assert "between two checks" in body
    assert "session high or low" in body


def test_a_body_with_no_events_still_reports_the_unchecked_rules() -> None:
    """Sent when something could not be evaluated even though nothing fired."""
    body = mail.alert_summary([], skipped=[("INFY.NS", "no direction is set")])

    assert "No watchlist levels reached" in body
    assert "INFY.NS" in body


def test_the_body_names_how_many_rules_were_checked() -> None:
    body = mail.alert_summary([event()], n_rules=4)

    assert "against 4 active rule(s)" in body


def test_the_body_falls_back_to_now_without_a_timestamp() -> None:
    body = mail.alert_summary([event()])

    assert "Checked" in body


def test_a_missing_currency_does_not_produce_a_stray_space() -> None:
    body = mail.alert_summary([event(currency="")])

    assert "above 1,350.00" in body


def test_the_alert_email_needs_no_attachment() -> None:
    """The whole message is a few lines; an attachment would be ceremony."""
    built, _notes = mail.compose_tearsheet_email(
        ["a@x.com"],
        mail.alert_subject([event()]),
        mail.alert_summary([event()]),
        (),
        config=config(),
    )

    assert list(built.iter_attachments()) == []
    assert built.get_body(("plain",)).get_content_type() == "text/plain"
