"""SMTP delivery for the HTML tearsheet.

Scope, deliberately narrow: this module composes an e-mail and hands it to an
SMTP server. It does not schedule (:mod:`src.reporting.schedule` does), it does
not build the report (:mod:`src.reporting.html_report` does) and it never
imports Streamlit, because the process that actually sends is a cron job with
no browser attached.

Credentials
-----------
There is exactly one way to supply them: a mapping handed in by the caller,
which in practice is ``st.secrets`` or ``os.environ``. No file in this
repository holds a password, no password is written to the SQLite store, and
``SmtpConfig.__repr__`` redacts it so a traceback or a debug log cannot leak
one. :func:`secrets_template` prints the block a user should add to their own
``.streamlit/secrets.toml`` — which ``.gitignore`` already excludes.

Transport security
------------------
The connection is TLS-verified against the system trust store
(``ssl.create_default_context``), with certificate and hostname checking left
ON. Port 465 is treated as implicit SSL, 587 and 25 as STARTTLS. Sending a
password over an unencrypted connection is refused unless the caller passes
``allow_insecure=True`` and means it — the common "it didn't work so I turned
off TLS" fix is how mail passwords end up on the wire in clear text.

Attachment size
---------------
A tearsheet carrying Plotly figures is measured in megabytes, and base64
inflates an attachment by about a third on the wire. Past
``MAX_ATTACHMENT_BYTES`` the attachment is dropped and the e-mail still goes
out, saying in the body that it was dropped and why: a report that silently
bounces at the recipient's gateway is worse than a short mail that explains
itself.
"""

from __future__ import annotations

import logging
import math
import os
import smtplib
import ssl
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import formataddr, formatdate, make_msgid
from typing import Any

from src.reporting.schedule import clean_recipients

logger = logging.getLogger(__name__)

__all__ = [
    "SMTP_FIELDS",
    "MAX_ATTACHMENT_BYTES",
    "SmtpConfig",
    "Attachment",
    "DeliveryResult",
    "load_smtp_config",
    "secrets_template",
    "redact",
    "text_summary",
    "alert_subject",
    "alert_summary",
    "compose_tearsheet_email",
    "send_message",
]

#: ``attribute -> accepted key names``, checked in order, case-insensitively.
#: Both the flat ``SMTP_HOST`` form (environment variables) and the bare
#: ``host`` form (a ``[smtp]`` table in ``secrets.toml``) are accepted, so one
#: loader serves both sources without the caller reshaping anything.
SMTP_FIELDS: dict[str, tuple[str, ...]] = {
    "host": ("smtp_host", "host", "server", "smtp_server"),
    "port": ("smtp_port", "port"),
    "username": ("smtp_username", "smtp_user", "username", "user"),
    "password": ("smtp_password", "smtp_pass", "password"),
    "sender": ("smtp_sender", "smtp_from", "sender", "from", "from_address"),
    "sender_name": ("smtp_sender_name", "sender_name", "from_name"),
    "use_ssl": ("smtp_ssl", "use_ssl", "ssl"),
    "use_starttls": ("smtp_starttls", "use_starttls", "starttls", "tls"),
    "timeout": ("smtp_timeout", "timeout"),
}

#: Raw attachment ceiling. Base64 adds ~33%, so 15 MiB of HTML arrives as about
#: 20 MB on the wire — under the 25 MB that Gmail, Outlook and most corporate
#: gateways enforce, with room for the rest of the message.
MAX_ATTACHMENT_BYTES: int = 15 * 1024 * 1024

#: Ports on which an unencrypted session is expected to be upgraded.
_STARTTLS_PORTS = (587, 25, 2525)

#: The port that means "TLS from the first byte" rather than "upgrade later".
_IMPLICIT_SSL_PORT = 465

_TRUE = {"1", "true", "yes", "on", "y", "t"}
_FALSE = {"0", "false", "no", "off", "n", "f", ""}

_NAN = float("nan")


def redact(value: object) -> str:
    """A loggable stand-in for a secret: length only, never a prefix.

    Showing even the first two characters of a password narrows a brute-force
    search, and a "helpfully" truncated secret in a log is still a secret in a
    log.
    """
    text = str(value or "")
    return f"<set, {len(text)} chars>" if text else "<not set>"


def _lookup(source: Mapping[str, Any], names: Sequence[str]) -> Any:
    """First present value among ``names``, matched case-insensitively."""
    lowered = {str(k).strip().lower(): v for k, v in source.items()}
    for name in names:
        if name in lowered:
            value = lowered[name]
            if value is not None and str(value).strip() != "":
                return value
    return None


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SmtpConfig:
    """Everything needed to open one authenticated SMTP session.

    Frozen, and never persisted: it is rebuilt from the caller's secret store
    on each use and discarded. Nothing in this dataclass reaches the SQLite
    store or a log line.
    """

    host: str
    port: int = 587
    username: str = ""
    password: str = ""
    sender: str = ""
    sender_name: str = ""
    use_ssl: bool = False
    use_starttls: bool = True
    timeout: float = 30.0
    #: Escape hatch for a relay that genuinely has no TLS (a container sidecar
    #: on localhost, say). Never defaulted on, and reported in the delivery
    #: result so it cannot be quietly forgotten.
    allow_insecure: bool = False

    def __repr__(self) -> str:
        """Redacted on purpose: a traceback must not carry the password."""
        return (
            f"SmtpConfig(host={self.host!r}, port={self.port}, "
            f"username={self.username!r}, password={redact(self.password)}, "
            f"sender={self.sender!r}, use_ssl={self.use_ssl}, "
            f"use_starttls={self.use_starttls})"
        )

    __str__ = __repr__

    @property
    def from_header(self) -> str:
        """``Name <address>`` when a display name is configured."""
        address = self.sender or self.username
        return formataddr((self.sender_name, address)) if self.sender_name else address

    @property
    def encrypted(self) -> bool:
        return bool(self.use_ssl or self.use_starttls)

    @property
    def problems(self) -> tuple[str, ...]:
        """Everything that would stop a send, as sentences a user can act on."""
        out: list[str] = []
        if not self.host:
            out.append("No SMTP host is configured.")
        if not 1 <= int(self.port) <= 65535:
            out.append(f"Port {self.port} is not a valid TCP port.")
        if not (self.sender or self.username):
            out.append("No sender address is configured.")
        address = self.sender or self.username
        if address and "@" not in address:
            out.append(f"The sender address {address!r} does not look like an address.")
        if self.use_ssl and self.use_starttls:
            out.append(
                "Both implicit SSL and STARTTLS are enabled; choose one "
                "(465 is usually SSL, 587 is usually STARTTLS)."
            )
        if self.password and not self.encrypted and not self.allow_insecure:
            out.append(
                "A password would be sent over an unencrypted connection. Enable "
                "STARTTLS or SSL, or set allow_insecure explicitly if this really "
                "is a trusted local relay."
            )
        return tuple(out)

    @property
    def ok(self) -> bool:
        return not self.problems


def load_smtp_config(
    source: Mapping[str, Any] | None = None,
    *,
    allow_insecure: bool = False,
) -> tuple[SmtpConfig | None, tuple[str, ...]]:
    """Build an :class:`SmtpConfig` from a secret mapping.

    ``source`` is normally ``st.secrets`` inside the app and ``os.environ`` in
    a cron job; it defaults to the environment. A nested ``smtp`` table is
    unwrapped automatically, so both

        [smtp]
        host = "smtp.gmail.com"

    and a flat ``SMTP_HOST=...`` environment variable work unchanged.

    Returns ``(None, reason)`` when nothing is configured at all — which is
    distinct from "configured wrongly", because the first is the normal state
    of a fresh checkout and should not be presented as an error.
    """
    mapping: Mapping[str, Any] = source if source is not None else os.environ
    try:
        nested = mapping.get("smtp")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001 - st.secrets raises on a missing key
        nested = None
    if isinstance(nested, Mapping):
        merged: dict[str, Any] = {str(k): v for k, v in mapping.items()}
        merged.update({str(k): v for k, v in nested.items()})
        mapping = merged

    values = {attr: _lookup(mapping, names) for attr, names in SMTP_FIELDS.items()}
    if values["host"] is None and values["sender"] is None:
        return None, (
            "No SMTP settings found. Add an [smtp] block to "
            ".streamlit/secrets.toml, or set the SMTP_* environment variables "
            "before running the scheduled send.",
        )

    port = _as_int(values["port"], 587)
    # Infer the transport from the port when it was not stated: the port is the
    # one setting users reliably copy from their provider's help page.
    use_ssl = _as_bool(values["use_ssl"], port == _IMPLICIT_SSL_PORT)
    starttls_default = port in _STARTTLS_PORTS and not use_ssl
    use_starttls = _as_bool(values["use_starttls"], starttls_default)

    config = SmtpConfig(
        host=str(values["host"] or "").strip(),
        port=port,
        username=str(values["username"] or "").strip(),
        password=str(values["password"] or ""),
        sender=str(values["sender"] or values["username"] or "").strip(),
        sender_name=str(values["sender_name"] or "").strip(),
        use_ssl=use_ssl,
        use_starttls=use_starttls,
        timeout=float(_as_int(values["timeout"], 30)),
        allow_insecure=bool(allow_insecure),
    )
    return config, config.problems


def secrets_template() -> str:
    """The TOML block a user should paste into their own secrets file.

    Shown in the UI *instead of* a form that would write a password into the
    application's database. ``.streamlit/secrets.toml`` is already git-ignored;
    a password typed into a Streamlit text input and saved to SQLite would sit
    in a file inside ``data/`` forever.
    """
    return (
        "# .streamlit/secrets.toml   (git-ignored - never commit this file)\n"
        "[smtp]\n"
        'host = "smtp.gmail.com"\n'
        "port = 587\n"
        'username = "you@example.com"\n'
        'password = "an-app-specific-password, not your login password"\n'
        'sender = "you@example.com"\n'
        'sender_name = "QuantFolio"\n'
        "\n"
        "# Or, for a cron job, export these instead:\n"
        "#   SMTP_HOST  SMTP_PORT  SMTP_USERNAME  SMTP_PASSWORD  SMTP_SENDER\n"
    )


# ---------------------------------------------------------------------------
# message composition
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Attachment:
    """One file to hang off the message."""

    filename: str
    payload: bytes
    maintype: str = "text"
    subtype: str = "html"

    @property
    def size(self) -> int:
        return len(self.payload)


@dataclass(frozen=True)
class DeliveryResult:
    """What happened to one send attempt. Never raises; always reports."""

    sent: bool
    recipients: tuple[str, ...] = ()
    subject: str = ""
    size_bytes: int = 0
    dry_run: bool = False
    error: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def summary(self) -> str:
        if self.dry_run:
            return (
                f"Dry run: {self.subject!r} would go to "
                f"{len(self.recipients)} recipient(s), {self.size_bytes / 1024:.0f} KB."
            )
        if self.sent:
            return f"Sent {self.subject!r} to {', '.join(self.recipients)}."
        return f"Not sent: {self.error}"


def _num(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return _NAN
    return out if math.isfinite(out) else _NAN


def _pct(value: Any, decimals: int = 2, *, ratio: bool = False) -> str:
    number = _num(value)
    if math.isnan(number):
        return "n/a"
    if ratio:
        number *= 100.0
    return f"{number:+.{decimals}f}%"


def _money(value: Any, currency: str) -> str:
    number = _num(value)
    if math.isnan(number):
        return "n/a"
    return f"{currency} {number:,.0f}"


def _ratio(value: Any, decimals: int = 2) -> str:
    number = _num(value)
    return "n/a" if math.isnan(number) else f"{number:.{decimals}f}"


def text_summary(
    result: Any,
    portfolio_name: str = "Portfolio",
    *,
    schedule_note: str = "",
) -> str:
    """A plain-text digest of one analysis run.

    This is the message *body*, not a fallback for it. The HTML tearsheet is an
    attachment, and attachments get stripped by gateways, blocked on phones and
    ignored on a commute — so the numbers a reader actually wants at a glance
    are in the text, and the attachment is for the detail.

    Every figure comes from ``result``; a statistic the pipeline could not
    compute prints ``n/a`` rather than a zero. Deliberately not HTML: a
    multipart body carrying a second rendering of the same numbers is one more
    place for them to disagree.
    """
    summary = dict(getattr(result, "summary", {}) or {})
    risk = dict(getattr(result, "risk", {}) or {})
    bench = dict(getattr(result, "benchmark", {}) or {})
    ledger = dict(getattr(result, "ledger", {}) or {})
    warnings = list(getattr(result, "warnings", []) or [])
    currency = str(summary.get("base_currency") or "INR")

    generated = getattr(result, "generated_at", None)
    stamp = (
        generated.strftime("%d %b %Y %H:%M")
        if isinstance(generated, datetime)
        else datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC")
    )

    lines: list[str] = [
        f"{portfolio_name} — portfolio tearsheet",
        f"Generated {stamp}",
    ]
    if schedule_note:
        lines.append(schedule_note)
    lines += [
        "",
        "POSITION",
        f"  Value                {_money(summary.get('current_value'), currency)}",
        f"  Invested             {_money(summary.get('total_investment'), currency)}",
        f"  Unrealised P&L       {_money(summary.get('total_pnl'), currency)}"
        f"  ({_pct(summary.get('total_return_pct'))})",
        f"  Change today         {_money(summary.get('day_change'), currency)}"
        f"  ({_pct(summary.get('day_change_pct'))})",
        f"  Holdings             {_ratio(summary.get('n_holdings'), 0)}",
        "",
        "RETURN AND RISK",
        f"  Annualised return    {_pct(risk.get('ann_return'), 2, ratio=True)}",
        f"  Annualised vol       {_pct(risk.get('ann_vol'), 2, ratio=True)}",
        f"  Sharpe               {_ratio(risk.get('sharpe'))}",
        f"  Sortino              {_ratio(risk.get('sortino'))}",
        f"  Max drawdown         {_pct(risk.get('mdd'), 2, ratio=True)}",
        f"  Daily VaR            {_pct(risk.get('var'), 2, ratio=True)}",
    ]
    if ledger.get("xirr") is not None:
        lines.append(f"  XIRR                 {_pct(ledger.get('xirr'), 2, ratio=True)}")
    if bench:
        lines += [
            "",
            f"VERSUS {str(bench.get('name') or 'benchmark').upper()}",
            f"  Beta                 {_ratio(bench.get('beta'))}",
            f"  Alpha (annualised)   {_pct(bench.get('alpha'), 2, ratio=True)}",
            f"  Tracking error       {_pct(bench.get('tracking_error'), 2, ratio=True)}",
            f"  Information ratio    {_ratio(bench.get('information_ratio'))}",
        ]

    if warnings:
        lines += ["", "DATA NOTES"]
        # Capped, not truncated silently: the count says how many were left out,
        # so a reader knows to open the attachment rather than assuming five.
        for note in warnings[:5]:
            lines.append(f"  - {note}")
        if len(warnings) > 5:
            lines.append(f"  - ...and {len(warnings) - 5} more, listed in the attachment.")

    lines += [
        "",
        "-" * 68,
        "The attached HTML tearsheet carries the full statistics, holdings table",
        "and charts. Open it in any browser; print to PDF if you want a copy.",
        "",
        "This is an educational analysis of data you supplied, computed from",
        "delayed public market data. It is not investment advice, not a",
        "recommendation, and not a solicitation to buy or sell anything.",
    ]
    return "\n".join(lines)


def alert_subject(events: Sequence[Any]) -> str:
    """The subject line for one batch of price alerts.

    The instruments and their levels go in the SUBJECT, not just the body: an
    alert read on a phone lock screen should already say what happened.
    Truncated past three names with a count, because a subject line that runs
    to 200 characters is shown to the reader as its first forty.
    """
    count = len(events)
    if not count:
        return "No price alerts"

    parts: list[str] = []
    for event in events[:3]:
        arrow = ">=" if str(getattr(event, "direction", "")) == "above" else "<="
        level = _num(getattr(event, "level", None))
        shown = f"{level:,.2f}" if not math.isnan(level) else "your level"
        parts.append(f"{getattr(event, 'ticker', '?')} {arrow} {shown}")
    if count > 3:
        parts.append(f"and {count - 3} more")

    noun = "alert" if count == 1 else "alerts"
    return f"{count} price {noun}: " + " · ".join(parts)


def alert_summary(
    events: Sequence[Any],
    skipped: Sequence[tuple[str, str]] = (),
    notices: Sequence[str] = (),
    *,
    checked_at: datetime | None = None,
    n_rules: int = 0,
) -> str:
    """Plain-text body for a batch of price alerts.

    Plain text and nothing else. An alert is read in a notification shade or a
    preview pane, and an HTML mail whose first screen is a logo has buried the
    one number the reader wanted. There is no attachment either -- nothing
    here needs one.

    The caveats are in the body, not just in the app, because this message is
    read away from the app and will be forwarded. Three of them matter enough
    to repeat every time: the feed is delayed, the evidence may be a session
    high rather than the current price, and a level being reached is not a
    reason to trade.

    ``skipped`` is printed as prominently as the alerts themselves. A rule
    that could not be evaluated -- no quote, no direction -- is silence that
    looks exactly like "your level was not reached", and that is the one
    failure an alert system must never hide.
    """
    stamp = (
        checked_at.strftime("%d %b %Y %H:%M")
        if isinstance(checked_at, datetime)
        else datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC")
    )

    count = len(events)
    noun = "level" if count == 1 else "levels"
    lines: list[str] = [
        f"{count} watchlist {noun} reached" if count else "No watchlist levels reached",
        f"Checked {stamp}" + (f" against {n_rules} active rule(s)" if n_rules else ""),
        "",
    ]

    for event in events:
        ticker = getattr(event, "ticker", "?")
        direction = str(getattr(event, "direction", ""))
        moved = "rose to or above" if direction == "above" else "fell to or below"
        currency = str(getattr(event, "currency", "") or "")
        prefix = f"{currency} " if currency else ""
        level = _num(getattr(event, "level", None))
        price = _num(getattr(event, "price", None))
        through = _num(getattr(event, "through_by_pct", None))
        evidence = str(getattr(event, "evidence", "last price"))
        as_of = getattr(event, "as_of", None)

        lines.append(f"  {ticker} {moved} {prefix}{level:,.2f}")
        past = f" ({through:+.2f}% past your level)" if not math.isnan(through) else ""
        lines.append(f"    seen at   {prefix}{price:,.2f} — {evidence}{past}")
        if isinstance(as_of, datetime):
            lines.append(f"    price from {as_of:%d %b %Y %H:%M}")
        if evidence != "last price":
            # The distinction that decides whether this is still actionable.
            lines.append(
                "    note      the level was touched during the session; the price "
                "may have since moved back"
            )
        note = str(getattr(event, "note", "") or "")
        if note:
            lines.append(f"    your note {note}")
        lines.append("")

    if skipped:
        lines += ["RULES THAT COULD NOT BE CHECKED"]
        for ticker, reason in skipped:
            lines.append(f"  {ticker}: {reason}")
        lines += [
            "  These were NOT evaluated. Silence about them does not mean your",
            "  level was not reached.",
            "",
        ]

    if notices:
        lines += ["ALSO"]
        lines += [f"  {notice}" for notice in notices]
        lines.append("")

    lines += [
        "-" * 68,
        "Prices are delayed — typically ~15 minutes for NSE/BSE, and outside",
        "market hours the latest available close. An alert means your level had",
        "been reached as of the timestamp shown, not that it is being reached now.",
        "",
        "A level is checked only when the alert command runs, so a move that",
        "crossed your level and came back between two checks is visible only if",
        "the session high or low caught it.",
        "",
        "These levels are numbers you typed. This is arithmetic against them —",
        "not investment advice, not a recommendation, and not a signal.",
    ]
    return "\n".join(lines)


def compose_tearsheet_email(
    recipients: Sequence[str] | str,
    subject: str,
    body: str,
    attachments: Sequence[Attachment] = (),
    *,
    config: SmtpConfig | None = None,
    sender: str = "",
    reply_to: str = "",
) -> tuple[EmailMessage | None, tuple[str, ...]]:
    """Build the message. ``(None, problems)`` if it could not be addressed.

    Notes are returned alongside the message rather than logged, so the UI and
    the CLI report the same thing — including an attachment that was dropped
    for size, which the recipient is also told about in the body.
    """
    notes: list[str] = []
    accepted, rejected = clean_recipients(recipients)
    notes.extend(rejected)
    if not accepted:
        return None, (*notes, "No valid recipient address, so nothing was composed.")

    from_header = config.from_header if config is not None else sender
    if not from_header:
        return None, (*notes, "No sender address, so nothing was composed.")

    message = EmailMessage()
    message["From"] = from_header
    message["To"] = ", ".join(accepted)
    message["Subject"] = subject
    message["Date"] = formatdate(localtime=True)
    # An explicit Message-ID keeps threading sane and stops some relays from
    # rewriting the header (which several spam filters treat as suspicious).
    message["Message-ID"] = make_msgid(domain=from_header.rsplit("@", 1)[-1].strip(">"))
    message["Auto-Submitted"] = "auto-generated"
    if reply_to:
        message["Reply-To"] = reply_to

    dropped: list[str] = []
    kept: list[Attachment] = []
    for item in attachments:
        if item.size > MAX_ATTACHMENT_BYTES:
            dropped.append(
                f"{item.filename} ({item.size / 1024 / 1024:.1f} MB) exceeds the "
                f"{MAX_ATTACHMENT_BYTES / 1024 / 1024:.0f} MB attachment limit"
            )
            continue
        kept.append(item)

    text = body
    if dropped:
        text += "\n\n" + "\n".join(
            f"NOTE: {reason} and was not attached. Generate the report from the "
            "Export page instead."
            for reason in dropped
        )
        notes.extend(f"{reason}; attachment omitted." for reason in dropped)

    message.set_content(text, subtype="plain", charset="utf-8")
    for item in kept:
        message.add_attachment(
            item.payload,
            maintype=item.maintype,
            subtype=item.subtype,
            filename=item.filename,
        )

    return message, tuple(notes)


# ---------------------------------------------------------------------------
# delivery
# ---------------------------------------------------------------------------
def _connect(config: SmtpConfig) -> smtplib.SMTP:
    """Open a verified SMTP session.

    ``create_default_context`` keeps certificate *and* hostname verification
    on. That is the point: an unverified TLS session protects against a passive
    listener but not against the active attacker who can actually read the
    password.
    """
    if config.use_ssl:
        context = ssl.create_default_context()
        return smtplib.SMTP_SSL(config.host, config.port, timeout=config.timeout, context=context)
    client = smtplib.SMTP(config.host, config.port, timeout=config.timeout)
    if config.use_starttls:
        client.ehlo()
        client.starttls(context=ssl.create_default_context())
        client.ehlo()
    return client


def send_message(
    config: SmtpConfig,
    message: EmailMessage,
    *,
    dry_run: bool = False,
) -> DeliveryResult:
    """Deliver one composed message. Reports failure, never raises.

    ``dry_run=True`` validates the configuration and the message and reports
    the size without opening a socket — the default for anything triggered by
    a page render, because sending mail is outward-facing and irreversible.

    Every failure mode SMTP has is folded into ``DeliveryResult.error`` as a
    sentence naming the cause: authentication refused, host unreachable,
    certificate rejected, recipients refused. A scheduled job that dies with a
    traceback into a cron log nobody reads is indistinguishable from one that
    never ran.
    """
    recipients = tuple(
        address.strip() for address in str(message.get("To", "")).split(",") if address.strip()
    )
    subject = str(message.get("Subject", ""))
    size = len(bytes(message))

    problems = config.problems
    if problems:
        return DeliveryResult(
            sent=False,
            recipients=recipients,
            subject=subject,
            size_bytes=size,
            error=" ".join(problems),
        )

    notes: list[str] = []
    if not config.encrypted:
        notes.append(
            "This message was sent over an UNENCRYPTED connection because allow_insecure was set."
        )

    if dry_run:
        return DeliveryResult(
            sent=False,
            recipients=recipients,
            subject=subject,
            size_bytes=size,
            dry_run=True,
            notes=tuple(notes),
        )

    try:
        with _connect(config) as client:
            if config.username and config.password:
                client.login(config.username, config.password)
            refused = client.send_message(message)
    except smtplib.SMTPAuthenticationError as exc:
        # Never echo the credential back, even in the error path. Gmail and
        # Outlook both reject a normal login password here and require an
        # app-specific one, which is the single most common cause.
        logger.warning("SMTP authentication refused by %s:%s", config.host, config.port)
        return DeliveryResult(
            sent=False,
            recipients=recipients,
            subject=subject,
            size_bytes=size,
            error=(
                f"{config.host} refused the login ({exc.smtp_code}). Gmail and "
                "Outlook require an app-specific password, not your account "
                "password."
            ),
            notes=tuple(notes),
        )
    except smtplib.SMTPRecipientsRefused as exc:
        rejected = ", ".join(str(k) for k in exc.recipients)
        return DeliveryResult(
            sent=False,
            recipients=recipients,
            subject=subject,
            size_bytes=size,
            error=f"Every recipient was refused by the server: {rejected}.",
            notes=tuple(notes),
        )
    except ssl.SSLError as exc:
        return DeliveryResult(
            sent=False,
            recipients=recipients,
            subject=subject,
            size_bytes=size,
            error=(
                f"The TLS handshake with {config.host}:{config.port} failed ({exc}). "
                "Check whether the port wants implicit SSL (465) or STARTTLS (587)."
            ),
            notes=tuple(notes),
        )
    except (OSError, smtplib.SMTPException) as exc:
        logger.warning("SMTP delivery to %s:%s failed: %s", config.host, config.port, exc)
        return DeliveryResult(
            sent=False,
            recipients=recipients,
            subject=subject,
            size_bytes=size,
            error=f"Delivery to {config.host}:{config.port} failed: {exc}",
            notes=tuple(notes),
        )

    if refused:
        # A partial success. Saying "sent" would hide the addresses that were
        # not, which is exactly the failure a scheduled report must surface.
        notes.append(
            "Some recipients were refused: "
            + ", ".join(f"{k} ({v[0]})" for k, v in refused.items())
        )
    delivered = tuple(a for a in recipients if a not in refused)
    return DeliveryResult(
        sent=bool(delivered),
        recipients=delivered,
        subject=subject,
        size_bytes=size,
        error="" if delivered else "Every recipient was refused.",
        notes=tuple(notes),
    )
