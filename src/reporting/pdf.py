"""HTML tearsheet -> real PDF, printed by the Edge that is already installed.

Streamlit can hand a user a download button, but what a user wants to file,
mail to a client or hand to an accountant is a PDF, not a ``.html`` they must
remember to open and "Save as PDF". This module is the last inch of that
journey: it takes the complete document :mod:`src.reporting.html_report`
already produces and returns real PDF bytes.

Three decisions are worth naming, because each one is a temptation avoided:

* **No new dependency.** A pure-Python PDF writer would have to be added to
  ``requirements.txt``, and it could not do the one thing that matters here:
  the report's charts are Plotly figures drawn by JavaScript at load time, so
  a library that only understands HTML and CSS would print the surrounding
  text and leave every chart as an empty box. The browsers already installed
  on every desktop run that JavaScript and are the reference implementation
  of CSS print layout - so the report goes to a browser. Nothing is installed
  to make this work.
* **A throwaway profile, always.** Edge refuses to start a second instance
  against a profile that is already open. Pointing ``--user-data-dir`` at a
  directory inside our own temporary folder means the print can never collide
  with the Edge window the user has open in front of them, and the profile is
  deleted with the temp directory when we are done.
* **A list argv, never a shell.** The HTML is written to a temp file and its
  ``file://`` URL is passed as one argument of a list, so a portfolio name
  containing quotes, spaces or an ``&`` cannot be reinterpreted as a command
  or as extra flags. There is no ``shell=True`` anywhere in this module.

The browser is waited on with a timeout and every failure mode - no browser,
a timeout, a non-zero exit, a run that claims success but wrote no file - is
raised as :class:`PdfError` carrying the exit code and an excerpt of stderr.
A silent empty PDF is worse than a loud error: the caller can fall back to
offering the HTML, but it cannot un-send a blank file.

This module does file I/O and runs a subprocess, which is why it lives in the
reporting layer, but it never imports Streamlit: the same call works from the
scheduled-mail command line as from a download button.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

__all__ = ["PdfError", "find_edge", "pdf_available", "html_to_pdf", "pdf_notice"]

#: Environment variables checked before anything else, so a deployment can
#: point at a Chromium that is not in a standard place without editing code.
ENV_VARS: tuple[str, ...] = ("EDGE_PATH", "CHROME_PATH")

#: The two locations a stock Windows install of Edge uses; the 32-bit path is
#: listed first because it is the one 64-bit Windows uses for an x86 app.
INSTALL_CANDIDATES: tuple[str, ...] = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
)

#: Last resort: a Chromium-family browser on ``PATH``. Edge's executable is
#: ``msedge``, but a Linux CI box or a macOS workstation more often has one of
#: the others, and any of them prints the same document.
PATH_NAMES: tuple[str, ...] = ("msedge", "chrome", "chromium")

#: A whole tearsheet, charts included, prints in a couple of seconds in
#: practice; ninety is generous enough for a cold start and a slow CDN and
#: short enough that a wedged browser does not hold a request open forever.
DEFAULT_TIMEOUT: float = 90.0

#: How much of the browser's stderr travels in the exception message. Edge is
#: chatty about unrelated quirks, so the tail is truncated rather than pasted
#: whole into a log line.
STDERR_LIMIT: int = 600

#: Suppress the console window Windows would otherwise flash when the app -
#: which normally runs behind a web server - spawns a GUI executable. Zero on
#: every other platform, where the flag does not exist.
_NO_WINDOW: int = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


class PdfError(RuntimeError):
    """A PDF could not be produced, with the reason in the message.

    Deliberately a :class:`RuntimeError`: callers that print reports catch it
    the same way they catch any other unforeseen failure, while the message
    stays specific enough to say *which* failure it was.
    """


def find_edge() -> str:
    """Path to a Chromium-family browser that can print, or ``""``.

    Searched in the order that respects the operator's intent before the
    machine's defaults: the ``EDGE_PATH`` / ``CHROME_PATH`` environment
    variables first (a deployment can pin a specific build), then the two
    standard Windows install locations, then ``msedge``/``chrome``/``chromium``
    on ``PATH``. An environment variable that names a file that does not exist
    is skipped rather than returned, so a stale override does not shadow a
    perfectly good browser sitting in the default location.

    Never raises: a missing browser is an ordinary, expected state on a
    headless server, and the answer is ``""``, not an exception.
    """
    for name in ENV_VARS:
        candidate = os.environ.get(name, "").strip()
        if not candidate:
            continue
        if os.path.isfile(candidate):
            return candidate
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    for candidate in INSTALL_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    for name in PATH_NAMES:
        resolved = shutil.which(name)
        if resolved:
            return resolved
    return ""


def pdf_available() -> bool:
    """Whether :func:`html_to_pdf` has a browser to call.

    The dashboard uses this to decide whether to offer a PDF download at all,
    rather than offering one that raises when it is clicked.
    """
    return bool(find_edge())


def pdf_notice() -> str:
    """Why a PDF cannot be produced, or ``""`` when it can.

    Phrased for a person, not a log: it names what was looked for so the
    reader can act on it (install Edge, or set ``EDGE_PATH``). Returning an
    empty string when PDFs *are* available lets a caller render the notice
    unconditionally - ``if pdf_notice(): warn(...)`` - without a second check.
    """
    if find_edge():
        return ""
    return (
        "PDF export needs a Chromium-based browser (Microsoft Edge, Google Chrome "
        "or Chromium), and none was found. Install Edge, or set the EDGE_PATH "
        "environment variable to the browser executable, to enable PDF downloads."
    )


def _excerpt(text: str, limit: int = STDERR_LIMIT) -> str:
    """The tail of a browser message, collapsed to one line and truncated.

    The *tail* matters more than the head: Chromium prints its "bytes written"
    confirmation last and its diagnostics before it, so when the run fails the
    useful line is often the final one. A trailing ellipsis marks a truncation
    so the reader knows the message is not complete.
    """
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return "..." + cleaned[-limit:]


def _print_argv(edge: str, source: str, target: str, profile: str) -> list[str]:
    """The full command line for one headless print, as a list of arguments.

    Only the flags that change the printed document are kept. In particular
    ``--virtual-time-budget`` is deliberately absent: ``src.reporting``'s
    ``html_report`` embeds Plotly behind a blocking ``<script src>`` that calls
    ``Plotly.newPlot`` while the document is still parsing, and headless Edge
    prints after the load event, so the charts are already drawn when the
    print happens - verified by decoding the chart's own axis labels out of
    the produced PDF. A virtual time budget would only spend wall-clock time
    waiting on a deadline nothing is racing.
    """
    return [
        edge,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        # A profile inside the temporary directory, so this can never attach to
        # (or be blocked by) the Edge the user already has open.
        f"--user-data-dir={profile}",
        f"--print-to-pdf={target}",
        source,
    ]


def html_to_pdf(html: str, *, timeout: float = DEFAULT_TIMEOUT) -> bytes:
    """Print a complete HTML document to PDF bytes using headless Edge.

    The document is written to a ``utf-8`` file inside a
    :class:`~tempfile.TemporaryDirectory` - so both the source and the browser
    profile it is printed with are removed when this function returns, whether
    it returns bytes or raises - and Edge is asked to print that file's URL.

    Raises :class:`PdfError` when no browser is found, when the process
    exceeds ``timeout`` seconds, when it exits non-zero, or when it exits zero
    without leaving a non-empty PDF behind. The exit code and an excerpt of
    the browser's stderr travel in that message; the HTML string itself is the
    caller's to keep.
    """
    edge = find_edge()
    if not edge:
        raise PdfError(pdf_notice())

    with tempfile.TemporaryDirectory(prefix="quantfolio-pdf-") as workspace:
        source = os.path.join(workspace, "report.html")
        target = os.path.join(workspace, "report.pdf")
        # utf-8 explicitly: the report is full of currency symbols and em
        # dashes, and the platform default on Windows is a codepage that
        # cannot represent them.
        with open(source, "w", encoding="utf-8") as handle:
            handle.write(html)

        argv = _print_argv(
            edge,
            source=_as_url(source),
            target=target,
            profile=os.path.join(workspace, "profile"),
        )
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
                creationflags=_NO_WINDOW,
            )
        except subprocess.TimeoutExpired as exc:
            raise PdfError(
                f"PDF export timed out after {timeout:.0f}s waiting for {edge!r}."
            ) from exc
        except OSError as exc:  # browser present but not runnable
            raise PdfError(f"Could not run the browser at {edge!r}: {exc}") from exc

        if completed.returncode != 0:
            detail = _excerpt(completed.stderr or completed.stdout or "")
            raise PdfError(
                f"PDF export failed: {edge!r} exited with code "
                f"{completed.returncode}. {detail}".strip()
            )

        if not os.path.isfile(target):
            detail = _excerpt(completed.stderr or completed.stdout or "")
            raise PdfError(
                f"PDF export failed: {edge!r} exited successfully but produced no "
                f"PDF file. {detail}".strip()
            )
        with open(target, "rb") as handle:
            data = handle.read()

    if not data:
        raise PdfError(f"PDF export failed: {edge!r} produced an empty file.")

    return data


def _as_url(path: str) -> str:
    """``file:///`` URL for a local path, correct for Windows drive letters.

    Edge accepts a bare path on Windows but a ``file://`` URL is unambiguous
    on every platform, and prefixing one here keeps a path with spaces or a
    ``#`` in it from being read as flags or a fragment.
    """
    return Path(path).resolve().as_uri()
