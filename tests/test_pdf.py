"""Tests for the HTML -> PDF step, kept entirely offline.

The one thing these tests must not do is launch a browser, for two reasons
that both matter in CI: it is the slowest thing in the suite by orders of
magnitude, and on a machine with no Chromium it would fail for a reason that
has nothing to do with the code under test. Everything here is therefore
arranged so that the browser is either genuinely absent or never reached.

What is worth asserting is the contract around the browser rather than the
browser itself:

* **Discovery is total.** :func:`find_edge` answers with a path that exists or
  with ``""`` - never a stale path, never an exception - because a caller uses
  it to decide whether to offer a download at all.
* **A missing browser is a specific error, not a traceback.** Forcing
  discovery to fail must produce :class:`PdfError` whose message says so, so a
  user reads "install Edge", not ``FileNotFoundError`` from the depths of
  ``subprocess``.
* **The notice is a string in every state**, so a page can render it without a
  guard.
"""

from __future__ import annotations

import os

import pytest

from src.reporting import pdf

# A document small enough to be legible in the test, complete enough that it
# would print if the browser were ever invoked.
MINIMAL_HTML = "<!DOCTYPE html><title>t</title><h1>Hello</h1>"


def test_find_edge_returns_a_path_that_exists_or_nothing() -> None:
    """The answer is a real executable, or the empty string - never a lie."""
    found = pdf.find_edge()
    assert isinstance(found, str)
    if found:
        assert os.path.isfile(found)


def test_pdf_available_agrees_with_find_edge() -> None:
    """``pdf_available`` is exactly "a browser was found", nothing more."""
    assert pdf.pdf_available() == bool(pdf.find_edge())


def test_pdf_notice_is_a_string_and_empty_only_when_available() -> None:
    """The notice explains the absence; when there is nothing to explain, it is ``""``."""
    notice = pdf.pdf_notice()
    assert isinstance(notice, str)
    assert (notice == "") == pdf.pdf_available()


def test_a_missing_browser_raises_pdf_error_that_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no browser, the error names the problem instead of leaking one."""
    monkeypatch.setattr(pdf, "find_edge", lambda: "")

    with pytest.raises(pdf.PdfError) as excinfo:
        pdf.html_to_pdf(MINIMAL_HTML)

    message = str(excinfo.value)
    assert "Chromium" in message or "Edge" in message


def test_pdf_error_is_a_runtime_error() -> None:
    """Callers that catch broad failures still catch this one."""
    assert issubclass(pdf.PdfError, RuntimeError)


def test_html_to_pdf_never_reaches_a_browser_when_none_is_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The missing-browser path must not shell out at all.

    ``subprocess.run`` is replaced by a tripwire that fails the test if it is
    called, which is what distinguishes "raised before spawning" from "spawned
    and happened to fail".
    """

    def _tripwire(*args: object, **kwargs: object) -> None:
        raise AssertionError("html_to_pdf invoked a subprocess despite find_edge() == ''")

    monkeypatch.setattr(pdf, "find_edge", lambda: "")
    monkeypatch.setattr(pdf.subprocess, "run", _tripwire)

    with pytest.raises(pdf.PdfError):
        pdf.html_to_pdf(MINIMAL_HTML)
