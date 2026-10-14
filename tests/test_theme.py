"""The visual system, and the one property that makes it exist at all.

:func:`test_the_stylesheet_survives_a_rerun` is the reason this module was
written. ``inject_css`` used to skip re-emitting the ``<style>`` block when a
session flag said that theme's CSS was "already mounted", to avoid bloating the
DOM. Streamlit does not work that way: it rebuilds the page from whatever the
*current* script run emits, so the flag did not preserve the stylesheet — it
deleted it. Every interaction after the first rendered the whole application as
unstyled text, with the markup for every card present and no rules to style it.

Nothing in the test suite caught that, because every page test mounts a page
in a fresh session and therefore only ever exercises the first run.
"""

from __future__ import annotations

import re

import pytest
from streamlit.testing.v1 import AppTest

from src.ui import theme


def _style_blocks(at: AppTest) -> list[str]:
    """Every emitted markdown element that carries the design tokens."""
    return [str(e.value) for e in at.markdown if "--qf-surface" in str(e.value)]


def _script():
    """Inject the stylesheet, then render something that depends on it.

    Deliberately unannotated and self-importing: ``AppTest.from_function``
    re-executes this source in a namespace with none of this module's imports.
    """
    import streamlit as _st

    from src.ui import components as _ui
    from src.ui import theme as _theme

    _theme.inject_css()
    _ui.kpi_card("Probe", "1.00")
    _st.button("rerun me")


# ---------------------------------------------------------------------------
# the regression guard
# ---------------------------------------------------------------------------
def test_the_stylesheet_survives_a_rerun() -> None:
    """The style block must be emitted on EVERY run, not just the first.

    A rerun is what happens on every widget interaction and every page switch,
    which is to say: almost always. If the stylesheet only survives the first
    run, the application is unstyled in practice and styled only in tests.
    """
    at = AppTest.from_function(_script, default_timeout=60)

    at.run()
    assert len(_style_blocks(at)) == 1, "nothing was emitted on the first run"

    at.button[0].click().run()
    assert len(_style_blocks(at)) == 1, (
        "the stylesheet vanished on rerun — every card on every page is now unstyled"
    )

    at.button[0].click().run()
    assert len(_style_blocks(at)) == 1, "still present on the third run"


def test_exactly_one_style_block_is_emitted_per_run() -> None:
    """The other half: emitting every run must not accumulate style blocks.

    Streamlit replaces the element at a given position rather than appending,
    so "always emit" costs exactly one tag — which is what makes the fix above
    free rather than a trade.
    """
    at = AppTest.from_function(_script, default_timeout=60)
    at.run()
    at.button[0].click().run()
    assert len(_style_blocks(at)) == 1


def test_the_markup_it_styles_is_rendered_alongside_it() -> None:
    """A stylesheet with no markup, or markup with no stylesheet, is the bug."""
    at = AppTest.from_function(_script, default_timeout=60)
    at.run()
    at.button[0].click().run()

    rendered = " ".join(str(e.value) for e in at.markdown)
    assert "qf-kpi" in rendered
    assert "--qf-surface" in rendered


# ---------------------------------------------------------------------------
# the stylesheet itself
# ---------------------------------------------------------------------------
def _placeholders() -> set[str]:
    """Token names the stylesheet template actually interpolates.

    Not every declared token is used by the CSS — ``grid`` exists for the chart
    palette — so the contract is "every placeholder is filled", not "every
    token is referenced".
    """
    return set(re.findall(r"(?<!\{)\{([a-z_0-9]+)\}(?!\})", theme._CSS_TEMPLATE))


@pytest.mark.parametrize("base", ["dark", "light"])
def test_every_placeholder_is_substituted_into_the_stylesheet(base: str) -> None:
    """No ``{placeholder}`` may survive into the emitted CSS.

    ``str.format`` on a stylesheet means every literal CSS brace is doubled. A
    single missed escape produces either a ``KeyError`` when the template is
    rendered or a stray brace that silently kills every rule after it — which
    is exactly the failure mode a large hand-escaped stylesheet invites.
    """
    css = theme.build_css(base)
    values = theme.tokens(base)

    assert "{{" not in css and "}}" not in css
    names = _placeholders()
    assert names, "the template interpolates nothing — the regex or the template moved"
    for name in names:
        assert name in values, f"{name} is interpolated but not a declared token"
        assert values[name] in css, f"{name} was not substituted for {base}"
    assert not re.search(r"(?<!\{)\{[a-z_0-9]+\}(?!\})", css), "an unfilled placeholder survived"


@pytest.mark.parametrize("base", ["dark", "light"])
def test_every_component_class_has_rules_in_both_themes(base: str) -> None:
    """Each design-system component must be styled in dark AND light."""
    css = theme.build_css(base)
    for css_class in (
        "qf-kpi",
        "qf-badge",
        "qf-insight",
        "qf-empty",
        "qf-header",
        "qf-section",
        "qf-freshness",
        "qf-hero",
        "qf-verdict",
        "qf-meter",
        "qf-model",
    ):
        assert f".{css_class}" in css, f"{css_class} is unstyled in {base}"


def test_the_two_themes_are_genuinely_different() -> None:
    assert theme.build_css("dark") != theme.build_css("light")
    assert theme.DARK_TOKENS.keys() == theme.LIGHT_TOKENS.keys()


def test_the_rendered_stylesheet_is_cached_per_theme() -> None:
    """Emitting on every run is only cheap because the render is memoised."""
    assert theme.build_css("dark") is theme.build_css("dark")
    assert theme.build_css("dark") is not theme.build_css("light")


def test_the_configured_base_outranks_the_viewers_system_preference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config pins what Streamlit paints; the browser's preference does not.

    The regression this guards: ``st.context.theme.type`` reports the viewer's
    OS appearance, not the app's rendered theme. Consulting it first gave white
    design-system cards on the configured near-black chrome for every viewer
    whose machine was set to light mode.
    """

    class _Theme:
        type = "light"

    class _Context:
        theme = _Theme()

    monkeypatch.setattr(theme.st, "context", _Context(), raising=False)
    monkeypatch.setattr(
        theme.st, "get_option", lambda name: "dark" if name == "theme.base" else None
    )

    assert theme.active_base() == "dark"
    assert theme.tokens() is theme.DARK_TOKENS


def test_the_viewers_preference_is_used_when_nothing_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no configured base, following the viewer IS the right answer."""

    class _Theme:
        type = "light"

    class _Context:
        theme = _Theme()

    monkeypatch.setattr(theme.st, "context", _Context(), raising=False)
    monkeypatch.setattr(theme.st, "get_option", lambda name: None)

    assert theme.active_base() == "light"


def test_the_theme_falls_back_to_dark_when_nothing_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither source available — a cron job, a bare import — must not raise."""

    def _boom(name: str) -> str:
        raise RuntimeError("no runtime")

    monkeypatch.setattr(theme.st, "context", None, raising=False)
    monkeypatch.setattr(theme.st, "get_option", _boom)

    assert theme.active_base() == "dark"


def test_an_unknown_theme_name_falls_back_to_light_tokens() -> None:
    """``tokens`` treats anything that is not "dark" as light, and must not raise."""
    assert theme.tokens("chartreuse") == theme.LIGHT_TOKENS
    assert theme.tokens("dark") == theme.DARK_TOKENS


def test_page_chrome_escapes_user_supplied_text() -> None:
    """Tickers and notes reach these helpers; a portfolio tool must not inject."""
    at = AppTest.from_function(_injection_probe, default_timeout=60)
    at.run()
    rendered = " ".join(str(e.value) for e in at.markdown)
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered


def _injection_probe():
    from src.ui import theme as _theme

    _theme.page_header("<script>alert(1)</script>", "sub", icon="📊")
