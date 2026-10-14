"""Display formatting -- ``src.ui.formatters``.

Formatting is the last thing that happens to a number before a human acts on it,
so to the person reading the screen a formatting bug is indistinguishable from an
analytics bug: a Sharpe ratio printed as ``0.00`` instead of a dash, a 42% return
printed as ``0.42%``, and a 12 lakh position printed as 1.2 crore are all "the
tool told me the wrong thing".

Verified here
-------------
* **Indian digit grouping.** ``1234567.89`` renders as ``12,34,567.89``: the last
  three integer digits form one group, everything to their left is paired. Every
  golden is worked out digit by digit in a comment next to the assertion, and two
  identities back them up -- grouping only ever inserts commas (strip them and
  you get the plain fixed-point rendering back), and every group except the last
  is exactly two digits.
* **The unit contract between fmt_pct and fmt_ratio_pct.** ``fmt_pct`` takes
  PERCENT units, ``fmt_ratio_pct`` takes DECIMALS; the exact 100x relation is
  asserted directly, because confusing the two is the most likely bug in the
  whole UI layer and it fails silently by a factor of a hundred.
* **Missing is missing.** ``None`` / NaN / +inf / -inf / a non-numeric string
  render as the em dash in every formatter -- never as ``0.00``. A tool that
  prints a zero for an undefined statistic is lying to its user.
* **Sign placement.** Money puts the minus OUTSIDE the currency symbol, and
  ``fmt_delta`` carries the sign in the arrow rather than repeating it.
* Currency symbols are read from ``config.CURRENCY_SYMBOLS`` and the
  Indian-notation currency set from ``config.LAKH_CRORE_CURRENCIES``, so these
  tests follow the configuration instead of hard-coding glyphs, and every
  configured currency is exercised.
* One test per branch of the compact ladders (Cr / L / K and T / B / M / K), of
  the 30/31, 365 and 730 day duration boundaries, and of the tone helpers.

Two genuine defects are pinned with ``xfail`` and described in the reason string
at the test: ``fmt_compact`` mangles ten-multiples at ``decimals=0``, and
``fmt_date`` raises on ``pd.NaT``.

Pure functions, no I/O, no RNG: deterministic by construction.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

# risk_score is imported only for the band-tone coupling test at the bottom of
# this file: the tone map must cover exactly the bands the scorer can emit.
from src.analytics.risk import risk_score
from src.config import CURRENCY_SYMBOLS, LAKH_CRORE_CURRENCIES
from src.ui.formatters import (
    DASH,
    DOWN,
    FLAT,
    UP,
    arrow,
    band_tone,
    currency_symbol,
    delta_tone,
    fmt_compact,
    fmt_date,
    fmt_datetime,
    fmt_delta,
    fmt_money,
    fmt_multiple,
    fmt_num,
    fmt_pct,
    fmt_ratio_pct,
    fmt_shares,
    fmt_signed,
    humanize_days,
    indian_group,
    safe,
)

# ---------------------------------------------------------------------------
# constants -- symbols come from the configuration and are never typed by hand,
# so re-pointing a symbol in config.py fails these tests loudly instead of
# leaving the UI and the tests disagreeing about what a rupee looks like.
# ---------------------------------------------------------------------------
RUPEE = CURRENCY_SYMBOLS["INR"]
DOLLAR = CURRENCY_SYMBOLS["USD"]
FRANC = CURRENCY_SYMBOLS["CHF"]  # "CHF " -- a symbol with a trailing space

#: Everything the module contract calls "missing". A non-numeric string counts:
#: it is what a spreadsheet cell containing "n/a" becomes after upload.
MISSING_VALUES = [
    pytest.param(None, id="None"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(np.nan, id="np-nan"),
    pytest.param(float("inf"), id="plus-inf"),
    pytest.param(float("-inf"), id="minus-inf"),
    pytest.param("not a number", id="non-numeric-str"),
    pytest.param("", id="empty-str"),
    pytest.param(pd.NA, id="pd-NA"),
]

#: Every formatter whose contract is "one number in, one display string out".
NUMERIC_FORMATTERS = [
    pytest.param(fn, id=fn.__name__)
    for fn in (
        fmt_num,
        fmt_pct,
        fmt_ratio_pct,
        fmt_money,
        fmt_compact,
        fmt_signed,
        fmt_multiple,
        fmt_delta,
        humanize_days,
        indian_group,
        fmt_shares,
    )
]

#: Inputs no caller should ever pass, but which a Streamlit widget, a CSV cell
#: or a half-populated dataclass eventually will.
HOSTILE_INPUTS = [
    None,
    float("nan"),
    float("inf"),
    float("-inf"),
    "abc",
    "",
    " ",
    True,
    False,
    0,
    -0.0,
    1e308,
    -1e308,
    np.float64("nan"),
    np.int64(5),
    pd.NA,
    pd.NaT,
    object(),
]


def decompact_inr(text: str) -> float:
    """Read a compact INR string back into a number, for round-trip checks.

    The sign sits outside the currency symbol, so it is stripped first.
    """
    sign = -1.0 if text.startswith("-") else 1.0
    body = text.lstrip("+-").removeprefix(RUPEE)
    for suffix, multiplier in ((" Cr", 1e7), (" L", 1e5), (" K", 1e3)):
        if body.endswith(suffix):
            return sign * float(body[: -len(suffix)].replace(",", "")) * multiplier
    return sign * float(body.replace(",", ""))


# ---- the missing-value contract ------------------------------------------
@pytest.mark.parametrize("formatter", NUMERIC_FORMATTERS)
@pytest.mark.parametrize("value", MISSING_VALUES)
def test_every_formatter_renders_a_missing_value_as_the_em_dash(formatter, value):
    # THE headline contract of this module: an undefined number is displayed as
    # "not available", never as a fabricated 0.00 that a user would act on.
    assert formatter(value) == DASH


def test_the_dash_is_an_em_dash_and_is_distinct_from_the_flat_arrow():
    # U+2014 EM DASH means "missing"; U+2013 EN DASH means "unchanged". They look
    # alike and mean different things, so both are pinned by code point.
    assert DASH == "—"
    assert FLAT == "–"
    assert DASH != FLAT
    assert FLAT != "-"


def test_the_direction_arrows_are_the_expected_glyphs():
    # Regression guard: the report writers embed these in HTML, where swapping
    # them inverts the meaning of every KPI card at a glance.
    assert UP == "▲"  # black up-pointing triangle
    assert DOWN == "▼"  # black down-pointing triangle
    assert UP != DOWN


@pytest.mark.parametrize("formatter", NUMERIC_FORMATTERS)
def test_a_numeric_string_is_formatted_rather_than_dashed(formatter):
    # "1234.5" is what a CSV upload hands the UI. It is a perfectly good number
    # and must not be swept into the missing bucket alongside "n/a".
    assert formatter("1234.5") != DASH


def test_a_numeric_string_formats_identically_to_the_float_it_denotes():
    assert fmt_num("1234.5") == fmt_num(1234.5) == "1,234.50"
    assert fmt_money("1234.5", "INR", 2) == RUPEE + "1,234.50"


# ---- safe() ---------------------------------------------------------------
def test_safe_stringifies_an_ordinary_value():
    assert safe(5) == "5"
    assert safe(5.5) == "5.5"


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), float("-inf")])
def test_safe_collapses_non_string_missing_values_to_a_dash(value):
    assert safe(value) == DASH


def test_safe_passes_a_string_through_untouched_even_though_it_is_not_numeric():
    # Deliberate asymmetry with the numeric formatters: a ticker or a sector name
    # is not a broken number, it is text, so safe() must not eat it.
    assert safe("Energy") == "Energy"
    assert safe("") == ""


def test_safe_honours_a_custom_default():
    assert safe(None, default="n/a") == "n/a"
    assert safe(float("nan"), default="") == ""


# ---- plain numbers --------------------------------------------------------
def test_fmt_num_uses_western_thousands_grouping_and_two_decimals():
    # 1234567.891 -> three-digit groups 1 | 234 | 567, and .891 -> .89
    assert fmt_num(1234567.891) == "1,234,567.89"


def test_fmt_num_is_currency_neutral_and_never_switches_to_indian_grouping():
    # CONVENTION GUARD: fmt_num knows nothing about currency, so it stays
    # Western even in an INR-first app. Money formatting is fmt_money's job.
    assert fmt_num(1234567.0, 0) == "1,234,567"
    assert indian_group(1234567.0, 0) == "12,34,567"


def test_fmt_num_rounds_at_the_requested_number_of_decimals():
    # 1234.56 at zero decimals rounds up to 1235.
    assert fmt_num(1234.56, 0) == "1,235"
    assert fmt_num(1234.5678, 3) == "1,234.568"


def test_fmt_num_keeps_the_minus_sign_on_negative_values():
    assert fmt_num(-1234.5) == "-1,234.50"


def test_fmt_num_renders_zero_as_zero_not_as_a_dash():
    # Zero is a known quantity, unlike NaN. It must survive.
    assert fmt_num(0) == "0.00"


def test_fmt_num_accepts_a_caller_supplied_dash():
    assert fmt_num(None, dash="n/a") == "n/a"


# ---- signed numbers and multiples ----------------------------------------
def test_fmt_signed_always_shows_an_explicit_sign():
    assert fmt_signed(1234.5) == "+1,234.50"
    assert fmt_signed(-1234.5) == "-1,234.50"


def test_fmt_signed_labels_zero_as_positive():
    # Documented quirk of the "+" format spec: exact zero prints "+0.00", and
    # negative zero prints "-0.00". Pinned so a change is deliberate.
    assert fmt_signed(0.0) == "+0.00"
    assert fmt_signed(-0.0) == "-0.00"


def test_fmt_multiple_suffixes_a_lower_case_x():
    # 1.35 -> "1.35x": the profit-factor / leverage rendering.
    assert fmt_multiple(1.35) == "1.35x"
    assert fmt_multiple(-1.5) == "-1.50x"
    assert fmt_multiple(1234.5, 1) == "1,234.5x"


# ---- percent units: the 100x trap ----------------------------------------
def test_fmt_pct_reads_its_input_as_percent_units():
    # 12.5 already means "12.5 percent", so it renders unchanged.
    assert fmt_pct(12.5) == "12.50%"


def test_fmt_ratio_pct_reads_its_input_as_a_decimal_ratio():
    # 0.125 means "12.5 percent", so it is multiplied by 100 on the way out.
    assert fmt_ratio_pct(0.125) == "12.50%"


@pytest.mark.parametrize("ratio", [0.0, 0.0042, 0.125, 0.42, 1.0, -0.0728, -1.5, 12.75])
def test_fmt_ratio_pct_is_exactly_fmt_pct_of_a_hundred_times_the_input(ratio):
    # THE unit identity of the UI layer: fmt_ratio_pct(x) == fmt_pct(100 * x) for
    # every x. If these two ever drift apart, every ratio on every page is wrong
    # by a factor of a hundred and nothing in the app will complain.
    assert fmt_ratio_pct(ratio) == fmt_pct(ratio * 100.0)


def test_the_two_percent_formatters_differ_by_exactly_a_factor_of_a_hundred():
    # Passing a decimal to fmt_pct understates by 100x; passing percent units to
    # fmt_ratio_pct overstates by 100x. Both failure directions are pinned here
    # so a reviewer can see the size of the mistake being guarded against.
    assert fmt_pct(0.42) == "0.42%"  # decimal into the percent formatter
    assert fmt_ratio_pct(0.42) == "42.00%"  # decimal into the ratio formatter
    assert fmt_pct(42.0) == "42.00%"  # percent into the percent formatter
    assert fmt_ratio_pct(42.0) == "4,200.00%"  # percent into the ratio formatter


def test_fmt_pct_prefixes_a_plus_only_when_asked_and_only_when_positive():
    assert fmt_pct(12.5, signed=True) == "+12.50%"
    assert fmt_pct(-12.5, signed=True) == "-12.50%"
    assert fmt_pct(0.0, signed=True) == "0.00%"  # zero gets no sign
    assert fmt_pct(12.5) == "12.50%"  # unsigned by default


def test_fmt_pct_signs_by_the_true_value_not_by_the_rounded_display():
    # 0.001% rounds to 0.00% but is genuinely positive, so it keeps its plus.
    assert fmt_pct(0.001, 2, signed=True) == "+0.00%"


def test_fmt_pct_honours_the_decimal_count_and_groups_thousands():
    assert fmt_pct(12.3456, 1) == "12.3%"
    assert fmt_pct(1234.5) == "1,234.50%"


def test_fmt_ratio_pct_forwards_decimals_and_the_signed_flag():
    # -0.0728 -> -7.28% -> one decimal -> "-7.3%"
    assert fmt_ratio_pct(-0.0728, 1, True) == "-7.3%"
    assert fmt_ratio_pct(0.0728, 1, True) == "+7.3%"


# ---- Indian digit grouping ------------------------------------------------
def test_indian_group_pairs_everything_left_of_the_last_three_digits():
    # 1234567.89 -> tail "567", head "1234" -> pairs "12","34" -> 12,34,567.89
    assert indian_group(1234567.89) == "12,34,567.89"


def test_indian_group_renders_one_lakh_without_decimals():
    # 100000 -> tail "000", head "100" -> pairs "1","00" -> 1,00,000
    assert indian_group(100000, 0) == "1,00,000"


def test_indian_group_leaves_three_digit_numbers_ungrouped():
    # len("999") <= 3, so no comma is inserted; the default is two decimals.
    assert indian_group(999) == "999.00"


def test_indian_group_keeps_the_minus_sign_outside_the_grouping():
    # -1234.5 -> sign "-" + group("1234.50") -> "-1,234.50"
    assert indian_group(-1234.5) == "-1,234.50"


def test_indian_group_handles_a_twelve_crore_integer():
    # 123456789 -> tail "789", head "123456" -> pairs "12","34","56"
    assert indian_group(123456789, 0) == "12,34,56,789"


@pytest.mark.parametrize(
    ("value", "decimals", "expected"),
    [
        (0, 2, "0.00"),
        (999, 0, "999"),
        (1000, 0, "1,000"),  # tail "000", head "1" -> 1,000
        (99999, 0, "99,999"),  # tail "999", head "99" -> 99,999
        (100000, 0, "1,00,000"),  # the first pairing boundary
        (1234567.89, 0, "12,34,568"),  # .89 rounds the whole part up
        (12345.6789, 3, "12,345.679"),
        (1e7, 0, "1,00,00,000"),  # one crore
        (-1e7, 0, "-1,00,00,000"),
    ],
)
def test_indian_group_goldens(value, decimals, expected):
    assert indian_group(value, decimals) == expected


@pytest.mark.parametrize(
    "value", [0.0, 5.0, 999.99, 1234.5, 100000.0, 1234567.89, 123456789.0, -1234567.89]
)
def test_indian_grouping_only_inserts_commas_and_never_changes_the_value(value):
    # IDENTITY: strip the separators and the plain fixed-point rendering comes
    # back, digit for digit. Grouping is presentation, never arithmetic.
    text = indian_group(value, 2)
    assert text.replace(",", "") == f"{value:.2f}"
    assert float(text.replace(",", "")) == pytest.approx(value, abs=5e-3)


@pytest.mark.parametrize("value", [1e5, 1e6, 1e7, 1e8, 1e9, 1e11, 123456789.0])
def test_every_indian_group_except_the_last_is_a_pair(value):
    # STRUCTURAL IDENTITY of the 2-2-3 convention: the last group is three
    # digits, every interior group exactly two, the leading group one or two.
    groups = indian_group(value, 0).split(",")
    assert len(groups[-1]) == 3
    assert all(len(g) == 2 for g in groups[1:-1])
    assert len(groups[0]) in (1, 2)


def test_indian_grouping_differs_from_western_grouping_above_one_lakh():
    # The whole reason this function exists: same digits, different commas.
    assert indian_group(1234567.89) == "12,34,567.89"
    assert fmt_num(1234567.89) == "1,234,567.89"
    assert indian_group(1234567.89).replace(",", "") == fmt_num(1234567.89).replace(",", "")


def test_indian_group_with_zero_decimals_emits_no_decimal_point():
    assert "." not in indian_group(1234567.89, 0)


def test_a_negative_value_rounding_to_zero_keeps_its_minus_sign():
    # Known cosmetic quirk, pinned deliberately: the sign comes from the value
    # and the digits from the rounded magnitude, so -0.4 at zero decimals shows
    # as "-0" -- honest about direction, if ugly.
    assert indian_group(-0.4, 0) == "-0"


# ---- currency symbols -----------------------------------------------------
def test_currency_symbol_is_case_insensitive():
    assert currency_symbol("inr") == currency_symbol("INR") == RUPEE


def test_currency_symbol_of_an_unknown_or_empty_code_is_the_empty_string():
    # Not a dash and not an exception: an unknown currency shows a bare number
    # rather than mislabelling it with somebody else's symbol.
    assert currency_symbol("XYZ") == ""
    assert currency_symbol("") == ""
    assert currency_symbol(None) == ""


@pytest.mark.parametrize("code", sorted(CURRENCY_SYMBOLS))
def test_every_configured_currency_resolves_to_its_configured_symbol(code):
    assert currency_symbol(code) == CURRENCY_SYMBOLS[code]


# ---- money ----------------------------------------------------------------
def test_fmt_money_uses_indian_grouping_for_rupees():
    # 1234567.891 at two decimals -> 12,34,567.89, prefixed by the INR symbol.
    assert fmt_money(1234567.891, "INR", 2) == RUPEE + "12,34,567.89"


def test_fmt_money_uses_western_grouping_for_dollars():
    assert fmt_money(1234567.891, "USD", 2) == DOLLAR + "1,234,567.89"


def test_fmt_money_defaults_to_rupees_and_to_whole_units():
    # Default currency INR, default decimals 0: .891 rounds the unit up.
    assert fmt_money(1234567.891) == RUPEE + "12,34,568"


def test_fmt_money_puts_the_minus_sign_outside_the_currency_symbol():
    # SIGN PLACEMENT GUARD. The accounting convention is "-<symbol>1,234.50",
    # never "<symbol>-1,234.50", and the report writers rely on it for alignment.
    text = fmt_money(-1234.5, "INR", 2)
    assert text == "-" + RUPEE + "1,234.50"
    assert text.index("-") < text.index(RUPEE)

    western = fmt_money(-1234.5, "USD", 2)
    assert western == "-" + DOLLAR + "1,234.50"
    assert western.index("-") < western.index(DOLLAR)


def test_fmt_money_puts_an_explicit_plus_outside_the_symbol_when_signed():
    assert fmt_money(1234.5, "INR", 2, signed=True) == "+" + RUPEE + "1,234.50"
    assert fmt_money(1234.5, "INR", 2) == RUPEE + "1,234.50"


def test_a_signed_zero_amount_is_marked_positive():
    # Pinned quirk: the sign branch treats "not negative" as positive, so an
    # exactly zero amount shows a plus when signed=True.
    assert fmt_money(0, "INR", 0, signed=True) == "+" + RUPEE + "0"


def test_fmt_money_of_an_unknown_currency_prints_a_bare_number():
    # No symbol is invented, and an unknown code is not an Indian-notation
    # currency, so it groups in the Western style.
    assert fmt_money(1000, "XYZ", 0) == "1,000"


def test_fmt_money_keeps_a_multi_character_symbol_intact():
    # The configured CHF symbol carries its own trailing space.
    assert fmt_money(1234.5, "CHF", 2) == FRANC + "1,234.50"


@pytest.mark.parametrize("code", sorted(CURRENCY_SYMBOLS))
def test_fmt_money_picks_its_grouping_from_the_lakh_crore_currency_set(code):
    # CONFIG COUPLING: the grouping style is a property of the currency set in
    # config, not a hard-coded "if INR" inside the formatter.
    text = fmt_money(1234567.0, code, 0)
    assert text.startswith(CURRENCY_SYMBOLS[code])
    body = text[len(CURRENCY_SYMBOLS[code]) :]
    expected = "12,34,567" if code in LAKH_CRORE_CURRENCIES else "1,234,567"
    assert body == expected


def test_the_lakh_crore_currency_set_is_exactly_the_rupee():
    # Regression guard on the configuration itself: adding USD here would
    # silently reformat every dollar figure in the app.
    assert "INR" in LAKH_CRORE_CURRENCIES
    assert "USD" not in LAKH_CRORE_CURRENCIES


def test_fmt_money_delegates_to_the_compact_ladder_when_asked():
    assert fmt_money(12345678, "INR", compact=True) == fmt_compact(12345678, "INR")


def test_compact_money_ignores_the_decimals_argument():
    # Documented wart, pinned: fmt_money does not forward `decimals` to
    # fmt_compact, so a compact amount always shows two significant decimals.
    assert fmt_money(12345678, "INR", decimals=0, compact=True) == RUPEE + "1.23 Cr"


# ---- compact magnitudes ---------------------------------------------------
def test_a_crore_scale_rupee_amount_is_rendered_in_crore():
    # 12,345,678 / 1e7 = 1.2345678 -> two decimals -> "1.23 Cr"
    assert fmt_compact(12345678, "INR") == RUPEE + "1.23 Cr"


def test_a_lakh_scale_rupee_amount_is_rendered_in_lakh():
    # 345,678 / 1e5 = 3.45678 -> two decimals -> "3.46 L"
    assert fmt_compact(345678, "INR") == RUPEE + "3.46 L"


def test_a_million_scale_dollar_amount_is_rendered_in_millions():
    # 1,234,567 / 1e6 = 1.234567 -> "1.23M", with no space before the suffix.
    assert fmt_compact(1234567, "USD") == DOLLAR + "1.23M"


@pytest.mark.parametrize(
    ("value", "expected_body"),
    [
        (1e7, "1 Cr"),  # exactly at the crore threshold
        (9999999, "100 L"),  # one below it: 9999999/1e5 = 99.99999 -> "100 L"
        (1e5, "1 L"),  # exactly at the lakh threshold
        (99999, "100 K"),  # 99999/1e3 = 99.999 -> "100 K"
        (1e3, "1 K"),  # exactly at the thousand threshold
        (999, "999"),  # below the ladder: plain, no suffix
        (0, "0"),
        (2e8, "20 Cr"),  # 2e8/1e7 = 20.00 -> decimal zeros trimmed -> "20"
    ],
)
def test_the_indian_compact_ladder_switches_at_the_documented_thresholds(value, expected_body):
    assert fmt_compact(value, "INR") == RUPEE + expected_body


@pytest.mark.parametrize(
    ("value", "expected_body"),
    [
        (1e12, "1T"),
        (1e9, "1B"),
        (1e6, "1M"),
        (1e3, "1K"),
        (999, "999"),
        (1234, "1.23K"),  # 1234/1e3 = 1.234 -> "1.23K"
    ],
)
def test_the_western_compact_ladder_switches_at_the_documented_thresholds(value, expected_body):
    assert fmt_compact(value, "USD") == DOLLAR + expected_body


def test_the_indian_ladder_spaces_its_suffix_and_the_western_one_does_not():
    # Typographic convention guard: "1.23 Cr" reads as words, "1.23M" as a unit.
    assert fmt_compact(1234, "INR") == RUPEE + "1.23 K"
    assert fmt_compact(1234, "USD") == DOLLAR + "1.23K"


def test_compact_trailing_zeros_are_trimmed_but_integer_digits_survive():
    # 1e8 / 1e7 = 10.00 -> "10.00" -> trim the decimal zeros -> "10 Cr".
    # The decimal point is what stops the trim from eating the "0" of "10".
    assert fmt_compact(1e8, "INR") == RUPEE + "10 Cr"
    assert fmt_compact(1.5e7, "INR") == RUPEE + "1.5 Cr"


# Regression guard for a fixed defect:
# fmt_compact trims with .rstrip('0') on a string that has no decimal point when
# decimals=0, so the scaled value 10 becomes '1' and 1e8 renders as one crore instead of
# ten -- any multiple of ten is understated 10x.
def test_a_zero_decimal_compact_value_keeps_its_integer_trailing_zero():
    # 1e8 / 1e7 = 10 -> at decimals=0 the correct rendering is "10 Cr".
    assert fmt_compact(1e8, "INR", decimals=0) == RUPEE + "10 Cr"


def test_compact_puts_the_sign_outside_the_symbol():
    assert fmt_compact(-12345678, "INR") == "-" + RUPEE + "1.23 Cr"
    assert fmt_compact(12345678, "INR", signed=True) == "+" + RUPEE + "1.23 Cr"


@pytest.mark.parametrize(
    "value", [0.0, 999.0, 1234.0, 345678.0, 12345678.0, 1.5e7, 4.2e9, -12345678.0]
)
def test_the_compact_form_never_misplaces_the_order_of_magnitude(value):
    # IDENTITY: reading the compact string back must recover the input to within
    # its two significant decimals. A ladder bug (Cr where L belongs) shows up
    # here as a factor of a hundred, not as a rounding difference.
    assert decompact_inr(fmt_compact(value, "INR")) == pytest.approx(value, rel=0.01, abs=1.0)


def test_compact_magnitude_is_monotonic_across_the_ladder():
    # Reading back a strictly increasing sequence must stay increasing: no
    # threshold may fold a larger number onto a smaller display value.
    values = [500.0, 1500.0, 99000.0, 250000.0, 9.9e6, 1.2e7, 5e8]
    read_back = [decompact_inr(fmt_compact(v, "INR")) for v in values]
    assert read_back == sorted(read_back)


# ---- share counts ---------------------------------------------------------
def test_fmt_shares_prints_a_whole_number_without_decimals():
    assert fmt_shares(100) == "100"
    assert fmt_shares(100.0) == "100"


def test_fmt_shares_trims_trailing_zeros_on_a_fractional_holding():
    # 12.5 -> "12.5000" -> trailing zeros trimmed -> "12.5"
    assert fmt_shares(12.5) == "12.5"
    assert fmt_shares(0.25) == "0.25"


def test_fmt_shares_keeps_four_decimals_of_precision():
    # Mutual-fund units are quoted to three or four places, so 1.23456789 must
    # round to 1.2346 rather than collapse to a whole share.
    assert fmt_shares(1.23456789) == "1.2346"


def test_fmt_shares_treats_a_float_within_a_nanoshare_of_an_integer_as_whole():
    # 3.0000000001 is float noise from a FIFO split, not a fractional share.
    assert fmt_shares(3.0000000001) == "3"
    assert fmt_shares(1e-12) == "0"


def test_fmt_shares_groups_thousands_in_the_western_style():
    # CONVENTION GUARD: a share COUNT is not money, so it is never lakh-grouped.
    assert fmt_shares(1234567) == "1,234,567"
    assert fmt_shares(1000.5) == "1,000.5"


def test_fmt_shares_keeps_the_sign_of_a_short_position():
    assert fmt_shares(-25) == "-25"


# ---- dates ----------------------------------------------------------------
def test_fmt_date_renders_a_date_in_the_house_format():
    assert fmt_date(date(2025, 9, 1)) == "01 Sep 2025"


def test_fmt_date_renders_a_datetime_and_a_timestamp_identically():
    assert fmt_date(datetime(2025, 9, 1, 14, 5)) == "01 Sep 2025"
    assert fmt_date(pd.Timestamp("2025-09-01 14:05")) == "01 Sep 2025"


def test_fmt_date_parses_an_iso_string():
    assert fmt_date("2025-09-01") == "01 Sep 2025"


def test_fmt_date_honours_a_custom_format():
    assert fmt_date(date(2025, 9, 1), "%Y-%m-%d") == "2025-09-01"


def test_fmt_date_of_none_or_an_unparseable_value_is_a_dash():
    assert fmt_date(None) == DASH
    assert fmt_date("not a date") == DASH
    assert fmt_date(float("nan")) == DASH


def test_fmt_datetime_adds_the_time_of_day():
    assert fmt_datetime(datetime(2025, 9, 1, 14, 5)) == "01 Sep 2025, 14:05"


def test_fmt_datetime_is_fmt_date_with_a_different_default_format():
    stamp = pd.Timestamp("2025-09-01 14:05")
    assert fmt_datetime(stamp) == fmt_date(stamp, "%d %b %Y, %H:%M")


# Regression guard for a fixed defect:
# pd.NaT is an instance of datetime, so fmt_date takes the isinstance branch and calls
# NaT.strftime, which raises ValueError('NaTType does not support strftime'). The guarded
# pandas fallback below it is never reached, so a missing date crashes the caller instead
# of rendering a dash.
def test_a_not_a_time_value_renders_as_a_dash_instead_of_raising():
    assert fmt_date(pd.NaT) == DASH


# ---- durations ------------------------------------------------------------
def test_humanize_days_uses_plain_days_below_the_month_boundary():
    # The switch is at 31, so 30 days is still "30 days".
    assert humanize_days(30) == "30 days"


def test_humanize_days_switches_to_months_at_thirty_one_days():
    # 31 -> years 0, remainder 31 -> months 31 // 30 = 1, left 31 - 30 = 1
    assert humanize_days(31) == "1m 1d"


def test_humanize_days_makes_a_single_day_singular():
    assert humanize_days(1) == "1 day"
    assert humanize_days(0) == "0 days"
    assert humanize_days(2) == "2 days"


def test_humanize_days_rounds_a_fractional_duration_before_bucketing():
    # 30.6 rounds to 31, which is on the far side of the month boundary.
    assert humanize_days(30.6) == "1m 1d"
    assert humanize_days(30.4) == "30 days"


def test_humanize_days_drops_the_day_remainder_on_a_whole_month():
    # 60 -> months 2, left 60 - 60 = 0, so no trailing day count.
    assert humanize_days(60) == "2m"


def test_one_year_is_reported_as_a_year_not_as_twelve_months():
    # 365 -> years 1, remainder 0 -> months 0 -> "1y"
    assert humanize_days(365) == "1y"


def test_two_years_is_reported_as_two_years():
    # 730 = 2 * 365 -> years 2, remainder 0 -> "2y"
    assert humanize_days(730) == "2y"


def test_a_year_and_a_month_shows_both_parts():
    # 395 -> years 1, remainder 30 -> months 1 -> "1y 1m"
    assert humanize_days(395) == "1y 1m"


def test_just_under_a_year_is_still_reported_in_months():
    # 364 -> years 0, remainder 364 -> months 12, left 364 - 360 = 4.
    # Pinned quirk of the mixed 365-day-year / 30-day-month radix: the last four
    # days before an anniversary read as "12m 4d", never as "1y".
    assert humanize_days(364) == "12m 4d"


def test_a_negative_duration_is_a_dash_rather_than_a_nonsense_string():
    # A negative recovery time means the episode never recovered.
    assert humanize_days(-1) == DASH
    assert humanize_days(-500) == DASH


# ---- tone helpers ---------------------------------------------------------
@pytest.mark.parametrize(
    ("value", "expected"),
    [(1.0, "gain"), (0.0001, "gain"), (-1.0, "loss"), (0.0, "muted"), (-0.0, "muted")],
)
def test_delta_tone_maps_the_sign_of_a_number_to_a_tone(value, expected):
    assert delta_tone(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1.0, "loss"), (-1.0, "gain"), (0.0, "muted")],
)
def test_delta_tone_inverts_for_metrics_where_less_is_better(value, expected):
    # invert=True is used for volatility, drawdown and tracking error: a rise is
    # bad news there, so it must be coloured as a loss.
    assert delta_tone(value, invert=True) == expected


@pytest.mark.parametrize("value", [1.0, -1.0, 0.5, -12345.0])
def test_inverting_swaps_gain_and_loss_for_every_non_zero_value(value):
    # IDENTITY: inversion exchanges the two signed tones and nothing else.
    tones = {delta_tone(value), delta_tone(value, invert=True)}
    assert tones == {"gain", "loss"}


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), "abc"])
def test_delta_tone_of_a_missing_value_is_muted_in_both_directions(value):
    assert delta_tone(value) == "muted"
    assert delta_tone(value, invert=True) == "muted"


def test_arrow_points_up_for_gains_down_for_losses_and_flat_for_zero():
    assert arrow(1.5) == UP
    assert arrow(-1.5) == DOWN
    assert arrow(0) == FLAT


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), "abc"])
def test_arrow_of_a_missing_value_is_empty_rather_than_a_dash(value):
    # Deliberate exception to the em-dash rule: the arrow sits NEXT TO a value
    # that is already showing a dash, so a second dash would read as "--".
    assert arrow(value) == ""


def test_fmt_delta_pairs_the_arrow_with_an_unsigned_magnitude():
    # 2.41 -> up arrow + "2.41%"
    assert fmt_delta(2.41) == UP + " 2.41%"


def test_fmt_delta_carries_the_sign_in_the_arrow_and_never_repeats_it():
    # -2.4056 -> down arrow + abs rounded to 2.41. The minus is not printed a
    # second time, which would read as "(down) -2.41%".
    text = fmt_delta(-2.4056)
    assert text == DOWN + " 2.41%"
    assert "-" not in text


def test_fmt_delta_of_zero_uses_the_flat_marker():
    assert fmt_delta(0) == FLAT + " 0.00%"


def test_fmt_delta_honours_decimals_and_a_custom_suffix():
    assert fmt_delta(1234.5, 1, " bps") == UP + " 1,234.5 bps"
    assert fmt_delta(2.41, 2, "") == UP + " 2.41"


@pytest.mark.parametrize("value", [3.2, -3.2, 0.0])
def test_fmt_delta_always_starts_with_the_arrow_for_the_same_value(value):
    # Consistency identity between the two helpers that share a code path.
    assert fmt_delta(value).startswith(arrow(value) + " ")


@pytest.mark.parametrize(
    ("band", "expected"),
    [("Low", "gain"), ("Moderate", "warn"), ("High", "loss")],
)
def test_band_tone_maps_each_risk_band_to_its_own_tone(band, expected):
    assert band_tone(band) == expected


@pytest.mark.parametrize("band", ["Insufficient data", "", "low", "HIGH", "unknown"])
def test_band_tone_falls_back_to_muted_for_anything_it_does_not_know(band):
    # Case matters: the rubric emits capitalised band names, and a mismatch must
    # grey the gauge out rather than colour it wrongly.
    assert band_tone(band) == "muted"


def test_band_tone_covers_every_band_the_risk_score_can_actually_emit():
    # CROSS-MODULE REGRESSION GUARD. Renaming a band in analytics.risk without
    # updating this map would silently grey out the whole risk gauge, so the
    # band names are collected from the scorer itself rather than assumed.
    bands = {
        risk_score(vol, mdd, [100.0 / n] * n, n, cvar=-0.01)["band"]
        for vol in (0.02, 0.10, 0.20, 0.35, 0.60)
        for mdd in (-0.01, -0.10, -0.25, -0.60)
        for n in (1, 3, 12)
    }
    assert bands == {"Low", "Moderate", "High"}
    assert {band_tone(b) for b in bands} == {"gain", "warn", "loss"}


def test_the_insufficient_data_band_is_greyed_out():
    # risk_score returns this exact string when it cannot score the portfolio.
    degraded = risk_score(float("nan"), float("nan"), None, 0)
    assert degraded["band"] == "Insufficient data"
    assert band_tone(degraded["band"]) == "muted"


# ---- module-wide contract -------------------------------------------------
@pytest.mark.parametrize("formatter", NUMERIC_FORMATTERS)
@pytest.mark.parametrize("value", HOSTILE_INPUTS)
def test_no_formatter_ever_raises_on_a_hostile_input(formatter, value):
    # Formatting runs inside Streamlit's render loop and inside the report
    # writers, where an exception blanks a whole page. Every formatter must
    # degrade to a non-empty string, whatever it is handed.
    result = formatter(value)
    assert isinstance(result, str)
    assert result != ""


@pytest.mark.parametrize("value", HOSTILE_INPUTS)
def test_the_tone_helpers_never_raise_and_stay_inside_their_vocabulary(value):
    assert delta_tone(value) in {"gain", "loss", "muted"}
    assert delta_tone(value, invert=True) in {"gain", "loss", "muted"}
    assert arrow(value) in {UP, DOWN, FLAT, ""}


@pytest.mark.parametrize("value", [None, "abc", 1, 1.5, object()])
def test_band_tone_stays_inside_its_vocabulary_for_any_input(value):
    assert band_tone(value) in {"gain", "warn", "loss", "muted"}
