"""Tests for Data Access Integration with Quarterly TTM Engine."""

import pytest
import numpy as np
import pandas as pd
from src.equity_research.data_access import fetch_research_bundle


def test_fetch_research_bundle_returns_valid_structure():
    # Test bundle assembly offline/mock test
    bundle, warnings = fetch_research_bundle("RELIANCE.NS", period="1y", benchmark="^NSEI")
    assert isinstance(bundle, dict)
    assert "ticker" in bundle
    assert "quote" in bundle
    assert "statements" in bundle
    assert "ohlcv" in bundle
    assert bundle["ticker"] == "RELIANCE.NS"
