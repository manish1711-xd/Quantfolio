"""Tests for UI Integration with Institutional Accuracy Panels."""

import pytest
from src.ui.pages.equity_research import render


def test_equity_research_page_imports():
    assert callable(render)
