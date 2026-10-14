"""Smart Portfolio Risk & Performance Analyzer — core package.

Layered design, dependencies pointing strictly downward with no cycles:

    config, models      constants, conventions and the frozen dataclasses that
                        are the contract between every layer below
    data/               fetcher (the ONLY module that touches the network) and
                        store (SQLite persistence for portfolios and watchlist)
    analytics/          15 pure modules — every formula in the application
    charts/             Plotly figure builders, no Streamlit
    reporting/          Excel workbook and HTML tearsheet writers
    ui/                 design system, session state, orchestration and pages

Three modules sit at this level rather than in a subpackage, because they are
pure, import-light and used by both the analytics and UI layers:

    stock_universe      search bundled stock-master CSVs by company name or
                        ticker, and resolve either onto a Yahoo Finance symbol
    portfolio_importer  detect what a broker export's columns mean, normalise
                        them onto the canonical schema, and report every row
                        that could not be imported and why
    watchlist           instruments tracked without owning them: targets,
                        52-week positioning, distance to a user-set level

The Streamlit app (app.py) is a thin shell over this package. Nothing outside
``ui/`` imports Streamlit and nothing outside ``data/fetcher.py`` imports
yfinance, which is what makes every calculation independently unit-testable
without a browser, a network or market hours.
"""

__version__ = "2.1.0"
