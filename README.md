<div align="center">

# 📊 QuantFolio

**Portfolio risk, performance, optimisation & machine-learning forecasting terminal — built for Indian markets first, working with any Yahoo Finance symbol.**

[![CI](https://github.com/your-handle/quantfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/your-handle/quantfolio/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20|%203.12-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.50%2B-ff4b4b)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5%2B-f7931e)
![License](https://img.shields.io/badge/license-MIT-green)

Full NSE + BSE listing · search by company name · broker-CSV import · FIFO tax lots · XIRR ·
Sortino/Calmar/Omega ·
Cornish–Fisher VaR & CVaR · drawdown episodes · Fama–French factor regression with
Newey–West errors · Black–Litterman · Indian STCG/LTCG tax-loss harvesting ·
Piotroski/Altman/Beneish scores · covered calls & protective puts · efficient frontier · risk parity · Monte Carlo ·
**ML price forecasting with purged walk-forward validation and conformal intervals** ·
market news & economic calendar · historical stress replay · rebalancing trade lists ·
watchlist price alerts ·
scheduled email tearsheets

**217 analytics functions · 51 chart builders · 17 pages · 4,544 tests · 92% coverage**

</div>

---

## Contents

**Start here** — [What this is](#what-this-is) · [How the project works](#how-the-project-works) · [Quickstart](#quickstart)

**Features** — [Getting your portfolio in](#getting-your-portfolio-in) · [Portfolio accounting](#portfolio-accounting) · [Risk analytics](#risk-analytics) · [Attribution](#attribution) · [Benchmark](#benchmark) · [Strategy](#strategy) · [Machine-learning price forecasting](#machine-learning-price-forecasting) · [Research & export](#research--export)

**Operating it** — [Docker](#docker) · [Make](#make) · [Refreshing the stock universe](#refreshing-the-stock-universe) · [Scheduled email tearsheets](#scheduled-email-tearsheets) · [Price alerts](#price-alerts) · [Input formats](#input-formats)

**Under the hood** — [Architecture](#architecture) · [The mathematics](#the-mathematics) · [The machine-learning forecaster](#the-machine-learning-forecaster) · [Project reference](#project-reference) · [Testing](#testing)

**Judging it** — [About "real-time" data](#about-real-time-data) · [Known limitations](#known-limitations) · [**Pros and cons**](#pros-and-cons) · [Troubleshooting](#troubleshooting) · [Version history](#version-history) · [Roadmap](#roadmap) · [Tech stack](#tech-stack)

**Deeper** — [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (layer diagram, data flow, design arguments) · [docs/METRICS.md](docs/METRICS.md) (every formula, convention and failure mode)

---

## What this is

Search "Reliance", or drop in your broker's CSV export unmodified, and get the analytics
a professional would actually run on it: money-weighted *and* time-weighted returns, a decomposition of
where your **risk** lives (which is rarely where your money lives), a mean-variance
efficient frontier with your current portfolio plotted against it, forward simulations,
and a replay of how these exact holdings would have behaved through the COVID crash and
the 2022 rate shock.

And — new in v3 — a **machine-learning price forecaster** that is built the other way
round from most of them: the forecast is one figure of seven, and the other six are all
about whether the model works. Skill is measured out-of-sample against a random walk,
the interval is the model's own held-out error quantiles, and when the fitted models lose
to "assume the price does not move" the page says so in the headline rather than the
footnote.

Every formula is documented in-app, every threshold is a named constant, and every
undefined statistic renders as `—` instead of a fabricated zero.

> **Disclaimer:** educational and analytical purposes only. Nothing here is investment
> advice, a recommendation, or a solicitation to trade. A forecast is a conditional
> statistical estimate, not a prediction of what will happen.

## How the project works

A single pass through the whole system, from a ticker you type to a number on screen.

### The one architectural rule

> **Computation never imports Streamlit. Presentation never computes.
> Every network call lives in `src/data/`.**

Four layers, one direction of travel:

```
     you ──▶ src/ui/          widgets, session state, page layout      (imports Streamlit)
              │
              ├──▶ src/data/         the ONLY network boundary — Yahoo, RSS, exchanges
              │
              ├──▶ src/analytics/    all the mathematics — 23 pure modules, no I/O
              │
              └──▶ src/charts/       51 Plotly builders on one registered template
```

That constraint is what makes ~22,000 lines of financial mathematics testable in
milliseconds, offline, against hand-computed values — no browser, no data subscription,
no market hours. It is also why swapping Yahoo Finance for a broker feed is a rewrite of
**one file** against the same signatures.

### 1 · Getting a portfolio in

Three ways, all landing on the same canonical `ticker, quantity, buy_price` shape:

| Route | What happens |
|---|---|
| **Search** | You type `Reliance`. `src/stock_universe.py` searches ~7,570 bundled NSE+BSE instruments **offline**, ranked by exchange-bhavcopy turnover so the company that actually trades wins the tie. No network call per keystroke. |
| **CSV / broker export** | `src/portfolio_importer.py` detects the columns, normalises `Tradingsymbol`→`ticker`, resolves bare `RELIANCE`→`RELIANCE.NS`, parses `"1,182.95"`, and shows you the mapping and **every rejected row with its reason** before anything is imported. |
| **ISIN only** | `src/isin.py` validates the ISO 6166 check digit, then resolves through a bundled seed, the generated universe (5,175 pairings), and anything learned from your own files — in that order of authority. A valid-but-unknown ISIN is reported as unresolvable, never guessed. |

Optionally you also supply a **trade ledger** (`ticker, trade_date, action, quantity,
price, fees`). That is what unlocks FIFO lots, realised P&L and XIRR — average cost alone
mathematically cannot produce them.

### 2 · One analysis run

Press **Analyze portfolio** and `src/ui/pipeline.py` executes ten stages in a fixed
order. It is the only module that knows the order, which is what keeps seventeen page
modules free of any computation at all:

```
 1. validate       clean_holdings()      bad rows reported and skipped, never fatal
 2. market data    fetch quotes, profiles, FX spot rates (cached by data type)
 3. positions      P&L, weights, FX conversion, portfolio summary
 4. history        adjusted closes → portfolio value series → daily returns
 5. risk           vol, Sharpe/Sortino/Calmar/Omega, VaR/CVaR, drawdown episodes, 0–100 score
 6. benchmark      beta, alpha, tracking error, capture ratios vs your chosen index
 7. structure      correlation, PCA, clustering, covariance (shrunk), risk attribution
 8. rolling        rolling vol / Sharpe / beta / correlation / VaR
 9. ledger         FIFO lots, realised P&L, XIRR on the real cash-flow schedule
10. narrative      rule-based insights, each citing the constant that fired it
```

The result is one `AnalysisResult` dataclass. Every page reads fields off it and renders
them — no page computes anything.

The run is fingerprinted by `session.run_key()`, so switching pages (which reruns the
entire Streamlit script) reuses the cached result instead of refetching the market.

### 3 · Caching, by how fast the data actually moves

| Data | TTL | Why |
|---|---|---|
| Quotes | 5 min | prices move; Yahoo is ~15 min delayed anyway |
| Daily history | 30 min | one more bar appears per day, at most |
| FX rates | 30 min | same cadence, for our purposes |
| Profiles / fundamentals | 24 h | sector and market cap do not change intraday |
| Symbol search / ISIN lookup | 1 h | "what is Reliance's ticker" is close to immutable |

Analytics functions are **not** cached: they are pure and fast, and a cache on a pure
function only adds a stale-state failure mode. The one exception is the forecast page,
which memoises fitted models in session state because a seven-model leaderboard is
seventeen seconds of compute, not a millisecond.

### 4 · Where the forecaster fits

The Price forecast page is the only other page allowed to fetch its own data, and it
fetches something the pipeline deliberately never collects: **5–10 years of daily OHLCV
bars for one instrument**. It has to, and the page says so on screen rather than silently
using a different window from the rest of the app:

- the longest feature lookback is 252 sessions, so a year of history yields no usable row;
- the walk-forward holds out 30% of what remains;
- each fold needs at least 200 rows to train on.

A one-year window cannot produce a single valid fold. Risk statistics elsewhere describe
*the period you chose*; a model needs every row it can get. Two different questions, two
different windows, both stated.

### 5 · Getting numbers back out

| Output | How |
|---|---|
| **Excel** | `src/reporting/excel.py` — formatted workbook with conditional formatting |
| **HTML tearsheet** | `src/reporting/html_report.py` — self-contained, prints cleanly to PDF |
| **Scheduled e-mail** | `python -m src.reporting.cli`, run by cron or Task Scheduler. The app stores the schedule; a real scheduler does the sending, because Streamlit has no background worker and pretending otherwise would mean a toggle that silently does nothing |
| **Price alerts** | The same command. Event-triggered, with direction, hysteresis and a cooldown, so a price parked on your level sends **one** e-mail, not ninety-six |
| **SQLite** | Multiple named portfolios, watchlists and schedules, in `data/quantfolio.db` |

---

## Screenshots

> Add screenshots of the Overview, Risk and Optimiser pages here.
> The efficient frontier and the risk-contribution chart are the two that make the point fastest.

| Overview | Risk analytics | Optimiser |
|---|---|---|
| _`docs/img/overview.png`_ | _`docs/img/risk.png`_ | _`docs/img/optimiser.png`_ |

---

## Features

### Getting your portfolio in
- **Search by company name across the whole Indian market** — type `Reliance`, `HDFC Bank`, `HUL`
  or `Bank Nifty` and pick the instrument. No need to know that the Yahoo Finance symbol is
  `RELIANCE.NS`, or that `.NS` means NSE and `.BO` means BSE. Runs **offline** against
  stock-master files built from the **official NSE and BSE listings** — ~7,570 Indian
  instruments, every one with its ISIN — plus curated US large caps, major ETFs and global
  indices. Typos are tolerated (`relaince` → Reliance Industries).
- **Ranked by what actually trades.** A full-market universe ranks *worse* than a curated one
  unless it knows which companies matter: `hdfc` returned HDFC AMC ahead of HDFC Bank, and the
  typo `relaince` returned Reliance *Communications*, purely on symbol length and alphabetical
  order. So each row carries its **turnover from the exchange bhavcopy**, which becomes a bounded
  ranking bonus — bounded below the gap between match tiers, so a big company can never outrank a
  better match. `python -m src.universe_build` refreshes the whole thing.
- **Smart broker import** — a Zerodha, Groww or Upstox export works *unmodified*.
  `Tradingsymbol` / `Qty` / `Average Price` are mapped automatically, bare `RELIANCE` becomes
  `RELIANCE.NS`, `"1,182.95"` parses, and `Previous Closing Price` is **never** mistaken for your
  cost basis. You see the column mapping and every rejected row **before** anything is imported,
  and no row is ever silently discarded.
- **ISIN → ticker** — an export whose only identifier is an ISIN now imports. Resolution is
  table-driven and never guessed: a bundled seed of 114 widely-held instruments, plus every
  pairing *learned from your own files* (most broker exports carry an ISIN column **and** a
  symbol column, so importing one teaches the mapping for next time), plus an optional provider
  lookup for the long tail. The **ISO 6166 check digit is enforced**, so a mistyped identifier is
  reported as a typo — naming the digit it should have had — rather than resolving to somebody
  else's company.
- **Per-column override** — when a header is one nobody could have an alias for
  (`Cost of acquisition per unit`), point the field at the right column yourself and the whole
  preview is re-derived. A mis-read column is a two-click correction, not a trip back to Excel to
  rename things. The picker shows your file's own first rows so you can tell which column is
  which, and an explicit choice may even override the market-price blocklist — with a loud warning,
  because that decision is yours to make.
- **Watchlist** — track instruments you do not own, with your own target prices, 52-week range
  positioning and a flag for names you already hold. Kept strictly separate from the portfolio: a
  watched name has no quantity and no cost basis, so it never touches P&L, weights or risk.

### Portfolio accounting
- **FIFO tax-lot engine** — realised vs unrealised P&L, per-lot holding periods, short/long-term
  split at the 365-day boundary. Buy fees capitalise into cost basis; sell fees are apportioned
  pro-rata across the lots that sale consumed.
- **XIRR** on the real cash-flow schedule, solved by Brent bracketing with the root verified
  against the NPV — reported next to the time-weighted return, with the difference explained.
- **Multi-currency** with live FX conversion; prices stay native, money columns convert.
- Duplicate tickers merge with a quantity-weighted average cost. Bad rows are reported and
  skipped, never fatal.
- **Tax-loss harvesting under Indian rules** — classifies every FIFO lot against the 12-month
  boundary and prices each unrealised loss by what the set-off rules actually let it absorb, not
  by loss × rate. Models Sections 111A/112A across the **23 July 2024 rate change** (15%/10% →
  20%/12.5%, exemption ₹1L → ₹1.25L), the Section 70/71 set-off hierarchy, the ₹1.25 lakh
  long-term exemption and the 8-year carry-forward. Two findings it surfaces that most
  harvesting tools miss:
  - **A loss can be worth exactly nothing.** With no realised gains of the right character, or
    with long-term gains already inside the exemption, harvesting saves ₹0 today. The app says so
    rather than quoting a notional figure you would pay brokerage and STT to not receive.
  - **For a loss, short-term beats long-term.** A losing lot near its first anniversary is worth
    *more* harvested before the boundary — short-term losses offset both kinds of gain at the
    higher rate. That is the opposite of the instinct that correctly applies to gains, and the
    app flags both cases.

### Risk analytics
- Volatility, **Sharpe, Sortino, Calmar, Omega, Ulcer index, Martin ratio**
- **VaR and CVaR** by three methods — historical, Gaussian and **Cornish–Fisher** (which corrects
  the normal quantile for observed skew and kurtosis) — plus a **Kupiec proportion-of-failures
  backtest** of the VaR model itself
- Skewness, excess kurtosis, tail ratio, hit ratio, gain-to-pain
- **Drawdown episodes** with peak/trough/recovery dates, time underwater and average recovery time
- **Rolling** volatility, Sharpe, Sortino, beta, correlation and VaR
- A transparent **0–100 risk score** whose rubric is rendered as a table, not hidden

### Attribution
- Return contribution per holding, summing exactly to the portfolio return
- **Marginal and component risk contribution (MCTR / CCTR)**, summing exactly to portfolio
  volatility — the table that shows a 10% position carrying 30% of the risk
- Concentration: HHI, effective number of bets, Gini, top-1/3/5 weight
- **PCA** of the return correlation matrix and **hierarchical clustering** of holdings
- **Fama–French style factor regression** — `r_p − r_f = α + Σ βᵢ·Fᵢ + ε` with
  **Newey–West (HAC) standard errors**, so every loading arrives with a t-statistic and an
  honest answer to "is this exposure distinguishable from zero?". Reports variance inflation
  factors (correlated factors make individual loadings unstable while R² stays high),
  Durbin–Watson, idiosyncratic volatility, a joint F-test, and **rolling loadings** so exposure
  drift is visible rather than averaged away. Factors come from a published file (Kenneth
  French's library, the IIM Ahmedabad Indian dataset) or from clearly-labelled ETF proxies —
  never presented as the academic factors when they are not.

### Benchmark
Beta, Jensen's alpha, tracking error, information ratio, Treynor, M², R², and
**up/down capture ratios** on monthly returns — against NIFTY 50, SENSEX, NIFTY Bank,
S&P 500, NASDAQ, FTSE, Nikkei, MSCI World or gold.

### Strategy
- **Efficient frontier** with a random-portfolio cloud, the capital market line, every named
  optimum, and *your current portfolio* plotted as a star so the gap is visible
- **Max Sharpe, min variance, risk parity, max diversification**, long-only with per-name caps
  and Ledoit-Wolf-style covariance shrinkage
- **Black–Litterman** — the answer to the health warning above it. Reverse-optimises a prior
  portfolio into the equilibrium returns the market must already expect (`Π = δΣw`), then moves
  them only as far as your stated views justify, weighted by your confidence. Absolute views
  ("A returns 12%") and relative views ("A beats B by 3%"), Idzorek-style confidence on a 0–100%
  slider, and the posterior fed straight into the same solvers so you can see **Black–Litterman
  weights next to historical-mean weights** with only the expected-return vector changed. With no
  views the posterior is *exactly* the equilibrium, which is what makes it safe to leave on.
- **Monte Carlo** — GBM, IID bootstrap, moving-block bootstrap and Student-t, with a fan chart,
  terminal distribution and reproducible seeds
- **Stress testing** — historical crisis replay with an honest *coverage* column, beta-implied
  shocks, and the worst realised windows
- **Rebalancing** — drift bands, an executable trade list with whole-share flooring, turnover
  and net cash

### Machine-learning price forecasting

A supervised model of the next *h* sessions' return, fitted to one instrument's own price
history — and, given equal billing, the out-of-sample evidence about whether it works.

- **Seven models raced on identical held-out blocks** — random walk, historical drift,
  ridge, elastic net, random forest, histogram gradient boosting, and an inverse-error
  blend. The two baselines are on the leaderboard rather than behind it, because on daily
  equity prices they frequently win, and **that is the finding**.
- **~42 leak-free, scale-free features** across eight families — lagged returns, momentum
  (including 12-1), realised and downside volatility with a regime ratio, distance from
  four moving averages, Wilder RSI / MACD / Bollinger %B / ATR, position in the trailing
  52-week range, volume z-scores, and cyclical calendar encodings. Not one carries the
  price level.
- **Purged walk-forward validation** — sequential expanding-window folds, never shuffled,
  with `h − 1` rows dropped between each training block and its test block because an
  *h*-session label physically overlaps the *h* sessions being scored.
- **Skill quoted against a random walk**, not against zero error. `1 − MAE/MAE_naive`, so
  a model worse than "assume the price does not move" reports a negative number and the
  page leads with it.
- **Diebold–Mariano significance test** with Newey–West errors at `h − 1` lags and the
  Harvey–Leybourne–Newbold small-sample correction — because "better than the benchmark"
  is a claim that needs a test.
- **Conformal prediction intervals** from the model's own held-out residual quantiles,
  signed rather than symmetric. A bad model draws a wide cone automatically.
- **A forecast cone from one model per horizon** (1, 5, 10, 21, 63 sessions), fitted
  directly rather than iterated — so nothing compounds its own error, and a kink between
  two markers is a real disagreement between two fits rather than a plotting artefact.
- **Permutation importance measured out of sample**, so a feature scores zero when the
  model was not using it, whatever its coefficient says.
- **Portfolio-level forecasting**, both top-down (the value series treated as one
  instrument) and bottom-up (a model per holding, weighted) — with the interval reported
  as a *range* between the perfect-correlation and zero-correlation bounds, because the
  forecaster never estimated a correlation between its own errors and inventing one would
  be the least defensible number on the page.

Full detail in **[The machine-learning forecaster](#the-machine-learning-forecaster)**
below.

### Research & export
- **AI-Powered Stock Equity Research Report Generator** (`src/equity_research/` & `src/ui/pages/equity_research.py`):
  - **Single-ticker institutional tearsheet** — enter any stock ticker (e.g., `RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`, `AAPL`) to generate a complete equity research report suitable for an Investment Analyst portfolio.
  - **Comprehensive Fundamental Analysis** — gross margin, operating margin, net profit margin, EPS, ROE, ROCE, Debt/Equity, Debt/Asset, and Free Cash Flow extracted across annual income, balance sheet, and cash flow statements with zero fabricated numbers.
  - **Technical & Risk Engine** — SMA (20, 50, 200), EMA (20, 50), 14-period Wilder exponential RSI (`ewm(alpha=1/period, adjust=False)`), MACD (12, 26, 9), sample annualized volatility (`ddof=1`), Beta vs. benchmark (^NSEI/^GSPC), Sharpe ratio, and maximum drawdown.
  - **5-Year DCF Valuation & Sensitivity Matrix** — geometric cash flow forecasting, WACC discounting, Gordon Growth terminal value `FCF₅ × (1+g) / (WACC - g)`, enterprise-to-equity value bridge, 2D sensitivity matrix heatmap across discount and growth rates, and Bull/Base/Bear scenario bands.
  - **Sector Peer Comparison** — automated comparative valuation against industry peers (P/E, P/B, Market Cap), sector averages, and target company percentile rankings.
  - **Automated Research Narrative** — institutional report structure covering Executive Summary, Company Overview, Financial Analysis, Valuation, Risks & Catalysts, Conclusion, Data Sources, and Modeling Assumptions (with optional LLM enhancement via Anthropic/OpenAI or transparent deterministic analyst templates).
  - **Export to PDF & Excel** — standalone CSS-styled tearsheet convertible to PDF via headless browser, and formatted multi-worksheet Excel financial models with formulas, percentages, currency formats, and sensitivity grids.
  - **SQLite Research History** — saved research teardowns persisted to `data/quantfolio.db` with reload, inspect, and delete capabilities.
- Per-security deep dive: fundamentals, 52-week positioning, candlesticks with Bollinger bands,
  **Wilder RSI**, MACD
- **Fundamental quality scores** — **Piotroski F-Score** (9 signals), **Altman Z-Score**
  (distress, with the right variant per market), **Beneish M-Score** (earnings-manipulation
  screen), **Sloan accruals** and fifteen ratios, computed from the company's own balance sheet,
  income statement and cash-flow statement. Every score reports **how many of its inputs were
  available**: a score built from six of nine signals shows as six of nine, never as a low nine,
  because counting a missing signal as a failure is what makes a healthy company look distressed.
- **Options overlay** — covered calls, protective puts and collars on a holding, with
  Black–Scholes–Merton pricing, all five Greeks, implied volatility solved by Brent bracketing,
  and payoff curves plotted **against holding the shares alone** (an overlay payoff on its own
  says nothing about what it cost). Reports the static return, the if-called return, the
  breakeven, the floor, the upside surrendered and the risk-neutral chance of assignment — and
  refuses to call a covered call "limited risk", because the underlying can still go to zero.
- Formatted **Excel workbook**, self-contained **HTML tearsheet** that prints cleanly to PDF, CSV
- **Scheduled email tearsheets** — daily, weekly or monthly, with the headline numbers as plain
  text in the body and the full tearsheet attached. The app stores the schedule; a `python -m
  src.reporting.cli` command run by **cron or Windows Task Scheduler** does the sending, because
  Streamlit has no background scheduler and pretending otherwise would mean a toggle that
  silently does nothing. Sending is idempotent — a five-minute cron delivers an 08:00 report
  exactly once, and a slot missed while the machine was asleep goes out once when it wakes.
  SMTP credentials are read from `st.secrets` or the environment and are **never** stored by the
  application.
- **Watchlist price alerts** — be e-mailed when a watched instrument reaches a level you set.
  Each rule carries a **direction** (*rises to* / *falls to*), which is the fact a bare target
  cannot supply, and fires on the **transition** rather than the condition: one e-mail when the
  level is reached, then silence until the price retreats past it by a re-arm band. A
  ninety-six-poll day on a price parked above the level sends one e-mail, not ninety-six. The
  session high/low is consulted as well as the last price so a crossing between two checks is
  not missed — and every alert names which observation saw it, because "the price is there now"
  and "it was touched at some point today" are different facts.
- **News & economic calendar** — live market headlines from Economic Times, Business Standard and
  LiveMint, per-stock headlines from Yahoo, and the ForexFactory economic calendar with its
  **High/Medium/Low impact** ratings so you can see which days a broad move is most likely.
  Headlines are matched to **your** holdings and watchlist, and every match names the term that
  found it. **A feed that answers is not a feed that works**: freshness is measured and a source
  that has stopped publishing is rejected with its age shown — MoneyControl's RSS endpoints still
  return valid XML frozen at April 2024, and presenting that as today's news is the failure this
  guard exists to prevent.
- **SQLite persistence** for multiple named portfolios

---

## Quickstart

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open <http://localhost:8501>, click **Load sample** in the sidebar — a nine-holding Indian
portfolio with a US name to exercise FX, plus an eighteen-trade ledger — then **Analyze portfolio**.

Then open **Research › Price forecast** and press nothing: it fits a ridge model on ten
years of daily bars, validates it out-of-sample against a random walk, and tells you in
one sentence whether the forecast it just drew is worth reading.

To try the other two ways in:

- **Search** — on *Portfolio › Build*, type `Reliance` (or `HUL`, or `Bank Nifty`) into
  **Search stock**, pick the row, enter a quantity and average cost.
- **ISIN-only import** — upload `data/sample_isin_only.csv`, an export whose only identifier is
  an ISIN. Six resolve through the bundled table, one is a real UK security the table does not
  cover (reported, with an online-lookup button), and one is a deliberate typo caught by the
  check digit.
- **Import** — on the same page, open **Import from CSV / broker export** and upload
  `data/sample_broker_export.csv`. It is a deliberately awkward Zerodha-shaped file: thousands
  separators, an ISIN column, a `Previous Closing Price` column that must *not* be read as your
  cost basis, `INFY` (which lists on two exchanges), and one unusable row. You will see the
  detected mapping, the nine rows that will import, and the one that will not — with the reason.

### Docker

```bash
docker build -t quantfolio .
docker run --rm -p 8501:8501 -v "$(pwd)/data:/app/data" quantfolio
```

Runs as an unprivileged user with a healthcheck; `data/` is a volume so saved portfolios and
your watchlist persist.

> **Mount the repo's own `data/` directory, not an empty one.** The bundled stock-master CSVs
> (`indian_stocks.csv`, `us_stocks.csv`, `global_indices.csv`) live in `data/`, which is also the
> declared volume — so bind-mounting an empty host directory over it hides them and offline search
> silently returns nothing. The app degrades gracefully rather than crashing (symbols pass straight
> through to the data layer and **Search online** still works), but name search stops finding
> anything until the files are present. `-v "$(pwd)/data:/app/data"`, as above, is correct.

### Make

```bash
make install-dev   # runtime + dev dependencies
make run           # launch the dashboard
make test          # offline test suite
make check         # lint + format check + types + tests (what CI runs)
make universe      # rebuild the stock master from the NSE/BSE listings
make universe-dry  # ...and report what would change without writing
make mail-list     # scheduled tearsheets: what is configured, and what is due
make mail-dry      # build them and validate SMTP, sending nothing
make mail-send     # send tearsheets and check price alerts (what cron should run)
make mail-alerts   # check watchlist price alerts only
```

---

## Refreshing the stock universe

The Indian half of the universe is generated from the exchanges' own files. Refresh it
whenever you want; nothing in the running app ever downloads it, so search stays instant,
offline and deterministic.

```bash
python -m src.universe_build --dry-run   # report what would change, write nothing
python -m src.universe_build             # refresh data/indian_stocks.csv
python -m src.universe_build -v          # ...and log every download attempt
```

Four public files, no API key and **no new dependency** (stdlib `urllib` with a browser
User-Agent, which NSE's CDN requires):

| Source | Rows | Supplies |
|---|--:|---|
| NSE `EQUITY_L.csv` | ~2,570 | symbol, company name, series, **ISIN** |
| BSE scrip master (JSON) | ~4,990 | `scrip_id`, issuer name, **ISIN**, industry |
| NSE bhavcopy (UDiFF, zipped) | ~3,630 | **turnover** for ranking |
| BSE bhavcopy (UDiFF) | ~5,000 | **turnover** for ranking |

Two inputs and one output, deliberately: `data/indian_stocks_curated.csv` is the
hand-maintained seed (aliases like `RIL`, sectors NSE does not publish, and the eleven NSE
ETFs that appear in no *equity* listing) and is **never written to**. `data/indian_stocks.csv`
is the generated merge, backed up to `.bak` before each overwrite. `data/universe_meta.json`
records which bhavcopy date the turnover came from, which the search UI displays.

What it refuses to do: write a truncated universe if a download fails, write to its own input,
drop a curated alias, or keep a symbol that has stopped existing. On the run that produced the
committed file it dropped four — `TATAMOTORS.NS` (demerged into `TMCV`/`TMPV`), `LTIM.NS`,
`AKZOINDIA.NS` and `GUJGASLTD.NS` — each absent from the listing *and* showing no turnover, and
each independently confirmed unpriceable by Yahoo. Two sources must agree before anything is
dropped, and every drop is named in the report.

---

## Scheduled email tearsheets

Two halves, because only one of them can live inside a Streamlit app.

**1. Give the app somewhere to send from.** Copy `.streamlit/secrets.toml.example` to
`.streamlit/secrets.toml` — the target is listed in `.gitignore` and `.dockerignore`, and the
app never writes a credential there or anywhere else:

```toml
[smtp]
host = "smtp.gmail.com"
port = 587                # 587 = STARTTLS, 465 = implicit SSL; inferred from the port
username = "you@example.com"
password = "an-app-specific-password"   # NOT your account password
sender = "you@example.com"
sender_name = "QuantFolio"
```

Gmail and Outlook both reject a normal login password for SMTP and require an app-specific one;
the app says so by name when the server refuses the login. For a cron job, export
`SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD` and `SMTP_SENDER` instead — the same
loader reads either. TLS is verified against the system trust store with hostname checking on,
and sending a password over an unencrypted connection is **refused** unless you pass
`--allow-insecure` and mean it.

**2. Configure the schedule in the app, then schedule the command.** Save a portfolio in the
sidebar (a cron run has no browser session, so it rebuilds the analysis from saved holdings),
then add a schedule on the **Export** page. Check it:

```bash
python -m src.reporting.cli --list       # what is configured, and what is due
python -m src.reporting.cli --dry-run    # build and validate; send nothing
python -m src.reporting.cli --force --only "Weekly review"
python -m src.reporting.cli --now 2026-09-07T08:00 --list   # verify the timing
```

Then hand it to a real scheduler:

```bash
# cron — every 15 minutes is fine; see the note on idempotence below
*/15 * * * * cd /srv/quantfolio && .venv/bin/python -m src.reporting.cli >> /var/log/quantfolio-mail.log 2>&1
```

```powershell
schtasks /Create /TN QuantFolioMail /SC MINUTE /MO 15 ^
  /TR "cmd /c cd /d C:\path\to\portfolio_analyzer && .venv\Scripts\python.exe -m src.reporting.cli"
```

### Price alerts

Alerts ride the same command. They are **event**-triggered rather than
time-triggered, so there is no cadence to configure: the rules are checked on every
invocation, and the cron interval simply *is* the check interval. Alerts want a short
one (five minutes); tearsheets do not care, which is why one entry covers both.

Set them up on the **Watchlist** page: give an instrument a target, choose whether you
want to hear about the price *rising to* or *falling to* it, tick the alert box, then add
the recipients under *Alert delivery and sensitivity*. Alerts are **off by default** and a
target alone never sends anything.

```bash
python -m src.reporting.cli --list        # every rule, and whether it can fire
python -m src.reporting.cli --alerts      # check alerts only (skip tearsheets)
python -m src.reporting.cli --no-alerts   # tearsheets only
```

Two knobs control how hard a rule is to re-trigger, and the defaults are usually right:

| Setting | Default | What it stops |
|---|---|---|
| Re-arm band | 1% past the level | A price wobbling on a round number producing a burst of e-mails |
| Minimum hours between alerts | 12 h | A genuinely volatile price clearing even that band repeatedly |

Measured: a price swinging 100 → 115 → 100 all day clears a 1% band on every leg, which is
**144 e-mails** over a day of five-minute polls. The 12-hour cooldown makes it **2**. Both
numbers are pinned in the test suite, because the contrast is the property.

**Why running it every 15 minutes is safe.** `is_due` compares `last_sent` against the most
recent scheduled *occurrence*, not against the wall clock, so each occurrence is delivered at
most once however often the command runs — and a slot missed while the machine was asleep is
still delivered once, whenever the scheduler next fires. `last_sent` is written only *after* a
successful send, and stamped with the occurrence rather than the send time, so a crash between
the two costs one duplicate at worst instead of losing a report.

Exit codes, for a scheduler that branches on them: `0` sent or nothing due · `1` a schedule
failed · `2` could not start (no SMTP settings, unreadable database).

---

## Input formats

You do not have to produce these shapes by hand — **Portfolio › Build › Import from CSV**
accepts a broker export as-is and shows you exactly how it read each column. These are the
canonical names it normalises onto.

**Holdings** (`data/sample_portfolio.csv`) — column names are matched case- and space-insensitively:

```csv
ticker,quantity,buy_price,buy_date
RELIANCE.NS,17,1268.40,2023-03-14
NIFTYBEES.NS,137,248.90,2023-02-06
AAPL,6,182.50,2024-02-14
```

**Trade ledger** (`data/sample_transactions.csv`) — optional, and what unlocks FIFO lots,
realised P&L and XIRR:

```csv
ticker,trade_date,action,quantity,price,fees
RELIANCE.NS,2023-03-14,BUY,8,1152.00,28.80
INFY.NS,2024-07-01,SELL,4,1806.00,27.10
```

Indian equities use the Yahoo suffix — `.NS` (NSE) or `.BO` (BSE). ETFs (`NIFTYBEES.NS`,
`GOLDBEES.NS`) and international tickers (`AAPL`, `MSFT`) work unchanged. **You rarely need to
type these**: search finds them by name, and import resolves a broker's bare symbols for you.

### Column names the importer recognises

| Canonical | Also accepted |
|---|---|
| `ticker` | `symbol`, `tradingsymbol`, `stock`, `security`, `instrument`, `scrip`, `stock_name`, `company_name` |
| `isin` | `isin_code`, `isin_number`, `security_isin`, `instrument_isin` |
| `quantity` | `qty`, `shares`, `units`, `quantity_available`, `no_of_shares`, `net_quantity` |
| `buy_price` | `average_price`, `avg_price`, `average_buy_price`, `avg_cost`, `cost_price`, `purchase_price`, `buy_rate` |
| `purchase_date` / `trade_date` | `date`, `transaction_date`, `order_date`, `execution_date`, `buy_date` |
| `action` | `trade_type`, `transaction_type`, `side`, `buy_sell`, `type` |
| `price` | `trade_price`, `execution_price`, `unit_price`, `rate` |
| `fees` | `brokerage`, `charges`, `commission`, `transaction_charges`, `stt` |

`BUY` also accepts `B`, `Bought`, `Purchase`; `SELL` also accepts `S`, `Sold`, `Sale`.

Deliberately **never** read as a cost basis, however they are named: `LTP`, `Last Price`,
`Current Price`, `Previous Closing Price`, `Market Value`, `P&L`. Mapping current market data onto
`buy_price` would report a ~0% return on the entire portfolio, and nothing on the screen would look
wrong — so the importer refuses the file instead and says which column it could not find.

---

## Architecture

> **Computation never imports Streamlit. Presentation never computes.
> Every network call lives in `src/data/`.**

```
app.py                    shell: page config, navigation, error boundary      124 lines
src/config.py             every constant and convention, in one auditable place
src/models.py             domain dataclasses shared by all layers
src/stock_universe.py     offline company-name search over bundled master CSVs
src/portfolio_importer.py broker-agnostic CSV import with a reviewable preview
src/isin.py               ISO 6166 validation and ISIN -> symbol resolution
src/watchlist.py          watched instruments, targets, 52-week positioning
src/alerts.py             price-alert rules: direction, hysteresis, fire-once state
src/universe_build.py     build the stock master from the NSE/BSE listings + bhavcopy
src/data/                 fetcher + exchange + news (the ONLY network modules) 3.3k lines
                          + SQLite store
src/analytics/            23 pure modules — all the mathematics                 22k lines
  ├─ features.py          leak-free, scale-free supervised feature matrix
  └─ forecast.py          forward-return models, purged walk-forward, conformal
src/charts/               51 Plotly builders on one registered template         9.2k lines
src/ui/                   design system, session, orchestration + 17 pages      21k lines
src/reporting/            Excel + HTML writers, SMTP delivery, cron runner      5.7k lines
data/                     stock-master CSVs (full NSE+BSE, US, indices) + samples
tests/                    40 modules, 4,544 tests, 92% coverage                 37k lines
```

That constraint is what makes the financial mathematics testable in milliseconds, offline,
against hand-computed values — no browser, no market data subscription, no market hours.
It also means swapping Yahoo Finance for a broker feed (Zerodha Kite, Upstox, IBKR) is a
rewrite of **one file** against the same signatures.

See **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for the layer diagram, data flow,
caching strategy and error philosophy.

---

## The mathematics

Every formula, convention and failure mode is documented in
**[docs/METRICS.md](docs/METRICS.md)**. A few of the choices that matter:

- **Undefined is not zero.** A Sharpe ratio at zero volatility is undefined. The app renders
  `—`. Printing `0.00` there is a lie, and lying about a number is worse than omitting it.
- **VaR is a negative decimal.** `−0.021` means a 2.1% daily loss. Sign conventions are stated
  in every docstring because mixing them is the most common bug in risk code.
- **RSI uses Wilder's smoothing**, not a rolling mean — and there is a regression test for it.
- **Factor standard errors are Newey–West, not OLS.** Return residuals are heteroskedastic and
  both factors and residuals are often serially correlated; classical OLS standard errors assume
  neither and are therefore too small. On persistent data the error is not marginal — a t-stat of
  16 becomes 8 once the correction is applied. Reporting the uncorrected number would turn noise
  into "significant" factor exposure, which is the single most common way a factor regression
  misleads its reader.
- **Risk contributions sum to portfolio volatility**, return contributions sum to the portfolio
  return. Both identities are asserted in the test suite; they are what make the numbers
  comparable at all.
- **Stress scenarios report coverage.** A stock listed in 2021 has no COVID-crash history.
  Treating it as flat would fabricate a reassuring number, so the app reports what fraction of
  portfolio weight actually had data — and refuses to score a scenario at zero coverage.
- **The optimiser is candid about its own fragility.** Expected returns from historical means
  are extremely noisy; the app shrinks the covariance, caps weights, and says plainly that
  min-variance and risk parity — which need no return forecast — are the more trustworthy
  outputs on that page. **Black–Litterman is the constructive fix**, and it is on the same page:
  equilibrium returns are a far better prior than a five-year sample mean, and a view only moves
  them in proportion to the confidence you attach to it.
- **Black–Litterman's "market" is not the market.** The real model uses the global cap-weighted
  market portfolio. A portfolio tool has your holdings, so it cap-weights *those* — a narrower
  claim, stated as such, with your current weights or equal weights offered as explicit
  alternatives.
- **A forecast is scored against a random walk, not against zero error.** Mean absolute
  error alone makes every price model look good, because daily log returns are small
  numbers. `skill = 1 − MAE/MAE_naive` puts the benchmark at zero, so a model worse than
  "assume no change" reports a negative number — and negative is the ordinary result.
- **The walk-forward is purged.** An *h*-session label spans *h* future sessions, so the
  last `h − 1` training labels overlap the test block. Dropping them is the difference
  between a measured error and a fabricated one, and it is the single most common way a
  published stock forecaster reports accuracy it does not have.
- **The prediction interval is conformal, not Gaussian.** It is the empirical quantile
  range of the model's own out-of-sample residuals, so a model with no skill draws a wide
  cone without anyone deciding to widen it. Signed quantiles, because equity return errors
  are skewed and a symmetric band would understate exactly the tail that matters.

---

## The machine-learning forecaster

Everything in this section lives in `src/analytics/features.py`,
`src/analytics/forecast.py`, `src/charts/forecast.py` and
`src/ui/pages/forecast.py`. Formula-level detail is in
**[docs/METRICS.md § 13](docs/METRICS.md)**; the design argument is in
**[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

### What it actually predicts

One number: the **log return over the next _h_ trading sessions**.

```
label:    y_t = log( C[t+h] / C[t] )
features: x_t = f(observations up to and including t)
model:    ŷ = m(x_t),   fitted on (x, y) pairs, validated on blocks it never saw
output:   Ŝ = C[t] · exp(ŷ)   ← the MEDIAN of the implied distribution, not its mean
```

Log rather than simple return for two reasons that both bite: it is additive across time,
so a 21-day label is exactly the sum of its 21 daily parts, and it is far closer to
symmetric — which every squared-error loss in the model layer implicitly assumes.

### The models

| Key | Model | Family | What it assumes | Where it breaks |
|---|---|---|---|---|
| `random_walk` | **Random walk** | baseline | Prices are a martingale: today's price is the best estimate of every future price | It cannot be right about anything specific. It is here because it is the opponent every other model must beat — and frequently is not beaten |
| `drift` | **Historical drift** | baseline | The mean return is stable and the future looks like the average of the past | The standard error of a mean return from a few years of daily data is usually larger than the estimate |
| `ridge` | **Ridge regression** (`RidgeCV`) | linear | A linear relationship, with correlated features sharing a coefficient rather than fighting over it | Cannot represent an interaction or a threshold; one coefficient for every regime |
| `elastic_net` | **Elastic net** (`ElasticNetCV`) | linear | Most features carry no signal; L1 sets the rest to exactly zero | With correlated features, which member of a group survives is close to arbitrary |
| `random_forest` | **Random forest** | tree ensemble | The relationship involves thresholds and interactions a linear model cannot see | Cannot extrapolate a trend; averaging pulls every prediction toward the sample mean |
| `gradient_boost` | **Histogram gradient boosting** | tree ensemble | A weak learner applied to what is left over can approximate almost any function | The most flexible model here, so on near-zero signal-to-noise it is the likeliest to fit noise |
| `ensemble` | **Inverse-error blend** | blend | The members err partly independently, so averaging cancels variance without adding bias | If every member is wrong the same way — which is what a regime change produces — blending changes nothing but the confidence projected |

Hyper-parameters live in `config.ForecastConfig`, not scattered through the code:
regularisation grids, tree depth (5), minimum leaf size (25), learning rate (0.05),
300 estimators, seed 42.

**Three implementation decisions worth naming:**

- **The elastic net standardises its target as well as its features.** A log return has a
  standard deviation near 0.02 and coordinate descent's tolerance scales with the target's
  variance; unscaled, it hits `max_iter` and warns on nearly every fold. It also uses an
  explicit alpha grid rather than the automatic path, whose lower end reaches alphas where
  descent on 42 collinear price features simply does not converge.
- **Gradient boosting runs with early stopping OFF.** scikit-learn carves its validation
  slice out with a *shuffled* split, which on a time series means stopping on rows inside
  the training window's own future. The iteration count is fixed and the regularisation is
  carried by depth, leaf size and an explicit L2 penalty instead.
- **The ensemble learns its weights on an inner, later holdout** inside each training
  block, then refits every member on the whole block. The weighting never sees a row
  outside the training block, so the blend is exactly as leak-free as its members.

**Why no LSTM, transformer or other deep model.** Not an oversight. A daily bar history is
a few thousand rows with a signal-to-noise ratio near zero; a recurrent or attention model
carries millions of parameters. That combination memorises its training set. It would also
add a dependency larger than the rest of this application put together and destroy the
"offline, deterministic, 160 seconds" property the whole test suite rests on. The
constraint is **data volume**, and it is stated rather than quietly worked around.

### The features

Eight families, ~42 columns, every one computed from observations **up to and including**
the session it sits on, and every one unit-free.

| Family | Columns | What it measures |
|---|---|---|
| `returns` | `ret_lag_1…10` | Individual lagged daily log returns — short-horizon autocorrelation |
| `momentum` | `mom_10…252`, `mom_12_1` | Cumulative log return per window, plus 12-month momentum skipping the last month |
| `volatility` | `vol_5/21/63`, `vol_ratio`, `downside_vol_21`, `skew_63`, `kurtosis_63` | Realised vol, the short/long regime ratio, downside-only vol, and higher moments |
| `trend` | `dist_sma_10/20/50/200`, `sma_20_50_spread`, `sma_50_200_spread` | Distance from each moving average and the golden/death-cross geometry, as continuous numbers |
| `oscillator` | `rsi_14`, `macd_hist`, `bb_percent_b`, `bb_bandwidth`, `atr_ratio` | Wilder RSI re-centred on 50, MACD histogram and ATR both divided by price |
| `range` | `range_position_252`, `high_252_gap`, `low_252_gap`, `bar_range`, `close_location`, `overnight_gap` | Position in the trailing 52-week range, the shape of the bar, the overnight gap |
| `volume` | `volume_z_21`, `volume_ratio` | Volume against its own 21-session norm, and the short/long ratio |
| `calendar` | `dow_sin/cos`, `month_sin/cos`, `turn_of_month` | Cyclical encodings — a linear model handed `day_of_week = 4` learns that Friday is four times Tuesday |

**Nothing carries the price level.** A model handed the raw close learns "the number went
up historically, so it goes up", which fits a trend perfectly in sample and transfers to
nothing. The property is pinned by a test: multiplying every price by 7.5 must not move a
single feature.

A close-only series — which is exactly what the portfolio value series is — has no bar and
no volume, so those families are **skipped rather than fabricated**, and the UI reports
which ones it is working without.

### How it is validated

This is the part that decides whether anything above is science or decoration.

```
       ├─── train ────┤ purge ├─ test ─┤
       ├──── train ─────────┤ purge ├─ test ─┤
       ├──── train ───────────────┤ purge ├─ test ─┤
       ├──── train ─────────────────────┤ purge ├─ test ─┤
       ├──── train ───────────────────────────┤ purge ├─ test ─┤
       0                                                      n
```

- **Sequential expanding-window folds.** Five of them, over the last 30% of the rows. The
  data is **never shuffled** — a shuffled split lets a model train on Thursday to predict
  Wednesday, and that is precisely how a published stock predictor reports 95% accuracy.
- **A purge of `h − 1 + 1` rows** between every training block and the test block that
  follows it. A label at horizon *h* spans *h* future sessions, so without the gap the
  last few training labels are built from the very returns the test block is about to be
  scored on. At a 63-session horizon that is **three months of the answer** sitting inside
  the training set.
- **Scaling happens inside the fold.** `StandardScaler` lives in a `Pipeline`, so no fold
  ever sees a mean computed from its own test block.
- **Nothing is reported from the training fit.** Every error, every hit rate, every
  interval on the page comes from held-out blocks.

### How to read the numbers

| Metric | Formula | Benchmark | Sign convention |
|---|---|---|---|
| **Skill** | `1 − MAE_model / MAE_random-walk` | **0** | Positive beats a random walk. **Negative is the normal result** and means the model is worse than assuming no change |
| **MASE** | `MAE_model / MAE_random-walk` | **1.0** | Below 1.0 beats the benchmark |
| **Out-of-sample R²** | `1 − Σ(y−ŷ)² / Σy²` | **0** | Campbell–Thompson. Negative means the model explains less than nothing. Low **single-digit percentages** are considered strong in the published literature |
| **Directional accuracy** | share of correct signs, where the model called one | **the base rate**, not 50% | An instrument that rose 58% of the time makes a 55% hit rate *worse* than always saying up — so the base rate sits next to it |
| **IC** | Spearman rank correlation of `ŷ` against `y` | **0** | A model can order outcomes usefully while getting every magnitude wrong |
| **Diebold–Mariano _p_** | test of equal squared-error loss | **≤ 0.10** | A positive statistic favours the model; a small _p_ says the difference is unlikely to be luck |

A constant-zero forecast calls no direction at all, so the random walk reports directional
accuracy as `—` rather than 0%: scoring it as zero would read as "always wrong" instead of
"never answered the question".

### The prediction interval

Split-conformal, straight from the model's own out-of-sample residuals `r = y − ŷ`:

```
interval = [ ŷ + Q(r, α/2),  ŷ + Q(r, 1 − α/2) ]        default α = 0.20 → an 80% band
P(higher) = share of (ŷ + r) above zero                 no Normal assumption anywhere
```

Three consequences, all deliberate:

- **A bad model produces a wide band automatically.** Nobody has to decide to widen it.
- **The quantiles are signed, not symmetric.** Equity returns are left-skewed and so are
  the errors of a model fitted to them; a symmetric `±Q(|r|)` band would overstate the
  upside and understate the downside by exactly the same amount.
- **The band is not centred on the point forecast** when the model has a measured bias —
  and the median residual is reported as exactly that, rather than the band being quietly
  re-centred on zero.

Coverage is **marginal, not conditional**: the band covers its nominal share of outcomes
*on average across the backtest*, and does not widen because today happens to be a
volatile day.

### What the page shows you, in order

1. **The verdict** — one sentence, before any chart. *"Ridge regression does not beat a
   random walk at 21 sessions"* is a headline, not a footnote.
2. **The forecast** — point, interval, probability of a rise, and the measured skill,
   in the same KPI row.
3. **The cone** — one independently fitted model per horizon, markers on the fits and a
   dotted line on the interpolation between them.
4. **The evidence** — rule-based insights, each citing the constant that fired it.
5. **Is it any good?** — directional accuracy against the base rate, out-of-sample R²,
   rank correlation, Diebold–Mariano _p_; then the prediction timeline, the
   predicted-vs-realised scatter, the error distribution, and a **fold-by-fold table**,
   which is the only thing that can show a model that worked until a regime change and
   then stopped.
6. **The leaderboard** — seven models, identical held-out rows, sorted by skill, with the
   baselines included.
7. **What the model used** — permutation importance by feature family, plus the full
   catalogue of what it was offered.

### Using it

```bash
streamlit run app.py     # → Research › Price forecast
```

Pick an instrument, a training history (5y / 10y / max), a horizon (1 / 5 / 10 / 21 / 63
sessions) and a model — or **Auto**, which races all seven and picks the winner. The
**Portfolio** tab forecasts your value series top-down and, on request, every holding
bottom-up.

From Python, with no Streamlit anywhere:

```python
from src.data.fetcher import fetch_ohlcv
from src.analytics import forecast as fc

bars = fetch_ohlcv("RELIANCE.NS", "10y")

board = fc.model_leaderboard(bars, horizon=21)      # race every model
run   = fc.run_forecast(bars, 21, fc.best_model(board))

print(run["predicted_price"], run["lower_price"], run["upper_price"])
print(run["metrics"]["skill"], run["metrics"]["dm_p"])
print(run["note"])                                   # plain-English verdict

path = fc.forecast_path(bars, (1, 5, 10, 21, 63), "ridge")   # the cone
```

### What it cannot do

Stated here rather than discovered later:

- **It cannot see anything but price.** No earnings, no news, no order flow, no macro, no
  other instrument. A model built on one series' own history is a weak-form efficiency
  test, and weak-form efficiency mostly holds.
- **It will usually report no skill, and that is the correct answer.** On real daily
  equity data the random walk is a genuinely hard opponent. A page that hid that would be
  the problem.
- **Forecast accuracy is not profit.** Nothing here models a transaction cost, a spread,
  slippage, market impact, position sizing, borrow cost or tax. A model with real
  statistical skill can lose money after all six. This is **not** a strategy backtest.
- **It cannot survive a regime change**, and has no way to know one has started. The
  fold-by-fold table is the only warning available, and it is retrospective.
- **Skill on one instrument over one window is a weak claim.** Test the same model on ten
  names before believing any of them; the leaderboard makes that easy and the result is
  usually humbling.
- **The interval is marginal, not conditional** (see above).
- **The provider's history is split- and dividend-adjusted as of today**, so features on
  an old row use numbers that were not knowable on that date. Small for a liquid equity,
  real for one with a corporate-action history.
- **The portfolio interval is a range, not a number.** The forecaster never estimated a
  correlation between its own errors across holdings, so the comonotonic and independent
  bounds are both shown and the truth is between them.

---

## Project reference

Everything the repository contains, in one place. Line counts are from the
working tree; every module's description is its own docstring's first line.

### The seventeen pages

| Group | Page | URL | What it answers |
|---|---|---|---|
| Portfolio | **Build** | `/portfolio` | Enter holdings and an optional trade ledger — search, CSV import, ISIN resolution, validation preview. The landing screen |
| Portfolio | **Overview** | `/overview` | The command centre: headline KPIs, allocation, insights, data freshness |
| Portfolio | **Holdings** | `/holdings` | The position blotter — per-name P&L, weight, sector, and the shape of the book |
| Portfolio | **Performance** | `/performance` | How the return was earned and when: growth, monthly seasonality, contribution |
| Portfolio | **Trades & lots** | `/trades` | FIFO tax lots, realised P&L, holding periods, XIRR on the real cash-flow schedule |
| Risk | **Risk analytics** | `/risk` | What this book can lose, how often, and where it comes from |
| Risk | **Benchmark** | `/benchmark` | Beta, alpha, tracking error, information ratio, up/down capture |
| Risk | **Correlation & factors** | `/correlation` | How many independent bets the book really owns — correlation, PCA, clustering, factor regression |
| Risk | **Stress testing** | `/stress` | Historical crisis replay with a coverage column, plus beta-implied shocks |
| Strategy | **Optimiser** | `/optimiser` | Efficient frontier, max-Sharpe / min-variance / risk parity / max diversification, Black–Litterman |
| Strategy | **Monte Carlo** | `/simulation` | GBM, IID bootstrap, block bootstrap and Student-t forward simulation |
| Strategy | **Rebalance** | `/rebalance` | Drift bands to an executable order ticket with whole-share flooring |
| Research | **Security deep dive** | `/security` | One instrument: fundamentals, candles, RSI/MACD/Bollinger, options overlay, quality scores |
| Research | **Price forecast** | `/forecast` | A machine-learning model, its forecast, and the out-of-sample evidence about whether it works |
| Research | **Watchlist** | `/watchlist` | Instruments tracked without holding them, with targets and price alerts |
| Research | **News & calendar** | `/news` | Market headlines matched to your book, plus the week's economic calendar |
| Research | **Export** | `/reports` | Excel, HTML tearsheet, CSV, and the scheduled-email configuration |

Every page module exposes exactly one public `render()`. That contract is a
test (`test_every_page_module_exposes_render`), and each page is additionally
mounted in isolation against a complete `AnalysisResult` *and* against no
result at all, so both the working state and the guard state are exercised.

### Module map

**Root — `src/`**

| Module | Lines | Responsibility |
|---|--:|---|
| `config.py` | 808 | Every tunable constant and convention, in one auditable place |
| `models.py` | 349 | Domain dataclasses shared by all layers |
| `stock_universe.py` | 896 | Offline company-name search over the bundled stock masters |
| `portfolio_importer.py` | 1,507 | Broker-agnostic CSV/Excel import with a reviewable preview |
| `isin.py` | 555 | ISO 6166 validation, ISIN → Yahoo symbol resolution and learning |
| `watchlist.py` | 682 | Instruments tracked without holding them |
| `alerts.py` | 709 | Price-alert rules: direction, crossing, hysteresis, fire-once state |
| `universe_build.py` | 829 | Build the stock masters from the official NSE/BSE files |

**Analytics — `src/analytics/`, 23 modules, ~22k lines, 217 public functions**

| Module | Lines | Fns | Responsibility |
|---|--:|--:|---|
| `positions.py` | 606 | 5 | Position P&L, weights, FX conversion, portfolio aggregation |
| `lots.py` | 655 | 7 | FIFO tax lots, realised trades, holding periods |
| `returns.py` | 826 | 12 | XIRR, TWR, Modified Dietz, monthly return tables |
| `risk.py` | 1,206 | 28 | Dispersion, tail risk, drawdown, the 0–100 risk score |
| `drawdown.py` | 438 | 6 | Peak → trough → recovery episodes |
| `rolling.py` | 430 | 7 | Rolling volatility, Sharpe, beta, correlation, VaR |
| `benchmark.py` | 689 | 9 | Relative return, relative risk and capture ratios |
| `attribution.py` | 806 | 7 | Return contribution and marginal/component risk contribution |
| `factors.py` | 677 | 6 | PCA and correlation-distance clustering (unsupervised) |
| `factor_model.py` | 1,021 | 11 | Fama–French style regression with Newey–West (HAC) errors |
| `black_litterman.py` | 806 | 11 | Equilibrium returns blended with explicit investor views |
| `optimize.py` | 1,603 | 12 | Efficient frontier, min-variance, max-Sharpe, risk parity |
| `montecarlo.py` | 867 | 6 | GBM, bootstrap, block bootstrap and Student-t simulation |
| `stress.py` | 1,076 | 5 | Historical scenario replay and beta-implied shocks |
| `options.py` | 1,013 | 13 | Black–Scholes, Greeks, covered-call / protective-put overlays |
| `tax.py` | 901 | 13 | Indian STCG/LTCG rules, set-off waterfall, loss harvesting |
| `fundamentals.py` | 1,128 | 9 | Piotroski, Altman, Beneish, accruals — scores, not forecasts |
| `technicals.py` | 877 | 12 | SMA/EMA, Wilder RSI, MACD, Bollinger, ATR |
| **`features.py`** | 669 | 9 | **Leak-free, scale-free supervised feature matrix and its label** |
| **`forecast.py`** | 1,792 | 13 | **Forward-return models, purged walk-forward, conformal intervals** |
| `news.py` | 778 | 8 | De-duplication, instrument matching, headline tone |
| `insights.py` | 2,243 | 3 | Rule-based sentences, each citing the constant that fired it |

**Charts — `src/charts/`, 51 builders across 8 modules + the theme**

| Module | Builders |
|---|---|
| `allocation.py` (6) | `allocation_donut` · `allocation_treemap` · `allocation_sunburst` · `weight_bar` · `weight_vs_target_bar` · `sector_bar` |
| `performance.py` (9) | `growth_line` · `cumulative_return_chart` · `benchmark_comparison` · `relative_strength_chart` · `value_bar` · `pnl_bar` · `return_bar` · `contribution_waterfall` · `monthly_heatmap` |
| `risk.py` (11) | `drawdown_chart` · `underwater_chart` · `drawdown_episodes_chart` · `return_distribution` · `var_backtest_chart` · `rolling_chart` · `rolling_multi_chart` · `risk_gauge` · `risk_contribution_chart` · `risk_return_scatter` · `stress_bar` |
| `correlation.py` (5) | `correlation_heatmap` · `rolling_correlation_chart` · `pca_scree` · `pca_loadings_heatmap` · `dendrogram` |
| `optimizer.py` (4) | `efficient_frontier_chart` · `weights_comparison` · `weights_donut` · `frontier_weights_area` |
| `simulation.py` (3) | `fan_chart` · `terminal_distribution` · `sample_paths_chart` |
| `technicals.py` (6) | `price_chart` · `rsi_chart` · `macd_chart` · `volume_chart` · `price_with_panels` · `range_52w_bar` |
| **`forecast.py` (7)** | `forecast_cone_chart` · `backtest_scatter` · `model_leaderboard_chart` · `feature_importance_chart` · `residual_distribution_chart` · `horizon_skill_chart` · `prediction_timeline_chart` |
| `theme.py` | Palette, registered Plotly template, layout helpers — every figure is built through `theme.apply()` |

Every builder obeys the same contract, asserted for all 51: it returns a
`go.Figure`, it never raises, it survives JSON serialisation, it degrades to a
titled placeholder on empty / one-row / all-NaN input, and every hovertemplate
suppresses Plotly's secondary trace box.

**Data — `src/data/`, the only network modules**

| Module | Lines | Responsibility |
|---|--:|---|
| `fetcher.py` | 1,096 | Yahoo Finance via `yfinance` — quotes, history, OHLCV, profiles, statements, dividends, FX, option chains, symbol search |
| `exchange.py` | 504 | The official NSE and BSE listing and bhavcopy files (build-time only) |
| `news.py` | 559 | RSS feeds, per-ticker headlines, the economic calendar, staleness guard |
| `store.py` | 1,182 | SQLite persistence: saved portfolios, watchlists, schedules, alert state |

**UI — `src/ui/`, 17 support modules + 17 pages, ~21k lines**

| Module | Lines | Responsibility |
|---|--:|---|
| `theme.py` | 534 | Design tokens, the injected stylesheet, page chrome |
| `components.py` | 459 | KPI cards, badges, insight cards, hero, verdict, meter, model card |
| `formatters.py` | 302 | Currency, percentages, Indian lakh/crore grouping, dates |
| `session.py` | 197 | Session-state schema — every key declared in one place |
| `pipeline.py` | 318 | The ten-stage analysis orchestration |
| `data_access.py` | 181 | The only UI module that talks to the fetcher; every call cached |
| `guards.py` | 119 | Page preconditions — result, history, benchmark, ledger |
| `sidebar.py` | 253 | The control rail |
| `stock_search.py` | 327 | Type a company name, get a resolved Yahoo symbol |
| `csv_import.py` | 449 | Two-step import: upload, review mapping and rejects, commit |
| `bl_panel.py` | 562 | Black–Litterman views editor and re-solve |
| `factor_panel.py` | 433 | Factor-source picker and regression display |
| `options_panel.py` | 502 | Contract entry, overlay pricing, payoff |
| `tax_panel.py` | 388 | Harvesting candidates, set-off detail, boundary alerts |
| `fundamentals_panel.py` | 332 | Quality scores and the horizons that are refused |
| `alerts_panel.py` | 431 | Alert rules, delivery settings, send-nothing preview |
| `email_panel.py` | 510 | Schedule editor, SMTP status, the cron command, test send |

**Reporting — `src/reporting/`**

| Module | Lines | Responsibility |
|---|--:|---|
| `excel.py` | 2,426 | Formatted workbook with conditional formatting |
| `html_report.py` | 1,207 | Self-contained tearsheet that prints cleanly to PDF |
| `email_report.py` | 804 | SMTP composition and delivery |
| `schedule.py` | 529 | Schedule model and due-time arithmetic |
| `cli.py` | 751 | `python -m src.reporting.cli` — what cron invokes |

### Configuration reference

Everything tunable lives in `src/config.py` as a frozen dataclass, so the
assumptions are auditable in one file rather than scattered through the code.

| Dataclass | Fields | Controls |
|---|--:|---|
| `AppConfig` | 11 | Per-session runtime: `period`, `benchmark_name`, `base_currency`, `risk_free_rate`, `var_confidence`, `convert_fx`, `rolling_window`, plus the five configs below |
| `RiskScoreRubric` | 11 | The 0–100 risk score: point weights and saturation caps for volatility, drawdown, concentration, holding count and tail risk |
| `InsightThresholds` | 17 | Every threshold that can fire a generated sentence — concentration, correlation, drawdown depth, beta, Sharpe/Sortino/Calmar, VaR, kurtosis, recovery time |
| `OptimizerConfig` | 6 | `frontier_points`, `max_weight`, `min_weight`, risk-parity iterations and tolerance, covariance `shrinkage` |
| `SimulationConfig` | 5 | `n_paths`, `horizon_days`, `seed`, `block_size`, fan-chart `percentiles` |
| `TechnicalConfig` | 11 | SMA/EMA windows, RSI period and bands, MACD spans, Bollinger period and width, ATR period |
| **`FeatureConfig`** | 8 | Forecaster lookbacks: `return_lags`, `momentum_windows`, `volatility_windows`, `ma_windows`, `moment_window`, `volume_windows`, `range_window`, `vol_regime_windows` |
| **`ForecastConfig`** | 18 | `horizons`, `min_train_rows`, `min_initial_train`, `n_splits`, `test_fraction`, `purge_extra`, `conformal_alpha`, `random_state`, `importance_repeats`, the regularisation grids, and the tree-ensemble shape |
| `IndianEquityTaxRegime` | 7 | A dated rate set — STCG, LTCG, exemption, long-term boundary, statute reference |
| `FactorSpec` | 6 | One factor as a long/short ETF spread, with its caveat |
| `StressWindow` | 4 | One historical crisis: name, peak, trough, note |

Also in `config.py`: `TRADING_DAYS` (252), `MIN_OBSERVATIONS` (30),
`DEFAULT_VAR_CONFIDENCE` (0.95), the benchmark table and its currencies, the
currency symbols and lakh/crore set, the default risk-free rates per currency,
the cache TTLs, the seven historical stress windows, the factor proxies per
market and the gaps where none exist, and the disclaimers.

**Files that configure the app rather than the analytics:**

| File | Purpose |
|---|---|
| `.streamlit/config.toml` | Host theme (base, palette, font), `maxUploadSize = 5` MB, XSRF protection on, toolbar mode, full error details |
| `.streamlit/secrets.toml` | SMTP credentials — **gitignored and dockerignored; the app never writes one** |
| `pyproject.toml` | Packaging, and the pytest / ruff / mypy / coverage configuration |
| `requirements.txt` | Runtime dependencies, each with the reason it is present |
| `requirements-dev.txt` | Test and lint tooling |

**Environment variables** (for a cron job with no browser session, read by the
same loader as `secrets.toml`): `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`,
`SMTP_PASSWORD`, `SMTP_SENDER`.

### Command-line reference

```bash
# Scheduled tearsheets and price alerts
python -m src.reporting.cli                      # send what is due (what cron runs)
python -m src.reporting.cli --list               # what is configured, and what is due
python -m src.reporting.cli --dry-run            # build and validate SMTP; send nothing
python -m src.reporting.cli --force              # ignore the schedule and send now
python -m src.reporting.cli --only "Weekly review"
python -m src.reporting.cli --now 2026-09-07T08:00   # pretend it is this time
python -m src.reporting.cli --alerts             # price alerts only
python -m src.reporting.cli --no-alerts          # tearsheets only
python -m src.reporting.cli --db path/to.db      # a different database
python -m src.reporting.cli --allow-insecure     # permit an unencrypted SMTP login
python -m src.reporting.cli --verbose
```

Exit codes, for a scheduler that branches on them: `0` sent or nothing due ·
`1` a schedule failed · `2` could not start (no SMTP settings, unreadable
database).

```bash
# Rebuild the stock master from the exchanges' own files
python -m src.universe_build                     # refresh data/indian_stocks.csv
python -m src.universe_build --dry-run           # report changes, write nothing
python -m src.universe_build --on 2026-09-02     # use a specific bhavcopy date
python -m src.universe_build --no-bhavcopy       # skip the turnover signal
python -m src.universe_build --keep-stale        # do not drop delisted symbols
python -m src.universe_build --no-backup         # do not write the .bak
python -m src.universe_build --data-dir path/    # a different data directory
python -m src.universe_build --verbose
```

### The data directory

| File | Size | What it is |
|---|--:|---|
| `indian_stocks.csv` | 917 KB | **Generated** merge of the NSE + BSE listings with turnover — 7,569 instruments, every one with an ISIN |
| `indian_stocks_curated.csv` | 22 KB | **Hand-maintained seed** — aliases, sectors NSE does not publish, NSE ETFs absent from the equity listing. Never written to |
| `indian_stocks.csv.bak` | 918 KB | Automatic backup taken before each rebuild |
| `us_stocks.csv` | 17 KB | Curated US selection — 189 names (NYSE 85, NASDAQ 60, NYSEARCA 44) |
| `global_indices.csv` | 1.2 KB | 15 index symbols. The three files together are the 7,773 the search caption reports |
| `isin_seed.csv` | 5.4 KB | 114 hand-checked ISIN → symbol pairings |
| `universe_meta.json` | 976 B | Which bhavcopy date the turnover came from — displayed in the search caption |
| `quantfolio.db` | — | SQLite: saved portfolios, watchlists, schedules, alert state |
| `sample_portfolio.csv` | 344 B | Nine holdings, Indian plus one US name to exercise FX |
| `sample_transactions.csv` | 1.3 KB | An 18-trade ledger for the same portfolio |
| `sample_broker_export.csv` | 586 B | A deliberately awkward Zerodha-shaped file with one unusable row |
| `sample_isin_only.csv` | 215 B | An export whose only identifier is an ISIN, including one deliberate typo |

### Performance characteristics

Measured on the development machine (Python 3.12, 10-year histories, live
Yahoo Finance). Yours will differ; the *shape* is what matters.

| Operation | Cost | Notes |
|---|---|---|
| One analysis run, 9 holdings | ~2–5 s | Dominated by network fetches, not computation |
| The same run, cached | instant | `run_key()` skips recomputation when no input changed |
| Every analytics function | sub-millisecond to ~50 ms | Which is why none of them is cached |
| Efficient frontier, 60 points | < 1 s | SLSQP per point |
| Monte Carlo, 5,000 paths × 252 days | ~1 s | The path matrix is ~10 MB and is held for the session |
| **Forecast, one model one horizon** | **~1–3 s** | Ridge is fastest; the tree ensembles dominate |
| **Forecast cone, 5 horizons** | **~5–10 s** | Five independent fits |
| **Full 7-model leaderboard** | **~17 s** | 7 models × 5 purged folds on identical rows |
| Full test suite | ~180 s | 4,544 tests, offline, deterministic |
| Test suite with coverage | ~325 s | |

The forecaster is the only part of the application where compute, rather than
the network, is the bottleneck — and it is synchronous, so the page blocks
while it fits. That is why the leaderboard is behind a button and every run is
memoised against its exact control tuple.

### Security and privacy

- **Nothing leaves the machine except market-data requests.** Your holdings,
  trades, watchlist and schedules live in a local SQLite file. There is no
  telemetry, no account, and no upload of a portfolio anywhere.
- **SMTP credentials are never stored by the application.** They are read from
  `st.secrets` or the environment at send time. `.streamlit/secrets.toml` is in
  both `.gitignore` and `.dockerignore`.
- **TLS is verified** against the system trust store with hostname checking on,
  and sending a password over an unencrypted connection is refused unless you
  pass `--allow-insecure` and mean it.
- **Every value reaching `unsafe_allow_html` is `html.escape`d first.** Tickers
  and notes are user-supplied, and a portfolio tool that renders them raw is one
  CSV upload away from script injection. There is a test for it.
- **XSRF protection is on** and uploads are capped at 5 MB — a holdings CSV is
  kilobytes.
- **The container runs as an unprivileged user** (uid 10001) with a healthcheck,
  and carries no compiler toolchain.
- **There is no authentication.** See the honest assessment below: anyone who
  can reach the port can see the portfolio.

---

## Testing

```bash
pytest                                    # full offline suite
pytest --cov=src --cov-report=term-missing
```

**4,544 tests, 92% coverage, ~180 seconds.** The entire suite runs **offline and
deterministically**: `yfinance` is monkeypatched, the SQLite store uses `tmp_path`, and every
RNG is seeded. Analytics you cannot test without the internet are analytics you cannot trust
in CI.

Two of those tests carry more weight than the rest, and they only mean anything as a pair:

| Test | Input | Asserts |
|---|---|---|
| `test_the_pipeline_finds_a_signal_that_is_genuinely_there` | AR(1) returns, ρ = −0.35 | skill > 2%, out-of-sample R² > 2%, direction > 55%, DM _p_ < 0.05 |
| `test_the_pipeline_reports_no_edge_on_a_pure_random_walk` | i.i.d. returns | skill < 5% |

Without the first, every "no skill" verdict the forecaster produces could be a broken
pipeline rather than an efficient market. Without the second, a leak would go unnoticed —
because a leaking forecaster simply looks excellent. Together they pin the property the
whole feature exists to provide: **the number on the screen tracks whether there is
anything to find.**

The suite earned its keep: writing it surfaced **twelve real defects**, every one of which is
now a named regression guard rather than an `xfail`. A representative few:

| Defect | Why it mattered |
|---|---|
| Cornish–Fisher VaR returned a **positive** number | One −20% day (skew ≈ −14) pushes the expansion outside its monotone domain, so the 99% *worst case* rendered as an 8% gain. Now checks monotonicity and falls back to the historical quantile. |
| `fmt_compact` understated round numbers **10×** | `.rstrip("0")` on a string with no decimal point turned `₹10 Cr` into `₹1 Cr`. |
| Benchmark growth silently blanked | A zero first observation produced an all-NaN column while still reporting `ok=True` — an invisible line that read as real data. |
| Stress coverage reported 1.00 on a 70%-priced book | Unpriced holdings were dropped from the denominator instead of being disclosed as uncovered. |
| Technicals never sorted the index | A newest-first price table (how most brokers export) produced a confidently wrong RSI. |
| `fetch_price_history` trimmed only the leading edge | A delisted ticker was carried forward at a stale price, biasing every correlation and volatility estimate downward. |
| Fuzzy search matched typos against **full** company names | `relaince` scored 0.42 against "reliance industries ltd" — below any usable cutoff — so a short unrelated name won and the top hit was Tesla. The pool now holds the short keys people actually type. |
| Rejected import rows also appeared under "symbols to confirm" | A row dropped for a blank quantity was additionally reported as a ticker problem, sending the user hunting for a symbol error that did not exist. |
| `str(value or "")` learned the symbol **"NAN"** | `float("nan")` is *truthy*, so a blank symbol cell survived the `or`, stringified to `"nan"` and was **written into the ISIN table** as a permanent wrong mapping. Found by a test asserting a blank symbol falls back to the row's ISIN. |
| The lexicon called "bonds **pare** gains" positive | Nouns every market headline contains — "profit", "record", "high", "order", "buy" — were scored as directional, classifying 64% of a live feed positive and scoring "profit falls 40%" as merely *mixed* because the noun cancelled the verb. Pruned to verbs and adjectives, with a `DAMPENERS` set for "pare/trim/erase". |
| "Tata" tagged every Tata headline to **both** TCS and Tata Power | A leading name word is only specific if no other tracked company shares it. Live output showed 7 mentions where 3 were real. |
| Filtering generic words first left Tata Power with **no name key at all** | Every word after "Tata" is in the generic list, so a major company became unmatchable. The two-word key is now built from the raw name. |
| The offline test suite started hitting the **live network** | `fetch_statements` was new and absent from the smoke test's stub, so the Security page test silently fetched real statements. Caught by the runtime: 63s down to 16s once stubbed. |
| A lowercased haystack compared against a mixed-case needle | `"book equity is NOT substituted" in n.lower()` could never match, so a test that looked like it guarded the Altman fallback guarded nothing. |
| BSE returned an **HTML error page with status 200** for a future date | `csv.DictReader` parsed it into 71 rows of `None`, which would have built the turnover signal from nothing and left ~4,900 symbols ranked at zero. Now every download is checked against the columns it must carry — a schema contract, which also catches a changed URL or a redirect to a login page. |
| Yahoo addresses BSE lines by scrip **id**, not scrip **code** | `RELIANCE.BO` prices; `500325.BO` returns nothing. Measured before generating anything: using the numeric code would have produced ~5,000 searchable rows whose prices never load. |
| The full universe made search **worse**, not better | `hdfc` returned HDFC AMC over HDFC Bank and the typo `relaince` returned Reliance Communications — ranked on symbol length and the alphabet. Fixed with bhavcopy turnover; a percentile transform was tried first and rejected because it *saturates at the top*, where search collisions actually happen. |
| Replacing the curated name with the exchange's deleted a working lookup key | "Reliance Industries Ltd" became "Reliance Industries Limited", and because BSE's own spelling happened to be the old one, CSV import started rewriting NSE holdings to `.BO` symbols. The superseded name is now kept as an alias. |
| The build wrote to its own input | Running it twice replaced the pristine curated seed's backup with a generated one, losing the hand-written aliases irrecoverably. The seed and the output are now separate files. |
| `load_map(data_dir=X)` read one source from `X` and another from the global directory | So an "isolated" ISIN map silently picked up the real universe file. Every source now resolves from one root. |
| Loading a watchlist without the alert columns switched alerting **on** for every row | `bool(float("nan"))` is **True**, and `pd.DataFrame(rows, columns=...)` fills a key the rows never supplied with NaN — so an upgraded database or a CSV round-trip would have started e-mailing about targets set years ago. The same NaN-truthiness trap as the `"NAN"` symbol above, with a worse consequence. |
| A newly saved schedule fired immediately for a slot that predated it | Configuring "daily at 08:00" at 14:00 made the 08:00 occurrence genuinely unsent, so the next cron tick mailed a report seven minutes after Save. Correct by the rule, surprising enough to read as a bug — a new schedule now starts from its *next* slot. |
| Preamble detection only fired on a single-column parse | The delimiter sniffer splits `Account Statement` on whitespace into two columns, so a broker file with a header block was refused outright. Detection is now by whether the mapping succeeded. |

Tests assert **hand-computed golden values** and **mathematical identities** rather than
smoke-checking that a function returns something:

| Identity | Why it cannot rot |
|---|---|
| `beta(x, x) == 1.0` | true for any correct implementation, forever |
| weights sum to `100` | catches every aggregation bug |
| `CVaR ≤ VaR` at 90/95/99% | catches sign and quantile errors |
| component risk contributions sum to `σ_p` | the definition of a decomposition |
| max-Sharpe Sharpe ≥ equal-weight Sharpe | catches a broken optimiser |
| min-variance vol ≤ every frontier point | catches a broken frontier sweep |
| `npv(xirr(f), f) == 0` | catches a wrong root |
| monotonic rising series ⇒ RSI → 100 | catches a rolling mean replacing Wilder |
| imported rows + rejected rows == source rows | the definition of "nothing was silently dropped" |
| `from_frame(to_frame(x)) == x` | catches a lossy or order-flipping watchlist round trip |
| every shipped ISIN passes its own ISO 6166 check digit | makes a mistyped seed row a build failure, not a wrong holding |
| every shipped ISIN's country prefix matches its symbol's country | catches the mis-pairing a checksum cannot see |
| a 1-factor regression beta == `risk.beta` (cov/var) to 1e-12 | two modules, two routes, one answer — neither has drifted |
| an exact linear combination recovers its own coefficients and R²=1 | catches a sign error or a mis-stacked design matrix |
| HAC at 0 lags == White HC1 == OLS on iid residuals | proves defaulting to robust costs nothing |
| `(δΣ)⁻¹(δΣw) == w` to 1e-12 | the defining property of equilibrium returns |
| no views ⇒ BL posterior == prior *exactly* | what makes the model safe to leave switched on |
| Woodbury posterior == textbook master formula | justifies the form that never inverts Ω |
| a loss set off against an already-exempt gain saves ₹0 | set-off is mandatory, so the loss is destroyed — pinned so no "optimisation" can hide it |
| `tax.is_long_term` == `RealisedTrade.is_long_term` | two modules cannot disagree about the 12-month boundary |
| `C − P == S·e^(−qT) − K·e^(−rT)` (put–call parity) | constrains both option formulas at once, to 1e-12 |
| every Greek == its numerical derivative | delta/vega/theta/rho to ~1e-10, gamma to ~1e-6 |
| a flat year makes every Beneish index exactly `1.0` | reduces the M-Score to a closed form, pinning all eight published coefficients in one assertion |
| Altman Z''-EM == Z'' + 3.25 | catches the emerging-market adjustment being applied to the wrong market |
| a missing Piotroski signal changes the *denominator*, not the score | the property the whole module exists to guarantee |
| a feed frozen 863 days ago yields **zero** items, not stale ones | HTTP 200 is not evidence of news |
| a group prefix shared by two tracked companies matches neither | "Tata" identifies neither Tata Power nor TCS |
| gamma and vega identical for a call and a put | catches a copy-paste error between the branches |
| `iv(price(σ)) == σ` from 5% to 200% | catches a broken root-finder or a mis-signed residual |
| 96 runs of the send command produce **1** e-mail | the whole point of a scheduled report: idempotence under any cron cadence |
| 96 alert checks on a price parked above the level produce **1** e-mail | an alert fires on the transition, not the condition |
| a 1% re-arm band turns 144 alerts into 2 (12 h cooldown) | pins what each anti-flap guard actually buys |
| a monthly "31st" schedule fires in all 12 months | catches the day-of-month clamp, including 29 February |
| `from_record(to_record(s)) == s` | catches a lossy schedule round trip through SQLite |
| prominence bonus ≤ (tier gap − max prefix penalty) | turnover reorders hits *within* a match tier and can never promote a weaker one |
| `_MIN_TIER_GAP` asserted against the tier constants themselves | lowering a tier cannot silently invalidate the inequality above |
| every NSE/BSE row in the shipped file ends `.NS` or `.BO` | a bare symbol cannot be priced, so one in the file is a build bug |
| the build is idempotent — running it twice yields byte-identical output | catches state accumulating in the generated file |
| truncating the price history leaves the last feature row **unchanged to 1e-12** | the lookahead test: any feature peeking at a future bar breaks it, and nothing downstream could tell |
| multiplying every price by 7.5 moves no feature at all | the scale-free property, without which a model learns the trend of the price axis |
| the train/test gap is exactly `h − 1 + purge_extra` rows, at every horizon | the purge that stops an overlapping label leaking the answer |
| a training row index is never inside a test block, stated as a set identity | cannot be satisfied by an off-by-one |
| the random walk scores skill `0.0` and MASE `1.0` **exactly** | the benchmark cannot beat or lose to itself |
| an 80% conformal band contains 80% of the residuals it was built from | the defining property of the interval |
| a left-skewed error law yields `|Q_low| > |Q_high|` | catches a symmetric band quietly replacing a signed one |
| the Diebold–Mariano statistic is antisymmetric in its two arguments | catches a sign error in the loss differential |
| `Ŝ == S₀ · exp(ŷ)` to 1e-12, on all three cone lines | the identity every forecast KPI card claims |
| a marked cone node is the **fitted** value, never the interpolation | catches a drawn line being passed off as a fit |
| interpolation between two nodes is log-linear, not price-linear | halfway between +2% and +6% is +4% compounded |
| a weighted portfolio forecast equals `Σ wᵢrᵢ` on simple returns | catches log returns being weighted directly (Jensen) |
| the comonotonic band strictly contains the independent one | the two correlation extremes, in the right order |
| four models agree with themselves across repeated runs | catches an unseeded estimator — which disagrees in the third decimal, not the fifteenth |
| the features' RSI equals `technicals.rsi` to 1e-9 | two modules, two implementations of Wilder smoothing, one answer |

`tests/conftest.py` generates **correlated** random walks rather than independent ones, because
uncorrelated toy data hides exactly the bugs a correlation, beta or optimiser test exists to catch.

CI runs lint, format check, mypy and the suite on Python 3.11/3.12 across Ubuntu and Windows,
then builds the Docker image and healthchecks the running container.

---

## About "real-time" data

Yahoo Finance is **not** a real-time feed. NSE/BSE quotes are typically delayed ~15 minutes,
and outside market hours the best available value is the previous close. So the app:

- labels every price as *"latest traded price (may be delayed ~15 min)"* or
  *"latest available closing price"*,
- shows per-ticker `as of` timestamps and the fetch time,
- caches by data type — 5 min for quotes, 30 min for history, 24 h for fundamentals —
  with a manual refresh.

It never claims to be live, because it isn't. True real-time Indian data needs a broker or
exchange feed (Zerodha Kite Connect, Upstox, official NSE products) — which would drop in
behind the existing `fetcher` interface.

---

## Known limitations

Stated here rather than discovered later:

- **Without a trade ledger**, historical portfolio value assumes today's share counts across the
  whole window. That answers *"how would my current portfolio have behaved?"* — the standard
  basis for volatility, drawdown and beta — but it is not the return you actually earned. Supply
  transactions and the app computes XIRR and realised P&L properly.
- **Risk statistics use the overlapping history of all holdings.** One recently listed name
  shortens the window for everyone; the app warns below 30 days.
- **The optimiser is single-period mean-variance**, optimal in sample only.
- **Beta-implied stress is first-order CAPM.** It ignores idiosyncratic risk, and betas converge
  toward 1 in a crisis — so it understates losses for low-beta portfolios.
- **Benchmarks are price indices**, excluding dividends.
- **Insights are rule-based heuristics** from documented thresholds — not predictions, not advice.
- **Headline tone reads the words, and nothing else.** A small hand-built lexicon of directional
  finance terms, with the matched words always shown. It is *not* Loughran-McDonald and does not
  pretend to be, it has no idea what the market had already priced in, and about a third of a
  market feed comes back unclassified — which is the honest answer for "Sensex ends flat".
- **No news impact is predicted.** The calendar says which releases have historically moved
  markets and when they land, not what will happen or in which direction. Measuring what news
  *did* to a price is an event study on realised returns — a separate calculation this page does
  not perform.
- **The economic calendar covers one week**, because `ff_calendar_thisweek.json` is the only
  horizon published (`nextweek` is a 404). The window actually covered is printed rather than
  implied.
- **MoneyControl's RSS feeds are abandoned.** All six endpoints answer HTTP 200 with valid XML
  whose newest item is 23 April 2024. They stay registered so the staleness guard is exercised in
  production and so the feature recovers by itself if the publisher resumes.
- **No return forecast is built from the *statements*, at any horizon, and the app says why.**
  Training a model that maps accounts at time *T* to the return from *T* to *T+N* needs pairs of
  exactly that shape, and each pair consumes *N* years of history. The provider supplies **five
  annual periods**, which gives four one-year pairs, two three-year pairs and **zero from five
  years out** — so a 5-, 7-, 10- or 11-year "prediction" would be fitted to no observations at
  all. The Security page lists every horizon with its training-pair count and the reason, rather
  than omitting the impossible ones or printing a number that came from nowhere. Unlocking them is
  a *paid data* problem (point-in-time fundamentals, 15+ years, delisted companies retained), not
  a modelling one.
- **That refusal is about annual accounts, and does not extend to prices.** Five annual
  periods give four training pairs; ten years of daily bars give ~2,500 rows. That is the whole
  difference, and it is why the Price forecast page *does* produce a number — fitted to prices,
  where there is enough data to hold out 30% of it in sequential blocks and quote an error worth
  reading. Neither page borrows the other's credibility: the fundamentals panel lists every
  impossible horizon with its zero pair count, and the forecast page reports skill against a
  random walk on every horizon it offers.
- **The ML forecaster will usually report no skill, and that is the correct answer.** Daily
  equity prices are close to a martingale, so the random walk is a genuinely hard opponent and
  beats the fitted models more often than not. The leaderboard shows it winning rather than
  hiding it, and the verdict banner leads with it.
- **Forecast accuracy is not profit, and this is not a strategy backtest.** No transaction
  cost, spread, slippage, market impact, position sizing, borrow cost or tax is modelled
  anywhere on the forecast page. A model with genuine statistical skill can lose money after all
  six.
- **The forecaster sees price and nothing else** — no earnings, no news, no order flow, no
  macro, no other instrument. It is a weak-form efficiency test, and weak-form efficiency mostly
  holds.
- **A conformal interval is marginal, not conditional.** It covers its nominal share of
  outcomes on average across the backtest; it does not widen because today is a volatile day.
- **Adjusted history is adjusted as of today.** Features on an old row are computed from
  split- and dividend-adjusted numbers that were not knowable on that date. Small for a liquid
  equity, real for one with a corporate-action history.
- **No deep learning, and the reason is data, not ambition.** An LSTM or transformer carries
  millions of parameters; a daily bar history is a few thousand rows with a signal-to-noise ratio
  near zero. That combination memorises its training set, and the dependency would be larger than
  the rest of this application put together.
- **The statements are restated, not point-in-time.** A figure labelled 2022 is today's view of
  2022 including later restatements. Correct for "how do these accounts look now", fatal for
  backtesting — which is the second reason the long horizons are refused.
- **A fundamental score is a description, not a recommendation.** A high F-Score describes past
  accounts. A Beneish score above −1.78 is a prompt to read the filings, not an accusation — the
  model's own false-positive rate is substantial and a large genuine acquisition raises it too.
- **The app cannot schedule anything by itself, and does not claim to.** Streamlit is
  request-response: it runs the script when a browser asks, has no background worker, and no
  process is alive at 08:00 on a Monday unless something starts one. A thread-based "scheduler"
  inside the app would fire zero times when the app is idle, several times when several tabs are
  open, and die on the next restart — silently. So scheduling is delegated to cron or Windows
  Task Scheduler, the UI puts that command at the *top* of the panel, and **nothing is ever sent
  until you set it up**. Reliability past that point is your scheduler's, not this application's.
- **The e-mailed tearsheet carries no charts.** They are rendered by the app's chart layer from a
  live session, which a cron job does not have. Building a second set for e-mail would mean two
  renderings of the same numbers that could disagree, so the attachment carries the figures as
  tables and the app is where you go for the charts.
- **Prices in a scheduled report are as of when the command ran**, not when you read the mail.
  The report stamps its own generation time and carries any data-quality warning the pipeline
  raised, rather than dropping it.
- **Yahoo Finance publishes no option chain for NSE or BSE.** Measured, not assumed: `AAPL`
  returns 21 expiries, `RELIANCE.NS` and `^NSEI` return zero. For an India-first tool that is the
  single biggest data gap in the product. The overlay analytics are therefore built to run from a
  strike, a premium and an expiry typed off your broker's screen — manual entry is the *primary*
  path, not a fallback — and the app names the data source as the gap rather than showing an empty
  table that would read as "this stock has no options".
- **Black–Scholes prices a European option.** NSE stock and index options *are* European, so the
  model is exactly right for the market this app targets. US equity options are American, so for
  the market whose chains Yahoo does carry it is an approximation that undervalues an in-the-money
  put and a call on a stock about to pay a large dividend. Stated wherever a US chain is priced.
- **"Probability of assignment" is risk-neutral**, the `N(d2)` from the model that priced the
  option — not a forecast. Under the real-world measure an equity drifts faster, so a call's true
  chance of finishing in the money is higher than the figure shown.
- **Not modelled:** early exercise, the volatility smile (one IV per contract, no surface),
  transaction costs and the spread you actually cross, margin and pledge mechanics, and rolling.
  Every figure assumes the contract is held to expiry.
- **The tax module is educational arithmetic, not tax advice.** It models STT-paid listed equity
  under a simplified reading of Sections 111A, 112A and 70–74. Out of scope: surcharge and cess,
  resident vs non-resident treatment, set-off against other income heads, advance-tax timing, and
  every asset class other than listed equity. Verify any action with a chartered accountant.
- **Section 112A grandfathering is flagged, not computed.** For equity bought on or before
  31 January 2018 the cost basis is the higher of actual cost and the lower of (31-Jan-2018 FMV,
  sale price). The statutory FMV is the *highest quoted price* that day, which this app does not
  hold — an adjusted close is neither the high nor unadjusted. Pre-cutoff lots are therefore
  reported with their gains **overstated**, with the limitation stated, rather than valued on a
  number that is not the statutory one.
- **The bundled factor "factors" are ETF proxies, not Fama–French factors.** The academic
  factors are built from the whole cross-section sorted on size and book-to-market and are not
  investable; an ETF spread is investable and measurably different (fund fees, the index
  provider's own screens, and the ETF's inception date, which also caps how far back the
  regression can run). For rigorous work, load a published factor file. **Indian proxy coverage
  is genuinely thin** — only market and a weak size tilt (Next 50 − Nifty 50) are offered,
  because the NSE factor ETFs are recent and thinly traded; value, momentum and quality proxies
  are deliberately *not* offered rather than shipped as noise.
- **The Indian universe is complete as of a date, not live.** It is built from end-of-day
  exchange files, so a company that listed this morning is not in it until you re-run
  `python -m src.universe_build`. The search caption states the date rather than implying
  currency, and a symbol missing from the files is *not* rejected — it is passed straight to the
  data layer, which is the only thing that can confirm it prices.
- **US and index coverage is still a curated selection** (189 US names across NYSE/NASDAQ/NYSEARCA,
  plus 15 indices — 204 in total). No equivalent listing
  is fetched for them, and the search caption says so instead of claiming completeness for
  everything.
- **Turnover is one day's trading, and it only breaks ties.** A good company that happened not to
  trade on the reference date ranks below one that did. It can never promote a weaker kind of
  match — the bonus is bounded below the gap between match tiers, and a test pins that inequality
  — but it does mean the ordering *within* a tier reflects one session, not a long-run average.
- **Yahoo does not cover every listed scrip.** Sampled across every BSE group, 35 of 36 resolved
  and 30 of 30 NSE symbols resolved; the failures are typically debt-like instruments. A row in
  the universe is not a promise that the provider will price it, which is why an unpriceable
  holding is reported per-ticker rather than silently dropped.
- **The ISIN table now covers the market, and is still not a directory.** The exchange listings
  publish an ISIN per instrument, so ISIN resolution draws on **5,175 pairings** (5,061 from the
  generated universe, plus the hand-checked seed, plus anything learned from your own imports —
  in that order of authority). A non-Indian ISIN, or one for an unlisted instrument, still
  resolves to nothing and says so. It ships 114 hand-checked pairings
  and grows from your own imports; there are tens of thousands of Indian ISINs alone. A valid ISIN
  that is in neither the seed nor your learned cache is *reported as unresolvable* — with a button
  to look it up online — never resolved to a guess. The check digit means a **typo** is always
  caught; an ISIN that is valid but simply unknown is a coverage gap, and the app says so.
- **`03/04/2024` is read day-first.** Indian broker exports are overwhelmingly DD-MM-YYYY. A US
  export using MM/DD/YYYY for an ambiguous date will be misread by a month — supply ISO
  (`YYYY-MM-DD`) dates if that matters to you.
- **A watchlist target price is a reminder you set**, not a forecast, a valuation or a
  recommendation. "Near your level" is arithmetic against your own number and nothing more.
- **A price alert says a level *had been* reached, never that it is being reached.** The feed is
  delayed (~15 min for NSE/BSE, and the latest close outside market hours), so every alert
  carries the quote's own timestamp rather than implying a live cross.
- **Alerts see samples, not a continuous price.** A move that crossed your level and came back
  between two checks is visible only if the session high or low caught it, and a crossing
  entirely inside an earlier, already-closed session cannot be recovered from a quote at all.
  Checking more often narrows the gap; it never closes it. A broker's own alerts sit on the
  order book and do not have this limitation — this is the honest ceiling of a delayed
  public feed.
- **A rule needs a direction.** A bare target cannot say whether you want to hear about the
  price rising to it or falling to it, so a target with alerting on but no direction is
  reported as *inactive* rather than shown as an armed alert that would never fire.

---

## Pros and cons

An honest assessment, written the way the app writes its own numbers. The
limitations above are per-feature; this is the whole-project view — what it is
genuinely good at, what it is genuinely not, and who should use something else.

### What it does well

**1 · The architecture actually holds.**
`src/analytics`, `src/charts`, `src/data` and `src/reporting` contain no
`import streamlit`; `src/ui` and `app.py` contain no formulas. That is not an
aspiration in a CONTRIBUTING file — it is why ~22,000 lines of financial
mathematics can be tested in milliseconds, offline, against hand-computed
values, with no browser and no market-data subscription. It is also why
swapping Yahoo Finance for a broker feed is a rewrite of **one file** against
the same signatures.

**2 · The test suite is doing real work.**
4,544 tests, 92% coverage, fully offline and deterministic. More importantly
they assert **mathematical identities and hand-computed golden values**, not
that a function returned something: `beta(x, x) == 1`, component risk
contributions sum to portfolio volatility, `npv(xirr(f), f) == 0`, put–call
parity to 1e-12, every Greek against its numerical derivative, imported rows +
rejected rows == source rows. Writing it surfaced **twenty-three real defects**,
each now a named regression guard and each named in the table above. Tests like those do not rot.

**3 · It refuses to fabricate, consistently.**
Undefined statistics render as `—`, never `0.00`. Stress scenarios report what
fraction of portfolio weight actually had data. The fundamentals panel lists
every return horizon it *cannot* model with its training-pair count of zero. A
feed that answers HTTP 200 with year-old content is rejected with its age
shown. The forecaster reports negative skill in the headline when it loses to a
random walk. Each of those is a place where it would have been easy, and more
flattering, to print a plausible number.

**4 · The forecaster is validated the way a forecaster should be.**
Purged walk-forward with no shuffling, skill quoted against a random walk,
Diebold–Mariano with HAC errors, conformal intervals from held-out residuals,
permutation importance measured out of sample, and a paired test suite proving
the pipeline finds a planted signal *and* reports none on a random walk. Most
published stock predictors do none of this, which is why most published stock
predictors report accuracy they do not have.

**5 · It is genuinely India-first, not localised as an afterthought.**
Full NSE + BSE listing built from the exchanges' own files, ISIN resolution
with check-digit validation, lakh/crore digit grouping, the Indian STCG/LTCG
regime *including the 23 July 2024 rate change*, `.NS`/`.BO` suffix handling,
and broker exports from Zerodha/Groww/Upstox that import unmodified.

**6 · Every assumption is a named constant in one file.**
Annualisation convention, risk-score weights, insight thresholds, optimiser
caps, forecaster geometry — all in `src/config.py` as frozen dataclasses. You
can audit what the app believes without reading the app.

**7 · It explains itself in place.**
Methodology expanders next to the numbers, the formula behind each statistic,
the caveat travelling with the chart into exports, and `docs/METRICS.md` for
the full derivations. The output is arguable-with, which is the point.

**8 · Breadth, at a consistent level of care.**
217 analytics functions, 51 chart builders, 17 pages, Excel and HTML export,
scheduled email, price alerts — and the twentieth feature is built to the same
standard as the second.

**9 · Zero-cost to start.** No data subscription, no API key, no account.
`pip install -r requirements.txt && streamlit run app.py`.

### What it does not do well

**1 · There is no authentication, and no multi-user story.**
Streamlit serves one app to whoever reaches the port, and SQLite is a single
file opened by a single process. Running it on a shared host exposes the
portfolio to anyone who can reach it. It is designed to run on your own
machine, and that is where it should stay unless you put a reverse proxy and
real auth in front of it.

**2 · The market data is free, delayed and unofficial.**
Yahoo Finance via `yfinance` is ~15 minutes behind for NSE/BSE and is the
previous close outside market hours. `yfinance` is an unofficial client against
an endpoint nobody promised to keep stable — it has broken before and will
again. There are no SLAs, no support, and rate limits nobody publishes. The app
labels every price honestly, but honest labelling does not make the data
better.

**3 · The headline ML feature will usually tell you it has no edge.**
That is the correct answer for daily equity prices, and it is reported
prominently rather than buried — but it does mean the most exciting-sounding
page most often concludes "do not act on this". If you wanted a forecaster that
always sounds confident, this is the wrong one on purpose.

**4 · Forecast accuracy is not profit, and there is no strategy backtest.**
No transaction cost, spread, slippage, market impact, position sizing, borrow
cost or STT is modelled anywhere on the forecast page. A model with genuine
statistical skill can lose money after all six. Turning a signal into a P&L is
the single biggest missing piece.

**5 · Compute is synchronous and in-request.**
A seven-model leaderboard is ~17 seconds during which the page is blocked.
There is no job queue, no worker, no progress you can navigate away from. It is
mitigated with memoisation and an explicit button, not solved.

**6 · Yahoo publishes no option chain for NSE or BSE.**
Measured, not assumed: `AAPL` returns 21 expiries, `RELIANCE.NS` and `^NSEI`
return zero. For an India-first tool that is the biggest data gap in the
product, and the overlay analytics are built around manual contract entry
because of it.

**7 · The fundamentals are restated, not point-in-time.**
A figure labelled 2022 is today's view of 2022, including later restatements,
and delisted companies are absent from the provider entirely. Correct for "how
do these accounts look now", fatal for backtesting anything built on them.

**8 · The optimiser is single-period mean-variance.**
Optimal in sample only, with famously unstable inputs. Covariance shrinkage,
weight caps and Black–Litterman all reduce the damage; none of them make a
historical mean a good return forecast.

**9 · Daily bars only.**
No intraday, no tick data, no order book. Which also means no execution
analytics, no microstructure, and alerts that see samples rather than a
continuous price.

**10 · The UI is desktop-first and dense.**
A deliberately information-rich analyst terminal. It has a print stylesheet and
a reduced-motion path, but it is not designed for a phone and a 400px viewport
will be uncomfortable.

**11 · Type checking is advisory, not clean.**
`mypy` reports 79 findings across 25 files — mostly `Any | None` in the UI
layer, where a value is known to be present but not provably so. CI runs it
with `continue-on-error: true`. The new analytics, chart and page modules are
clean; the older UI is not. There is also a configuration drift worth knowing:
`python_version = "3.11"` in `pyproject.toml` against a numpy whose stubs use
3.12 syntax, which makes a bare `mypy src` fail at the stubs before it checks
anything.

**12 · The test suite is three minutes, and the ML added most of that.**
It was ~100 seconds before the forecaster; it is ~180 now, and ~325 with
coverage. Still fast enough to run on every commit, but the trend has one
direction and the fold count is the lever.

**13 · Coverage of the world is uneven.**
The Indian universe is complete as of a rebuild date — a company that listed
this morning is not searchable until you re-run the build. US and global
coverage is a curated 204 instruments, not a listing. Benchmarks are price
indices and exclude dividends. There is no fixed income, no commodities beyond
gold ETFs, no mutual-fund NAV handling, no derivatives P&L, and no currency
hedging.

**14 · Memory grows with the interesting settings.**
A 5,000-path Monte Carlo matrix and a cached forecast's residual vectors live
in session state for the session. Bounded (six memo entries, one simulation),
but a long session at maximum settings will hold tens of megabytes per user.

**15 · Nothing here is audited, regulated, or advice.**
It is a demonstration of quantitative portfolio analytics and production Python
architecture that happens to be genuinely useful. It has not been reviewed by
anyone's compliance function, and the tax module is educational arithmetic that
a chartered accountant should check before you act on it.

### Should you use it?

| If you are… | Verdict |
|---|---|
| An Indian retail investor who wants professional analytics on a broker export | **Yes.** This is the case it was built for |
| Someone learning quantitative finance who wants readable, documented, tested implementations | **Yes.** `docs/METRICS.md` plus the source is a course |
| An engineer evaluating Python application architecture | **Yes.** The layering rule and the identity-based test suite are the point |
| Looking for a forecaster that will tell you what a stock will do | **No.** It will usually tell you it cannot, with evidence |
| Wanting to backtest and trade a strategy | **Not yet.** No costs, no execution, no P&L — see the roadmap |
| Needing real-time data, intraday, or an order book | **No.** Use a broker feed; the `fetcher` interface is where it would drop in |
| Running a multi-user or client-facing service | **No.** No auth, no tenancy, single-process SQLite |
| Managing institutional money | **No.** Not audited, not regulated, not advice |

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| **"No market data could be fetched for any holding"** | Indian symbols need a Yahoo suffix: `RELIANCE.NS` (NSE) or `RELIANCE.BO` (BSE), not `RELIANCE`. Search by company name instead of typing symbols |
| **One holding is skipped with "no usable quote"** | Yahoo does not price every listed scrip — typically debt-like instruments. The holding is reported per-ticker rather than silently dropped |
| **Risk statistics say the window is too short** | Risk uses the *overlapping* history of all holdings, so one recently listed name shortens it for everyone. Widen the history window or remove the newest holding |
| **The Price forecast page refuses to fit** | It needs ~250 labelled rows after a 252-session lookback and a purged walk-forward. Choose a longer training history (10 years / max), or a shorter horizon — a 63-session horizon alone costs 63 rows |
| **The forecast says the model has no skill** | That is a result, not a failure. Try the leaderboard: if the random walk wins too, this instrument is not forecastable from its own price at this horizon |
| **Search finds nothing / "offline search covers 0 instruments"** | The bundled stock masters live in `data/`. In Docker, bind-mounting an empty host directory over `/app/data` hides them — mount the repo's own `data/`, as the Docker section shows |
| **A very recent listing is missing from search** | The universe is built from end-of-day exchange files. Run `python -m src.universe_build`, or type the symbol directly — an unknown symbol is passed straight to the data layer, not rejected |
| **CSV import read the wrong column as my cost basis** | Open the per-column override, point `buy_price` at the right column, and the whole preview re-derives. `LTP`, `Last Price`, `Previous Closing Price` and `Market Value` are blocked from mapping to `buy_price` by default |
| **An ISIN will not resolve** | The check digit catches typos and names the digit it should have had. A valid-but-unknown ISIN is a coverage gap — use the online-lookup button, and importing a file with both an ISIN and a symbol column teaches the pairing for next time |
| **Scheduled email never arrives** | The app cannot schedule anything; cron or Task Scheduler must run `python -m src.reporting.cli`. Check `--list` for what is due and `--dry-run` to validate SMTP. Gmail and Outlook reject a normal password — an app-specific one is required |
| **Alerts fire too often / not often enough** | The re-arm band (default 1% past the level) and the minimum hours between alerts (default 12 h) are both on the Watchlist page. A price parked above your level sends one email, not ninety-six |
| **Dates are read a month wrong** | `03/04/2024` is parsed day-first, because Indian broker exports are overwhelmingly DD-MM-YYYY. Supply ISO `YYYY-MM-DD` if that matters |
| **The page looks unstyled** | Fixed in v3.0.1 — the stylesheet was emitted only on the first script run. If you see it on an older build, upgrade |
| **`mypy src` fails at numpy's stubs** | A known configuration drift: `python_version = "3.11"` against numpy stubs using 3.12 syntax. Run `mypy --python-version 3.12 src`. It is advisory in CI |

---

## Version history

| Version | What it added |
|---|---|
| **v1** | Holdings, positions, P&L, core risk statistics, allocation charts, the layering rule and the offline test suite |
| **v2** | `scipy`: the efficient frontier and risk parity (SLSQP), XIRR by Brent bracketing, Kupiec χ² VaR backtest, hierarchical clustering. Plus FIFO tax lots, drawdown episodes, Monte Carlo, stress replay, rebalancing, Excel and HTML export |
| **v2.x** | Full NSE + BSE universe built from the exchanges' own files, ISIN resolution, broker-CSV import, Black–Litterman, Fama–French regression with Newey–West errors, Indian tax-loss harvesting, options overlays, fundamental quality scores, news and the economic calendar, watchlist price alerts, scheduled email tearsheets |
| **v3.0.0** | **The machine-learning forecaster**: `scikit-learn`, 42 leak-free features, seven models, purged walk-forward validation, Diebold–Mariano significance, conformal prediction intervals, a multi-horizon cone, permutation importance, portfolio-level forecasting — and 7 new charts, a new page, and 114 new tests |
| **v3.0.1** | Two theme fixes found by running the app: the stylesheet was emitted only on the first script run (leaving every page after the first interaction unstyled), and `active_base()` trusted the browser's OS appearance over the configured theme (light cards on dark chrome). Both now have regression tests |

---

## Roadmap

- [ ] Real-time feed via Zerodha Kite / Upstox behind the same `fetcher` interface
- [ ] **Cross-sectional forecasting** — rank the whole universe on one model rather than
      fitting one model per name. Where a rank correlation of 0.02 is actually tradeable,
      and the single-name page cannot express it
- [ ] **A strategy backtest behind the forecast** — turn a signal into positions and charge
      it costs, spread, slippage and Indian STT, so the gap between accuracy and profit is
      measured rather than asserted
- [ ] **Regime detection** so the forecaster can flag that its training distribution has
      moved, instead of the fold-by-fold table being the only retrospective warning
- [ ] **Point-in-time fundamentals** as forecaster features, which is the paid-data problem
      that also unlocks the long horizons the Security page currently refuses

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | |
| Data | pandas, numpy | |
| Optimisation & stats | **scipy** | SLSQP for the frontier, Brent/Newton for XIRR, χ² for the Kupiec test, hierarchical clustering |
| Machine learning | **scikit-learn** | RidgeCV / ElasticNetCV with inner time-series CV, random forest, histogram gradient boosting, permutation importance, `TimeSeriesSplit` |
| Market data | yfinance | free, broad coverage, honest about being delayed |
| Charts | Plotly | interactive, one registered template, exports to HTML |
| Dashboard | Streamlit 1.50+ | `st.navigation` multipage, `cache_data` TTL tiers |
| Persistence | SQLite (stdlib) | zero-config, file-backed, no server to run |
| Export | XlsxWriter | formatted workbooks with conditional formatting |
| Tests | pytest | offline, seeded, identity-based |

`matplotlib`, `requests`, `torch` and `tensorflow` remain deliberately absent — every chart is
an interactive Plotly figure, yfinance manages its own HTTP layer, and the forecasting models are
the ones a few thousand daily bars can actually support. `scipy` was added in v2 because SLSQP,
root-finding, χ² p-values and linkage clustering are all genuinely needed; `scikit-learn` was
added in v3 for the same reason — a purged walk-forward over regularised linear models and tree
ensembles is not something to hand-roll, and doing so would be worse engineering, not better.

---

## License

MIT — see [LICENSE](LICENSE).

<div align="center">
<sub>Built as a demonstration of quantitative portfolio analytics and production Python architecture.<br/>
Not investment advice.</sub>
</div>
