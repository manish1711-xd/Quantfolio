# Architecture

## The one rule everything else follows

> **Computation never imports Streamlit. Presentation never computes.**

`src/analytics`, `src/data`, `src/charts` and `src/reporting` contain no
`import streamlit`. `src/ui` and `app.py` contain no formulas. That single
constraint is what makes ~1,900 lines of financial mathematics testable in
milliseconds, offline, with hand-computed expected values — and it is why the
test suite needs neither a browser nor a market data subscription.

A second rule sits under it:

> **Every network call lives in `src/data/`.**

`src/data/fetcher.py` is the only file that imports `yfinance`;
`exchange.py` and `news.py` are the only files that import `urllib`. Nothing
outside that package touches the wire, and everything else receives plain
`pandas` objects.

The rule is the **package boundary**, deliberately, and not a module count —
it was "exactly one module" and then "exactly two" before the count stopped
being the interesting part. One module per provider concern is the shape that
keeps each one readable.

It was "exactly one module" until the full NSE/BSE listing arrived. The
sentence was **restated rather than quietly broken**, because what it exists to
protect is the property, not the count: swapping Yahoo Finance for a broker
feed (Zerodha Kite, Upstox, Interactive Brokers) still means rewriting
`fetcher.py` against the same signatures — no analytics change, no UI change,
no test change. `exchange.py` answers a different question on a different
cadence with different failure modes (see *The stock universe* below), and
folding it into `fetcher.py` would have preserved the sentence while making the
file that matters most harder to read.

---

## Layer diagram

```
                        ┌──────────────────────────────┐
                        │           app.py             │  shell, navigation, routing
                        └──────────────┬───────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
┌───────▼────────┐            ┌────────▼─────────┐          ┌─────────▼────────┐
│  src/ui/       │            │  src/ui/pages/   │          │ src/reporting/   │
│  theme         │            │  overview        │          │  excel           │
│  components    │◄───────────┤  holdings        ├─────────►│  html_report     │
│  formatters    │            │  performance     │          └─────────┬────────┘
│  session       │            │  risk … (13)     │                    │
│  data_access   │            └────────┬─────────┘                    │
│  sidebar       │                     │                              │
│  pipeline      │                     │                              │
└───────┬────────┘                     │                              │
        │                              ▼                              │
        │                     ┌──────────────────┐                    │
        │                     │   src/charts/    │◄───────────────────┘
        │                     │   theme + 7 mods │  Plotly figure builders
        │                     └────────┬─────────┘
        │                              │
        │                     ┌────────▼─────────┐
        └────────────────────►│  src/analytics/  │  23 pure modules
                              │  positions lots  │  no I/O, no globals,
                              │  returns risk    │  no Streamlit
                              │  optimize …      │
                              └────────┬─────────┘
                                       │
                              ┌────────▼─────────┐
                              │    src/data/     │  ← the ONLY network boundary
                              │  fetcher  store  │
                              └────────┬─────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │  src/config.py    src/models.py     │  frozen contracts
                    └─────────────────────────────────────┘

  off to the side, imported by both analytics and ui, network-free:

     src/stock_universe.py      search a bundled stock-master CSV by name
     src/isin.py                validate an ISO 6166 ISIN, resolve it to a symbol
     src/portfolio_importer.py  map any broker's column names onto the schema
     src/watchlist.py           instruments tracked without owning them
     src/alerts.py              when a watched level counts as reached
     src/universe_build.py      rebuild the stock master from exchange files

  and one module that runs with no app at all, invoked by cron:

     src/reporting/cli.py       loads schedules, rebuilds the analysis, sends
```

Dependencies point strictly downward. `config` and `models` import nothing from
the project; `data` imports only those two; `analytics` imports `config`,
`models` and other `analytics` modules; and so on. There are no cycles, which is
what lets any module be imported and exercised in isolation.

---

## Package responsibilities

| Package | Responsibility | Imports Streamlit? | Touches network? |
|---|---|:--:|:--:|
| `src/config.py` | Every tunable constant and convention in one auditable place | no | no |
| `src/models.py` | Domain dataclasses shared by all layers | no | no |
| `src/stock_universe.py` | Search bundled stock-master CSVs by company name or ticker | no | no |
| `src/isin.py` | ISO 6166 check-digit validation; ISIN -> symbol resolution and learning | no | no |
| `src/portfolio_importer.py` | Detect a broker export's columns, normalise, report every rejected row | no | no |
| `src/watchlist.py` | Watched instruments, targets, 52-week positioning | no | no |
| `src/alerts.py` | Price-alert rules: direction, crossing, hysteresis, fire-once state | no | no |
| `src/universe_build.py` | Merge the NSE/BSE listings and bhavcopies into the stock master | no | via `data/exchange` |
| `src/data/exchange.py` | Download the exchanges' own listing and bhavcopy files | no | **yes** |
| `src/data/news.py` | RSS feeds, per-ticker headlines, the economic calendar | no | **yes** |
| `src/analytics/news.py` | De-duplication, instrument matching, headline tone | no | no |
| `src/data/` | Market data access + SQLite persistence | no | **yes** (fetcher only) |
| `src/analytics/` | All financial mathematics | no | no |
| `src/analytics/fundamentals.py` | Statement scores: Piotroski, Altman, Beneish, accruals | no | no |
| `src/analytics/features.py` | Leak-free, scale-free supervised feature matrix and its forward-return label | no | no |
| `src/analytics/forecast.py` | Forward-return models, purged walk-forward validation, conformal intervals | no | no |
| `src/ui/factor_panel.py` | Factor-source picker and regression display | **yes** | no |
| `src/ui/bl_panel.py` | Black-Litterman prior, views editor and re-solve | **yes** | no |
| `src/ui/tax_panel.py` | Harvesting candidates, set-off detail, boundary alerts | **yes** | no |
| `src/ui/options_panel.py` | Contract entry (manual or chain), overlay pricing, payoff | **yes** | no |
| `src/ui/email_panel.py` | Schedule editor, SMTP status, the cron command, test send | **yes** | no |
| `src/ui/alerts_panel.py` | Alert rules, delivery settings, a send-nothing preview | **yes** | no |
| `src/charts/` | Plotly figure builders | no | no |
| `src/reporting/` | Excel workbook and HTML tearsheet writers, schedule model | no | no |
| `src/reporting/email_report.py` | Compose one message and deliver it by SMTP | no | **yes** (SMTP only) |
| `src/reporting/cli.py` | The `python -m` entry point a scheduler invokes | no | via the two above |
| `src/ui/` | Design system, widgets, session state, orchestration | **yes** | no |
| `app.py` | Page config, navigation, routing | **yes** | no |

---

## Why the constants live in `config.py`

A portfolio analytics tool is only as trustworthy as its stated conventions. If
`252` is typed into nine files, nobody can answer "what annualisation does this
use?" without grepping. So every judgement call — the trading-day count, the
VaR confidence level, the risk-score weights, the insight thresholds, the
optimiser's shrinkage intensity, the historical stress windows — is a named
constant in `src/config.py`, and the UI surfaces the rubric it produced.

That has a second benefit: the risk score is not a black box. `RiskScoreRubric`
is a dataclass the app renders as a table, so a user can see that a 72/100 came
from 26 points of volatility, 19 of drawdown, 14 of concentration and 13 of
tail risk — and disagree with the weighting if they want to.

---

## Data flow of one analysis run

```
sidebar widgets
   └─► AppConfig  ──────────────────────────────────────────────┐
holdings grid / CSV / SQLite                                    │
   └─► analytics.positions.clean_holdings ──► validated frame   │
                                                  │             │
              ui.data_access (st.cache_data TTLs) │             │
   ┌──────────────────────────────────────────────┘             │
   ├─► fetcher.fetch_quotes      (5 min TTL)                    │
   ├─► fetcher.fetch_profiles    (24 h TTL)                     │
   ├─► fetcher.fetch_fx_rates    (30 min TTL)                   │
   ├─► fetcher.fetch_price_history (30 min TTL)                 │
   └─► fetcher.fetch_benchmark_history                          │
                    │                                           │
                    ▼                                           │
     analytics.positions.build_positions ──► positions frame    │
                    │                                           │
                    ├─► risk.portfolio_value_series ──► values ─┤
                    ├─► risk.risk_summary                       │
                    ├─► drawdown.drawdown_episodes              │
                    ├─► benchmark.compare                       │
                    ├─► attribution.attribution_summary         │
                    ├─► factors / rolling / lots                │
                    └─► insights.generate_insights ◄────────────┘
                                    │
                                    ▼
                        models.AnalysisResult   (stored in st.session_state)
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              13 UI pages     chart builders   report writers
```

The pipeline runs **once** per input change. `session.run_key()` fingerprints
the holdings and every setting that could alter a number; switching pages
reruns the script (Streamlit always does) but the fingerprint matches, so the
cached `AnalysisResult` is reused and no recomputation or refetch happens.

---

## Input resolution: search, then import, then the grid

Three of the layers above exist only to answer one question — *which instrument
did the user mean?* — before any mathematics runs.

```
"Reliance"                    a Zerodha CSV                 typed directly
     │                              │                              │
     ▼                              ▼                              │
stock_universe.search      portfolio_importer                      │
  bundled CSVs, offline      detect_columns  (alias table)          │
  ranked, deduplicated       build_preview   (per-row validation)   │
     │                              │                              │
     │                       stock_universe.resolve                 │
     │                        (prefer_exchange = file's market)     │
     ▼                              ▼                              ▼
              ticker, quantity, buy_price  ──►  positions.clean_holdings
```

**Why the universe is a local file.** Downloading a full exchange listing on
every app start would be slow, fragile and rude to the provider. The bundled
CSVs are parsed once with the stdlib `csv` module and cached in an `lru_cache`,
so search runs on every keystroke with no I/O at all. The provider's own search
endpoint (`fetcher.search_symbols`) is the fallback for the long tail, and it
fires **only on an explicit button press** — never per character typed.

**Why the local file is never a gate.** A symbol absent from the curated CSVs is
still perfectly usable: `resolve()` returns `None`, the raw text passes through
unchanged, and the data layer — the only thing that can actually confirm a
symbol prices — decides. Inventing a `.NS` suffix for an unknown symbol would
turn a working ticker into a broken one, so nothing here does that.

**Why resolution refuses to fuzzy-match.** `search()` offers typo suggestions
because the user then picks one. `resolve()` does not, because its output is
used to *rewrite* what the user typed. Silently deciding that `RELAINCE` meant
`RELIANCE` and adding that to a portfolio is the kind of helpfulness a financial
tool must not have.

**Why import is two steps.** The importer can map "Tradingsymbol" onto `ticker`
and "Average Price" onto `buy_price`, but a mapping is an inference — and an
inference about which column holds your cost basis has to be visible before it
becomes a portfolio. So `ImportPreview` is rendered first and nothing reaches
session state until Import is pressed.

**Why the preview is editable.** No alias table covers every header that
exists, so detection will sometimes fail outright ("Cost of acquisition per
unit"). `remap()` re-derives the entire preview — every row validation, ticker
resolution and rejection reason — from `ImportPreview.source_frame` under a
corrected mapping, so a failure is recoverable in place rather than a trip back
to a spreadsheet. Three rules keep that honest:

- **One column, one field.** Re-pointing a field at a column another field
  holds takes it from that field, which then shows up in `missing` rather than
  two canonical names quietly reading the same data.
- **A named column that is not in the file unmaps the field.** Your instruction
  could not be carried out, so the field is absent and says so — it does not
  silently keep its old value.
- **An override is recorded only when it was honoured.** Deliberately setting a
  field to "not present" is your decision and is remembered; naming a column
  that does not exist is not, so the table never shows "you chose this" beside
  a blank.

`BLOCKED_HEADERS` is deliberately *not* enforced on an override. It exists to
stop automatic detection reading a market price as a cost basis; a human who
explicitly picks that column may know something the alias table does not, so
`ColumnMapping.blocked()` reports the choice and the UI warns rather than
refusing.

The single hardest bug this design prevents: a holdings export contains both
what you paid *and* what the position is worth now. Reading "Previous Closing
Price" or "LTP" as the cost basis would report a ~0% return on the whole
portfolio, and **nothing on the screen would look wrong**. Hence
`BLOCKED_HEADERS`, and hence the alias table being ordered so explicit cost
aliases claim their column before the ambiguous bare `price` is considered.

### Identifier resolution: symbol, then ISIN

A holdings row can name its instrument three ways, and they are tried in
descending order of directness:

```
"Tradingsymbol" column        "ISIN" column only          neither
        │                            │                       │
        ▼                            ▼                       ▼
stock_universe.resolve      isin.is_valid  ─ fails ─►  reject: "check digit
  RELIANCE -> RELIANCE.NS           │                   should be 8, not 9"
                                    ▼
                            isin.resolve
                          learned ▸ seed         ─ miss ─►  reject: "valid ISIN,
                                    │                        not in the table"
                                    ▼
                            RELIANCE.NS
```

**Why a symbol always wins.** ISIN resolution is a fallback. A file that
carries both is read from the symbol column, and the ISIN is used only to
*learn* the pairing.

**Why learning is the primary source.** Most broker exports carry both columns,
which means they have already told us the mapping for every row they contain.
Recording that costs nothing and is strictly better data than anything this
repository could ship: it is the user's own broker's authority, and it grows
exactly the coverage that particular user needs. `data/isin_learned.csv` is
therefore a cache, gitignored, and safe to delete.

**Why the seed is small and machine-checked.** 114 pairings exist so the
feature works on a fresh install. That table is the one place in the codebase
where a silent transcription error would be *severe* — a wrong ISIN puts
another company in the portfolio while every number still looks plausible — so
two mechanical gates police it in CI: every ISIN must satisfy its own ISO 6166
check digit, and every ISIN's country prefix must agree with the country of the
symbol it maps to. A mistyped seed row is a build failure.

**Why the check digit matters more than it looks.** It is the difference
between "this identifier is wrong" (arithmetic, certain) and "we could not find
this identifier" (coverage, uncertain). Those need different messages and
different fixes, and conflating them would send users hunting for the wrong
problem. It is also why an ISIN is never looked up online before validating
locally: asking a provider about a typo can only produce a plausible wrong
answer.

---

### The invariant that makes "nothing is dropped" checkable

```
ImportPreview.n_rows + ImportPreview.n_rejected == ImportPreview.source_rows
```

Asserted as `accounted_for`, and asserted directly by the test suite across
every file shape it can produce. An importer that quietly drops two of your
twelve holdings is worse than one that refuses the file, because the resulting
portfolio still looks plausible — so every rejected row is returned with the
reason it was rejected, and the UI shows them with a download button.

---

## The two "factor" modules, and why they are separate

`src/analytics/factors.py` and `src/analytics/factor_model.py` sound like the
same thing and answer different questions:

| | `factors.py` | `factor_model.py` |
|---|---|---|
| Kind | Unsupervised | Supervised |
| Input | Only the portfolio's own returns | Returns **plus** external factor series |
| Method | PCA, correlation-distance clustering | OLS with Newey–West standard errors |
| Answers | "How many independent bets is this really?" | "How much size / value / momentum is in here?" |
| Needs external data | No | **Yes** |

Splitting them is what lets the first one work offline for every portfolio
while the second is honest about needing data it cannot conjure.

### Where factor returns come from, and what is claimed about them

```
published file                        ETF proxies
(French, IIM Ahmedabad)               (VTI, IWM, VTV, VUG, MTUM…)
        │                                      │
   parse_factor_csv                    build_proxy_factors
   percent-vs-decimal                  long − short, or long − rf
   detected AND REPORTED                       │
        │                                      │
        └──────────────┬───────────────────────┘
                       ▼
              factor_regression
        excess returns, inner-joined dates
                       ▼
      RegressionResult: β, SE, t, p, VIF per factor
                       + α, R², adj R², F, DW, resid vol
```

**The percent trap.** Published factor files are quoted in percent (`0.53`
means 0.53%) while everything here is decimal. Getting it wrong scales every
loading by 100, and the failure is *invisible* — the betas simply look absurd
with no error anywhere. Auto-detection therefore inspects the magnitude and
**always reports the decision it made**, and can be overridden explicitly.

**Why proxies are labelled relentlessly.** An ETF spread is correlated with the
academic factor and is not it. `FactorSpec.caveat` carries the specific
limitation per factor — "a long-only momentum ETF minus the market omits the
short leg" — so the UI can state the real objection instead of one blanket
disclaimer nobody reads. And where no defensible proxy exists,
`FACTOR_PROXY_GAPS` says so: India gets market and a weak size tilt and
nothing else, because padding the list with thinly-traded recent ETFs would
shorten the regression window for every other factor and add more tracking
noise than signal.

**Why Newey–West is the default.** The correction applies to the
autocorrelation of the score `u_t · x_t`. With a serially uncorrelated
regressor it barely moves the standard error; with a persistent factor *and* a
persistent residual — the realistic case — it roughly doubles it, taking a
t-statistic of 16 down to 8. At zero lags it reduces to White's HC1, which in
turn matches classical OLS on iid homoskedastic residuals, so the default
costs nothing when the simpler assumptions happen to hold. All three of those
relationships are asserted in the test suite.

---

## Caching strategy

Streamlit reruns the whole script on every widget interaction, so caching is not
an optimisation here — it is a correctness requirement. Without it, dragging a
slider would refetch every ticker.

Three tiers:

1. **`st.cache_data` on `src/ui/data_access.py`** — network responses, keyed by
   `(tickers_tuple, period)` with a TTL matched to how fast that data actually
   changes. Tuples, not lists, because the cache key must be hashable.
2. **`AnalysisResult` in `st.session_state`** — the computed result, invalidated
   by `run_key()`.
3. **No caching in `src/analytics`** — those functions are pure and fast
   (the whole pipeline is well under a second for a 30-name portfolio), and a
   cache on a pure function would only add a stale-state failure mode.
4. **A bounded memo in `src/ui/pages/forecast.py`** — the one exception, and
   it exists because the forecaster breaks the assumption behind tier 3. A
   seven-model leaderboard is ~17 seconds of fitting, not a millisecond, and
   its result carries a path matrix and a residual vector that
   `st.cache_data` would have to hash and serialise on every rerun. So the
   page keeps a plain dict in session state, keyed by the exact control
   tuple plus a fingerprint of the price history, holding at most six
   entries FIFO. Pure functions, memoised at the call site, with the bound
   stated — rather than a cache decorator quietly retaining fitted models
   for the life of the session.

The **Refresh market data** button clears tier 1 and forces tier 2 to recompute.

---

## Black-Litterman: where the optimiser's inputs come from

The Optimiser page carries a section called "THE HONEST HEALTH WARNING", and it
is honest: mean-variance optimisation is exquisitely sensitive to expected
returns, and a five-year sample mean is a poor estimator of them. Black-Litterman
is the constructive answer, so it sits on the same page immediately above that
warning.

```
prior portfolio                     investor views
(cap / current / equal weights)     P·E[R] = Q, with confidence
        │                                    │
   reverse optimisation              Omega = ((1-c)/c)·P τΣ P'
   Pi = delta · Sigma · w                    │
        │                                    │
        └────────────────┬───────────────────┘
                         ▼
     E[R] = Pi + τΣP'(P τΣ P' + Omega)^-1 (Q − P·Pi)
     Sigma_post = Sigma + τΣ − τΣP'(…)^-1 P τΣ
                         ▼
              posterior_mu() + risk-free rate
                         ▼
          src/analytics/optimize.py  (unchanged)
```

**Why it drops in without touching the optimiser.** `optimize.max_sharpe` and
friends take `(mu, cov)`. Black-Litterman produces exactly those, so the
integration is a substitution rather than a rewrite — and the panel solves the
*same* objectives on both input sets so the user sees what their views actually
changed. A single set of weights with no baseline would hide the whole point.

**The three identities that pin it.** Each is asserted in the test suite:

1. `(δΣ)⁻¹(δΣw) == w` to 1e-12. Reverse optimisation and forward optimisation
   are inverses, which is *what makes* Π an equilibrium rather than a rescaling
   of the weights. An implementation failing this is not computing an
   equilibrium at all.
2. **No views ⇒ posterior == prior, exactly.** Not to a tolerance. This is why
   the model is safe to leave switched on: an asset nobody has an opinion about
   keeps its equilibrium return instead of inheriting sample-mean noise.
3. **Both posterior formulas agree.** The production code uses the Woodbury
   form, which inverts one k×k matrix (k = number of views) and never inverts
   `Omega`. The textbook form inverts both `Omega` and `τΣ` — and a confident
   view drives `Omega` towards singular, so it is a numerical trap. Keeping the
   textbook form available purely to assert agreement is what justifies using
   the faster, safer one.

**The property worth understanding.** A view propagates to assets it never
mentions, through the covariance. Give a bullish view on D and a correlated C
moves too, ordered by correlation strength. That is the single feature
separating Black-Litterman from simply overwriting an expected return, and the
UI names the spillover assets explicitly so it reads as intent rather than as a
bug.

**Honest scoping.** True Black-Litterman reverse-optimises the *global market
portfolio*. This has the user's holdings, so `market_cap_weights` cap-weights
those and the docstring, the UI and the methodology block all say so. Since
Yahoo reports no market cap for most ETFs, the cap prior falls back to current
weights with an explicit notice rather than dead-ending — an ETF-heavy Indian
book hits that path routinely.

---

## The tax module, and why it refuses to flatter the user

`src/analytics/tax.py` is the only place in this application where a plausible
wrong answer has consequences off the screen. Someone acting on an invented
tax saving pays real brokerage, real STT and a real spread to achieve nothing.
That shapes the whole design.

```
FIFO lots (src/analytics/lots.py)          realised trades, this FY only
        │                                            │
   classify_lots                              realised_gains
   per LOT, not per position                   STCG / STCL / LTCG / LTCL
        │                                            │
        └──────────────────┬─────────────────────────┘
                           ▼
                   harvest_candidates
        each loss priced MARGINALLY: re-run the whole
        waterfall with it added, take the difference
                           ▼
                       set_off
        1. STCL -> STCG      (elective order: highest rate)
        2. LTCL -> LTCG      (it can go nowhere else)
        3. STCL -> LTCG      (mandatory, in full)
                           ▼
        tax + carry-forward + wasted_on_exempt_gains
```

**Per lot, never per position.** Two FIFO tranches of one ticker can have
different tax characters, and harvesting is a lot-level decision. Aggregating
to the position first would destroy exactly the information the feature needs.

**A loss is priced by what it absorbs.** Not `loss x rate`. Each candidate
re-runs the full set-off waterfall with that one loss added and takes the
difference in tax. The two figures diverge whenever the exemption or the
hierarchy binds, which is most of the time — and the honest answer is often
zero.

### The correction that mattered most

An early draft of `set_off` "optimised" step 3 by applying short-term losses
only to long-term gains *above* the exemption, on the reasoning that
offsetting an already-exempt rupee saves nothing. That is sound economics and
wrong law. Sections 70 and 71 say a loss **shall** be set off against
available gains of the permitted character in the same year: the quantum is
mandatory, and only the ordering is elective. The draft would have reported a
lower tax than the statute permits — the most dangerous class of bug this
module could carry.

The fix does the mandatory thing and then reports the damage:
`SetOffResult.wasted_on_exempt_gains` is the loss consumed against gains the
exemption already sheltered. It buys nothing and it cannot be carried forward
instead. Surfacing that number is more useful than any optimisation would have
been, because the only way to avoid the waste is to harvest in a *different
financial year* — which is an actionable insight, unlike a wrong number.

### The inversion worth understanding

For a **gain**, crossing the 12-month boundary lowers the rate: wait. For a
**loss**, crossing it makes the loss *less* useful, because a short-term loss
offsets both short- and long-term gains at the higher rate while a long-term
loss offsets only long-term gains. Holding a loss for "long-term treatment" is
the wrong instinct, and `boundary_alerts` gives the two cases opposite advice
with the reasoning attached.

### Dated rules, not constants

India changed the rates mid-year on 23 July 2024, so `INDIAN_TAX_REGIMES` is a
dated list and `regime_for(d)` a lookup. A sale on 22 July 2024 is taxed at
15%/10% with a ₹1 lakh exemption; one on 24 July at 20%/12.5% with ₹1.25 lakh.
A tool using one rate for the whole year would be wrong for anyone reviewing
FY2024-25.

### Where it stops

Section 112A grandfathering needs the **highest quoted price** on 31 January
2018. This application holds split-adjusted closes, which are neither. So a
pre-cutoff lot is flagged and its gain reported as *overstated*, rather than
computed on a number that is not the statutory one — the same principle the
ISIN table follows, and the reason both are trustworthy.

---

## The options overlay, and designing around a missing feed

This feature is the clearest case in the codebase of a data limitation
dictating an architecture rather than being papered over.

```
                    is there a chain for this symbol?
                                 │
              ┌──────────────────┴──────────────────┐
        yes (US listings)                    no (NSE / BSE)
              │                                     │
     fetch_option_chain                    the reason, named
     bid/ask/IV/OI, untouched               "the gap is the feed,
              │                              not the market"
              └──────────────────┬──────────────────┘
                                 ▼
                    OptionQuote(strike, premium, expiry, …)
                                 ▼
        black_scholes_price · greeks · implied_volatility
                                 ▼
     covered_call / protective_put / collar  ->  payoff_curve
```

**The measurement that shaped it.** `AAPL` returns 21 expiries;
`RELIANCE.NS` and `^NSEI` return zero. That was checked, not assumed. Indian
equity and index options are among the most heavily traded contracts anywhere,
so an empty table would have been actively misleading — hence
`option_chain_available()`, which answers *before* any request and returns a
sentence explaining that the gap is in the data source.

**Manual entry is the primary path.** The premium is the only input that
genuinely requires a live market, and the user reading their broker's option
screen already has it. So `OptionQuote` is the seam: a chain row and four typed
numbers produce the same object, and every calculation downstream is identical.
For an India-first product this is the difference between a working feature and
a US-only one.

**Two conventions differ here from the rest of the package**, both deliberately:

- **365 calendar days, not 252 trading days.** An option decays over a weekend.
  This is the one module where the project-wide `TRADING_DAYS` would be wrong,
  and using it would understate theta by about 30%.
- **Compounded annualisation** of overlay yields. A 2.1% premium over 30 days is
  28.8% a year, not 25.6% — and for a structure written repeatedly, that gap is
  the reason to write it.

**The European/American irony.** Black–Scholes prices a European option. NSE
options *are* European, so the model is exact for the market with no chain
data; US equity options are American, so it is an approximation precisely where
the chains exist. That is stated in the UI whenever a US chain is priced,
rather than left for the reader to already know.

**What the module refuses to say.** A covered call's `max_loss` is the full
position net of premium, not a comforting small number — the underlying can go
to zero. `implied_volatility` returns NaN below the no-arbitrage floor instead
of a plausible figure, because a price under the floor means the quote is
stale, crossed, or carries an American early-exercise premium the European
model cannot represent. And writing more contracts than the shares cover
produces an explicit **NAKED** warning: it is the one error in this module that
could cost a user unbounded money.

**Testing without golden numbers.** Option maths is almost entirely pinned by
identities, so the suite needs no values copied from a spreadsheet: put–call
parity to 1e-12, every Greek against a central-difference derivative, gamma and
vega identical across call and put, `delta_call − delta_put == e^(−qT)`,
implied volatility round-tripping from 5% to 200%, and every payoff against its
closed form at, above and below each strike.

---

## Scheduled e-mail, and where the clock lives

This is the clearest case in the codebase of a *platform* limitation dictating
an architecture. The other such cases were data limitations; this one is about
what a Streamlit process actually is.

**Streamlit cannot schedule.** It is a request-response server: it runs the
script when a browser asks it to. There is no background worker, no timer, and
no process alive at 08:00 on a Monday unless something else starts one. A
`threading.Timer` inside the app would fire zero times when the app is idle,
*N* times when *N* tabs are open, and die on the next restart without a word.
An "Enabled" toggle that silently does nothing is the most dishonest thing this
application could ship, so the clock is explicitly outside it:

```
   +----------------------+          +-------------------------------+
   |  Streamlit app       |          |  cron / Task Scheduler        |
   |  src/ui/email_panel  |          |  (the only real clock)        |
   |                      |          +---------------+---------------+
   |  stores WHAT and     |                          | every N minutes
   |  WHEN; prints the    |                          v
   |  command; sends one  |          +-------------------------------+
   |  test on request     |          |  python -m src.reporting.cli  |
   +----------+-----------+          +---------------+---------------+
              |                                      |
              |  schedules table                     | load_schedules()
              +------------->  SQLite  <-------------+ mark_schedule_sent()
                              store                  |
                                                     |
                     schedule.due_schedules(now) <---+
                                                     |
                                                     v
                     pipeline.run(cfg, holdings, txns)   <- the SAME pipeline
                                                     |
                                                     v
                     html_report.build_html_report()
                                                     |
                                                     v
                     email_report.compose / send_message
```

**The same pipeline, not a second one.** `pipeline.run` grew two optional
arguments -- `holdings` and `transactions` -- defaulting to the session, so the
runner passes frames loaded from SQLite while every other caller is unchanged.
A scheduled report computed by a separate simplified path would eventually
disagree with the dashboard, and the entire value of the e-mail is that it says
what the app says.

**Idempotence is the property that really matters.** A cron entry running every
fifteen minutes must deliver an 08:00 report once, not four times an hour. So
`is_due` compares `last_sent` against the most recent *occurrence* rather than
against the wall clock:

```
due  <=>  enabled AND usable
          AND (an occurrence exists at or before now)
          AND last_sent < that occurrence
```

That single formulation buys three behaviours at once: any cron cadence sends
once; a slot missed while the machine was asleep is delivered once, however
late; and nothing is ever sent twice. Two ordering decisions defend it:

* **`last_sent` is written only after a successful send**, so a crash between
  the two costs one duplicate rather than losing a report. A scheduled report
  that quietly stops arriving is the failure nobody notices for months.
* **It is stamped with the occurrence, not the send time.** An 08:00 slot
  delivered late at 11:20 and recorded as 11:20 would suppress the *next*
  day's 08:00 report if the runner happened to fire before 11:20.

The write is a single targeted `UPDATE ... WHERE name = ?` rather than a
read-modify-replace, so the runner cannot clobber a schedule the user edited
while a report was building.

**A new schedule starts from its next slot.** Configuring "daily at 08:00" at
14:00 leaves today's 08:00 occurrence genuinely unsent, so a strict reading
mails a report seven minutes after Save. Correct, and surprising enough to read
as a bug -- hence `starting_next_slot`, applied on creation only. Applying it
on *load* would swallow whatever was outstanding on every restart.

**No credential is stored anywhere.** The `schedules` table holds when and to
whom; the SMTP host, username and password are read from `st.secrets` or the
environment at send time. A password typed into a Streamlit text input would be
persisted into `data/quantfolio.db` -- a plaintext secret inside the project
directory, which `.gitignore` cannot save you from once that file has been
copied. So the panel has **no password field**: it prints the `secrets.toml`
block to write instead, and `SmtpConfig.__repr__` redacts the value so a
traceback cannot leak it either. TLS uses `ssl.create_default_context()` with
certificate *and* hostname verification left on, and authenticating over an
unencrypted connection is refused unless explicitly overridden.

**Nothing outward-facing fires on a render.** A Streamlit script re-runs on
every widget interaction anywhere on the page, and an e-mail cannot be
recalled. So `send_message` opens no connection unless asked, the panel's test
send sits behind a form *and* a confirmation checkbox, and a test asserts that
rendering the panel with a live schedule and working credentials opens no
socket at all. The test send also passes `record=False`: proving the plumbing
works must not consume the occurrence the user configured.

**What is deliberately not in the e-mail: the charts.** They are built by the
Streamlit chart layer from a live session, and a cron job has none. Rendering a
second set for e-mail would mean two depictions of the same numbers that could
disagree, so the attachment carries the figures as tables and stays in the tens
of kilobytes rather than the megabytes a Plotly bundle costs. The plain-text
body is not a fallback either -- attachments get stripped by gateways and
ignored on a commute, so the headline numbers live in the text.

**Two smaller things a cron job needs and a web app does not.** Streamlit's
cache decorators log a "no runtime" warning as each one is *constructed*, which
is twenty-odd lines emitted at import time; the runner lowers Streamlit's log
level before importing the pipeline, using Streamlit's own setter because it
re-levels its private logger registry from its configuration. And stdout is
reconfigured to UTF-8 with `errors="replace"`, because on Windows `print`
encodes with the locale codepage and a rupee sign in a pipeline warning would
otherwise kill the runner with a `UnicodeEncodeError` while it was reporting a
problem.

**Testing a scheduler without waiting for one.** Every clock is injected, so
idempotence is asserted by simulation rather than by hope: ninety-six
fifteen-minute ticks produce one e-mail, 1,440 one-minute ticks produce the
same one, a fortnight produces fourteen, a quarter produces thirteen weeklies,
and a leap year produces twelve monthlies with the 31st clamped every short
month. Delivery is tested through a recording fake in place of `smtplib`, which
is the only way to assert the transport decisions -- STARTTLS versus implicit
SSL, a verified context, login only when credentials exist -- since they are
invisible from outside. And the runner is driven through `main(argv)` and
asserted on its exit code, because that is all a scheduler ever reads.

---

## Price alerts, and the difference between a condition and an event

The watchlist could already answer "how far is this from my target". Turning
that into an alert needed two things it did not have, and both are visible in
the design.

**A target has no direction.** `watchlist.triggered` reports *proximity* --
`|to_target_pct| <= tolerance` -- and its docstring is explicit that it can do
no better, because the table stored a target but never which side of the price
that target was on. So it could not tell an upside level the price rose to from
an entry level it fell to, and an alert that cannot tell those apart is
useless. `WatchItem` therefore gained `direction`, plus `reference_price` --
the price when the target was set -- so a direction can be *inferred* for a
rule saved before the field existed. Inference is a fallback, never a silent
default: a target with alerting on and no direction is reported as **inactive**
rather than displayed as an armed rule that would never fire.

**A condition is not an event.** A rule that e-mailed whenever `price >= level`
would send on every check for as long as the price stayed there. So each rule
is a two-state machine and only one edge sends:

```
                    price reaches the level
                   (last price, or session high/low)
        ┌──────────────────────────────────────────────┐
        │                                              ▼
   ┌─────────┐                                   ┌────────────┐
   │  ARMED  │                                   │ TRIGGERED  │
   │ waiting │                                   │  silent    │
   └─────────┘                                   └────────────┘
        ▲                                              │
        └──────────────────────────────────────────────┘
              price retreats past the level by
              rearm_pct  AND  cooldown elapsed
                     (no e-mail on this edge)
```

`ARMED -> TRIGGERED` is the only transition that produces an e-mail. The return
edge is silent, because "your level is no longer reached" is not news.

**Why both guards.** The re-arm band and the cooldown look redundant and are
not, and the numbers are pinned in the suite rather than asserted loosely:

| Guard | Stops | Measured |
|---|---|---|
| 1% re-arm band | A price wobbling on a round number | 96 polls on a parked price -> 1 e-mail |
| 12 h cooldown | A price volatile enough to clear the band repeatedly | 100/115 swings all day: 144 e-mails -> 2 |

The band handles noise; the cooldown bounds the worst case. Neither alone is
enough, and a user who receives 144 e-mails in a day has effectively lost the
feature.

**Event-triggered, so no cadence to configure.** This is where alerts differ
structurally from scheduled tearsheets, which share the same runner:

* A tearsheet's idempotence key is the scheduled **occurrence** -- one e-mail
  per 08:00 slot, however often the command runs.
* An alert's is the **transition** -- one e-mail per crossing. There is no
  frequency setting at all: the rules are checked on every invocation, so the
  cron interval simply *is* the check interval.

Forcing alerts into `Schedule.frequency` would have muddied a clean guarantee
to reuse a field, so they got their own single-row `alert_settings` table
instead. Alerts want a five-minute cron and tearsheets do not care, which is
exactly why one command does both and why running it often has to be safe.

**Only rules with an alert are priced.** A fifty-name watchlist with three
alerts costs three quote requests per run, not fifty. This runs every few
minutes against a free endpoint, and fetching prices nothing is waiting on is
the fastest available way to get throttled. Alerting switched off fetches
nothing at all and is **not** reported as a failure -- a cron entry must not
log an error because the user turned the feature off deliberately.

**The commit point.** `store.update_alert_state` is a single targeted
`UPDATE ... WHERE ticker = ?` and the only write the alert runner performs.
`save_watchlist` is a wholesale replace, so a cron job using it would undo a
note or a target edited in the browser while quotes were being fetched -- the
same reasoning as `mark_schedule_sent`, and `watchlist.with_alert` is the pure
counterpart that merges only the alert fields. State is written *after* a
successful send, so a delivery failure leaves the rule armed for the next run
rather than swallowing the alert.

**What the module refuses to claim.** Three limitations are stated in
`src/alerts.py` itself rather than left to the UI, because they decide whether
a user should act on an alert:

* The feed is delayed, so an alert says the level *had been* reached as of a
  timestamp -- which every event carries.
* Between two checks only the session high/low is evidence, so it is consulted
  as well as the last price, and every event names **which observation saw the
  crossing**. A session-high trigger is never presented as the current price.
  A crossing entirely inside an earlier, already-closed session cannot be
  recovered from a quote at all; checking more often narrows that gap and never
  closes it.
* A rule that could not be evaluated -- no quote, no direction -- travels in
  `skipped`, appears in the e-mail under its own heading, and prints on stdout
  even when nothing fired. Silence about a rule looks exactly like "your level
  was not reached", and that is the one failure mode an alert system must not
  have.

**A migration, for the first time in this codebase.** The alert fields are new
columns on an existing table, and `CREATE TABLE IF NOT EXISTS` alters nothing
-- so without a migration every alert write would fail with "no such column" on
the one database that already holds the user's data. `_MIGRATIONS` plus
`PRAGMA table_info` handles it: additive `ALTER TABLE ... ADD COLUMN` only,
idempotent, no version counter (the columns present *are* the version), and a
failure is logged and skipped rather than raised, because an un-migrated column
costs one feature whereas a store that will not open costs every saved
portfolio. A migrated row comes back with alerting **off** -- an upgrade must
not start e-mailing about targets set years ago.

---

## The stock universe, and what a full market listing costs

The universe was a hand-curated 243 NSE rows. It is now ~7,570 instruments
generated from the exchanges' own files. That change is mostly *data*, but it
forced three architectural decisions worth recording.

### 1. A second network module, declared rather than smuggled

`fetcher.py` used to open with "every network call in the whole project lives
in this module". Exchange downloads could not honestly live there, so the
invariant was **restated**:

> Every network call lives in `src/data/`, in exactly two modules, and nothing
> outside that package imports `urllib` or any HTTP client.

The two are genuinely different problems -- yfinance versus the exchanges'
CDNs, prices-for-symbols-you-name versus the-list-of-what-exists, every few
minutes versus once a day, one bad ticker versus the whole file missing. What
the original invariant existed to protect -- that swapping the price provider
is a rewrite of one file -- still holds. Quietly adding `urllib` to
`fetcher.py` would have preserved the sentence and lost the property.

No new dependency: stdlib `urllib` with a cookie jar, `zipfile`, `csv`,
`json`. A browser `User-Agent` is required, not optional -- NSE's CDN answers
403 to `Python-urllib/3.x`.

### 2. Build-time, not request-time

```
   ONCE A DAY, ON DEMAND                      EVERY SEARCH
   ─────────────────────                      ────────────
   python -m src.universe_build               stock_universe.search()
             │                                          ▲
             ▼                                          │
   data/exchange.py  ─── NSE EQUITY_L ────┐             │
                     ─── BSE scrip master ┤             │
                     ─── NSE bhavcopy ────┤             │
                     ─── BSE bhavcopy ────┘             │
             │                                          │
             ▼                                          │
   universe_build.merge()  ◄── data/indian_stocks_curated.csv
             │                 (hand-maintained, never written)
             ▼                                          │
   data/indian_stocks.csv  ────────────────────────────┘
   data/universe_meta.json    (committed; read offline)
```

The running app never downloads a listing. Search stays instant, offline and
deterministic, and the whole thing still works on a plane. The cost is that
the universe is *as of a date*, which is why `universe_meta.json` exists and
why the search caption prints the date rather than implying currency.

### 3. Two inputs, one output

They were briefly the same file, and running the build twice replaced the
pristine seed's backup with a generated one -- losing hand-written aliases
irrecoverably, on a repository with no version control. Now:

* `indian_stocks_curated.csv` is **input only**: aliases ("RIL", "HUL"),
  sectors NSE does not publish, and the eleven NSE ETFs that appear in no
  *equity* listing.
* `indian_stocks.csv` is the generated merge, `.bak`-backed and written
  atomically via `os.replace` from a temp file in the same directory.

That makes the build idempotent, which a test pins by running it twice and
comparing bytes.

### Why the bhavcopy, when the listing already has the names

Because a full-market universe **ranks worse than a curated one** unless it
knows which companies matter. Measured on the real files, before turnover:

| Query | Returned | Should have returned |
|---|---|---|
| `hdfc` | HDFC AMC (₹241 cr) | HDFC Bank (₹2,595 cr) |
| `tata` | Tata Capital (₹25 cr) | Tata Power (₹439 cr) |
| `rel` | Ravindra Energy (₹2.2 cr) | Reliance Industries (₹1,767 cr) |
| `relaince` | Reliance **Communications** (₹0.3 cr) | Reliance Industries |

The first three lost to a rule that prefers a *shorter* symbol for a prefix
match; the fourth lost to the alphabet, because every record sharing a fuzzy
key scores identically on the difflib ratio alone. The listings carry no
measure of size, and the bhavcopy carries `TtlTrfVal` -- the day's traded
value. That is the only prominence signal available anywhere in this data.

**The transform took three attempts, and the failures are the interesting
part.** The CSV stores the raw turnover -- the measured number -- so the
ranking rule can change without re-downloading anything:

* `log10(t)/11`, a raw log over a fixed divisor, put a ₹86 cr name and a
  ₹25 cr name **0.05 apart** -- less than the cost of one extra symbol
  character. No effect.
* A **percentile rank** fixed the mid-caps but *saturates at the top*, which is
  exactly where search collisions happen: HDFC Bank and HDFC AMC came out
  0.016 apart despite 10x the turnover. Worse, ranked *within* each exchange it
  inverted every dual listing, because BSE's ~5,000 mostly-thin names put a
  mid-cap at a higher percentile there than the same company's busier NSE line
  reaches among NSE's more liquid population. `INFY.BO` outranked `INFY.NS`,
  contradicting the India-first rule `EXCHANGE_PRIORITY` exists to enforce.
* **A log normalised over the observed range**, pooled across exchanges, keeps
  the ratio information at both ends. The HDFC pair lands 0.12 apart, the Tata
  cluster spreads across 0.15, and the busier line of a dual listing wins on
  its own merits. 0.0 is reserved for "did not trade" so that it ranks below
  "traded a little", which is real information about something a user may be
  about to buy.

**The bonus is bounded, and the bound is load-bearing.** Turnover may only
reorder hits that matched the same *way*; it must never promote a substring
match over a name match:

```
_PROMINENCE_BONUS  <=  _MIN_TIER_GAP - _MAX_PREFIX_PENALTY
        2.0        <=       4.0      -        0.8
```

A test pins that inequality *and* asserts `_MIN_TIER_GAP` against the tier
constants themselves, so lowering a tier cannot silently invalidate it. Exact
identifier matches carry no bonus at all -- typing a ticker means that ticker.

The prefix-length penalty was cut from 0.5 to 0.1 per character in the same
change. Its original justification, "TCS should beat TCSPLUS for the query
TCS", is handled by the exact-symbol tier sixteen points above; at 0.5 it was
outvoting a tenfold difference in traded value.

### What the build refuses to do

Every one of these was a real failure mode, not a hypothetical:

* **Accept a 200 as a file.** Asked for a future date, BSE serves an HTML error
  page at status 200, which `csv.DictReader` turns into 71 rows of `None`.
  Every download is checked against the columns it must carry -- a schema
  contract, not a row-count threshold, so it also catches a changed URL, a
  redirect to a login page or a silently altered format.
* **Address BSE by the wrong identifier.** Yahoo prices `RELIANCE.BO` and
  returns nothing for `500325.BO`. Measured before generating anything; using
  the numeric scrip code would have produced ~5,000 searchable rows whose
  prices never load, which is the same harm as inventing symbols.
* **Assume a date is a trading day.** There is no calendar in these files, so
  `latest_bhavcopy` walks backwards -- skipping weekends without a request,
  spending one 404 on each holiday -- and reports the date it actually got.
* **Write a truncated universe.** If a listing download fails, nothing is
  written and the existing file is untouched. A refresh that half-worked is
  worse than one that did not run, because search would quietly stop finding
  things.
* **Delete a curated alias.** "RIL" exists in no exchange file. The merge is
  additive, and when the exchange's spelling replaces the curated `name` the
  superseded name is kept as an alias -- dropping it removed a lookup key that
  worked, and because BSE's own spelling of Reliance happened to *be* the old
  one, CSV import started rewriting NSE holdings to `.BO` symbols.
* **Keep a symbol that has stopped existing.** A curated row absent from the
  listing *and* showing no turnover is dropped and named. Two independent
  sources must agree. On the run that produced the committed file that was
  `TATAMOTORS.NS` (demerged into `TMCV`/`TMPV`), `LTIM.NS`, `AKZOINDIA.NS` and
  `GUJGASLTD.NS` -- all four independently confirmed unpriceable by Yahoo. The
  eleven curated ETFs are kept by the *same* rule: absent from an equity
  listing by definition, but really trading.

### A side effect worth naming: ISIN coverage

The listings publish an ISIN per instrument, so `isin.load_map()` now draws on
three sources in increasing order of authority -- the generated universe
(5,061 pairings, unreviewed), the hand-checked seed (114), and anything learned
from the user's own imports. ISIN-only broker exports went from working for the
seeded names to working for the market.

Wiring that up exposed a real bug: `load_map(data_dir=X)` read the seed from
`X` but the universe from the *global* data directory, so an "isolated" map was
not isolated. Every source now resolves from one root, which is also what
restored hermetic tests.

---

## Fundamental scores, and the forecast the *statements* cannot support

The request that produced this module was "an ML model predicting the return
for next month, year, 3 year, 5 year, 7 year, 10 year, 11 year and so on, of
the balance sheet". Half of that is buildable and half of it is not, and the
split is worth recording because it is entirely a *data* question.

### What the provider actually supplies

Measured before anything was written: **five annual periods** of balance
sheet, income statement and cash flow, with identical row labels on NSE and US
filings, and every field the three classic scores need present. Enough for
real analysis.

### Why no horizon beyond a year gets a number

A model for an *N*-year return needs pairs of (accounts at year *T*, return
from *T* to *T+N*). Each pair consumes *N* years of history, so:

| Horizon | Training pairs from 5 periods |
|---|--:|
| 1 month / 3 months / 1 year | 4 |
| 3 years | 2 |
| 5, 7, 10, 11 years | **0** |

A ten-year figure would not be a weak prediction. It would be a prediction
fitted to zero observations, printed next to a portfolio value, in a tool
someone allocates real money with. `horizon_availability()` therefore returns
a `HorizonCheck` per horizon — pair count, verdict, reason — and the UI lists
**every horizon that was asked for**, marking the impossible ones and saying
what data would unlock them. Listing them beats omitting them: a missing row
reads as an oversight, whereas a row saying "0 training pairs" is an answer.

Two further biases apply even where the arithmetic works, and travel with the
result: the statements are **restated** (a 2022 input contains information
published in 2024), and delisted companies are **absent from the provider
entirely** — a fact this project discovered the hard way when `TATAMOTORS.NS`
vanished mid-session.

### Where this constraint does *not* apply

The refusal above is about **annual accounts**, and it is a data-volume
argument rather than a philosophical one. Five annual periods give four usable
training pairs. A ten-year daily price history gives about 2,500 rows.

That is the whole difference, and it is why `src/analytics/forecast.py` exists
and does produce a number: it is fitted to *prices*, where there is enough data
to hold out 30% of it in sequential blocks and still quote an error with a
standard error worth reading. The two statements are consistent, and each is
made in the UI at the point where it applies — the fundamentals panel lists
every impossible horizon with its zero pair count, and the forecast page
reports skill against a random walk on every horizon it does offer. **Neither
page borrows the other's credibility.**

### The one rule every score obeys

**A score computed from incomplete inputs is reported as incomplete, never as
a low score.** This is the standard bug in implementations of these scores,
and it is not cosmetic: a debt-free company that reports no long-term debt
figure loses Piotroski's leverage signal, and scoring that as a *failure* ranks
it below a leveraged competitor. Here it changes the denominator instead —
"6 of 8 signals passed, 1 unavailable" — and the verdict words ("strong",
"weak") are withheld entirely unless all nine ran, because those thresholds are
defined on the full nine.

The same rule, applied per score:

| Score | Behaviour when an input is missing |
|---|---|
| Piotroski F-Score | Denominator shrinks; the signal is shown as `n/a`, never as a fail |
| Altman Z-Score | **Withheld entirely** — a Z-Score missing a term is not a lower Z-Score |
| Beneish M-Score | **Withheld entirely** — the published coefficients are only meaningful together |
| Interest cover | `NaN`, never infinity: no reported interest expense means *undefined* cover |

`_div` returns `NaN` rather than `inf` for a zero denominator precisely so that
last row holds — an infinite interest cover would flatter a company whose
filing is merely incomplete.

### Two calibration decisions

**The Altman variant is chosen from the issuer's market.** The original 1968 Z
includes an asset-turnover term that makes a software company look distressed
and a retailer look safe on business model alone, so Z″ (which drops it) is
used throughout. The +3.25 emerging-market adjustment is applied to Indian
issuers and withheld from developed-market ones — applying it everywhere
inflated Apple's score from 19.7 to 22.9 against thresholds never calibrated
for it. The variant used is printed with the score, because a Z-Score quoted
without its variant is unreadable.

**Piotroski scales ROA by *beginning*-of-year assets**, per the paper. Using
the closing balance is a common shortcut that shifts every ROA and can flip the
improvement signal; a test pins 10.5% (100,000 over the prior 950,000) against
the 10.0% the shortcut would give.

### Testing scores without a spreadsheet

Two fixtures make the golden values hand-checkable:

* A **flat** company, identical in both years, collapses every Beneish index
  to exactly 1.0 — reducing the M-Score to
  `-4.84 + 2.360 + 4.679 x TATA = -2.760740` and pinning all eight published
  coefficients in a single assertion.
* An **improving** company, built line by line so each of the nine Piotroski
  signals has a definite direction. Scaling one company uniformly was tried
  first and does not work: multiplying every line by one factor leaves every
  *ratio* unchanged, so five of the nine signals have no direction at all and
  "improving on every dimension" is not what the fixture expresses.

---

## The ML forecaster, and the four ways it could have lied

`src/analytics/features.py` and `src/analytics/forecast.py` answer the question
users ask most and most tools answer dishonestly: *what will this be worth in a
month?* The architecture is shaped almost entirely around the ways that
question gets answered wrongly, because every one of them produces a model that
**looks excellent** and is worthless.

### 1. Lookahead in the features

A feature that peeks at a future bar — a centred rolling window, a `bfill`, a
scaler fitted on the whole sample — inflates accuracy and is invisible
downstream. Three structural choices prevent it:

* every feature at row *t* is a function of observations up to and including
  *t*, with no exceptions and no centred windows;
* standardisation happens **inside each training fold** (`StandardScaler` in a
  `Pipeline`), never in `features.py`, so no fold ever sees a mean computed
  from its own test block;
* the label is returned as a **separate object** from the feature matrix, so a
  careless `X = frame.drop(...)` cannot leave it behind.

The guard is a test rather than a convention:
`test_truncating_the_history_does_not_change_the_last_feature_row` computes the
matrix on `prices[:k]` and on the full series and demands the boundary row
match to 1e-12. Any lookahead breaks it immediately.

### 2. Leakage across the train/test boundary

This is the subtle one, and it is specific to multi-day horizons. The label at
row *t* is the return from *t* to *t+h*. Train right up to the test block and
the last *h − 1* training labels are built from the very returns the test block
is about to be scored on. For a 63-session horizon that is three months of the
answer inside the training set.

`make_folds` therefore **purges** `h − 1 + purge_extra` rows between every
training block and the test block that follows it. The folds are sequential and
expanding — never shuffled — because a shuffled split lets a model train on
Thursday to predict Wednesday, which is how a published stock predictor reports
95% accuracy.

### 3. A benchmark that is too easy to beat

Reporting mean absolute error alone makes every model look good, because daily
log returns are small numbers. Every skill figure in this application is
therefore quoted against a **random walk** — a forecast of exactly zero:

```
skill = 1 − MAE_model / MAE_random-walk
```

Negative skill means the model was worse than assuming the price does not
move, and that is the *ordinary* result. The leaderboard races both baselines
alongside the fitted models for exactly that reason: when `random_walk` wins,
that is the finding, and a board that hid it would be hiding the answer.

Statistical significance is a separate question, so a **Diebold–Mariano** test
on the squared-error loss differential sits next to the skill number, with a
Newey–West variance at *h − 1* lags (overlapping horizons make the differential
serially correlated by construction) and the Harvey–Leybourne–Newbold
small-sample correction. The same HAC convention as the factor-regression
module, for the same reason.

### 4. An interval that came from an assumption

A Gaussian band around a point forecast is a statement about the model's
assumptions, not its performance. The band here is **split-conformal**: the
empirical quantile range of the model's own *out-of-sample* residuals, shifted
onto the point forecast. Three consequences, all deliberate:

* a bad model produces a wide band automatically — nobody has to decide to
  widen it;
* the quantiles are **signed**, not symmetric, so a left-skewed error law stays
  left-skewed instead of averaging the downside away;
* the band is **not centred** on the point forecast when the model has a
  measured bias, and the median residual is reported as exactly that.

The direction probability falls out of the same object: the share of
`point + residual` above zero. No Normal assumption, and it collapses toward
the base rate precisely when the model has nothing to say.

### Why direct multi-horizon, not recursive

The cone is built from one independently fitted model **per horizon**. A
recursive forecast feeds its own one-step prediction back as an input and
iterates, which compounds the error and — worse — requires inventing the
features it cannot derive from a single predicted close (volatility, RSI,
volume). Direct fitting costs more compute and answers the question that was
actually asked. The consequence is stated rather than hidden: the nodes are
independent fits, so the cone is not guaranteed to widen monotonically, and a
kink between two horizons is a real disagreement between two models.

### Why no deep learning

Not an oversight, and not a resource constraint. An LSTM or a transformer
carries millions of parameters; a daily bar history is a few thousand rows with
a signal-to-noise ratio near zero. That combination memorises its training set.
It would also add a dependency larger than the rest of this application put
together, and break the "offline, deterministic, seconds" property the test
suite depends on. The models here — ridge, elastic net, random forest,
histogram gradient boosting, and an inverse-error blend — are the ones the data
volume can actually support.

### How the suite proves the machinery works at all

Two tests, and they only mean something as a pair:

| Test | Input | Asserts |
|---|---|---|
| `test_the_pipeline_finds_a_signal_that_is_genuinely_there` | AR(1) returns, ρ = −0.35 | skill > 2%, out-of-sample R² > 2%, direction > 55%, DM *p* < 0.05 |
| `test_the_pipeline_reports_no_edge_on_a_pure_random_walk` | i.i.d. returns | skill < 5% |

Without the first, every "no skill" verdict could be a broken pipeline rather
than an efficient market. Without the second, a leak would go unnoticed,
because a leaking forecaster simply looks good. Together they pin the property
the feature exists to provide: **the number on the screen tracks whether there
is anything to find.**

### Where the portfolio-level forecast gets its interval

Expected return aggregates exactly — a portfolio's simple return *is* the
weighted mean of its holdings' simple returns, which is why each holding's log
forecast is converted to simple before weighting and back afterwards
(weighting log returns directly is wrong by Jensen's inequality, and it is a
common enough error to be worth naming).

The interval does not aggregate exactly, because the forecaster never estimated
a correlation between its own errors across holdings. Inventing one would be
the least defensible number on the page, so two bounds are returned instead:
**comonotonic** (every holding at its own bound simultaneously — perfect
correlation, the widest defensible band, the crash case) and **independent**
(half-widths added in quadrature — zero correlation, which no equity book has).
The truth is between them, the gap is visible, and the uncovered weight is
reported rather than treated as flat.

---

## The news pipeline, and why a 200 is not news

### The finding

MoneyControl's RSS endpoints — the source originally asked for — **still
answer HTTP 200 with well-formed XML, and every one of them is frozen.**
Measured on 3 September 2026:

| Feed | Newest item | Age |
|---|---|--:|
| `rss/latestnews.xml` | 23 Apr 2024 | 863 days |
| `rss/business.xml` | 23 Apr 2024 | 862 days |
| `rss/results.xml` | 23 Apr 2024 | 863 days |
| `rss/MCtopnews.xml` | 2016 | 3,620 days |

A build that checked only the status code would have shipped a "Latest news"
page presenting April-2024 headlines as today's market news. That is worse
than shipping no news page: a reader would act on it.

Three other Indian market feeds *were* live at age 0 — Economic Times,
Business Standard, LiveMint — so the feature is fully deliverable. It simply
could not be deliverable on faith.

### The guard, and why MoneyControl stays registered

`fetch_feed` measures the age of the newest item and **rejects** a feed past
`MAX_FEED_AGE_DAYS` (7), returning zero items plus the age. `stale` is a
distinct state from a failure: the feed answered correctly and its content is
merely too old, and conflating the two would lose the only diagnosis a reader
can act on.

MoneyControl is left in `FEEDS` on purpose. It is the guard's regression test
*in production*, and the feature will start working again on its own if the
publisher resumes. Nothing about the mechanism is MoneyControl-specific: any
feed can rot, and it does so silently.

### Two precision problems the live data exposed

Both were visible only by running the thing against real headlines, which is
why it was run before being written up.

**The lexicon called "bonds pare gains" positive.** The first version scored
the nouns every market headline contains — "profit", "record", "high",
"order", "stake", "buy" — as directional. That classified 64% of a live feed
positive, and rated *"profit falls 40% on weak demand"* merely **mixed**,
because the noun cancelled the verb. Pruning to verbs and adjectives, and
adding a `DAMPENERS` set ("pare", "trim", "erase", "tames") treated as
negators of the term they precede, fixed both:

| Headline | Before | After |
|---|---|---|
| bonds **pare** gains … **tames** boost | positive | neutral |
| **profit falls** 40% on **weak** demand | mixed | negative |
| **No fall** in profit | negative | neutral |

The classified share fell from 78% to 65%, which is the honest direction:
"Sensex ends flat" *should* come back unclassified.

**"Tata" tagged every Tata headline to both TCS and Tata Power.** Matching on
the leading distinctive word of a company name is fine for "Infosys" and
useless for a business house — and India has many. The rule is now
*specificity computed from the records supplied*: a leading word shared by
more than one tracked company identifies neither, so the two-word form is
required. Live mentions went from 7 to 3, and all three are correct.

That fix then exposed a second bug: filtering generic words *before* taking
the pair left "Tata Power Company Limited" with **no name key at all**, since
every word after "Tata" is in `GENERIC_NAME_WORDS`. A major company had become
unmatchable. The two-word key is now built from the raw name — `"tata power"`,
which is both specific and how headlines actually write it.

### What the module refuses to say

* **Tone is a reading of the words.** Every result carries the matched terms
  and the UI shows them, because a label nobody can interrogate is a random
  number in a confident font. It has no idea what was already priced in.
* **`neutral` and `mixed` are different answers.** Nothing matched, versus
  terms on both sides cancelling. Collapsing them would overstate how much of
  the feed was actually read.
* **A match found only in the article body is labelled `(in body)`.**
  Otherwise a tag appears beside a headline that visibly does not mention the
  company, which reads as a bug in the matcher rather than as a feature of the
  source.
* **No impact is predicted.** The calendar says which releases have
  historically moved markets and when they land — not what will happen, and
  not in which direction. Measuring what news *did* to a price is an event
  study on realised returns, a separate calculation.
* **The calendar covers one week**, because `ff_calendar_thisweek.json` is the
  only horizon published. `CalendarResult.window` prints what was actually
  covered rather than implying a month of visibility.

### A pattern that recurred three times

`quality_summary`, `horizon_availability` and `summarise` were each written
returning `dict[str, object]`, and each cost the caller its types — the UI
could not do arithmetic on a value without the checker objecting, correctly.
All three are now frozen dataclasses (`QualityReport`, `HorizonCheck`,
`FeedSummary`). The lesson is small and general: a result with named fields
should be a type, not a dictionary, the first time.

---

## Error philosophy

Market data is unreliable in specific, predictable ways: symbols get renamed,
Yahoo rate-limits, a stock listed in 2021 has no 2008 history, an ETF reports no
sector, a newly-listed name shortens the overlapping window for everyone. The
codebase treats each of those as a normal case, not an exception.

- **Batch fetches return `(results, errors)`**, never raise. One dead ticker
  degrades to a warning; the other holdings still analyse.
- **Undefined statistics return `NaN`**, and the UI renders an em dash. A Sharpe
  ratio at zero volatility is *undefined*, not zero — printing `0.00` there
  would be a lie, and lying about a number is worse than omitting it.
- **Every divisor is guarded**: zero variance, zero investment, empty series,
  single observation, all-NaN column.
- **Coverage is reported, not assumed.** The stress page states what fraction of
  portfolio weight actually had data in each historical window rather than
  quietly treating missing holdings as flat.
- **`app.py` has a last-resort try/except** so a genuinely unexpected failure
  shows a message and a collapsible traceback instead of a blank page.

---

## Testing strategy

The suite runs offline, deterministically, in seconds.

| Layer | Approach |
|---|---|
| `analytics` | Hand-computed golden values and mathematical identities |
| `data/fetcher` | `yfinance` monkeypatched with a fake `Ticker` |
| `data/store` | Round-trip against a `tmp_path` SQLite file |
| `charts` | Every builder returns a `go.Figure` that survives `to_json()` |
| `ui` | Streamlit `AppTest` drives the real app with a stubbed data layer |
| `reporting/schedule` | Injected clocks; idempotence asserted by simulating a cron |
| `alerts` | Injected clocks and hand-built quotes; fire-once asserted by simulating a poll loop |
| `reporting/email_report` | A recording fake replaces `smtplib`; no socket is opened |
| `reporting/cli` | Driven through `main(argv)`, asserted on its exit code |
| `data/exchange` | `build_opener` replaced by a fake serving canned bytes |
| `universe_build` | Fixtures shaped like the real downloads; no socket, no file |
| `data/news` | `urlopen` replaced by a router; one feed served deliberately stale |
| `analytics/news` | Hand-written headlines with known tone and known mentions |

Identities are more valuable than fixtures because they cannot rot:

- beta of a series against itself is exactly `1.0`
- portfolio weights sum to `100`
- component risk contributions sum to portfolio volatility
- return contributions sum to the portfolio's total return
- the max-Sharpe portfolio's Sharpe ≥ equal-weight's
- min-variance volatility ≤ every point on the efficient frontier
- `CVaR ≤ VaR` at every confidence level
- both benchmark growth series equal exactly `100.0` at the window start
- 96 runs of the send command produce exactly one e-mail, at any cron cadence
- 96 alert checks on a price parked above its level produce exactly one e-mail
- a monthly "31st" schedule fires in all twelve months, 29 February included

`tests/conftest.py` builds correlated random walks rather than independent ones,
because uncorrelated toy data hides precisely the bugs a correlation, beta or
optimiser test exists to catch.

---

## Extension points

| To add… | Change only… |
|---|---|
| A real-time broker feed | `src/data/fetcher.py` (same signatures) |
| A new risk metric | `src/analytics/risk.py` + a test + one KPI card |
| A new benchmark | one line in `config.DEFAULT_BENCHMARKS` |
| A new optimisation objective | one function in `src/analytics/optimize.py` |
| A new stress scenario | one `StressWindow` in `config.HISTORICAL_STRESS` |
| Re-skin the whole app | `src/ui/theme.py` tokens + `src/charts/theme.py` palette |
| A new page | one module in `src/ui/pages/` + one `st.Page` entry |

---

## Known trade-offs

These are deliberate, and the app states them where they matter:

- **Historical portfolio value assumes today's share counts across the whole
  window** unless a transaction ledger is supplied. That answers "how would my
  current portfolio have behaved?", which is the standard basis for volatility,
  drawdown and beta — but it is not the same as the return you actually earned.
  Supply transactions and the app computes XIRR and realised P&L properly.
- **Risk statistics use the overlapping history of all holdings.** One recently
  listed name shortens the window for everyone; the app warns below 30 days.
- **The optimiser is single-period mean-variance.** Expected returns estimated
  from historical means are famously noisy, which is why the covariance is
  shrunk, weights are capped, and the page says so out loud.
- **Beta-implied stress is a first-order CAPM approximation.** It ignores
  idiosyncratic risk and the empirical fact that betas converge toward 1 in a
  crisis.
- **`yfinance` is not a real-time feed.** NSE/BSE quotes are typically delayed
  ~15 minutes and outside market hours the best available value is the last
  close. Every price carries its own timestamp and freshness label.
