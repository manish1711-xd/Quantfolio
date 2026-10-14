"""Central configuration: constants, market metadata and tunable thresholds.

Everything that is a *judgement call* (annualisation convention, risk-score
weights, insight thresholds, default risk-free rates) lives here rather than
being scattered through the analytics code. That makes the assumptions
auditable in one place -- which matters, because a portfolio analytics tool is
only as trustworthy as its stated conventions.

No imports from the rest of the package: this module is the bottom of the
dependency graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------
APP_NAME = "QuantFolio"
APP_TAGLINE = "Portfolio Risk, Performance & Optimisation Terminal"
APP_VERSION = "3.0.1"

DISCLAIMER = (
    "For educational and analytical purposes only. Nothing here is investment "
    "advice, a recommendation, or a solicitation to trade."
)

DATA_SOURCE = (
    "Yahoo Finance via yfinance - delayed / latest-available data, not guaranteed real-time"
)

# ---------------------------------------------------------------------------
# annualisation & statistical conventions
# ---------------------------------------------------------------------------
#: Equity trading days per year. NSE averages ~250 and NYSE ~252; the standard
#: convention is 252 and the difference is immaterial at this precision.
TRADING_DAYS: int = 252

#: Below this many overlapping daily observations risk statistics are flagged
#: as unreliable in the UI (they are still computed and shown).
MIN_OBSERVATIONS: int = 30

#: Minimum observations before a rolling-window statistic is emitted at all.
MIN_ROLLING_WINDOW: int = 20

#: Default confidence level for Value-at-Risk / Expected Shortfall.
DEFAULT_VAR_CONFIDENCE: float = 0.95

#: Minimum acceptable return used by the Omega ratio (annualised decimal).
DEFAULT_MAR: float = 0.0

# ---------------------------------------------------------------------------
# history windows
# ---------------------------------------------------------------------------
#: History windows offered in the UI (yfinance period strings).
VALID_PERIODS: tuple[str, ...] = ("1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max")

PERIOD_LABELS: dict[str, str] = {
    "1mo": "1 month",
    "3mo": "3 months",
    "6mo": "6 months",
    "1y": "1 year",
    "2y": "2 years",
    "5y": "5 years",
    "10y": "10 years",
    "max": "Max available",
}

#: Approximate trading days in each window -- used to warn when a window is too
#: short for the statistic the user asked for.
PERIOD_TRADING_DAYS: dict[str, int] = {
    "1mo": 21,
    "3mo": 63,
    "6mo": 126,
    "1y": 252,
    "2y": 504,
    "5y": 1260,
    "10y": 2520,
    "max": 5040,
}

DEFAULT_PERIOD = "1y"

# ---------------------------------------------------------------------------
# benchmarks
# ---------------------------------------------------------------------------
#: Display name -> Yahoo Finance symbol.
DEFAULT_BENCHMARKS: dict[str, str] = {
    "NIFTY 50": "^NSEI",
    "NIFTY Midcap 100": "^NSEMDCP50",
    "NIFTY Bank": "^NSEBANK",
    "SENSEX": "^BSESN",
    "S&P 500": "^GSPC",
    "NASDAQ 100": "^NDX",
    "NASDAQ Composite": "^IXIC",
    "Dow Jones": "^DJI",
    "FTSE 100": "^FTSE",
    "Nikkei 225": "^N225",
    "MSCI World (URTH)": "URTH",
    "Gold (GLD)": "GLD",
}

DEFAULT_BENCHMARK = "NIFTY 50"

#: Benchmark symbol -> the currency its level is quoted in (used for FX notes).
BENCHMARK_CURRENCY: dict[str, str] = {
    "^NSEI": "INR",
    "^NSEMDCP50": "INR",
    "^NSEBANK": "INR",
    "^BSESN": "INR",
    "^GSPC": "USD",
    "^NDX": "USD",
    "^IXIC": "USD",
    "^DJI": "USD",
    "^FTSE": "GBP",
    "^N225": "JPY",
    "URTH": "USD",
    "GLD": "USD",
}

# ---------------------------------------------------------------------------
# currencies & FX
# ---------------------------------------------------------------------------
CURRENCY_SYMBOLS: dict[str, str] = {
    "INR": "₹",
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "AUD": "A$",
    "CAD": "C$",
    "CHF": "CHF ",
    "SGD": "S$",
    "HKD": "HK$",
    "CNY": "CN¥",
    "AED": "AED ",
}

SUPPORTED_BASE_CURRENCIES: tuple[str, ...] = ("INR", "USD", "EUR", "GBP")

DEFAULT_BASE_CURRENCY = "INR"

#: Illustrative annual risk-free rates (decimals) per currency. These are
#: starting points for the sidebar input, NOT live yields -- the user is told
#: to replace them with the current sovereign yield for their market.
DEFAULT_RISK_FREE: dict[str, float] = {
    "INR": 0.065,  # ~10y G-Sec ballpark
    "USD": 0.043,  # ~10y UST ballpark
    "EUR": 0.025,  # ~10y Bund ballpark
    "GBP": 0.040,  # ~10y Gilt ballpark
    "JPY": 0.010,
}

#: Currencies for which Yahoo quotes the pair as CCYUSD=X (USD is the quote
#: currency) rather than USDCCY=X. EURUSD=X and GBPUSD=X are the classic cases.
USD_QUOTED_PAIRS: frozenset[str] = frozenset({"EUR", "GBP", "AUD", "NZD"})

# ---------------------------------------------------------------------------
# Indian number formatting
# ---------------------------------------------------------------------------
#: Currencies formatted with the Indian lakh/crore grouping by default.
LAKH_CRORE_CURRENCIES: frozenset[str] = frozenset({"INR"})


# ---------------------------------------------------------------------------
# risk score rubric (transparent, rule-based, NOT a regulatory standard)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RiskScoreRubric:
    """Weights and saturation points of the 0-100 portfolio risk score."""

    vol_points: float = 30.0
    vol_cap: float = 0.40  # >= 40% annualised vol scores full points
    drawdown_points: float = 25.0
    drawdown_cap: float = 0.50  # >= 50% peak-to-trough scores full points
    concentration_points: float = 18.0
    holdings_points: float = 12.0
    holdings_floor: int = 10  # >= 10 names scores zero points
    tail_points: float = 15.0
    tail_cap: float = 0.05  # daily 95% CVaR >= 5% scores full points
    low_band: float = 35.0
    high_band: float = 65.0

    @property
    def total_points(self) -> float:
        return (
            self.vol_points
            + self.drawdown_points
            + self.concentration_points
            + self.holdings_points
            + self.tail_points
        )


RISK_RUBRIC = RiskScoreRubric()


# ---------------------------------------------------------------------------
# insight thresholds (every generated sentence traces back to one of these)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class InsightThresholds:
    concentration_warn_pct: float = 40.0
    concentration_info_pct: float = 25.0
    sector_concentration_pct: float = 45.0
    min_diversified_holdings: int = 5
    high_correlation: float = 0.80
    deep_drawdown: float = -0.20
    severe_drawdown: float = -0.35
    vol_tolerance: float = 0.10
    aggressive_beta: float = 1.20
    defensive_beta: float = 0.80
    good_sharpe: float = 1.00
    good_sortino: float = 1.50
    weak_calmar: float = 0.50
    high_var_pct: float = 3.0  # daily 95% VaR worse than -3%
    fat_tail_kurtosis: float = 3.0  # excess kurtosis above this = fat tails
    long_recovery_days: int = 180
    effective_holdings_ratio: float = 0.50  # ENB / N below this = lopsided


INSIGHTS = InsightThresholds()


# ---------------------------------------------------------------------------
# optimiser defaults
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OptimizerConfig:
    frontier_points: int = 60
    max_weight: float = 0.35  # long-only cap per name, UI-adjustable
    min_weight: float = 0.0
    risk_parity_iterations: int = 2000
    risk_parity_tolerance: float = 1e-10
    #: Ledoit-Wolf style shrinkage intensity applied to the sample covariance
    #: matrix. Sample covariance is notoriously unstable when the number of
    #: assets approaches the number of observations; shrinking toward a scaled
    #: identity stabilises the optimiser.
    shrinkage: float = 0.10


OPTIMIZER = OptimizerConfig()


# ---------------------------------------------------------------------------
# Monte Carlo defaults
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SimulationConfig:
    n_paths: int = 2000
    horizon_days: int = 252
    seed: int = 42
    block_size: int = 10  # for block bootstrap
    percentiles: tuple[int, ...] = (5, 25, 50, 75, 95)


SIMULATION = SimulationConfig()


# ---------------------------------------------------------------------------
# historical stress windows (peak -> trough of well-documented drawdowns)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StressWindow:
    name: str
    start: str
    end: str
    note: str


HISTORICAL_STRESS: tuple[StressWindow, ...] = (
    StressWindow(
        "COVID-19 crash",
        "2020-02-19",
        "2020-03-23",
        "Global pandemic sell-off; NIFTY 50 fell ~38% in five weeks.",
    ),
    StressWindow(
        "2022 rate shock",
        "2022-01-03",
        "2022-06-17",
        "Inflation surprise and the fastest Fed hiking cycle since 1994.",
    ),
    StressWindow(
        "Global financial crisis",
        "2008-01-01",
        "2009-03-09",
        "Credit crisis; broad equity drawdowns beyond 50%.",
    ),
    StressWindow(
        "Taper tantrum",
        "2013-05-22",
        "2013-08-28",
        "Fed taper signal; heavy pressure on emerging-market assets and the rupee.",
    ),
    StressWindow(
        "Demonetisation",
        "2016-11-08",
        "2016-12-26",
        "India-specific liquidity shock to consumption and small caps.",
    ),
    StressWindow(
        "Volmageddon",
        "2018-01-26",
        "2018-02-08",
        "Short-volatility unwind; a sharp two-week equity drawdown.",
    ),
    StressWindow(
        "China devaluation",
        "2015-08-10",
        "2015-09-29",
        "CNY devaluation and a global growth scare.",
    ),
)

#: Instantaneous benchmark shocks translated through portfolio beta.
FACTOR_SHOCKS: tuple[tuple[str, float], ...] = (
    ("Benchmark -5%", -0.05),
    ("Benchmark -10%", -0.10),
    ("Benchmark -20%", -0.20),
    ("Benchmark -30%", -0.30),
    ("Benchmark +10%", 0.10),
)

# ---------------------------------------------------------------------------
# caching
# ---------------------------------------------------------------------------
QUOTE_TTL_SECONDS: int = 300  # 5 minutes for latest prices
HISTORY_TTL_SECONDS: int = 1800  # 30 minutes for daily history
PROFILE_TTL_SECONDS: int = 86400  # 24 hours for slow-moving fundamentals
#: Symbol search. "What is Reliance's ticker" is close to immutable, so this
#: is cached aggressively - the online lookup is a fallback for the long tail
#: the bundled stock-master files do not cover, not a hot path.
SEARCH_TTL_SECONDS: int = 3600  # 1 hour


# ---------------------------------------------------------------------------
# stock search
# ---------------------------------------------------------------------------
#: Rows rendered by the stock-search result list before the user must refine.
MAX_SEARCH_RESULTS: int = 8

#: Exchange preferred when a bare symbol matches on more than one venue.
#: "INFY" is both the NYSE ADR and the root of NSE's "INFY.NS"; an India-first
#: tool should lean NSE, while still SHOWING the ADR so the user can choose.
PREFERRED_EXCHANGE = "NSE"


# ---------------------------------------------------------------------------
# technical indicator defaults
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TechnicalConfig:
    sma_windows: tuple[int, ...] = (20, 50, 200)
    ema_windows: tuple[int, ...] = (12, 26)
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    atr_period: int = 14


TECHNICALS = TechnicalConfig()


# ---------------------------------------------------------------------------
# machine-learning forecaster: feature construction
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureConfig:
    """Windows used to turn a price history into a supervised learning matrix.

    Every window here is a LOOKBACK: the feature at row ``t`` is computed from
    observations up to and including ``t`` and never beyond it. That is the
    property the whole forecaster rests on, and it is enforced by construction
    in :mod:`src.analytics.features` rather than assumed.

    The windows are deliberately scale-free (returns, ratios, z-scores,
    positions inside a range) so a model trained on a 200-rupee stock transfers
    its coefficients meaningfully to a 4,000-rupee one -- and so that nothing
    the model sees carries the price level, which is the fastest way to fit a
    trend that does not generalise.
    """

    #: Individual lagged daily log returns (short-horizon autocorrelation).
    return_lags: tuple[int, ...] = (1, 2, 3, 5, 10)
    #: Cumulative log return over each window (momentum).
    momentum_windows: tuple[int, ...] = (10, 21, 63, 126, 252)
    #: Realised volatility windows (standard deviation of daily log returns).
    volatility_windows: tuple[int, ...] = (5, 21, 63)
    #: Moving averages the close is measured against, as a distance ratio.
    ma_windows: tuple[int, ...] = (10, 20, 50, 200)
    #: Windows for the higher-moment (skew / kurtosis) features.
    moment_window: int = 63
    #: Volume averaging windows for the z-score and the short/long ratio.
    volume_windows: tuple[int, ...] = (5, 21)
    #: Window defining the "52-week" range position feature.
    range_window: int = 252
    #: Short and long window of the volatility-regime ratio.
    vol_regime_windows: tuple[int, int] = (5, 63)


FEATURES = FeatureConfig()


# ---------------------------------------------------------------------------
# machine-learning forecaster: models, validation and intervals
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ForecastConfig:
    """Defaults for the supervised forward-return models.

    Three numbers here decide whether the output is science or decoration:

    ``n_splits`` and ``test_fraction``
        How much of the history is held out, and in how many *sequential*
        blocks. Every estimate the UI prints -- error, directional accuracy,
        skill against a random walk, the prediction interval -- comes from
        these out-of-sample blocks, never from the fitted training residuals.

    ``purge_extra``
        Rows dropped between a training block and the test block that follows
        it. A target at horizon ``h`` spans ``h`` future days, so the last
        ``h - 1`` training labels overlap the test window. Keeping them leaks
        the answer, and it is the single most common way a published stock
        forecaster reports accuracy it does not have.

    ``conformal_alpha``
        Miscoverage rate of the prediction interval. 0.20 gives an 80% band
        whose width is read off the out-of-sample residual quantiles, so it
        widens when the model is bad instead of pretending otherwise.
    """

    #: Forecast horizons offered, in trading sessions.
    horizons: tuple[int, ...] = (1, 5, 10, 21, 63)
    default_horizon: int = 21

    #: Minimum rows in the supervised matrix before anything is fitted at all.
    min_train_rows: int = 250
    #: Minimum rows in the FIRST training block of the walk-forward.
    min_initial_train: int = 200

    #: Sequential out-of-sample blocks in the walk-forward backtest.
    n_splits: int = 5
    #: Fraction of the supervised matrix reserved for those blocks.
    test_fraction: float = 0.30
    #: Extra rows purged either side of the horizon overlap, for safety.
    purge_extra: int = 1

    #: Miscoverage of the conformal prediction interval (0.20 -> 80% band).
    conformal_alpha: float = 0.20

    #: Seed for every estimator that draws random numbers.
    random_state: int = 42
    #: Permutation-importance repeats. More is smoother and linearly slower.
    importance_repeats: int = 5

    #: Ridge / ElasticNet regularisation grid, searched inside each fold.
    ridge_alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)
    elastic_l1_ratios: tuple[float, ...] = (0.1, 0.5, 0.9)
    #: Explicit elastic-net alpha grid. Stated rather than left to the
    #: estimator's automatic path because that path reaches alphas near zero,
    #: where coordinate descent on 40 collinear price features does not
    #: converge in any reasonable iteration budget. The floor is the real
    #: regularisation decision and it belongs here, in the open.
    elastic_alphas: tuple[float, ...] = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)

    #: Tree-ensemble shape. Shallow and heavily leaf-regularised on purpose:
    #: a daily bar series carries a few thousand rows and a signal-to-noise
    #: ratio near zero, so an unconstrained forest memorises the sample.
    n_estimators: int = 300
    max_depth: int = 5
    min_samples_leaf: int = 25
    max_features: float = 0.5
    learning_rate: float = 0.05

    @property
    def confidence(self) -> float:
        """Nominal coverage of the prediction interval, as a decimal."""
        return 1.0 - self.conformal_alpha


FORECAST = ForecastConfig()

#: Skill (1 - MASE against a random walk) below which the UI refuses to present
#: a forecast as anything but a failed experiment. Zero is the honest cut: a
#: model that cannot beat "tomorrow's price is today's price" has no business
#: drawing a cone on a chart.
FORECAST_MIN_SKILL: float = 0.0

#: Directional accuracy above which a model is called better than a coin flip
#: in the UI copy. Not a significance test -- the Diebold-Mariano p-value that
#: sits next to it is.
FORECAST_DIRECTION_EDGE: float = 0.55

FORECAST_DISCLAIMER = (
    "A forecast is a conditional statistical estimate, not a prediction of what will "
    "happen. Every model here is fitted to this instrument's own past prices, measured "
    "out-of-sample against a random walk, and reported with the interval its own errors "
    "imply. Markets are close to unforecastable at these horizons; treat a narrow "
    "interval as a warning sign, not a result."
)


# ---------------------------------------------------------------------------
# factor model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FactorSpec:
    """One factor built as a long/short spread of two tradeable instruments.

    ``short_symbol=None`` means the series is an EXCESS return over the
    risk-free rate rather than a spread — the convention for the market factor.

    These are PROXIES. The real Fama-French factors are built from the whole
    cross-section of stocks sorted on size and book-to-market, rebalanced
    annually, and are not investable. An ETF spread is investable, correlated
    with the academic factor, and measurably different from it: the gap
    includes fund fees, the index provider's own screening rules, and the
    ETF's inception date. ``caveat`` records the specific limitation so the UI
    can state it per factor instead of one blanket disclaimer.
    """

    name: str
    label: str
    long_symbol: str
    short_symbol: str | None = None
    description: str = ""
    caveat: str = ""

    @property
    def is_excess_return(self) -> bool:
        return self.short_symbol is None

    @property
    def formula(self) -> str:
        if self.short_symbol is None:
            return f"{self.long_symbol} − risk-free"
        return f"{self.long_symbol} − {self.short_symbol}"

    @property
    def symbols(self) -> tuple[str, ...]:
        return (
            (self.long_symbol,)
            if self.short_symbol is None
            else (
                self.long_symbol,
                self.short_symbol,
            )
        )


#: ETF-based factor proxies per market. US coverage is good; Indian coverage is
#: genuinely poor and the entry says so rather than padding the list with
#: thinly-traded ETFs whose short histories would silently shorten the
#: regression window for everything else.
FACTOR_PROXIES: dict[str, tuple[FactorSpec, ...]] = {
    "US": (
        FactorSpec(
            "MKT",
            "Market",
            "VTI",
            None,
            "Total US market return in excess of the risk-free rate.",
        ),
        FactorSpec(
            "SMB",
            "Size (small minus big)",
            "IWM",
            "VTI",
            "Russell 2000 minus the total market: a small-cap tilt.",
            "IWM is small-cap BLEND, so this mixes size with the sector "
            "composition of the Russell 2000.",
        ),
        FactorSpec(
            "HML",
            "Value (high minus low)",
            "VTV",
            "VUG",
            "Large-cap value minus large-cap growth.",
            "Index-provider value/growth screens are not book-to-market "
            "sorts, so this is a softer value signal than academic HML.",
        ),
        FactorSpec(
            "WML",
            "Momentum (winners minus losers)",
            "MTUM",
            "VTI",
            "US momentum factor ETF minus the market.",
            "A long-only momentum ETF minus the market is not the "
            "long/short winners-minus-losers portfolio; it omits the short leg.",
        ),
        FactorSpec(
            "QMJ",
            "Quality (quality minus market)",
            "QUAL",
            "VTI",
            "US quality factor ETF minus the market.",
            "Long-only, and 'quality' has no single agreed definition.",
        ),
        FactorSpec(
            "LOWVOL",
            "Low volatility",
            "USMV",
            "VTI",
            "Minimum-volatility ETF minus the market.",
            "Related to, but not the same as, the betting-against-beta factor.",
        ),
    ),
    "India": (
        FactorSpec(
            "MKT",
            "Market",
            "NIFTYBEES.NS",
            None,
            "NIFTY 50 return in excess of the risk-free rate.",
        ),
        FactorSpec(
            "SMB",
            "Size (small minus big)",
            "JUNIORBEES.NS",
            "NIFTYBEES.NS",
            "NIFTY Next 50 minus NIFTY 50: a size tilt within large caps.",
            "Next 50 is still large/mid cap, so this is a much weaker size "
            "signal than an academic SMB built on true small caps.",
        ),
    ),
}

#: Markets with no usable proxy for these factors, and what to do instead.
#: Stated explicitly because an empty column in a factor table looks like a
#: bug, whereas "this data does not exist here" is a fact the user should know.
FACTOR_PROXY_GAPS: dict[str, str] = {
    "India": (
        "Value, momentum and quality proxies are NOT offered for India: the NSE "
        "factor ETFs are recent and thinly traded, so including them would "
        "shorten the regression window for every other factor and add more "
        "tracking noise than signal. For rigorous Indian factor work, load the "
        "IIM Ahmedabad Fama-French dataset as a factor file instead."
    ),
}

#: Rolling window (trading days) for the factor-loading drift chart.
FACTOR_ROLLING_WINDOW: int = 126


# ---------------------------------------------------------------------------
# Indian capital-gains tax (listed equity and equity mutual funds, STT paid)
# ---------------------------------------------------------------------------
#: Everything below is a SIMPLIFIED model of Sections 111A, 112A, 70, 71 and 74
#: of the Income-tax Act 1961, sufficient for the harvesting arithmetic this
#: application does and NOT a substitute for professional advice. It ignores
#: surcharge, cess, the resident/non-resident distinction, set-off against
#: other heads, and every asset class other than STT-paid listed equity.
@dataclass(frozen=True)
class IndianEquityTaxRegime:
    """The rates and thresholds in force from a given date.

    India changed these mid-year on 23 July 2024 (Finance (No. 2) Act 2024),
    which is why this is a dated list rather than four constants: a sale on
    22 July 2024 and one on 24 July 2024 are taxed differently, and a
    harvesting tool that used one rate for both would be wrong for anyone
    reviewing FY2024-25.
    """

    label: str
    effective_from: date
    #: Section 111A short-term rate on STT-paid listed equity.
    stcg_rate: float
    #: Section 112A long-term rate.
    ltcg_rate: float
    #: Section 112A annual exemption on long-term gains, in rupees.
    ltcg_exemption: float
    #: Days held BEYOND which listed equity is long-term (i.e. > 12 months).
    long_term_days: int = 365
    statute: str = ""

    @property
    def stcg_pct(self) -> float:
        return self.stcg_rate * 100.0

    @property
    def ltcg_pct(self) -> float:
        return self.ltcg_rate * 100.0


#: In ascending date order. ``tax.regime_for`` picks the one in force.
INDIAN_TAX_REGIMES: tuple[IndianEquityTaxRegime, ...] = (
    IndianEquityTaxRegime(
        label="Pre-23 July 2024",
        effective_from=date(2018, 4, 1),
        stcg_rate=0.15,
        ltcg_rate=0.10,
        ltcg_exemption=100_000.0,
        statute="Sections 111A / 112A as they stood before Finance (No. 2) Act 2024",
    ),
    IndianEquityTaxRegime(
        label="From 23 July 2024",
        effective_from=date(2024, 7, 23),
        stcg_rate=0.20,
        ltcg_rate=0.125,
        ltcg_exemption=125_000.0,
        statute="Sections 111A / 112A as amended by Finance (No. 2) Act 2024",
    ),
)

#: Grandfathering date under Section 112A: for equity ACQUIRED on or before
#: this date, the cost of acquisition is the higher of actual cost and the
#: lower of (31-Jan-2018 fair market value, sale consideration). The FMV is
#: the HIGHEST QUOTED PRICE on that date, which this application does not
#: hold -- see ``src.analytics.tax`` for how that is reported rather than
#: guessed.
GRANDFATHER_ACQUISITION_CUTOFF: date = date(2018, 1, 31)

#: Indian financial year starts 1 April.
FINANCIAL_YEAR_START_MONTH: int = 4

#: Assessment years a capital loss may be carried forward (Section 74), and
#: only if the return was filed by the due date under Section 139(1).
LOSS_CARRY_FORWARD_YEARS: int = 8

#: Within this many days of the long-term boundary, the UI flags a lot as
#: time-sensitive: the tax character of the gain or loss is about to change.
TAX_BOUNDARY_WARNING_DAYS: int = 45

TAX_DISCLAIMER = (
    "Educational tax arithmetic, not tax advice. It models STT-paid listed equity "
    "under a simplified reading of Sections 111A, 112A and 70-74, and ignores "
    "surcharge, cess, residency status, set-off against other income heads and every "
    "other asset class. Verify any action with a qualified chartered accountant "
    "before you trade on it."
)


# ---------------------------------------------------------------------------
# rebalancing
# ---------------------------------------------------------------------------
DEFAULT_REBALANCE_BAND_PCT: float = 5.0  # absolute weight drift tolerated

# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------
DB_FILENAME = "quantfolio.db"

# ---------------------------------------------------------------------------
# sample tickers offered as quick-add chips in the UI
# ---------------------------------------------------------------------------
TICKER_SUGGESTIONS: dict[str, tuple[str, ...]] = {
    "India - large cap": (
        "RELIANCE.NS",
        "TCS.NS",
        "HDFCBANK.NS",
        "INFY.NS",
        "ICICIBANK.NS",
        "BHARTIARTL.NS",
        "ITC.NS",
        "LT.NS",
        "SBIN.NS",
        "HINDUNILVR.NS",
    ),
    "India - ETFs": (
        "NIFTYBEES.NS",
        "JUNIORBEES.NS",
        "BANKBEES.NS",
        "GOLDBEES.NS",
    ),
    "US - mega cap": (
        "AAPL",
        "MSFT",
        "NVDA",
        "GOOGL",
        "AMZN",
        "META",
        "TSLA",
    ),
    "US - ETFs": ("SPY", "QQQ", "VTI", "AGG", "GLD", "TLT"),
}


@dataclass(frozen=True)
class AppConfig:
    """Runtime knobs the UI can override per session."""

    period: str = DEFAULT_PERIOD
    benchmark_name: str = DEFAULT_BENCHMARK
    base_currency: str = DEFAULT_BASE_CURRENCY
    risk_free_rate: float = DEFAULT_RISK_FREE[DEFAULT_BASE_CURRENCY]
    var_confidence: float = DEFAULT_VAR_CONFIDENCE
    convert_fx: bool = True
    rolling_window: int = 63  # ~one quarter
    optimizer: OptimizerConfig = field(default_factory=lambda: OPTIMIZER)
    simulation: SimulationConfig = field(default_factory=lambda: SIMULATION)
    forecast: ForecastConfig = field(default_factory=lambda: FORECAST)
    features: FeatureConfig = field(default_factory=lambda: FEATURES)

    @property
    def benchmark_symbol(self) -> str:
        return DEFAULT_BENCHMARKS.get(self.benchmark_name, "^NSEI")

    @property
    def currency_symbol(self) -> str:
        return CURRENCY_SYMBOLS.get(self.base_currency, "")
