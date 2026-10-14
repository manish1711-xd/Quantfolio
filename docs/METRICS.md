# Metrics reference

Every number the application can display, with its formula, its convention and
its failure mode. If a metric is undefined for the given input the app renders
an em dash (`—`) rather than a zero — a Sharpe ratio at zero volatility is
*undefined*, not `0.00`, and rounding that distinction away is how analytics
tools quietly mislead people.

**Annualisation convention:** 252 trading days (`config.TRADING_DAYS`). NSE
averages ~250 and NYSE ~252; 252 is the standard and the difference is
immaterial at this precision.

---

## 1. Position and portfolio economics

| Metric | Formula | Notes |
|---|---|---|
| Investment | `quantity × avg_buy_price × fx_rate` | FX applied only to money columns |
| Current value | `quantity × last_price × fx_rate` | prices stay in native currency |
| Unrealised P&L | `current_value − investment` | |
| Return % | `P&L / investment × 100` | guarded: `investment > 0` |
| Weight % | `value_i / Σ value × 100` | guarded: total `> 0` |
| Day change % | `last / previous_close − 1` | NaN when the provider omits prev close |
| HHI | `Σ wᵢ²`, w as fractions | `1/n` equal-weight … `1` single stock |
| Effective bets | `1 / HHI` | "how many equally-weighted positions is this really?" |
| Top-5 weight | `Σ` five largest weights | |
| Gini coefficient | `Σᵢ(2i − n − 1)wᵢ / (n Σwᵢ)` on sorted w | 0 = equal weight, →1 = one holding |

**Effective bets** is the concentration number worth internalising. A portfolio
of 20 names where one is 60% has an HHI near 0.38, so roughly **2.6 effective
bets** — it is a two-stock portfolio wearing twenty tickers.

---

## 2. Returns

| Metric | Formula | Answers |
|---|---|---|
| Daily return | `rₜ = Pₜ / Pₜ₋₁ − 1` | one-session move |
| Cumulative return | `V_end / V_start − 1` | total over the window |
| Annualised (geometric) | `(V_end / V_start)^(252/(n−1)) − 1` | CAGR-style growth rate |
| CAGR | `(V_end / V_start)^(365/days) − 1` | calendar-time growth (ACT/365) |
| Time-weighted (TWR) | chain-link `rₜ = (Vₜ − Fₜ)/Vₜ₋₁ − 1`, then annualise | **how the strategy performed** |
| Modified Dietz | `(EV − BV − F) / (BV + Σ wᵢFᵢ)` | money-weighted, single period |
| XIRR | solve `Σ CFᵢ / (1+r)^(dᵢ/365) = 0` | **what your money actually earned** |

### TWR vs XIRR — the distinction that matters

They answer different questions and they disagree whenever contributions were
not evenly spaced.

- **TWR** strips out the timing and size of deposits. It is how you judge a
  *manager* or a *strategy*, and it is what every fund factsheet reports.
- **XIRR** is money-weighted: it accounts for the fact that you had ₹50,000
  invested during the good year and ₹5,00,000 during the bad one. It is how you
  judge *your own outcome*.

A SIP into a market that fell and then recovered typically shows a decent TWR
and a *better* XIRR — because the later, larger instalments bought lower. The
reverse happens when you add money into a top. The app reports TWR on the
Performance page and XIRR on the Trades page, and says which is which.

`xirr` is solved with Brent bracketing over a widening interval, falling back to
Newton, with the root re-substituted into the NPV and rejected unless it
genuinely zeroes. It returns NaN — never a wrong number, never an exception —
when the flows contain no sign change or no root exists in `(−99.9%, 10000%)`.

---

## 3. Risk

| Metric | Formula | Reading |
|---|---|---|
| Annualised volatility | `std(r, ddof=1) × √252` | dispersion, yearly scale |
| Downside deviation | `√(mean(min(r − MAR, 0)²)) × √252` | only the bad half |
| Sharpe ratio | `(Rₐ − R_f) / σ` | excess return per unit of *total* risk |
| Sortino ratio | `(Rₐ − R_f) / σ_downside` | per unit of *harmful* risk |
| Calmar ratio | `Rₐ / |MDD|` | return per unit of worst loss |
| Omega ratio | `Σ max(r − MAR, 0) / Σ max(MAR − r, 0)` | full-distribution gain/loss odds |
| Ulcer index | `√(mean(drawdown%²))` | depth *and* duration of pain |
| Martin ratio | `(Rₐ − R_f) / Ulcer` | return per unit of sustained drawdown |
| Drawdown | `dₜ = Vₜ / max(V₀..Vₜ) − 1` | fall from the running peak |
| Max drawdown | `min(dₜ)` | worst peak-to-trough decline |
| Tail ratio | `|P95(r)| / |P5(r)|` | right tail vs left tail |
| Hit ratio | `count(r > 0) / count(r)` | share of positive days |
| Gain-to-pain | `Σr / Σ|min(r, 0)|` | total return per unit of loss endured |
| Skewness | third standardised moment | negative = crash-prone |
| Excess kurtosis | fourth standardised moment − 3 | > 0 = fatter tails than normal |

Sharpe treats upside and downside volatility as equally bad, which is wrong for
anyone who only minds losing money — hence Sortino. Calmar and Martin replace
volatility with drawdown, which is what investors actually experience: nobody
abandons a strategy because of its standard deviation, they abandon it 30% below
the peak.

### Value at Risk and Expected Shortfall

All VaR and CVaR figures are **negative daily decimals**: `−0.021` means a 2.1%
loss.

| Method | Formula | When it is right |
|---|---|---|
| Historical | empirical quantile at `1 − c` | makes no distributional assumption; limited by the worst day in the sample |
| Gaussian | `μ + z_α σ` | clean, but assumes normality and so understates tails |
| Cornish–Fisher | `z_cf = z + (z²−1)S/6 + (z³−3z)K/24 − (2z³−5z)S²/36` | corrects the normal quantile for observed skew `S` and excess kurtosis `K` |

**CVaR (expected shortfall)** is the mean of the returns at or beyond VaR — the
average size of a bad day *given* that it was bad. VaR tells you the threshold;
CVaR tells you what is behind it. `CVaR ≤ VaR` always, and the app asserts it.

A historical VaR materially worse than the Gaussian VaR is direct evidence of
fat left tails, which is exactly what equity returns have.

**Kupiec proportion-of-failures test** backtests the VaR itself:

```
LR = −2 ln[ (1−p)^(n−x) p^x ] + 2 ln[ (1−x/n)^(n−x) (x/n)^x ]     ~ χ²(1)
```

with `x` observed breaches in `n` days at expected rate `p = 1 − c`. Rejection
means the VaR model is miscalibrated for this portfolio — too many breaches
(understating risk) or suspiciously few (overstating it).

---

## 4. Drawdown episodes

A drawdown *episode* opens when value falls below the running peak, records its
trough, and closes only when a **new high** is reached. The app reports, per
episode: peak date, trough date, recovery date (`None` if still underwater),
depth, days peak→trough, and days trough→recovery.

Depth alone is half the story. A 20% fall that recovers in five weeks and a 20%
fall that takes three years are entirely different experiences, and only the
second one makes people sell at the bottom. That is why the app reports
**time underwater** and **average recovery time** alongside max drawdown, and
why the Ulcer index — which squares the drawdown at every point in time — is a
better single summary of pain than the maximum ever is.

---

## 5. Benchmark-relative

| Metric | Formula | Reading |
|---|---|---|
| Beta | `cov(r_p, r_b) / var(r_b)` | sensitivity to index moves |
| Jensen's alpha | `R_p − [R_f + β(R_b − R_f)]` | return beyond what beta explains |
| Tracking error | `std(r_p − r_b) × √252` | how far you stray from the index |
| Information ratio | `(R_p − R_b) / tracking_error` | active return per unit of active risk |
| Treynor ratio | `(R_p − R_f) / β` | excess return per unit of *systematic* risk |
| M² (Modigliani) | `R_f + Sharpe_p × σ_b` | your return, rescaled to index-level risk |
| R² | `corr(r_p, r_b)²` | share of your variance the index explains |
| Up capture | `mean(r_p | r_b > 0) / mean(r_b | r_b > 0)` | participation in rallies |
| Down capture | `mean(r_p | r_b < 0) / mean(r_b | r_b < 0)` | participation in falls |
| Capture ratio | `up / down` | **> 1 is the asymmetry you want** |

Capture ratios are computed on **monthly** compounded returns; daily capture is
dominated by noise.

An information ratio around 0.5 is respectable for an active strategy; a
*negative* one means the active risk you took was not merely unrewarded but
actively costly, and an index fund would have done better with less deviation.

Note that the comparison uses **price indices**, which exclude dividends. That
flatters the index slightly relative to a total-return portfolio.

---

## 6. Attribution

**Return contribution.** Each holding's contribution to the portfolio return is
`pnlᵢ / total_investment × 100`, so the column sums exactly to the portfolio's
total return %. This is *not* the holding's own return — a 200% gain on a 1%
position contributes 2 percentage points, not 200.

**Risk contribution.** With portfolio volatility `σ_p = √(wᵀΣw)`:

```
MCTRᵢ = (Σw)ᵢ / σ_p          marginal: dσ_p / dwᵢ
CCTRᵢ = wᵢ × MCTRᵢ           component: sums exactly to σ_p
risk share %ᵢ = CCTRᵢ / σ_p × 100     sums to 100
```

Both summation identities are asserted in the test suite, because they are what
make the numbers comparable at all.

This is the most actionable table in the application. Weight tells you where
your *money* is; risk contribution tells you where your *risk* is, and they are
routinely very different. A 10% position in a high-volatility, high-correlation
name can carry 30% of portfolio risk.

**Diversification ratio** `= (w · σ) / √(wᵀΣw)`. It compares the weighted
average of individual volatilities to the actual portfolio volatility. A value
of 1.0 means diversification is doing nothing for you; higher is better.

---

## 7. Optimisation

Single-period mean-variance, long-only, fully invested, with per-name caps.

| Objective | Optimises | Needs a return forecast? |
|---|---|---|
| Max Sharpe (tangency) | `max (μᵀw − R_f)/√(wᵀΣw)` | **yes** |
| Min variance | `min wᵀΣw` | no |
| Risk parity | `min Σ(RCᵢ − σ_p²/n)²` | no |
| Max diversification | `max (wᵀσ)/√(wᵀΣw)` | no |
| Efficient frontier | `min wᵀΣw` s.t. `μᵀw = target` | yes |

Solved with SLSQP (`scipy.optimize.minimize`) under `Σw = 1` and
`min_weight ≤ wᵢ ≤ max_weight`.

**Covariance shrinkage.** The sample covariance matrix is badly conditioned when
the number of assets approaches the number of observations, and the optimiser
will happily exploit that noise. The app shrinks toward a scaled identity:

```
Σ_shrunk = (1 − a)·Σ_sample + a·(trace(Σ)/n)·I
```

### Read this before trusting the frontier

Mean-variance optimisation is famously fragile. Expected returns estimated from
historical means carry enormous standard errors, and the optimiser piles into
whatever happened to perform best in-sample — small input changes swing the
weights dramatically. That is precisely why the covariance is shrunk, why
per-name caps exist, and why **min-variance and risk parity, which need no
return forecast at all, are usually the more trustworthy outputs on that page.**

Everything the optimiser produces is optimal *in sample*. It is not a
recommendation.

---

## 8. Simulation

| Method | Process | Honest limitation |
|---|---|---|
| GBM | `Sₜ₊₁ = Sₜ·exp((μ − σ²/2) + σz)` | normal log-returns → **understates tail risk** |
| IID bootstrap | resample observed daily returns | preserves fat tails; destroys autocorrelation |
| Block bootstrap | resample contiguous blocks | preserves volatility clustering |
| Student-t | fat-tailed innovations | heavier tails, still i.i.d. |

The `− σ²/2` term in GBM is the Itô drift adjustment: without it the median path
drifts upward incorrectly, because `E[exp(X)] ≠ exp(E[X])`.

Every method assumes the future is drawn from the same distribution as the
past. A bootstrap **cannot generate a crash larger than any in its sample**, and
none of these models a regime change. The output is a description of this
sample propagated forward — not a forecast. The seed is exposed in the UI so any
run is exactly reproducible, which is what makes a simulation auditable.

---

## 9. Stress testing

Three complementary approaches:

1. **Historical replay** — apply today's weights to each asset's *actual*
   returns during a documented crisis window (COVID-19 crash, 2022 rate shock,
   GFC, taper tantrum, demonetisation, Volmageddon, China devaluation).

   Every scenario reports a **coverage** figure: the fraction of today's
   portfolio weight that actually existed during that window. A stock listed in
   2021 has no COVID-crash history, and treating it as flat would fabricate a
   reassuring number. Coverage 0 means the scenario cannot be evaluated at all,
   and the app says so instead of guessing.

2. **Beta-implied shock** — `impact = β × shock`. First-order CAPM, and
   deliberately labelled as such: it ignores idiosyncratic risk, and empirically
   betas converge toward 1 in a crisis, so this **understates** losses for
   low-beta portfolios exactly when it matters.

3. **Worst realised windows** — the worst non-overlapping N-day returns this
   exact portfolio actually produced inside the selected window. No model, no
   assumption.

---

## 10. Correlation structure

**PCA on the correlation matrix** (standardised returns) finds the independent
directions of shared movement. PC1 is almost always "the market". If PC1
explains 80% of your variance, you own one bet wearing many tickers, however
many rows the holdings table has.

**Hierarchical clustering** on the correlation distance
`d(i,j) = √(0.5(1 − ρᵢⱼ))` groups near-substitutes. Names inside a cluster
diversify each other very little, and the correlation heatmap is reordered by
that clustering so the block structure is visible.

Correlations are **not stable**. They rise in crises — precisely when
diversification is needed most — which is why the app charts rolling
correlation rather than reporting a single number.

---

## 11. Risk score (0–100, rule-based)

A transparent heuristic, **not** a regulatory measure and not comparable to any
published risk rating. The rubric lives in `config.RiskScoreRubric` and the app
renders the component breakdown so you can disagree with the weighting.

| Component | Max points | 0 points at | Full points at |
|---|---|---|---|
| Annualised volatility | 30 | 0% | ≥ 40% |
| Maximum drawdown | 25 | 0% | ≥ 50% fall |
| Concentration (HHI, rescaled) | 18 | equal weight | single stock |
| Number of holdings | 12 | ≥ 10 names | 1 name |
| Tail risk (daily 95% CVaR) | 15 | 0% | ≥ 5% |

**Bands:** `< 35` Low · `35–65` Moderate · `> 65` High.

When a component cannot be computed (no history → no CVaR) it contributes
nothing and the score is **rescaled over the points that were actually
available**, so a partial score is still on a 0–100 basis. The thresholds are
judgement calls chosen to be reasonable and legible, not calibrated against any
standard.

---

## 12. Technical indicators

| Indicator | Definition |
|---|---|
| SMA(n) | arithmetic mean of the last n closes |
| EMA(n) | exponential mean, `α = 2/(n+1)` |
| RSI(14) | `100 − 100/(1 + avg_gain/avg_loss)`, **Wilder's smoothing** (`α = 1/n`) |
| MACD | `EMA₁₂ − EMA₂₆`; signal `EMA₉(MACD)`; histogram = MACD − signal |
| Bollinger(20, 2) | `SMA₂₀ ± 2·std₂₀`; `%B = (P − lower)/(upper − lower)` |
| ATR(14) | Wilder-smoothed mean of `max(H−L, |H−C₋₁|, |L−C₋₁|)` |

RSI uses **Wilder's exponential smoothing**, not a simple rolling mean. The
difference is small on any single day and compounds badly over a series, and
using a rolling mean is the standard tell of an implementation copied from a
blog post. The test suite asserts that a monotonically rising series drives RSI
to 100, which is the regression guard for it.

These are mechanical pattern flags computed from price alone. They are widely
known, therefore already reflected in the price, and they are not predictions.

---

## 13. Forecasting (machine learning)

Everything in this section comes from `src/analytics/features.py` and
`src/analytics/forecast.py`. The sign conventions matter more here than
anywhere else in this document, because most of these quantities are
**negative for a normal, correctly-behaved model** and reading one as a
failure is the easiest mistake to make on the page.

### Label and features

| Quantity | Definition |
|---|---|
| Label | `y_t = log(C_{t+h} / C_t)` — the log return over the next *h* sessions |
| Feature at row *t* | a function of observations **up to and including** *t*, only |
| Feature scale | every feature is unit-free: a log return, a ratio, a z-score or a position inside a range |

Log rather than simple return for two reasons that both bite here: it is
additive across time, so a 21-day label is exactly the sum of its 21 daily
parts, and it is far closer to symmetric — which every squared-error loss in
the model layer implicitly assumes.

The eight feature families are lagged returns, momentum (10–252 sessions plus
12-1), volatility (realised, downside, regime ratio, skew, kurtosis), trend
(distance from the 10/20/50/200 averages and their spreads), oscillators
(Wilder RSI, MACD histogram, Bollinger %B and bandwidth, ATR ratio), range
(position in the trailing 252-session range, bar shape, overnight gap), volume
(z-score, short/long ratio) and calendar (cyclical day/month encodings, turn of
month). The volume family and the bar-shape features need columns a close-only
series does not have, and are **skipped rather than fabricated** for one.

### Validation geometry

| Quantity | Definition |
|---|---|
| Folds | `n_splits` sequential expanding-window blocks over the last `test_fraction` of rows |
| Purge | `h − 1 + purge_extra` rows dropped between each training block and its test block |
| Shuffling | **none, ever** — a shuffled split trains on Thursday to predict Wednesday |

The purge is not a safety margin. An *h*-session label physically spans *h*
future sessions, so without it the last `h − 1` training labels are built from
the very returns the test block is about to be scored on.

### Accuracy

| Metric | Formula | Sign convention |
|---|---|---|
| MAE | `mean(|y − ŷ|)` | in log-return units over the whole horizon |
| RMSE | `sqrt(mean((y − ŷ)²))` | same units |
| MAE (naive) | `mean(|y|)` | the error of a forecast of exactly zero — the random walk |
| MASE | `MAE / MAE_naive` | **below 1.0 beats the random walk** |
| Skill | `1 − MASE` | **positive beats the random walk; negative is the normal result** |
| Out-of-sample R² | `1 − Σ(y − ŷ)² / Σy²` | Campbell–Thompson. **Negative is normal** and means the model explains less than nothing. Low single-digit percentages are considered strong in the published literature |
| Directional accuracy | `mean(sign(ŷ) = sign(y))`, over rows where the model called a direction **and** the move was not flat | compare against `up_rate`, **not** 50% |
| `up_rate` | `mean(y > 0)` | the base rate. An instrument that rose 58% of the time makes a 55% hit rate *worse* than always saying up |
| IC | Spearman rank correlation of `ŷ` against `y` | does the model at least *order* outcomes correctly? |

A forecast of exactly zero calls no direction at all, so the random walk
reports directional accuracy as `—` rather than 0%. Scoring it as zero would
read as "always wrong" instead of "never answered".

### Significance

Diebold–Mariano on the squared-error loss differential
`d_t = e²_naive − e²_model`, so a **positive statistic means the model beat the
benchmark**:

```
DM = mean(d) / sqrt(V̂(d) / n)
V̂(d) = γ₀ + 2·Σ_{k=1..L} (1 − k/(L+1))·γ_k      Newey–West, L = h − 1
DM  ← DM · sqrt( (n + 1 − 2(L+1) + (L+1)L/n) / n )   Harvey–Leybourne–Newbold
```

Referred to `t(n − 1)`, two-sided. The HAC variance is not optional: overlapping
horizons make `d_t` serially correlated *by construction*, and the uncorrected
variance is too small — which is precisely how noise becomes "significant".

Returns `—` when the two forecasts are identical (the differential is
degenerate) or fewer than eight usable points exist.

### Prediction interval

Split-conformal, from the model's own out-of-sample residuals `r = y − ŷ`:

```
interval = [ ŷ_new + Q_{α/2}(r),  ŷ_new + Q_{1−α/2}(r) ]
```

**Signed** quantiles, not a symmetric `±Q(|r|)`: equity returns are left-skewed
and so are the errors of a model fitted to them, and a symmetric band would
overstate the upside and understate the downside by the same amount.

Coverage is **marginal, not conditional** — it holds on average across the
backtest and does not widen because today happens to be a volatile day.

| Quantity | Definition |
|---|---|
| Bias | `median(r)` — the model's measured out-of-sample bias, and why the band is not centred on the point forecast |
| P(up) | `mean(ŷ_new + r > 0)` — the share of the empirical predictive distribution above zero. No Normal assumption |
| Forecast price | `S₀ · exp(ŷ)` — the **median** of the implied distribution, not its mean. The mean sits above it, because compounding is multiplicative |

### Portfolio aggregation

| Quantity | Definition |
|---|---|
| Expected return | `Σ wᵢ · (exp(ŷᵢ) − 1)`, renormalised across covered weight | 
| Comonotonic band | every holding at its own bound simultaneously — perfect correlation, the widest defensible band |
| Independent band | per-holding half-widths added in quadrature — zero correlation, which no equity book has |
| Coverage | share of portfolio weight that produced a usable forecast |

Each holding's **log** forecast is converted to a simple return before
weighting and back afterwards. Weighting log returns directly is wrong by
Jensen's inequality. The truth about the interval lies between the two bands
and depends on a correlation the forecaster never estimated, so both are shown
rather than one invented.

### What none of this measures

Forecast accuracy is **not profit**. Nothing in this section models a
transaction cost, a spread, slippage, market impact, position sizing, borrow
cost or tax. A model with genuine statistical skill can lose money after all
six, and this application does not backtest a strategy.

---

## Guard rails

Every metric returns `NaN` rather than raising or fabricating when:

- there are fewer than 2 observations
- the starting value is non-positive
- volatility, benchmark variance or the Omega denominator is zero
- investment or total portfolio value is zero
- a covariance matrix is singular
- fewer than `MIN_OBSERVATIONS // 2` returns exist for a tail statistic
- a walk-forward cannot produce a single fold with a trainable block
- fewer than eight out-of-sample residuals exist for a conformal interval
- a forecast model predicts a constant, so it calls no direction at all

The UI renders `NaN` as an em dash and, where it matters, explains what would be
needed to compute the number instead.
