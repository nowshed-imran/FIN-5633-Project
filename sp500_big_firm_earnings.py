"""
Do index moves get bigger on days when the largest S&P 500 firms report
earnings?

Author: Nowshed Hossain Imran
Course: FIN 5633 Computational Finance (course project, 2025, revised 2026)

Big firm   = index weight of 1% or more on the day before it reports
Small firm = index weight below 1%

The revision follows the standard steps for time series regression:
  1. Set up the project folders
  2. Download the data (skip if it already exists)
  3. Build one row per trading day with Big and Small earnings features
  4. Check the data: weak dependence and a Dickey-Fuller unit root test
  5. H1 model: absolute index return on Big-day and Small-day dummies with
     controls (yesterday's VIX, year dummies, month dummies), estimated by OLS
  6. Check the assumptions: functional form (RESET), serial correlation
     (AR(1) test, Breusch-Godfrey, Durbin-Watson), heteroskedasticity
     (Breusch-Pagan, ARCH) and strict exogeneity
  7. Fix the problems: Prais-Winsten FGLS vs Newey-West standard errors
  8. Structural break: Chow test for 2008 and 2020
  9. H2 (short): do Big earnings misses move the index more than Big beats?
 10. Save the tables and the chart to the output folder

Reference: Wooldridge, J. M. (2020). Introductory econometrics: A modern
approach (7th ed.). Cengage.

AI assistance: The study design, hypotheses, sample choices and interpretation
are my own. The code was written with the help of Claude (Anthropic), which I
used to rewrite my original course scripts into this single script, find and
fix errors, and implement the econometric corrections. I have reviewed the
code extensively and would appreciate any suggestions for improvement.

Run from any folder:  python sp500_big_firm_earnings.py
The script works inside its own folder, so data/ and output/ are created
next to it.
"""
import os
from io import StringIO
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import (het_breuschpagan, het_arch,
                                          acorr_breusch_godfrey, linear_reset)
from statsmodels.stats.stattools import durbin_watson
from scipy import stats


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
START = "2006-10-01"       # fixed sample window: October 2006 to
END = "2026-10-01"         # September 2026 (END itself is not included)
BIG_CUTOFF = 0.01          # 1% index weight
SURPRISE_CAP = 100         # clip surprises to +/-100% (EPS near zero explodes)

# The folder this script is saved in
PROJECT = os.path.dirname(os.path.abspath(__file__))

TICKERS_FILE = "data/sp500_tickers.csv"
PRICES_FILE = "data/prices.csv"
EARNINGS_FILE = "data/earnings.csv"
VIX_FILE = "data/vix.csv"


# ---------------------------------------------------------------------------
# 1. Project folders
# ---------------------------------------------------------------------------
# Work inside the project folder (same as setwd() in R), so every file
# path below is relative to it
os.chdir(PROJECT)

for folder in ["data", "output"]:
    if not os.path.exists(folder):
        os.mkdir(folder)
        print("created", folder)


# ---------------------------------------------------------------------------
# 2. Data (each file is downloaded once, then read from disk)
# ---------------------------------------------------------------------------

def download_firm(ticker):
    """Shares outstanding and earnings history for one firm."""
    try:
        stock = yf.Ticker(ticker)
        shares = stock.info.get("sharesOutstanding")
        e = stock.get_earnings_dates(limit=100)   # 100 quarters is the max
    except Exception:
        return None
    if shares is None or e is None or e.empty:
        return None

    when = e.index.tz_convert("America/New_York")
    earn = pd.DataFrame({
        "ticker": ticker,
        "announce_date": when.tz_localize(None).normalize(),
        "hour": when.hour,
        "surprise": e["Surprise(%)"].values,
    })
    return shares, earn


# 2a. S&P 500 members, shares outstanding and earnings history
if os.path.exists(TICKERS_FILE) and os.path.exists(EARNINGS_FILE):
    firms = pd.read_csv(TICKERS_FILE)
    earnings = pd.read_csv(EARNINGS_FILE, parse_dates=["announce_date"])
else:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    # Wikipedia blocks requests that do not look like they come from a browser
    html = urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"})).read()
    html = html.decode("utf-8")
    tickers = pd.read_html(StringIO(html))[0]["Symbol"]
    tickers = tickers.str.replace(".", "-").tolist()   # BRK.B -> BRK-B

    # Each firm needs two requests to Yahoo. Most of the time is spent waiting
    # for replies, so 8 firms are downloaded at once instead of one by one.
    # pool.map works like lapply in R: run download_firm on every ticker.
    print("Downloading shares and earnings for", len(tickers), "firms...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(download_firm, tickers))

    firm_rows, earn_rows = [], []
    for t, res in zip(tickers, results):
        if res is not None:
            firm_rows.append({"ticker": t, "shares": res[0]})
            earn_rows.append(res[1])
    print(len(tickers) - len(firm_rows), "firms failed and were skipped")

    firms = pd.DataFrame(firm_rows)
    earnings = pd.concat(earn_rows).dropna(subset=["surprise"])  # drops future dates
    firms.to_csv(TICKERS_FILE, index=False)
    earnings.to_csv(EARNINGS_FILE, index=False)

tickers = firms["ticker"].tolist()
print(len(tickers), "firms")

# 2b. Daily closing prices for every firm and the index (^GSPC)
# yf.download already fetches many tickers at once, so this part is fast
if os.path.exists(PRICES_FILE):
    prices = pd.read_csv(PRICES_FILE, index_col=0, parse_dates=True)
else:
    prices = yf.download(tickers + ["^GSPC"], start=START, end=END,
                         auto_adjust=False)["Close"]
    prices.to_csv(PRICES_FILE)

# 2c. VIX (market volatility index), used as a control
if not os.path.exists(VIX_FILE):
    yf.download("^VIX", start=START, end=END,
                auto_adjust=False)["Close"].to_csv(VIX_FILE)
vix = pd.read_csv(VIX_FILE, index_col=0, parse_dates=True).iloc[:, 0]

# Same date format everywhere so the joins below line up
prices.index = prices.index.astype("datetime64[ns]")
vix.index = vix.index.astype("datetime64[ns]")

# Apply the fixed window, also to files saved by an earlier run
prices = prices[(prices.index >= START) & (prices.index < END)]
vix = vix[(vix.index >= START) & (vix.index < END)]
print(prices[tickers].notna().any().sum(), "firms with price data in the window")


# ---------------------------------------------------------------------------
# 3. Daily dataset
# ---------------------------------------------------------------------------

# Index return
spx = pd.DataFrame({"ret": prices["^GSPC"].pct_change()}).dropna()

# Approximate index weight of each firm on each day:
# market cap = price x shares (today's share count), weight = cap / total cap
caps = prices[tickers] * firms.set_index("ticker")["shares"]
weights = caps.div(caps.sum(axis=1), axis=0)

# Use the weight from the day BEFORE the reaction day, so nothing from the
# event day itself is used to classify a firm as Big or Small
weights = weights.shift(1).stack().reset_index()
weights.columns = ["trade_date", "ticker", "weight"]

# Reaction day: reports at or after 4 pm hit the market the next trading day.
# Reports before the open or during the day are matched to the same day.
# merge_asof moves weekend/holiday dates forward to the next trading day.
earnings["react_date"] = earnings["announce_date"]
after_close = earnings["hour"] >= 16
earnings.loc[after_close, "react_date"] += pd.Timedelta(days=1)

earnings["react_date"] = earnings["react_date"].astype("datetime64[ns]")
trade_days = pd.DataFrame({"trade_date": spx.index})

# Keep only reports inside the price window. Without this, merge_asof would
# push every older report forward onto the first trading day.
in_window = earnings["react_date"].between(trade_days["trade_date"].min(),
                                           trade_days["trade_date"].max())
earnings = earnings[in_window]

events = pd.merge_asof(earnings.sort_values("react_date"), trade_days,
                       left_on="react_date", right_on="trade_date",
                       direction="forward")
events = events.merge(weights, on=["trade_date", "ticker"]).dropna()

events["surprise"] = events["surprise"].clip(-SURPRISE_CAP, SURPRISE_CAP)

# Weighted surprise, split into beats and misses (used in H2).
# A 5% beat from a firm with 6% weight counts as 0.06 x 5 = 0.3.
events["pos"] = events["weight"] * events["surprise"].clip(lower=0)
events["neg"] = events["weight"] * events["surprise"].clip(upper=0)

# Market conditions known before each day starts: yesterday's VIX. Today's VIX would move together with today's return.
spx["vix_lag"] = vix.shift(1)
spx["year"] = spx.index.year
spx["ret_pct"] = spx["ret"] * 100
spx["abs_ret"] = spx["ret"].abs() * 100


def build_daily(cutoff):
    """One row per trading day with Big and Small earnings features.
    The cutoff is a function input so other cutoffs can be tried in step 7."""
    ev = events.copy()
    ev["group"] = np.where(ev["weight"] >= cutoff, "big", "small")

    # Collapse to one row per day and group (like GROUP BY in SQL), then
    # spread Big and Small into separate columns
    daily = (ev.groupby(["trade_date", "group"])
               .agg(n=("ticker", "count"),
                    weight=("weight", "sum"),
                    surprise=("surprise", "mean"),
                    pos=("pos", "sum"),
                    neg=("neg", "sum"))
               .unstack("group"))
    daily.columns = [f"{grp}_{stat}" for stat, grp in daily.columns]

    d = spx.join(daily)
    feature_cols = list(daily.columns)
    d[feature_cols] = d[feature_cols].fillna(0)
    # Like CASE WHEN in SQL: the first true condition wins, so a day with
    # both Big and Small reports is a Big day
    d["day_type"] = np.select([d["big_n"] > 0, d["small_n"] > 0],
                              ["Big", "Small only"], default="None")
    return d


data = build_daily(BIG_CUTOFF)
print(data["day_type"].value_counts())


# ---------------------------------------------------------------------------
# 4. Check the data
# ---------------------------------------------------------------------------
data["month"] = data.index.month
data["big_day"] = (data["day_type"] == "Big").astype(int)
data["small_day"] = (data["day_type"] == "Small only").astype(int)
data = data.dropna(subset=["vix_lag"])
n = len(data)
print("\nTrading days used:", n)

h1_table = data.groupby("day_type")["abs_ret"].agg(["count", "mean", "std"])
print("\nAbsolute index return (%) by day type")
print(h1_table.round(3))

# Weak dependence: how strongly is today's value related to yesterday's?
# A coefficient well below 1 means the series is weakly dependent.
checks = []
for col in ["ret_pct", "abs_ret", "vix_lag"]:
    ar1 = smf.ols(f"{col} ~ {col}_l1", data=data.assign(
        **{f"{col}_l1": data[col].shift(1)})).fit()
    adf = adfuller(data[col], autolag="AIC")
    checks.append({"series": col, "AR1_coef": ar1.params[f"{col}_l1"],
                   "DF_stat": adf[0], "DF_p": adf[1]})
checks = pd.DataFrame(checks)
print("\nWeak dependence and Dickey-Fuller tests")
print(checks.round(4).to_string(index=False))


# ---------------------------------------------------------------------------
# 5. H1 model, estimated by OLS
# ---------------------------------------------------------------------------
# All trading days. No-report days are the base group, so:
#   big_day   = extra absolute return on days a Big firm reports
#   small_day = extra absolute return on days only Small firms report
# H1: big_day > small_day (and big_day > 0)
CONTROLS = " + vix_lag + C(year) + C(month)"
H1 = "abs_ret ~ big_day + small_day" + CONTROLS
H1_NO_CONTROLS = "abs_ret ~ big_day + small_day"

ols_raw = smf.ols(H1_NO_CONTROLS, data=data).fit()
ols = smf.ols(H1, data=data).fit()
print("\nOLS without controls:", ols_raw.params[["big_day", "small_day"]].round(4).to_dict())
print("OLS with controls:   ", ols.params[["big_day", "small_day", "vix_lag"]].round(4).to_dict(),
      " R2 =", round(ols.rsquared, 3))
# Month dummies: joint F test
month_terms = [t for t in ols.params.index if t.startswith("C(month)")]
month_p = float(ols.f_test(", ".join(f"{t} = 0" for t in month_terms)).pvalue)
print("Month dummies jointly zero? p =", month_p)


# ---------------------------------------------------------------------------
# 6. Check the assumptions
# ---------------------------------------------------------------------------
u = ols.resid
lags = int(4 * (n / 100) ** (2 / 9))       # Newey-West rule of thumb

# a) Functional form: RESET
reset = linear_reset(ols, power=3, use_f=True)

# b) Serial correlation: regress the residual on its own lag and the
#    regressors (this version does not need strict exogeneity), then
#    Breusch-Godfrey with more lags, and Durbin-Watson
X = pd.DataFrame(ols.model.exog, index=data.index, columns=ols.model.exog_names)
z = X.copy()
z["u_l1"] = u.shift(1)
ar_fit = sm.OLS(u.iloc[1:], z.iloc[1:]).fit()
rho = ar_fit.params["u_l1"]
bg = acorr_breusch_godfrey(ols, nlags=lags)

# c) Heteroskedasticity: Breusch-Pagan, and the ARCH test for time series
bp = het_breuschpagan(u, ols.model.exog)
arch = het_arch(u, nlags=1)

# d) Strict exogeneity (needed for FGLS): add next day's value of the
#    controls. If tomorrow's lagged VIX (= today's VIX) matters, the
#    regressors are not strictly exogenous.
lead = data.assign(vix_lead=data["vix_lag"].shift(-1),
                   big_lead=data["big_day"].shift(-1)).dropna()
se_test = smf.ols(H1 + " + vix_lead + big_lead", data=lead).fit(
    cov_type="HAC", cov_kwds={"maxlags": lags})

tests = pd.DataFrame([
    {"assumption": "Functional form (RESET)", "statistic": float(reset.fvalue), "p": float(reset.pvalue)},
    {"assumption": "Serial correlation: AR(1) rho", "statistic": rho, "p": ar_fit.pvalues["u_l1"]},
    {"assumption": f"Serial correlation: Breusch-Godfrey ({lags} lags)", "statistic": bg[0], "p": bg[1]},
    {"assumption": "Serial correlation: Durbin-Watson", "statistic": durbin_watson(u), "p": np.nan},
    {"assumption": "Heteroskedasticity: Breusch-Pagan", "statistic": bp[0], "p": bp[1]},
    {"assumption": "Heteroskedasticity: ARCH(1)", "statistic": arch[0], "p": arch[1]},
    {"assumption": "Strict exogeneity: tomorrow's VIX", "statistic": se_test.tvalues["vix_lead"], "p": se_test.pvalues["vix_lead"]},
    {"assumption": "Strict exogeneity: tomorrow's Big day", "statistic": se_test.tvalues["big_lead"], "p": se_test.pvalues["big_lead"]},
])
print("\nAssumption tests for the H1 model")
print(tests.round(4).to_string(index=False))

# RESET rejects, so try the usual fixes for the form of the model and check
# whether the Big-day effect changes
data["abs_l1"] = data["abs_ret"].shift(1)
form_rows = []
for label, extra in [("Linear VIX (main)", ""), ("+ VIX squared", " + I(vix_lag**2)"),
                     ("log VIX instead of VIX", "LOG"), ("+ yesterday's |return|", " + abs_l1"),
                     ("+ VIX squared + yesterday's |return|", " + I(vix_lag**2) + abs_l1")]:
    f = H1.replace("vix_lag", "np.log(vix_lag)") if extra == "LOG" else H1 + extra
    dd = data.dropna(subset=["abs_l1"]) if "abs_l1" in f else data
    m = smf.ols(f, data=dd).fit()
    r = linear_reset(m, power=3, use_f=True)
    m_nw = smf.ols(f, data=dd).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    form_rows.append({"model": label, "R2": m.rsquared, "RESET_F": float(r.fvalue),
                      "RESET_p": float(r.pvalue), "big_day": m_nw.params["big_day"],
                      "p_big": m_nw.pvalues["big_day"]})
form = pd.DataFrame(form_rows)
print("\nFunctional form checks (Big-day effect with Newey-West SE)")
print(form.round(4).to_string(index=False))


# ---------------------------------------------------------------------------
# 7. Fixing the problems
# ---------------------------------------------------------------------------
# Option 1, FGLS (Prais-Winsten): transform the data with the estimated AR(1)
# rho. Only valid if the regressors are strictly exogenous.
def prais_winsten(formula, df, iters=20):
    model = smf.ols(formula, data=df)
    y, Xm, names = model.endog, model.exog, model.exog_names
    b = np.linalg.lstsq(Xm, y, rcond=None)[0]
    for _ in range(iters):
        e = y - Xm @ b
        r = np.sum(e[1:] * e[:-1]) / np.sum(e[:-1] ** 2)
        ys = np.r_[np.sqrt(1 - r ** 2) * y[0], y[1:] - r * y[:-1]]
        Xs = np.vstack([np.sqrt(1 - r ** 2) * Xm[0], Xm[1:] - r * Xm[:-1]])
        b = np.linalg.lstsq(Xs, ys, rcond=None)[0]
    fit = sm.OLS(ys, Xs).fit()
    fit.model.exog_names[:] = names
    return fit, r

pw, pw_rho = prais_winsten(H1, data)
pw_params = pd.Series(pw.params, index=ols.model.exog_names)
pw_bse = pd.Series(pw.bse, index=ols.model.exog_names)

# Option 2, OLS with Newey-West (HAC) standard errors: valid without strict
# exogeneity, and also robust to heteroskedasticity
nw = smf.ols(H1, data=data).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
hc1 = smf.ols(H1, data=data).fit(cov_type="HC1")


def big_minus_small(fit, params=None, cov=None):
    """Big-day effect minus Small-day effect, with t and two-sided p."""
    names = list(fit.model.exog_names)
    p = fit.params if params is None else params
    V = fit.cov_params() if cov is None else cov
    V = pd.DataFrame(np.asarray(V), index=names, columns=names)
    p = pd.Series(np.asarray(p), index=names)
    d = p["big_day"] - p["small_day"]
    se = np.sqrt(V.loc["big_day", "big_day"] + V.loc["small_day", "small_day"]
                 - 2 * V.loc["big_day", "small_day"])
    t = d / se
    return d, se, t, 2 * stats.norm.sf(abs(t))


rows = []
for name, fit in [("OLS, usual SE", ols), ("OLS, HC1 SE", hc1),
                  ("OLS, Newey-West SE", nw), ("Prais-Winsten FGLS", pw)]:
    if name.startswith("Prais"):
        prm = pw_params; V = pw.cov_params()
        b_big, se_big = pw_params["big_day"], pw_bse["big_day"]
    else:
        prm = fit.params; V = fit.cov_params()
        b_big, se_big = fit.params["big_day"], fit.bse["big_day"]
    d, se_d, t_d, p_d = big_minus_small(ols, prm, V)
    rows.append({"method": name, "big_day": b_big, "se_big": se_big,
                 "p_big": 2 * stats.norm.sf(abs(b_big / se_big)),
                 "big_minus_small": d, "se_diff": se_d, "p_diff": p_d})
h1_results = pd.DataFrame(rows)
print(f"\nH1 estimates (Newey-West lags = {lags}, Prais-Winsten rho = {pw_rho:.3f})")
print(h1_results.round(4).to_string(index=False))
nw_raw = smf.ols(H1_NO_CONTROLS, data=data).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
print("Without controls (Newey-West): big_day =", round(nw_raw.params["big_day"], 4))

# Other Big cutoffs, Newey-West
cut_rows = []
for cutoff in [0.005, 0.01, 0.02]:
    dc = build_daily(cutoff)
    dc["month"] = dc.index.month
    dc["big_day"] = (dc["day_type"] == "Big").astype(int)
    dc["small_day"] = (dc["day_type"] == "Small only").astype(int)
    dc = dc.dropna(subset=["vix_lag"])
    f = smf.ols(H1, data=dc).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    d, se_d, t_d, p_d = big_minus_small(f)
    cut_rows.append({"cutoff": cutoff, "big_days": int(dc["big_day"].sum()),
                     "big_day": f.params["big_day"], "p_big": f.pvalues["big_day"],
                     "big_minus_small": d, "p_diff": p_d})
cutoffs = pd.DataFrame(cut_rows)
print("\nOther cutoffs (Newey-West)")
print(cutoffs.round(4).to_string(index=False))


# ---------------------------------------------------------------------------
# 8. Structural break: is the Big-day effect different in 2008 and 2020?
# ---------------------------------------------------------------------------
data["crisis"] = data.index.year.isin([2008, 2020]).astype(int)
chow = smf.ols(H1 + " + big_day:crisis + small_day:crisis", data=data).fit(
    cov_type="HAC", cov_kwds={"maxlags": lags})
chow_f = chow.f_test("big_day:crisis = 0, small_day:crisis = 0")
print("\nChow test, H1 (Newey-West): big_day in normal years =",
      round(chow.params["big_day"], 4), " extra in crisis =",
      round(chow.params["big_day:crisis"], 4), " joint p =", round(float(chow_f.pvalue), 4))
h1_calm = smf.ols(H1, data=data[data["crisis"] == 0]).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
print("H1 without 2008 and 2020: big_day =", round(h1_calm.params["big_day"], 4),
      " p =", round(h1_calm.pvalues["big_day"], 4))


# ---------------------------------------------------------------------------
# 9. H2 (short): do Big misses move the index more than Big beats?
# ---------------------------------------------------------------------------
# Index return on weighted beats (>= 0) and misses (<= 0). Both slopes should
# be positive. H2: the Big miss slope is larger than the Big beat slope.
H2_FORMULA = "ret_pct ~ big_pos + big_neg + small_pos + small_neg" + CONTROLS
h2 = smf.ols(H2_FORMULA, data=data).fit()
h2_nw = smf.ols(H2_FORMULA, data=data).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
h2_bg = acorr_breusch_godfrey(h2, nlags=lags)
h2_bp = het_breuschpagan(h2.resid, h2.model.exog)


def miss_minus_beat(fit):
    test = fit.t_test("big_neg - big_pos = 0")
    t = test.tvalue.item(); p2 = test.pvalue.item()
    return test.effect.item(), t, (p2 / 2 if t > 0 else 1 - p2 / 2)


h2_rows = []
for name, fit in [("All years", h2_nw),
                  ("Without 2008 and 2020", smf.ols(H2_FORMULA, data=data[data["crisis"] == 0]).fit(
                      cov_type="HAC", cov_kwds={"maxlags": lags}))]:
    e, t, p = miss_minus_beat(fit)
    h2_rows.append({"sample": name, "big_pos": fit.params["big_pos"],
                    "big_neg": fit.params["big_neg"], "miss_minus_beat": e, "t": t, "p_one_sided": p})
h2_table = pd.DataFrame(h2_rows)
print(f"\nH2 (Newey-West). Breusch-Godfrey p = {h2_bg[1]:.4g}, Breusch-Pagan p = {h2_bp[1]:.4g}")
print(h2_table.round(4).to_string(index=False))
h2_chow = smf.ols(H2_FORMULA + " + big_pos:crisis + big_neg:crisis", data=data).fit(
    cov_type="HAC", cov_kwds={"maxlags": lags})
h2_chow_p = float(h2_chow.f_test("big_pos:crisis = 0, big_neg:crisis = 0").pvalue)
print("H2 Chow test, joint p =", round(h2_chow_p, 4))


# ---------------------------------------------------------------------------
# 10. Save results and chart
# ---------------------------------------------------------------------------
# Numbers quoted in the text that do not sit in the tables above
other = pd.DataFrame([
    {"item": "H1 big_day without controls (Newey-West)", "estimate": nw_raw.params["big_day"], "p": nw_raw.pvalues["big_day"]},
    {"item": "H1 month dummies jointly zero (F test)", "estimate": np.nan, "p": month_p},
    {"item": "H1 big_day in normal years (Chow model)", "estimate": chow.params["big_day"], "p": chow.pvalues["big_day"]},
    {"item": "H1 extra big_day effect in 2008 and 2020", "estimate": chow.params["big_day:crisis"], "p": chow.pvalues["big_day:crisis"]},
    {"item": "H1 Chow test, both day-type effects change (joint p)", "estimate": np.nan, "p": float(chow_f.pvalue)},
    {"item": "H1 big_day without 2008 and 2020", "estimate": h1_calm.params["big_day"], "p": h1_calm.pvalues["big_day"]},
    {"item": "H2 Chow test, beat and miss slopes change (joint p)", "estimate": np.nan, "p": h2_chow_p},
    {"item": "Prais-Winsten rho", "estimate": pw_rho, "p": np.nan},
])
other.to_csv("output/chow_and_other_tests.csv", index=False)
h1_table.to_csv("output/h1_abs_return_by_day_type.csv")
checks.to_csv("output/data_checks.csv", index=False)
tests.to_csv("output/assumption_tests.csv", index=False)
form.to_csv("output/functional_form.csv", index=False)
h1_results.to_csv("output/h1_results.csv", index=False)
cutoffs.to_csv("output/h1_cutoffs.csv", index=False)
h2_table.to_csv("output/h2_results.csv", index=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
roll = data["abs_ret"].rolling(21).mean()
ax1.plot(data.index, data["abs_ret"], color="lightgray", linewidth=0.5, label="Daily")
ax1.plot(data.index, roll, color="tab:blue", linewidth=1.2, label="21-day average")
ax1.set_title("(a) Absolute S&P 500 return (%)")
ax1.legend(loc="upper left")
labels = ["OLS\nusual", "OLS\nHC1", "OLS\nNewey-West", "Prais-Winsten\nFGLS"]
est = h1_results["big_day"].values; se = h1_results["se_big"].values
ax2.bar(labels, est, yerr=1.96 * se, capsize=5, color="tab:blue")
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_title("(b) H1: Big-day effect with 95% intervals")
ax2.set_ylabel("Extra absolute return (pp)")
plt.tight_layout()
plt.savefig("output/results.png", dpi=150)
print("\nDone. Results are in the output folder.")
