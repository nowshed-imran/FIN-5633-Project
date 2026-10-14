"""
Do earnings announcements from the largest S&P 500 firms move the index more
than announcements from the rest of the index?

Author: Nowshed Hossain Imran
Course: Computational Finance

Big firm   = index weight of 1% or more on the day before it reports
Small firm = index weight below 1%

The script does seven things:
  1. Sets up the project folders
  2. Downloads the data (only if it is not already saved)
  3. Builds one row per trading day with Big and Small earnings features
  4. H1: compares the size of index moves on Big vs Small earnings days
  5. H2: tests whether earnings features predict index direction better
     than always guessing the most common direction
  6. H3: tests whether Big earnings misses move the index more than
     Big earnings beats of the same size, then re-runs it without the
     2008 and 2020 crisis years as a robustness check
  7. Extension: re-tests H1 and H3 with Newey-West standard errors, market
     controls (yesterday's VIX, year dummies), a Chow-style crisis test and
     other Big cutoffs (Wooldridge, Introductory Econometrics, Ch. 7, 10, 12)

Before running, set PROJECT below to your project folder.
Then run:  python sp500_big_firm_earnings.py
"""

import os
from io import StringIO
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
START = "2006-10-01"       # fixed sample window: October 2006 to
END = "2026-10-01"         # September 2026 (END itself is not included)
BIG_CUTOFF = 0.01          # 1% index weight
TRAIN_SHARE = 0.7          # first 70% of days train, last 30% test
SURPRISE_CAP = 100         # clip surprises to +/-100% (EPS near zero explodes)

# Change this to the folder where the project lives on your computer.
# The r before the quotes lets Windows backslashes work as they are.
PROJECT = r"C:\Users\nowsh\OneDrive - Oklahoma A and M System\Desktop\Course\Fall 25\Computational Finance\Project\FIN-5633-Project"

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

# 2c. VIX (market volatility index), used as a control in the extension
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


# ---------------------------------------------------------------------------
# 3. Daily dataset
# ---------------------------------------------------------------------------

# Index return and direction
spx = pd.DataFrame({"ret": prices["^GSPC"].pct_change()}).dropna()
spx["up"] = (spx["ret"] > 0).astype(int)

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

# Weighted surprise, split into beats and misses (used in H3).
# A 5% beat from a firm with 6% weight counts as 0.06 x 5 = 0.3.
events["pos"] = events["weight"] * events["surprise"].clip(lower=0)
events["neg"] = events["weight"] * events["surprise"].clip(upper=0)

# Market conditions known before each day starts (used in the extension):
# yesterday's VIX. Today's VIX would move together with today's return.
spx["vix_lag"] = vix.shift(1)
spx["year"] = spx.index.year
spx["ret_pct"] = spx["ret"] * 100
spx["abs_ret"] = spx["ret"].abs() * 100


def build_daily(cutoff):
    """One row per trading day with Big and Small earnings features.
    The cutoff is a function input so the extension can try other cutoffs."""
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
# 4. H1: Are index moves larger on days when Big firms report?
# ---------------------------------------------------------------------------
h1 = data.groupby("day_type")["abs_ret"].agg(["count", "mean"])
big_days = data.loc[data["day_type"] == "Big", "abs_ret"]
small_days = data.loc[data["day_type"] == "Small only", "abs_ret"]
t_stat, p_val = stats.ttest_ind(big_days, small_days, equal_var=False)

print("\nH1: average absolute index return (%) by day type")
print(h1.round(3))
print(f"Big vs Small-only: t = {t_stat:.2f}, p = {p_val:.4f}")


# ---------------------------------------------------------------------------
# 5. H2: Do earnings features predict index direction?
# ---------------------------------------------------------------------------
# Only days with at least one report. Split by time, not at random,
# so the model is never trained on days that come after the test days.
event_days = data[data["day_type"] != "None"]
cut = int(len(event_days) * TRAIN_SHARE)
train, test = event_days.iloc[:cut], event_days.iloc[cut:]

# Baseline: always guess the most common direction in the training data.
# The index rises on more than half of all days, so 0.5 is too easy a bar.
majority = train["up"].mode()[0]
baseline_acc = (test["up"] == majority).mean()

# Logistic regression, same formula style as glm(..., family = binomial) in R
big = "big_n + big_weight + big_surprise"
small = "small_n + small_weight + small_surprise"
models = {"Big only": big, "Small only": small, "Both": big + " + " + small}

results = []
for name, features in models.items():
    model = smf.logit("up ~ " + features, data=train).fit(disp=0)
    pred = (model.predict(test) > 0.5).astype(int)   # predict up if P(up) > 0.5

    correct = int((pred == test["up"]).sum())
    acc = correct / len(test)
    # One-sided binomial test: is accuracy above the baseline accuracy?
    p = stats.binomtest(correct, len(test), baseline_acc,
                        alternative="greater").pvalue

    results.append({"model": name, "accuracy": acc,
                    "baseline": baseline_acc, "p_value": p,
                    "share_pred_up": pred.mean()})

results = pd.DataFrame(results)
print("\nH2: out-of-sample accuracy vs always-guess-majority baseline")
print(results.round(3).to_string(index=False))


# ---------------------------------------------------------------------------
# 6. H3: Do Big misses move the index more than Big beats?
# ---------------------------------------------------------------------------
# Regression on all earnings days (same formula style as lm() in R):
#   index return (%) ~ Big beats + Big misses + Small beats + Small misses
# Beats are >= 0 and misses are <= 0, so both slopes should be positive.
# H3 says the Big miss slope is larger than the Big beat slope.
# Small beats and misses are included as controls.
# HC1 = standard errors that are robust to uneven variance across days.
H3_FORMULA = "ret_pct ~ big_pos + big_neg + small_pos + small_neg"


def miss_minus_beat(model):
    """One-sided test that the Big miss slope is larger than the Big beat slope.
    The t-test gives a two-sided p-value, so it is halved."""
    diff = model.t_test("big_neg - big_pos = 0")
    t = diff.tvalue.item()
    p_two = diff.pvalue.item()
    p_one = p_two / 2 if t > 0 else 1 - p_two / 2
    return diff.effect.item(), t, p_one


h3 = smf.ols(H3_FORMULA, data=event_days).fit(cov_type="HC1")
print("\nH3: index return on weighted beats and misses")
print(h3.summary().tables[1])

effect, t, p_h3 = miss_minus_beat(h3)
print(f"Big miss slope - Big beat slope = {effect:.3f}, "
      f"t = {t:.2f}, one-sided p = {p_h3:.4f}")

h3_table = pd.DataFrame({"coef": h3.params, "std_err": h3.bse,
                         "p_value": h3.pvalues})

# Robustness 1: how many days actually have a Big miss?
big_miss_days = event_days[event_days["big_neg"] < 0]
print(f"\nDays with at least one Big miss: {len(big_miss_days)}")
print("Index return (%) on those days:")
print(big_miss_days["ret_pct"].describe().round(2))

# Robustness 2: drop the two crisis years (2008 and 2020) and re-run H3
calm = event_days[~event_days.index.year.isin([2008, 2020])]
h3_calm = smf.ols(H3_FORMULA, data=calm).fit(cov_type="HC1")
effect_c, t_c, p_c = miss_minus_beat(h3_calm)
print("\nH3 without 2008 and 2020:")
print(h3_calm.params[["big_pos", "big_neg"]].round(3).to_string())
print(f"Big miss slope - Big beat slope = {effect_c:.3f}, "
      f"t = {t_c:.2f}, one-sided p = {p_c:.4f}")


# ---------------------------------------------------------------------------
# 7. Extension: H1 and H3 again, with time-series econometrics
#    (Wooldridge, Introductory Econometrics, Ch. 7, 10 and 12)
# ---------------------------------------------------------------------------
# Problems with the tests above, and the fix used for each:
#  a) Daily errors can be correlated across days (assumption TS.5').
#     Fix: Newey-West (HAC) standard errors.
#  b) Market conditions are left out, so the error is related to the
#     regressors (assumption TS.3'). Fix: control for yesterday's VIX and add
#     year dummies, C(year), which works like factor(year) in R.
#  c) The Big slopes may be different in crisis years. Fix: a Chow-style
#     test that lets the Big slopes change in 2008 and 2020.
#  d) The 1% cutoff is arbitrary. Fix: repeat with 0.5% and 2%.
# H1 becomes a regression on a Big-day dummy. Without controls this gives the
# same difference in means as the t-test, but now HAC errors and controls fit.

CONTROLS = " + vix_lag + C(year)"


def fit_hac(formula, df):
    """OLS with Newey-West standard errors. Lag length uses Wooldridge's
    rule of thumb: integer part of 4 x (n / 100) ^ (2 / 9)."""
    lags = int(4 * (len(df) / 100) ** (2 / 9))
    return smf.ols(formula, data=df).fit(cov_type="HAC",
                                         cov_kwds={"maxlags": lags})


def run_extension(df, cutoff, specs):
    """H1 (Big-day effect) and H3 (miss minus beat) for each specification."""
    ev = df[df["day_type"] != "None"].copy()
    ev["big_day"] = (ev["day_type"] == "Big").astype(int)
    rows = []
    for spec, extra in specs:
        m1 = fit_hac("abs_ret ~ big_day" + extra, ev)
        rows.append({"test": "H1: Big day effect on |return|", "spec": spec,
                     "cutoff": cutoff, "estimate": m1.params["big_day"],
                     "t": m1.tvalues["big_day"], "p": m1.pvalues["big_day"],
                     "n": int(m1.nobs)})
        m3 = fit_hac(H3_FORMULA + extra, ev)
        effect, t, p = miss_minus_beat(m3)
        rows.append({"test": "H3: Big miss - Big beat", "spec": spec,
                     "cutoff": cutoff, "estimate": effect, "t": t, "p": p,
                     "n": int(m3.nobs)})
    return rows


# a) and b): HAC errors alone, then HAC errors plus market controls
ext = run_extension(data, BIG_CUTOFF,
                    [("HAC", ""), ("HAC + VIX + year", CONTROLS)])

# c) Chow-style test. big_neg:crisis is the extra Big miss slope in 2008 and
# 2020. The F-test asks whether both Big slopes change in those years.
ev = event_days.copy()
ev["crisis"] = ev.index.year.isin([2008, 2020]).astype(int)
chow = fit_hac(H3_FORMULA + " + big_pos:crisis + big_neg:crisis" + CONTROLS, ev)
chow_f = chow.f_test("big_pos:crisis = 0, big_neg:crisis = 0")
effect, t, p = miss_minus_beat(chow)   # miss minus beat in normal years
ext.append({"test": "H3: Big miss - Big beat, normal years",
            "spec": "HAC + VIX + year + crisis", "cutoff": BIG_CUTOFF,
            "estimate": effect, "t": t, "p": p, "n": int(chow.nobs)})
ext.append({"test": "Chow: extra Big miss slope in crisis",
            "spec": "HAC + VIX + year + crisis", "cutoff": BIG_CUTOFF,
            "estimate": chow.params["big_neg:crisis"],
            "t": chow.tvalues["big_neg:crisis"],
            "p": float(chow_f.pvalue), "n": int(chow.nobs)})

# d) Other cutoffs, full specification only
for cutoff in [0.005, 0.02]:
    ext += run_extension(build_daily(cutoff), cutoff,
                         [("HAC + VIX + year", CONTROLS)])

ext = pd.DataFrame(ext)
print("\nExtension: H1 and H3 with Newey-West errors and market controls")
print("(p is two-sided for H1, one-sided for H3, joint F-test for Chow)")
print(ext.round(3).to_string(index=False))


# ---------------------------------------------------------------------------
# Save results and chart
# ---------------------------------------------------------------------------
h1.to_csv("output/h1_abs_return_by_day_type.csv")
results.to_csv("output/h2_prediction_results.csv", index=False)
h3_table.to_csv("output/h3_beat_vs_miss_regression.csv")
ext.to_csv("output/extension_results.csv", index=False)

# Chart: how each estimate changes from the first pass to the corrected tests


def estimate_bars(ax, labels, estimates, t_values, title, ylabel):
    """Bars with 95% confidence intervals (standard error = estimate / t)."""
    se = [abs(e / t) for e, t in zip(estimates, t_values)]
    ax.bar(labels, estimates, yerr=[1.96 * s for s in se], capsize=5,
           color="tab:blue")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelsize=8)


def ext_row(test, spec, cutoff=BIG_CUTOFF):
    """Estimate and t-statistic of one row of the extension table."""
    row = ext[(ext["test"] == test) & (ext["spec"] == spec)
              & (ext["cutoff"] == cutoff)].iloc[0]
    return row["estimate"], row["t"]


H1_TEST, H3_TEST = "H1: Big day effect on |return|", "H3: Big miss - Big beat"
FULL = "HAC + VIX + year"

h1_rows = [(big_days.mean() - small_days.mean(), t_stat),
           ext_row(H1_TEST, "HAC"), ext_row(H1_TEST, FULL),
           ext_row(H1_TEST, FULL, 0.005), ext_row(H1_TEST, FULL, 0.02)]
h3_rows = [miss_minus_beat(h3)[:2], miss_minus_beat(h3_calm)[:2],
           ext_row(H3_TEST, FULL),
           ext_row("H3: Big miss - Big beat, normal years",
                   "HAC + VIX + year + crisis")]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.8))

estimate_bars(ax1, ["t-test", "Newey-West", "+ VIX\n+ year", "0.5%\ncutoff",
                    "2%\ncutoff"],
              [e for e, _ in h1_rows], [t for _, t in h1_rows],
              "(a) H1: Big-day effect", "Extra absolute return (pp)")

ax2.bar(results["model"], results["accuracy"], color="tab:blue")
ax2.axhline(baseline_acc, color="black", linestyle="--",
            label=f"Always guess majority ({baseline_acc:.3f})")
ax2.set_ylim(0.45, 0.60)
ax2.set_title("(b) H2: Out-of-sample accuracy")
ax2.set_ylabel("Accuracy")
ax2.legend(loc="upper left")

estimate_bars(ax3, ["First pass", "Without\n2008, 2020", "Newey-West\n+ VIX + year",
                    "Normal years\n(Chow)"],
              [e for e, _ in h3_rows], [t for _, t in h3_rows],
              "(c) H3: Big miss minus Big beat", "Difference in slopes")

plt.tight_layout()
plt.savefig("output/results.png", dpi=150)
print("\nDone. Results are in the output folder.")
