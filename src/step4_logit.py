#!/usr/bin/env python3
# Making the script directly executable from terminal

"""
==================================
Project Update 4: Logistic Regression
Submitted by: Nowshed Imran
Date: 11/21/2025
==================================

Purpose of Step 4
-----------------
This script study whether daily aggregate earnings activity of large (“Big” - 
more than 1% weight) and small (“Small” - less than 1% weight) S&P 500 firms 
helps to predict whether the S&P 500 index goes up or down on a given day.

It takes the firm samples and data prepared in Steps 1-3 and, for a range of
sampling designs, builds simple logistic regression models that forecast
index direction using only information from earnings announcement days.

Prerequisites (from previous steps)
-----------------------------------
Before running this script, the following scripts must be axecuted:

1. Step 1 (bootstrap): Setting root folder and creating folders to populate
2. Step 2 (weights): Calculating weight of each stocks and split in big and small group
3. Step 3 (select_sample): Select samples, small companies multiples and download data

Overview of this script
----------------------------------
This script takes the samples constrcted in Step 3 (for each
SMALL_MULTIPLIER m = 1, 2, ..., MAX_MULTIPLIER) and checks whether
daily earnings activity from Big and Small firms helps to predict
whether the S&P 500 index would go up or down.

For each sampling design m (how many Small firms are selected per Big firm):

1. Build daily Big/Small earnings features
   - Read sample_firms_m{m}.csv to identify which firms are "Big" and which are "Small",
     along with their index weights (w_latest).
   - For each firm in the sample, read its earnings file from
     data/raw/earnings/{ticker}.csv.
   - For each calendar date and each bucket (Big vs Small), compute:
       - Number of earnings reports on that date (n_reports).
       - Sum of index weights of the firms reporting (total_weight).
       - Average earnings surprise (%) among reporting firms (avg_surprise).
   - After aggregation, rename these to the daily features used later:
       big_n, small_n,
       big_weight, small_weight,
       big_surprise, small_surprise.

2. Construct the index return label
   - Load daily S&P 500 (^GSPC) OHLCV data from data/raw/ohlcv/^GSPC.csv.
   - Compute the daily percentage return of the close price.
   - Create a binary label for index direction:
       y = 1  if the daily return > 0  (index goes up),
       y = 0  otherwise                (index goes down or is flat).

3. Merge index returns with earnings features
   - Merge the daily S&P 500 returns with the Big/Small earnings features using
     the calendar date as the key.
   - For days where there are no Big or Small earnings in the sample, earnings
     features are filled with zeros (interpreted as “no earnings news from that
     bucket on that day”).
   - Define **event days** as dates where at least one earnings report occurs:
       big_n + small_n > 0.
   - All logistic regressions are estimated only on these event days.

4. Estimate three logistic regression models (for each m)
   - For the event days, estimate a logistic model of the form:
       P(index up on day t) = f(Big/Small earnings features on day t).
   - Use three different feature sets:

     1) "overall" model:
        Features = {big_n, small_n,
                    big_weight, small_weight,
                    big_surprise, small_surprise}

     2) "big_only" model:
        Features = {big_n, big_weight, big_surprise}

     3) "small_only" model:
        Features = {small_n, small_weight, small_surprise}

   - Data cleaning and model estimation:
       - Drop rows with missing or infinite values in any of the chosen features
         or in y.
       - If there are fewer than 50 valid event days for a given (m, model),
         skip that model as the sample is too small.
       - Split the remaining data into training (70%) and test (30%) sets using
         a stratified split to preserve the proportion of up vs down days.
       - Fit a standard sklearn LogisticRegression (with default regularization
         and max_iter=1000) on the training data.
       - Compute out-of-sample accuracy on the test set, defined as the fraction
         of test days where the predicted label matches y.

5. Collect and summarize results
   - For each combination of:
       - SMALL_MULTIPLIER m
       - model type includes {"overall", "big_only", "small_only"}
     the script records:
       - n_obs_total  = total number of valid event days used in the model.
       - n_test       = number of observations in the test set.
       - accuracy     = test-set classification accuracy.
   - All results are combined and saved to:
       data/processed/logit_index_summary.csv.

   - The script also creates a grouped bar chart showing test accuracy by
     SMALL_MULTIPLIER and model type and saves it as:
       data/processed/logit_index_accuracy.png.

     A horizontal line at accuracy = 0.5 is added as a reference “coin-flip”
     benchmark. This makes it easy to see whether any model for any multiplier
     delivers accuracy meaningfully above random guessing.

Disclaimer:
The code of this script particularly almost all of it was inititially from 
ChatGPT. It made multiple model error at first, and I have removed complex 
codes and fixed error as much as possible. As this is my first time learning
about Logistic Regression, I may have made some mistakes which I intend to 
fix on Update 5.
"""
### Importing libraries and Statements
# System libraries
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
# Dataframe, calculation and plot libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Logistic regression libraries
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Path safety to make the script reproducible.
# This code block is copied from ChatGPT.
def _detect_project_root() -> Path:
    # 1) If running as a script, prefer this
    if "__file__" in globals():
        here = Path(__file__).resolve()
        cand = here.parent.parent if here.parent.name == "src" else here.parent
        if (cand / "src").exists():
            return cand

    # 2) If in IPython, walk upward from current working dir to find 'src'
    p = Path.cwd().resolve()
    for _ in range(8):  # walk up to 8 levels just in case
        if (p / "src").exists():
            return p
        p = p.parent

    raise SystemExit("[error] Could not find project root containing 'src' folder. "
                     "Run this from your project folder or adjust the search.")

ROOT = _detect_project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Setting paths based on root folder
DIR_PROCESSED = ROOT / "data" / "processed"
DIR_OHLCV     = ROOT / "data" / "raw" / "ohlcv"
DIR_EARN      = ROOT / "data" / "raw" / "earnings"

INDEX_TIC     = "^GSPC"
SEED          = 5621  # random seed

# Using maximum multiples compared to big bucket companies
MAX_MULTIPLIER = 20   # change to anything 1–20

RESULTS_SUMMARY = DIR_PROCESSED / "logit_index_summary.csv"
RESULTS_PLOT    = DIR_PROCESSED / "logit_index_accuracy.png"

### Step 1: loading S&P 500 daily data and calculating retrun
def load_index_returns() -> pd.DataFrame:
    """
    Load ^GSPC daily OHLCV and compute daily returns and labels.

    Returns a DataFrame with columns:
        - date: datetime.date
        - ret:  daily return 
        - y:    1 if ret > 0 else 0 # for training model
    """
    path = DIR_OHLCV / f"{INDEX_TIC}.csv"
    if not path.exists():
        raise SystemExit(f"[error] Missing index OHLCV file: {path}")

    df = pd.read_csv(path)

    if "Date" not in df.columns or "Close" not in df.columns:
        raise SystemExit(f"[error] Index file {path} has no Date/Close columns.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    df = df.dropna(subset=["Date"]).sort_values("Date")

    if df.empty:
        raise SystemExit(f"[error] Index file {path} is empty after cleaning.")

    # Ensure one row per calendar date (just in case)
    df["date"] = df["Date"].dt.date
    df = df.sort_values("date")

    df = df.dropna(subset=["Close"])
    df = df[["date", "Close"]].drop_duplicates(subset=["date"], keep="last")

    df["ret"] = df["Close"].pct_change()
    df = df.dropna(subset=["ret"])

    df["y"] = (df["ret"] > 0).astype(int)

    out = df[["date", "ret", "y"]].copy()
    print(f"[info] Loaded index returns: {len(out)} daily observations.")
    return out

### Step 2: Aggregating earnings into daily Big/Small features
def build_earnings_features_for_multiplier(m: int) -> Optional[pd.DataFrame]:
    """
    For a given SMALL_MULTIPLIER m, read sample_firms_m{m}.csv to get the
    Big vs Small membership and firm weights. Then read per-firm earnings from
    data/raw/earnings/{ticker}.csv, aggregate by date & bucket, and pivot to
    daily Big/Small features.

    Returns a DataFrame with index 'date' and columns:
        - big_n,       small_n        (# of reports per day)
        - big_weight,  small_weight   (sum of w_latest)
        - big_surprise,small_surprise (average Surprise(%))
    or None if no valid earnings rows.
    """
    sample_path = DIR_PROCESSED / f"sample_firms_m{m}.csv"
    if not sample_path.exists():
        print(f"[warn] {sample_path} not found, skipping earnings aggregation.")
        return None

    sample_df = pd.read_csv(sample_path)
    if "ticker" not in sample_df.columns:
        print(f"[warn] 'ticker' missing in {sample_path}, skipping earnings aggregation.")
        return None

    # Clean basic fields
    sample_df["ticker"] = sample_df["ticker"].astype(str).str.upper().str.strip()
    sample_df["bucket"] = sample_df.get("bucket", "NA").astype(str)
    if "w_latest" not in sample_df.columns:
        print(f"[warn] 'w_latest' missing in {sample_path}, skipping earnings aggregation.")
        return None

    # Only keep Big & Small buckets
    firms = sample_df[["ticker", "bucket", "w_latest"]].drop_duplicates()
    firms = firms[firms["bucket"].isin(["Big", "Small"])].copy()

    if firms.empty:
        print(f"[warn] No Big/Small firms in {sample_path}, skipping earnings aggregation.")
        return None

    earnings_rows: List[pd.DataFrame] = []

    for firm in firms.itertuples(index=False):
        tic = firm.ticker
        bucket = firm.bucket
        weight = firm.w_latest

        earn_path = DIR_EARN / f"{tic}.csv"
        if not earn_path.exists():
            # Some stocks mightn't have earnings file or yfinance is down
            continue

        try:
            earn_df = pd.read_csv(earn_path)
        except Exception as e:
            print(f"[warn] Could not read earnings for {tic}: {e}")
            continue

        # For troubleshooting if a file is corrupted
        if "EarningsDate" not in earn_df.columns:
            print(f"[warn] Earnings file for {tic} has no 'EarningsDate', skipping.")
            continue

        # Parse date, drop invalid
        earn_df["EarningsDate"] = pd.to_datetime(
            earn_df["EarningsDate"], errors="coerce", utc=True
        )
        earn_df = earn_df.dropna(subset=["EarningsDate"])

        if earn_df.empty:
            continue

        earn_df["date"] = earn_df["EarningsDate"].dt.date

        # Surprise(%) column may or may not exist
        if "Surprise(%)" in earn_df.columns:
            surprise = earn_df["Surprise(%)"].astype(float)
        else:
            surprise = pd.Series(np.nan, index=earn_df.index)

        # One row per firm-report
        tmp = pd.DataFrame(
            {
                "date":     earn_df["date"].values,
                "bucket":   bucket,
                "w_latest": float(weight),
                "surprise": surprise.values,
            }
        )
        earnings_rows.append(tmp)

    if not earnings_rows:
        print(f"[warn] No earnings rows constructed for m={m}.")
        return None

    all_earn = pd.concat(earnings_rows, axis=0, ignore_index=True)

    # Aggregate by date & bucket
    agg = (
        all_earn
        .groupby(["date", "bucket"], as_index=False)
        .agg(
            n_reports=("w_latest", "count"),
            total_weight=("w_latest", "sum"),
            avg_surprise=("surprise", "mean"),
        )
    )

    # Pivot to wide: columns of form (metric, bucket)
    wide = agg.pivot(index="date", columns="bucket")

    # Flatten MultiIndex columns: (metric, bucket) -> "metric_bucket"
    wide.columns = [
        f"{metric}_{bucket.lower()}"
        for (metric, bucket) in wide.columns.to_flat_index()
    ]

    wide = wide.reset_index()

    # Ensure the key columns exist, else create them with zeros / NaNs
    for col in [
        "n_reports_big", "n_reports_small",
        "total_weight_big", "total_weight_small",
        "avg_surprise_big", "avg_surprise_small",
    ]:
        if col not in wide.columns:
            wide[col] = np.nan

    # Rename colums for better readability
    wide = wide.rename(
        columns={
            "n_reports_big":      "big_n",
            "n_reports_small":    "small_n",
            "total_weight_big":   "big_weight",
            "total_weight_small": "small_weight",
            "avg_surprise_big":   "big_surprise",
            "avg_surprise_small": "small_surprise",
        }
    )

    # Fill counts & weights with 0 when missing; surprises with 0 (neutral)
    for c in ["big_n", "small_n", "big_weight", "small_weight"]:
        wide[c] = wide[c].fillna(0.0)

    for c in ["big_surprise", "small_surprise"]:
        wide[c] = wide[c].fillna(0.0)

    print(f"[info] Earnings features for m={m}: {len(wide)} distinct days.")
    return wide

### Step 3: Logistic Regression
def fit_logit_for_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    model_name: str,
    m: int
) -> Optional[Dict[str, Any]]:
    """
    Fit a logistic regression using the specified feature columns and
    return accuracy + counts, or None if not enough data.
    """
    # Keep only needed columns + label
    cols_needed = feature_cols + ["y"]
    tmp = df[cols_needed].copy()

    # Clean NaNs/Infs in X
    X = tmp[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = tmp["y"].astype(int)

    mask_valid = ~X.isna().any(axis=1)
    X = X.loc[mask_valid]
    y = y.loc[mask_valid]

    n_all = len(X)
    if n_all < 50:
        # Too few event days for a meaningful split
        print(
            f"[warn] m={m}, model='{model_name}': "
            f"only {n_all} valid event days, skipping."
        )
        return None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=SEED,
            stratify=y,
        )
    except ValueError as e:
        print(f"[warn] m={m}, model='{model_name}': train_test_split failed: {e}")
        return None

    model = LogisticRegression(max_iter=1000, random_state=SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(
        f"[info] m={m}, model='{model_name}': "
        f"n_all={n_all}, n_test={len(y_test)}, accuracy={acc:.3f}"
    )

    return {
        "small_multiplier": m,
        "model": model_name,
        "n_obs_total": int(n_all),
        "n_test": int(len(y_test)),
        "accuracy": float(acc),
    }

### Step 4: Binding all small functions to one main fuction
def main():
    print("=== Step 4: Logistic Regression of Index on Earnings (Big vs Small) ===")
    print(f"[info] MAX_MULTIPLIER = {MAX_MULTIPLIER}")

    # Load index returns once
    idx_df = load_index_returns()

    results_rows: List[Dict[str, Any]] = []

    for m in range(1, MAX_MULTIPLIER + 1):
        print("\n" + "=" * 60)
        print(f"[info] SMALL_MULTIPLIER = {m}")
        print("=" * 60)

        earn_feat = build_earnings_features_for_multiplier(m)
        if earn_feat is None or earn_feat.empty:
            print(f"[warn] No earnings features for m={m}, skipping this multiplier.")
            continue

        # Merge with index returns on 'date'
        merged = idx_df.merge(earn_feat, on="date", how="left")

        # Fill missing earnings features with zeros (no event those days)
        for c in [
            "big_n", "small_n",
            "big_weight", "small_weight",
            "big_surprise", "small_surprise",
        ]:
            if c not in merged.columns:
                merged[c] = 0.0
            else:
                merged[c] = merged[c].fillna(0.0)

        # Restrict to "event days" where at least one earnings occurs
        merged["total_n"] = merged["big_n"] + merged["small_n"]
        event_df = merged[merged["total_n"] > 0].copy()

        if event_df.empty:
            print(f"[warn] No event days (earnings) for m={m}, skipping.")
            continue

        print(f"[info] m={m}: {len(event_df)} event days with at least one earnings.")

        # -----------------------------------------------------
        # Define feature sets for three models:
        #   overall   : Big + Small features
        #   big_only  : Big-only features
        #   small_only: Small-only features
        # -----------------------------------------------------
        features_overall = [
            "big_n", "small_n",
            "big_weight", "small_weight",
            "big_surprise", "small_surprise",
        ]
        features_big_only = ["big_n", "big_weight", "big_surprise"]
        features_small_only = ["small_n", "small_weight", "small_surprise"]

        # Fit each model and collect results
        for model_name, fcols in [
            ("overall",   features_overall),
            ("big_only",  features_big_only),
            ("small_only", features_small_only),
        ]:
            res = fit_logit_for_model(event_df, fcols, model_name, m)
            if res is not None:
                results_rows.append(res)

    # --------- Save summary + grouped bar chart ----------
    if not results_rows:
        print("\n[warn] No models were successfully estimated.")
        print("=== Step 4 Complete ===")
        return

    summary_df = pd.DataFrame(results_rows)
    summary_df = summary_df.sort_values(["small_multiplier", "model"]).reset_index(drop=True)
    summary_df.to_csv(RESULTS_SUMMARY, index=False)

    print(f"\n[write] {RESULTS_SUMMARY} (rows={len(summary_df)})")
    print("\n=== Logistic Regression Accuracy by SMALL_MULTIPLIER and Model ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))

### Step 5: Chart for visualization
    try:
        pivot_df = summary_df.pivot(
            index="small_multiplier",
            columns="model",
            values="accuracy"
        ).sort_index()

        # Ensure columns exist
        for col in ["overall", "big_only", "small_only"]:
            if col not in pivot_df.columns:
                pivot_df[col] = np.nan

        plt.figure(figsize=(10, 6))

        x = np.arange(len(pivot_df.index))  # multipliers
        width = 0.25                        # width of each bar

        plt.bar(x - width, pivot_df["overall"],   width, label="Overall",   color="#1f77b4")
        plt.bar(x,         pivot_df["big_only"],  width, label="Big-only",  color="#ff7f0e")
        plt.bar(x + width, pivot_df["small_only"], width, label="Small-only", color="#2ca02c")

        # Baseline at 0.5 (random guessing)
        plt.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Baseline (0.5)")

        plt.xticks(x, pivot_df.index)
        plt.xlabel("SMALL_MULTIPLIER (m)")
        plt.ylabel("Accuracy")
        plt.title("Logistic Regression: Index Up/Down vs Big/Small Earnings Activity")
        plt.ylim(0.0, 1.0)
        plt.grid(axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plt.savefig(RESULTS_PLOT, dpi=150)
        print(f"[write] {RESULTS_PLOT}")
        plt.show()

    except Exception as e:
        print(f"[warn] Could not plot grouped bar chart: {e}")

    print("=== Step 4 Complete ===")

if __name__ == "__main__":
    main()