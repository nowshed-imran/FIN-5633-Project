#!/usr/bin/env python3
# Making the script directly executable from terminal

"""
==================================
Project Update 3
Submitted by: Nowshed Imran
Date: 10/25/2025
==================================
Important Note:
1. Execute Step 1 (bootstrap) to ensure the folder structre exists.
2. Execute Step 2 (weights) to make sure the market-cap weights file is ready.

Goal of this script:
1. Selects "Big" firms:
   - By default, includes ALL Big firms.
   - Optionally, user can limit how many Big firms to randomly pick.
2. Determines the target number of "Small" firms:
   - target Small count = SMALL_MULTIPLIER * (number of selected Big firms).
   - The script loops SMALL_MULTIPLIER from 1 up to MAX_MULTIPLIER (capped at 20),
     so we obtain multiple sampling designs (m = 1, 2, ..., MAX_MULTIPLIER).
3. Selects "Small" firms by sector (for each multiplier m):
   - Equal number (x) from each sector.
   - Remainder (y) from sectors that still have unused Small firms.
4. Saves sample membership for each multiplier:
   - For each SMALL_MULTIPLIER = m, combines the selected Big and Small firms
     into a single DataFrame with columns: ticker, w_latest, bucket, sector
   - Writes:
      data/processed/sample_firms_m{m}.csv  (firm-level sample)
      data/processed/sample_meta_m{m}.csv   (metadata: seed, n_big, n_small, etc.)
5. Downloads OHLCV price data (up to YEARS years) for ^GSPC and all sampled firms:
   - Sample length is 20 years but it can be modified
   - Based on each multiplier (m) OHLCV data gets downloaded
   - Skip download if it is downloaded previously
6. Downloads quarterly earnings dates and surprises (up to EARNINGS_LIMIT):
   - Download earning information for each ticker upto 20 years
   - Skip download if it is downloaded previously

Limitations:
I am not good enough coder to write this project. I have figured out the project
and logics first. How many small companies to select for training is my own, and
it can be flawed. I would love some feedback on that. I have tried coding first,
and used ChatGPT to solve errors and ensure robustness. The code I have directly 
copied in mentioned in the comment.
"""

# Importing statements and libraries
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

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

# Paths & constants. ChatGPT helped with the path
CSV_WEIGHTS = ROOT / "data" / "processed" / "weights_latest.csv"
CSV_SAMPLE  = ROOT / "data" / "processed" / "sample_firms.csv"
DIR_OHLCV   = ROOT / "data" / "raw" / "ohlcv"
DIR_EARN    = ROOT / "data" / "raw" / "earnings"

INDEX_TIC = "^GSPC"            # S&P 500 Index
YEARS = 20                     # 20 years of data to train model
SAMPLE_LEN = f"{YEARS}y"       # To dowload S&P 500 Data
SEED = 5621                    # Random seed for reproducibility

### User Override
BIG_OVERRIDE = None             # Upto 14 is allowed
SMALL_MULTIPLIER = 3            # Small = SMALL_MULTIPLIER × Big (max 20)
EARNINGS_LIMIT = YEARS * 4      # Four Earning Call each year

# Modifying how many multiple of big companies to keep for logistic regression training
MAX_MULTIPLIER = 20   # change to anything 1–20
MAX_MULTIPLIER = min(max(1, int(MAX_MULTIPLIER)), 20)

# Make sure folders exist for outputs
DIR_OHLCV.mkdir(parents=True, exist_ok=True)
DIR_EARN.mkdir(parents=True, exist_ok=True)

# User input sanity check 
SMALL_MULTIPLIER = min(max(1, int(SMALL_MULTIPLIER)), 20)
EARNINGS_LIMIT   = min(max(4, int(EARNINGS_LIMIT)), 32)

### Step 1: Load Weight files
if not CSV_WEIGHTS.exists():
    raise SystemExit("[error] Missing weights file even after Step 2.")

# Read weights file (produced in Step 2 script)
dfw = pd.read_csv(CSV_WEIGHTS)
dfw["ticker"] = dfw["ticker"].astype(str).str.upper().str.strip()
if "sector" not in dfw.columns:
    dfw["sector"] = "NA"

rng = np.random.default_rng(SEED)

### Step 2: Select "Big" firms
big_all = dfw[dfw["bucket"] == "Big"].copy()

# By defualt, use all the big frims unless user overrides. ChatGPT helped here.
if BIG_OVERRIDE is None or BIG_OVERRIDE <= 0 or BIG_OVERRIDE >= len(big_all):
    big_df = big_all.reset_index(drop=True)
    n_big = len(big_df)
    print(f"[info] Big: using ALL ({n_big})")
else:
    # random subset of Big firms (reproducible)
    big_df = (big_all.sample(n=BIG_OVERRIDE,
                            random_state=int(rng.integers(0, 2**32 - 1)))
                            .reset_index(drop=True))
    n_big = len(big_df)
    print(f"[info] Big: OVERRIDE to {n_big} random firms")

### Step 3: Select Small firms by sector and based on multiples
# MAIN LOOP: repeat Step 3–5 for SMALL_MULTIPLIER = 1,...,10. Upto 20 can be done.
for SMALL_MULTIPLIER in range(1, MAX_MULTIPLIER + 1):
    print("\n" + "=" * 50)
    print(f"[info] SMALL_MULTIPLIER = {SMALL_MULTIPLIER}")
    print("=" * 50)
    # select small firm by sector
    small_pool = dfw[(dfw["bucket"] == "Small") & dfw["sector"].notna()].copy()
    small_pool["sector"] = small_pool["sector"].astype(str).replace("", "NA")
    print("[info] Step 3 started for multiplier", SMALL_MULTIPLIER)

    # Calculate how many Small firm to pick
    n_small_target = min(SMALL_MULTIPLIER * n_big, len(small_pool))
    print(f"[info] Small target = {SMALL_MULTIPLIER} * {n_big} = {SMALL_MULTIPLIER*n_big} "
        f"(capped at available: {n_small_target})")

    if n_small_target == 0:
        small_df = small_pool.head(0).copy()
    else:
        # Figure out how many sectors in the S&P 500. 
        sectors = sorted(small_pool["sector"].unique().tolist())
        k = len(sectors)

        # Compute equal companies for each sector (x) and remainder (y)
        x = (n_small_target // k) if k > 0 else 0
        y = n_small_target - (k * x) if k > 0 else 0
        print(f"[info] Equal-per-sector x = {x}, remainder y = {y}, sectors = {k}")

        picked_list = []
        # Store remaining companies so that they are not picked again
        leftover = {}

        # Selecting equal number of comapnies  from each sector. ChatGPT helped here.
        for s in sectors:
            grp = small_pool[small_pool["sector"] == s]
            take = min(x, len(grp))
            if take > 0:
                first_take = grp.sample(n=take,
                                        random_state=int(rng.integers(0, 2**32 - 1)))
                picked_list.append(first_take)
                # remove picked from this sector to form a leftover pool
                remaining = grp.loc[~grp["ticker"].isin(first_take["ticker"])]
            else:
                first_take = grp.head(0)
                remaining = grp  # nothing picked yet
            leftover[s] = remaining

        # Combine all equal selections
        picked = (pd.concat(picked_list, axis=0).reset_index(drop=True)
                if picked_list else small_pool.head(0).copy())
        
        # Pick y extra stocks from sectors that still have capacity. ChatGPT helped here.
        rem = y
        while rem > 0:
            progressed = False
            for s in sectors:
                if rem <= 0:
                    break
                pool_s = leftover.get(s, small_pool.head(0))
                if not pool_s.empty:
                    # take 1 from this sector
                    one = pool_s.sample(n=1,
                                        random_state=int(rng.integers(0, 2**32 - 1)))
                    picked = pd.concat([picked, one], axis=0, ignore_index=True)
                    # update leftover (remove the selected stocks)
                    pool_s = pool_s.loc[~pool_s["ticker"].isin(one["ticker"])]
                    leftover[s] = pool_s
                    rem -= 1
                    progressed = True
            if not progressed:
                # no sector had remaining capacity
                break

        small_df = picked.reset_index(drop=True)

    

    ### Save sample firms and print summary
    keep_cols = ["ticker", "w_latest", "bucket", "sector"]
    sample_df = pd.concat([big_df[keep_cols], small_df[keep_cols]],
                        axis=0).reset_index(drop=True)

    # Build file names that depend on SMALL_MULTIPLIER
    csv_sample_path = ROOT / "data" / "processed" / f"sample_firms_m{SMALL_MULTIPLIER}.csv"
    meta_path       = ROOT / "data" / "processed" / f"sample_meta_m{SMALL_MULTIPLIER}.csv"

    # Writng the output to CSV
    sample_df.to_csv(csv_sample_path, index=False)
    print(f"[write] {csv_sample_path}  (rows={len(sample_df)}, "
        f"Big={len(big_df)}, Small={len(small_df)})")

    # Writing the output to CSV
    meta = pd.DataFrame([{
        "seed": SEED,
        "big_override": BIG_OVERRIDE,
        "small_multiplier": SMALL_MULTIPLIER,
        "n_big": len(big_df),
        "n_small": len(small_df),
        "sample_len": SAMPLE_LEN,
    }])
    meta.to_csv(meta_path, index=False)
    print(f"[write] {meta_path} (metadata for this sample)")

    ### Download 1-year OHLCV and earnings data
    # Download ^GSPC index. This block of code is from ChatGPT.
    idx_hist = yf.Ticker(INDEX_TIC).history(SAMPLE_LEN)
    if not idx_hist.empty:
        (idx_hist.reset_index()
                .rename(columns=str.title)
                .to_csv(DIR_OHLCV / f"{INDEX_TIC}.csv", index=False))
        print(f"[write] OHLCV: {INDEX_TIC}.csv ({len(idx_hist)})")
    else:
        print(f"[warn] Empty index data for {INDEX_TIC}")

    # Download data for all selected firms. ChatGPT helped here a lot. Still couldn't
    # figure out to check for existing data and skiping if correct information is 
    # already available.
    for tic in sample_df["ticker"]:
        # checking if the data file already exist or not
        price_path = DIR_OHLCV / f"{tic}.csv"
        if price_path.exists():
            print(f"[skip] OHLCV already exists for {tic}, skipping download.")
        else:
            try:
                prc = yf.Ticker(tic).history(SAMPLE_LEN)
                if not prc.empty:
                    (prc.reset_index()
                        .rename(columns=str.title)
                        .to_csv(price_path, index=False))
                    print(f"[write] OHLCV: {tic}.csv ({len(prc)})")
                else:
                    print(f"[warn] Empty OHLCV for {tic}")
            except Exception as e:
                print(f"[error] OHLCV {tic}: {e}")

        # Earning (quarterly with timestamp ONLY)
        earn_path = DIR_EARN / f"{tic}.csv"

        # Check if earnings already exist or not
        if earn_path.exists():
            print(f"[skip] Earnings already exists for {tic}, skipping download.")
        else:
            try:
                tk = yf.Ticker(tic)
                earn_q = tk.get_earnings_dates(limit=EARNINGS_LIMIT)

                # If no valid quarterly table, skip
                if not (isinstance(earn_q, pd.DataFrame) and not earn_q.empty):
                    print(f"[warn] Skipping earnings for {tic}: no quarterly table available.")
                else:
                    earn_q = earn_q.reset_index()

                    # Standardize earnings date column name
                    if "index" in earn_q.columns:
                        earn_q = earn_q.rename(columns={"index": "EarningsDate"})
                    elif "Earnings Date" in earn_q.columns:
                        earn_q = earn_q.rename(columns={"Earnings Date": "EarningsDate"})

                    # Enforce timestamp and keep only needed columns
                    if "EarningsDate" in earn_q.columns:
                        earn_q["EarningsDate"] = pd.to_datetime(
                            earn_q["EarningsDate"], errors="coerce"
                        )
                        earn_q = earn_q.dropna(subset=["EarningsDate"])

                        keep = [
                            c
                            for c in [
                                "EarningsDate",
                                "EPS Estimate",
                                "Reported EPS",
                                "Surprise(%)",
                                "Quarter",
                            ]
                            if c in earn_q.columns
                        ]
                        earn_q = earn_q[keep]

                        if not earn_q.empty:
                            earn_q.to_csv(earn_path, index=False)
                            print(
                                f"[write] Earnings (quarterly): {tic}.csv ({len(earn_q)})"
                            )
                        else:
                            print(
                                f"[warn] Skipping earnings for {tic}: "
                                "required columns missing after cleaning."
                            )
                    else:
                        print(
                            f"[warn] Skipping earnings for {tic}: "
                            "no timestamp column in get_earnings_dates()."
                        )

            except Exception as e:
                print(f"[error] Earnings {tic}: {e}")

# Script closing message
print("[info] Step 3 complete.")