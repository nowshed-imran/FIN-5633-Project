#!/usr/bin/env python3
# Making the script directly executable from terminal

"""
==================================
Project Update 2: Step 2
Submitted by: Nowshed Imran
Date: 10/03/2025
==================================
Important Note:
1. html5lib package is a requirement to run this script.
2. "[info] tickers to fetch: 503" may look stuck but it is downloading data.
    Improvements will be made here.
3. S&P 500 has 503 companies but some company info gets stuck while 
    dowloading. In that case a few less companies will be used for
    weight calculation

Goal:
1. Makes sure S&P 500 tickers and sector data by web scraping Wikipedia
2. Pull shares outstanding and current price via yfinance.
3. Compute market cap and normalized weights
4. Classify big and small companies based on weights
5. Save results to data/processed/weights_latest.csv

I have tried to code by myself, and gotten help with ChatGPT to solve error
or refine codes. The codes I have directly copied has comment with it.
"""

# Importing statements and libraries
import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from urllib.request import Request, urlopen # Need for data scraping
from pathlib import Path 

# Declaring constants
USER_AGENT = "Mozilla/5.0"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CSV_TICKERS = "data/raw/sp500_tickers.csv"
CSV_WEIGHTS = "data/processed/weights_latest.csv"
PROGRESS_EVERY = 25       # helps to get 25 company update
REQUEST_TIMEOUT = 30      # in seconds

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

os.chdir(ROOT)  # ensure all relative paths like data/... work
print(f"[info] Working directory set to: {ROOT}")

from src.step1_bootstrap import main as bootstrap_main

# This function is totally ChatGPT. I don't understand it yet.
# It took me a whole day just to get it working.
def sp500_from_wikipedia(csv_path: str) -> None:
    """
    Pull current S&P 500 tickers by scraping Wikipedia, clean them to Yahoo 
    style, and save to CSV at csv_path.
    """
    try:
        # use a browsery User-Agent for avoiding website block
        req = Request(WIKI_URL, headers={"User-Agent": USER_AGENT})
        html = urlopen(req, timeout=REQUEST_TIMEOUT).read()
        # NOTE: requires html5lib
        tables = pd.read_html(html, flavor="html5lib")
    except ImportError:
        raise SystemExit("Missing dependency: pip install html5lib")
    except Exception as e:
        raise SystemExit(f"Could not download S&P 500 tickers: {e}")

    df = next(t for t in tables if 'Symbol' in t.columns)      # find the table contains the constituents
    # Pulling ticker and sector data
    out = pd.DataFrame({
        "ticker": (df["Symbol"].astype(str)
                   .str.strip().str.upper().str.replace(".", "-", regex=False)),
        "sector": df.get("GICS Sector", pd.Series(["NA"] * len(df))).astype(str).str.strip(),
    })
    out = (out.loc[out["ticker"] != ""]
              .drop_duplicates(subset=["ticker"])
              .sort_values("ticker")
              .reset_index(drop=True))
    
    out.to_csv(csv_path, index=False)
    print(f"[write] refreshed S&P500 stock tickers with sector: {csv_path} "
          f"({len(out)} tickers)")
    
    if out["sector"].astype(str).str.upper().eq("NA").all():
        print("[warn] Wikipedia did not provide GICS Sector; sectors will be NA.")

def load_or_refresh_sp500(csv_path: str) -> pd.DataFrame:
    """
    Load tickers from CSV. If file missing or effectively empty (no rows),
    refresh from Wikipedia and load again. Returns a single 'ticker' column.
    """
    need_refresh = False
    if not os.path.exists(csv_path):
        need_refresh = True
    else:
        try:
            tmp = pd.read_csv(csv_path)
            if ("ticker" not in tmp.columns or tmp.empty
                or "sector" not in tmp.columns
                or tmp["sector"].isna().all()
                or (tmp["sector"].astype(str).str.upper() == "NA").all()):
                need_refresh = True
        except Exception:
            need_refresh = True

    if need_refresh:
        sp500_from_wikipedia(csv_path)

    df = pd.read_csv(csv_path)
    df["ticker"] = (df["ticker"].astype(str)
                    .str.strip().str.upper()
                    .str.replace(".", "-", regex=False))
    df["sector"] = df.get("sector", "NA")

    df = (df.loc[df["ticker"] != ""]
            .drop_duplicates(subset=["ticker"])
            .sort_values("ticker")
            .reset_index(drop=True))
    return df

def main():
    print("=== Step 2: Computing Weights ===")

    # Bootsraping to be safe
    bootstrap_main()

    # Skip Weight calculation if done previously
    if os.path.exists(CSV_WEIGHTS):
        try:
            df_existing =  pd.read_csv(CSV_WEIGHTS)
            if not df_existing.empty and "w_latest" in df_existing.columns:
                print(f"[info] {CSV_WEIGHTS} already exists — skipping recalculation.")
                print(f"  Total rows: {len(df_existing)}")
                print(f"  Sum of weights: {df_existing['w_latest'].sum():.6f}")
                return
        except Exception:
            pass 

    # 1) Load tickers
    df_tickers = load_or_refresh_sp500(CSV_TICKERS)
    tickers = df_tickers["ticker"].tolist()   # converting to python list
    print(f"[info] tickers to fetch: {len(tickers)}") # show the number of ticker

    # 2) Fetch share price and volume to calculate market capitalization
    stock_prcs = []
    for i, t in enumerate(tickers, start=1):
        try:
            tk = yf.Ticker(t)
            price = None
            try:
                price = tk.fast_info.get("last_price")
            except Exception:
                price = None
            if price is None:
                try:
                    price = tk.info.get("currentPrice")
                except Exception:
                    price = None

            # Shares Outstanding
            shares = None
            try:
                shares = tk.info.get("sharesOutstanding")
            except Exception:
                shares = None

            if not shares or not price or shares <= 0 or price <= 0:
                print(f"[skip] {t}: missing/invalid shares or price")
            else:
                cap = float(shares) * float(price)
                stock_prcs.append((t, int(shares), float(price), float(cap)))

        except Exception as e:
            print(f"[skip] {t}: {e}")

        if i % PROGRESS_EVERY == 0:
            print(f"[info] fetched ~{i}/{len(tickers)} tickers...")
        
    if not stock_prcs:
        raise SystemExit("No valid observations. Check connection or yfinance.")
    
    dfw = pd.DataFrame(
        stock_prcs, columns=["ticker", "shares_today", "last_close", "cap"]
    )

    # 3) Normalize weights. ChatGPT helped here.
    total_cap = dfw["cap"].sum()
    if total_cap <= 0:
        raise SystemExit("Cannot normalize weights.")
    dfw["w_latest"] = dfw["cap"]/ total_cap

    # 4) Creating Big/ Small stock buckets
    dfw["bucket"] = np.where(dfw["w_latest"] >= 0.01, "Big", "Small")
    # Adding sector column to dfw
    dfw = dfw.merge(df_tickers[["ticker", "sector"]], on="ticker", how="left")
    dfw["sector"] = dfw["sector"].fillna("NA")

    # 5) Save Output. Save the file in data/processed
    # Sorting by stock weight. Drop old index
    dfw = dfw.sort_values("w_latest", ascending=False).reset_index(drop=True)
    dfw.to_csv(CSV_WEIGHTS, index=False)
    
    # print summary
    print(f"[write] {CSV_WEIGHTS}")
    print(f"  Total rows: {len(dfw)}")
    print(f"  Big firms: {(dfw['bucket'] == 'Big').sum()}")
    print(f"  Small firms: {(dfw['bucket'] == 'Small').sum()}")
    print(f"  Sum of weights: {dfw['w_latest'].sum():.6f}")
    print("=== Step 2 Complete ===")

if __name__ == "__main__":
    main()