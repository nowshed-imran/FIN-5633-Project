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
1. Makes sure S&P 500 tickers by web scraping Wikipedia
2. Pull shares outstanding and current price via yfinance.
3. Compute market cap and normalized weights
4. Classify big and small companies based on weights
5. Save results to data/processed/weights_latest.csv

I have tried to code by myself, and gotten help with ChatGPT to solve error
or refine codes. The codes I have directly copied has comment with it.
"""

# Importing statements and libraries
import os
import pandas as pd
import numpy as np
import yfinance as yf
from urllib.request import Request, urlopen # Need for data scraping
from pathlib import Path
from src.step1_bootstrap import main as bootstrap_main  # Import bootstrap

# Path safety to make the script reproducible. 
# This code block is copied from ChatGPT.
try:
    THIS_FILE = Path(__file__).resolve()
    ROOT = (THIS_FILE.parent.parent if THIS_FILE
            .parent.name == "src" else THIS_FILE.parent)
except NameError:
    ROOT = Path(os.getcwd())
os.chdir(ROOT)
print(f"[info] Working directory set to: {ROOT}")

# This function is totally ChatGPT. I don't understand it yet.
# It took me a whole day just to get it working.
def sp500_from_wikipedia(csv_path: str):
    """
    Pull current S&P 500 tickers by scraping Wikipedia, clean them to Yahoo 
    style, and save to CSV at csv_path.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        # use a browsery User-Agent for avoiding website block
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        html = urlopen(req, timeout=30).read()
        tables = pd.read_html(html, flavor="html5lib")  # html5lib installed
    except ImportError:
        raise SystemExit("install html5lib")
    except Exception as e:
        raise SystemExit(f"Could not download S&P 500 table. : {e}")

    df = tables[0]                  # first table contains the constituents
    tickers = df["Symbol"].tolist() # column name is 'Symbol'

    # Cleaning the ticker info
    cleaned = []
    for t in tickers:
        if not isinstance(t, str):
            continue
        t = t.strip().upper().replace(".", "-")
        if t:
            cleaned.append(t)

    tickers = sorted(set(cleaned))
    # ChatGPT helped with export
    pd.DataFrame({"ticker": tickers}).to_csv(csv_path, index=False)
    print(f"[write] refreshed S&P500 stock tickers: {csv_path} "
          f"({len(tickers)} tickers)")

def load_or_refresh_sp500(csv_path: str) -> pd.DataFrame:
    """
    Load tickers from CSV. If file missing or effectively empty (no rows),
    refresh from yfinance and load again. Returns a single 'ticker' column.
    """
    need_refresh = False
    if not os.path.exists(csv_path):
        need_refresh = True
    else:
        try: # ChatGPT helped to understand and code with try/ except
            df = pd.read_csv(csv_path)
            if "ticker" not in df.columns or df.empty:
                need_refresh = True
        except Exception:
            need_refresh = True
    # Execute refresh_sp500 function when needed
    if need_refresh:
        sp500_from_wikipedia(csv_path)
    
    # After ensuring refresh clean the data frame
    df = pd.read_csv(csv_path)
    df["ticker"] = (df["ticker"] 
                            .astype(str) # converting to string
                            .str.strip() # removing blank space
                            .str.upper() # making the words capitalized
                            .str.replace(".", "-", regex=False) # Replacing . to -
                            )
    df = (
        df.loc[df["ticker"] != ""]          # remove blank rows
                  .drop_duplicates()        # remove duplicates
                  .sort_values("ticker")    # sort alphabetically
                  .reset_index(drop=True)
                )
    return df 

def main():
    print("=== Step 2: Computing Weights ===")

    # Bootsraping to be safe
    bootstrap_main()

    # Fixing file path
    tickers_path = "data/raw/sp500_tickers.csv"
    out_path = "data/processed/weights_latest.csv"

    # Skip Weight calculation if done previously
    if os.path.exists(out_path):
        try:
            df_existing = pd.read_csv(out_path)
            if not df_existing.empty and "w_latest" in df_existing.columns:
                print(f"[info] {out_path} already exists — skipping recalculation.")
                print(f"  Total rows: {len(df_existing)}")
                print(f"  Sum of weights: {df_existing['w_latest'].sum():.6f}")
                return
        except Exception:
            pass 

    # 1) Load tickers
    df_tickers = load_or_refresh_sp500(tickers_path)
    tickers = df_tickers["ticker"].tolist()   # converting to python list
    print(f"next step may look stuck but it is pulling data of each stock")
    print(f"[info] tickers to fetch: {len(tickers)}") # show the number of ticker

    # 2) Fetch share price and volume to calculate market capitalization
    stock_prcs = []
    for i, t in enumerate(tickers, start=1):
        if i % 25 == 0:
        print(f"[info] fetched ~{i}/{len(tickers)} tickers...")
        try:
            info = yf.Ticker(t).info
            shares = info.get("sharesOutstanding", None)
            price  = info.get("currentPrice", None)
            if not shares or not price or shares <= 0 or price <= 0:
                print(f"[skip] {t}: missing/invalid shares or price")
                continue
            cap = float(shares) * float(price)
            stock_prcs.append((t, int(shares), float(price), float(cap)))
        except Exception as e:
            print(f"[skip] {t}: {e}")
    if not stock_prcs:
        raise SystemExit("Check connection or yfinance.")

    dfw = pd.DataFrame(stock_prcs, columns=["ticker", "shares_today", 
                                      "last_close", "cap"])
    
    # 3) Normalize weights. ChatGPT helped here.
    total_cap = dfw["cap"].sum()
    if total_cap <= 0:
        raise SystemExit("Cannot normalize weights.")
    dfw["w_latest"] = dfw["cap"]/ total_cap

    # 4) Creating Big/ Small stock buckets
    dfw["bucket"] = np.where(dfw["w_latest"] >= 0.01, "Big", "Small")

    # 5) Save Output. Save the file in data/processed
    out_path = "data/processed/weights_latest.csv"
    # Sorting by stock weight. Drop old index
    dfw = dfw.sort_values("w_latest", ascending=False).reset_index(drop=True)
    dfw.to_csv(out_path, index=False)
    
    # print summary
    total_rows = len(dfw)
    big_count = (dfw["bucket"] == "Big").sum()
    sum_weights = dfw["w_latest"].sum()
    print(f"[write] {out_path}")
    print(f"  Total rows: {total_rows}")
    print(f"  Big firms: {big_count}")
    print(f"  Sum of weights: {sum_weights:.6f}")

if __name__ == "__main__":
    main()