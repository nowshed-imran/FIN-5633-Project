#!/usr/bin/env python3
# Making the script directly executable from terminal

"""
==================================
Project Update 2: Step 1
Submitted by: Nowshed Imran
Date: 10/02/2025
==================================

Purpose:
This script initializes the folder and file structure for a reproducible 
project. It executes a main function that will check whether necessary folder 
and file structure is present or not, and create those if missing. Those
files will be used to later populate with data and model training.

ChatGPT has been used to learn how to convert difficult ideas to code. I have 
attempted to code first and used ChatGPT's input to fix them.
"""

# Importing statements and third-party libraries
import os
import pandas as pd
from pathlib import Path

# Path safety to make the script reproducible. #
# This code block is from ChatGPT.
try:
    THIS_FILE = Path(__file__).resolve()
    if THIS_FILE.parent.name == "src":
        ROOT = THIS_FILE.parent.parent
    else:
        ROOT = THIS_FILE.parent
except NameError:
    # Fallback if __file__ doesn't exist (e.g., in Jupyter)
    ROOT = Path(os.getcwd())

os.chdir(ROOT)
print(f"[info] Working directory set to: {ROOT}")

# Combining all the steps under main function
def main():
    """
    Executes the bootstrap process.
    This function performs the following actions:
        1. Defines the required directory structure.
        2. Creates folders if they do not already exist.
        3. Creates a placeholder ticker CSV file in data/raw/
           for S&P 500 company tickers.
        4. Creates an empty weights file in data/processed/ that will later
           store computed market-cap weights and bucket labels.
        5. Prints clear messages ("[create]" or "[skip]") for transparency.
    """
    print("=== Bootstrap Stage Started ===")

    # 1. Defining the folders to create
    folders = [
        "data",
        "data/raw",
        "data/raw/ohlcv",
        "data/raw/earnings", 
        "data/processed",
        "reports"
    ]

    # 2. Create folders if they don't exist
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"[create] folder: {folder}")
        else:
            print(f"[skip] folder exists: {folder}")
    
    # 3. Empty placeholder for S&P 500 index
    tickers_path = "data/raw/sp500_tickers.csv"
    if not os.path.exists(tickers_path):
        pd.DataFrame(columns=["ticker"]).to_csv(tickers_path, index=False)
        print(f"[create] empty file: {tickers_path}")
    else:
        print(f"[skip] file exists: {tickers_path}")

    # 4. Create an empty placeholder for weights
    weights_path = "data/processed/weights_latest.csv"
    if not os.path.exists(weights_path):
        header = ["ticker", "shares_today", "last_close", 
                  "cap", "w_latest", "bucket"]
        pd.DataFrame(columns=header).to_csv(weights_path, index=False)
        print(f"[create] file: {weights_path}")
    else:
        print(f"[skip] file exists: {weights_path}")
    
    print("=== Bootstrap Stage Complete ===")

if __name__ == "__main__":
    main()