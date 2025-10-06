# FIN-5633-Project
Project Status

| Stage  | Description                                            | Status     |
| :----- | :----------------------------------------------------- | :----------|
| Step 1 | Folder / file bootstrap                                | Complete   |
| Step 2 | Market-cap weights generation                          | Complete   |
| Step 3 | Sampling and data pull (OHLCV + earnings + index)      | Incomplete |
| Step 4 | Event-window definition & abnormal-return construction | To do      |
| Step 5 | Modeling & visualization of index vs. firm reactions   | To do      |

Step 1: Bootstrap (step1_bootstrap.py)
    - Creates the project folder structure
    - Generates empty placeholder files for data to populate
    - Checks each path and prints status

Step 2: Market-Cap Weights Computation (step2_compute_weights.py)
    - Calls the bootstrap function for confirming file structure
    - Pulls S&P 500 tickers directly from Wikipedia (yfinance doesn't have it)
        + Uses a browser-like user agent to avoid 403 errors
        + Requires html5lib (pip install html5lib)
        + Cleans symbols into Yahoo Finance format
    - Pulling company data from yfinance
        + sharesOutstanding
        + currentPrice
    - Computes market capitalization = shares × price
    - Based on market cap calculates the weight of all stocks in S&P 500 index
    - Categorizes firms
        + Big = 1% or more weight
        + Small = less than 1% weight
    - Exports results to data/processed/weights_latest.csv
    - Skips re-calculation if weight is already calculated
    - Print summary

Install dependencies:
pip install pandas numpy yfinance html5lib



