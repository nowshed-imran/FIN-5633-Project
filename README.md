# FIN-5633-Project

### Install dependencies (Important):
```
pip install pandas numpy yfinance html5lib
```

### Where to place the scripts?
1. Create a main **project folder** (for example: `Python Project`).  
2. Inside it, create a subfolder called **`src`**.  
3. Copy these three scripts into the `src` folder:
   ```
   step1_bootstrap.py
   step2_compute_weights.py
   step3_select_sample.py
   ```

### How to run the code?
There are **three** ways to run the code.
#### **Option 1 – VS Code (workspace preferred)**
Open the **project folder** as a workspace.  
Then select each file and click **“Run Python File in Terminal.”**

#### **Option 2 – Command Line / Terminal**
Go to the **project root**, open terminal from that folder, and run:
```
python3 src/step1_bootstrap.py
python3 src/step2_compute_weights.py
python3 src/step3_select_sample.py
```

#### **Option 3 – IPython**
Open terminal in the 'src' folder and run ipython. From ipython code can be 
run block by block maintaining script sequence. 

## Project Status
| Stage  | Description                                         | Status     |
| :----- | :---------------------------------------------------| :----------|
| Step 1 | Folder / file bootstrap                             | Complete   |
| Step 2 | Market-cap weights generation                       | Complete   |
| Step 3 | Sampling and data pull (OHLCV + earnings + index)   | Complete   |
| Step 4 | Logistic Regression Model Training                  | Complete   |
| Step 5 | Measuring model prediction impact                   | Complete   |

### Step 1: Bootstrap (`step1_bootstrap.py`)
- Creates the project folder structure
- Generates empty placeholder files for data to populate
- Checks each path and prints status

### Step 2: Market-Cap Weights Computation (`step2_compute_weights.py`)
- Calls the bootstrap function for confirming file structure
- Pulls S&P 500 tickers directly from Wikipedia (yfinance doesn't have it)
    + Uses a browser-like user agent to avoid 403 errors
    + Requires `html5lib`
    + Cleans symbols into Yahoo Finance format
- Pulling company data from yfinance
    + `sharesOutstanding`
    + `currentPrice`
- Computes market capitalization = shares × price
- Based on market cap calculates the weight of all stocks in S&P 500 index
- Categorizes firms
    + Big = 1% or more weight
    + Small = less than 1% weight
- Exports results to data/processed/weights_latest.csv
- Skips re-calculation if weight is already calculated
    + Print summary

### Step 3: Sample Selection and Data Pull (`step3_select_sample.py`)
- Loads `weights_latest.csv`.  
- **Big firms:** uses all by default, or a random subset if `BIG_OVERRIDE` is set.  
- **Small firms:** selects sector-balanced samples where  
  ```
  Small = SMALL_MULTIPLIER × Big (default 3×)
  ```
  and distributes them equally across sectors (`x per sector + y remainder`).  
- Saves final firm list to `data/processed/sample_firms.csv`.  
- Downloads 1 year of OHLCV price history for each firm and for ^GSPC index.  
- Downloads **quarterly earnings data** (with timestamp only) for each firm.  
- All data saved under:
  ```
  data/raw/ohlcv/
  data/raw/earnings/
  ```

### Step 4 — Logistic Regression (step4_logit.py)
For each multiplier m:
- Builds daily Big/Small earnings features:
  - big_n, small_n
  - big_weight, small_weight
  - big_surprise, small_surprise
- Creates index UP/DOWN label:
  - y = 1 if return > 0 else 0
- Merges earnings features + index data
- Restricts to event days (big_n + small_n > 0)
- Fits 3 models:
  - overall
  - big_only
  - small_only
- Performs 70/30 stratified train/test split
- Stores results in:
  - logit_index_summary.csv
  - logit_index_accuracy.png



