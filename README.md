# Big Firm Earnings and the S&P 500

A few firms now carry 1% or more of the S&P 500 index. This project tests whether index moves are bigger on days when these firms report earnings (H1), and whether their earnings misses move the index more than their beats (H2). The revised version follows the steps for time series regression in Wooldridge's *Introductory Econometrics*: check the data, set up the model, test each assumption, choose the matching fix, and check for a structural break. The full write-up is in `report/FIN-5633-Project-Findings.pdf`.

## Main results

October 2006 to September 2026, 502 firms, 5,029 trading days.

- H1: On days a Big firm reports, the absolute index return is about 0.07 percentage points (about 9%) larger than on days without reports (Newey-West p = 0.032), and larger than on days when only Small firms report (p = 0.007). The effect grows with firm size and holds without 2008 and 2020.
- H2: Big misses move the index more than Big beats over all years (one-sided p < 0.001), but not without 2008 and 2020, so the evidence comes mostly from the crisis years.

![Results](output/results.png)

## Method in brief

A firm is Big if its index weight was 1% or more on the trading day before the market reacts to its announcement, and Small otherwise. Weights are approximated as price times current shares outstanding. Announcements released at or after 4 pm Eastern time are matched to the next trading day.

1. Data checks: Weak dependence and Dickey-Fuller unit root tests for the index return, absolute return and VIX.
2. Model: Absolute index return on a Big-day and a Small-day dummy, with yesterday's VIX, year dummies and month dummies, over all trading days.
3. Assumption tests: RESET, AR(1) residual test, Breusch-Godfrey, Breusch-Pagan, and a test for strict exogeneity. The script also computes Durbin-Watson and ARCH; they are in `output/assumption_tests.csv` but not in the report.
4. Fix: The regressors are not strictly exogenous, so Prais-Winsten FGLS is not appropriate. The main results use OLS with Newey-West standard errors (9 lags, the usual rule of thumb). Prais-Winsten is shown for comparison.
5. Robustness: Big cutoffs of 0.5% and 2%, and a Chow-style test for 2008 and 2020.

An earlier version also tested whether earnings information predicts index direction. That part was dropped in the revision.

## Repository

| File | Content |
| :--- | :--- |
| `sp500_big_firm_earnings.py` | The full analysis in one script: download, sample, tests |
| `report/` | Findings report: PDF, LaTeX source, `elsarticle.cls`, `elsarticle-harv.bst` and the figure |
| `output/` | Results produced by the script |
| `LICENSE` | MIT |

## Data

All data is public and downloaded by the script on the first run.

| File | Source | Content |
| :--- | :--- | :--- |
| `data/sp500_tickers.csv` | Wikipedia, yfinance | Current members and shares outstanding |
| `data/earnings.csv` | yfinance | Announcement date, hour, EPS surprise (%) |
| `data/prices.csv` | yfinance | Daily closes for all members and ^GSPC |
| `data/vix.csv` | yfinance | Daily VIX close |

The data files are not included in this repository because Yahoo Finance data is intended for personal use under Yahoo's terms. Running the script downloads them. The sample window is fixed in the script (`START` and `END`), but the member list, shares outstanding and earnings history are a snapshot from October 2026 and will differ if downloaded later.

## How to run

1. Clone or download the repository. The script works inside its own folder, so nothing needs to be edited.
2. Install the packages and run the script (Python 3):
   ```
   pip install pandas numpy matplotlib lxml yfinance scipy statsmodels
   python sp500_big_firm_earnings.py
   ```

The first run downloads the data in a few minutes. Later runs read from `data/` and finish in seconds. Delete a file in `data/` to download it again.

## Output

| File | Table or figure in the write-up |
| :--- | :--- |
| `output/data_checks.csv` | Table 1 |
| `output/assumption_tests.csv` | Table 2 (the file also holds the Durbin-Watson and ARCH rows) |
| `output/functional_form.csv` | Section 2.6 (functional form) |
| `output/h1_abs_return_by_day_type.csv` | Table 3 |
| `output/h1_results.csv` | Table 4 |
| `output/h1_cutoffs.csv` | Table 5 |
| `output/h2_results.csv` | Table 6 |
| `output/chow_and_other_tests.csv` | Chow tests and the other numbers quoted in the text (Tables 5 and 6, Section 3.1) |
| `output/results.png` | Figure 1 |

## Report format

The findings report is written in LaTeX using Elsevier's elsarticle template (elsarticle.cls), the template for the Journal of Financial Economics and other Elsevier journals. The class file is kept in `report/` so the source compiles as is. `elsarticle.cls` and `elsarticle-harv.bst` are copyright Elsevier Ltd and are included unmodified under the LaTeX Project Public License; the MIT license of this repository does not cover them.

## AI assistance

The study design, hypotheses, sample choices and interpretation are my own. The code was written with the help of Claude (Anthropic), which I used to rewrite my original course scripts into this single script, find and fix errors, and implement the econometric corrections. I have reviewed the code extensively and would appreciate any suggestions for improvement.
