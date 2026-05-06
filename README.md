# Empirical Project: EV Transition Analysis
**Is the electric vehicle transition driven by real consumer demand, or industry hype?**

Live Blog: https://dcf204.github.io/Empirical-Project-EV-transition-analysis-/blog.html

## Overview
This project investigates whether the EV transition is driven by genuine consumer 
demand or speculative hype. Using data from Google Trends, Yahoo Finance, and the 
IEA Global EV Outlook 2025, it analyses UK consumer search interest, company stock 
prices, and real-world EV adoption across five countries: Norway, China, the UK, 
Germany, and the United States. A key distinction is made between fully electric 
vehicles (BEV) and plug-in hybrids (PHEV). The blog also aims to explain the differences across economies, diving into the transition gap and why it exists. Finally, a simple OLS regression is conducted to understand whether UK consumer search interest is a predictive measure of EVs sales the following year. 

## Repository Structure
```
Empirical-Project-EV-transition-analysis-/
├── data/
│   ├── raw/              # Raw data
│   └── clean/            # Cleaned and processed datasets ready for analysis
├── outputs/
│   ├── figures/          # All 6 generated plots (PNG)
│   └── tables/           # Regression results (TXT)
├── scripts/
│   ├── 01_scrape.py      # Collects data from Google Trends, Yahoo Finance etc
│   ├── 02_clean.py       # Cleans, validates and merges all raw datasets
│   ├── 03_analysis.py    # Runs OLS regression, saves results to outputs/tables/
│   ├── 04_figures.py     # Generates all 6 figures, saves to outputs/figures/
│   └── runAll.py         # Runs the full pipeline in sequence
├── .gitignore            # Excludes venv, pycache and raw data files
├── Makefile              # Run entire pipeline with: make all
├── README.md             # Project documentation
├── blog.html             # complete script for live blog
└── requirements.txt      # Python dependencies
```
## How to Replicate
1. Clone the repository
```bash
git clone https://github.com/dcf204/Empirical-Project-EV-transition-analysis-
cd Empirical-Project-EV-transition-analysis-
```
2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Run the full pipeline
```bash
make all
```
Or run scripts individually in order:
```bash
python scripts/01_scrape.py
python scripts/02_clean.py
python scripts/03_analysis.py
python scripts/04_figures.py
```

**Note:** Google Trends may occasionally rate limit requests, the scraper will automatically retry up to 3 times.

## Requirements
Requires Python 3.10+, pip, and the following packages (all installable via `pip install`):
See  `requirements.txt`
- `pandas`, `numpy`, `matplotlib`, `scipy`, `statsmodels` for data processing and analysis
- `pytrends`  Google Trends 
- `yfinance` Yahoo Finance stock price data
- `requests`  HTTP requests for Our World in Data downloads

## Data Sources

| Dataset | Source | Details |
|---|---|---|
| UK Google Trends | [Google Trends](https://trends.google.com) via pytrends | Monthly search interest for 'electric car', 'Tesla', 'EV charging', 'hybrid car' — UK only, 2018–2024 |
| Company Stock Prices | [Yahoo Finance](https://finance.yahoo.com) via yfinance | Daily closing prices for Tesla (TSLA), Volkswagen (VWAGY), BYD (BYDDY), 2018–2024 |
| Company Financials | [Yahoo Finance](https://finance.yahoo.com) via yfinance | Annual revenue, gross profit and net income for Tesla, VW, BYD |
| EV Sales & Adoption | [Our World in Data](https://ourworldindata.org) | IEA Global EV Outlook 2025 — BEV vs PHEV breakdown, absolute sales, market share, EV stocks on road |

## Data Dictionary

| File | Column | Type | Description |
| --- | --- | --- | --- |
| `google_trends_clean.csv` | `Date` | datetime | Year-month of observation (monthly) |
| | `electric car` | int 0–100 | UK relative search interest for "electric car" |
| | `hybrid car` | int 0–100 | UK relative search interest for "hybrid car" |
| | `EV charging` | int 0–100 | UK relative search interest for "EV charging" |
| | `Tesla` | int 0–100 | UK relative search interest for "Tesla" |
| `company_stocks_clean.csv` | `Date` | datetime | Daily date of observation |
| | `Company` | str | Company name (Tesla, Volkswagen, BYD) |
| | `Ticker` | str | Stock ticker (TSLA, VWAGY, BYDDY) |
| | `Close` | float | Raw daily closing price (local currency) |
| | `Close_Normalised` | float | Price indexed to 100 at first trading day of 2018 |
| | `Volume` | float | Daily trading volume |
| `company_stocks_monthly.csv` | `YearMonth` | str | Calendar month (YYYY-MM) |
| | `Company` | str | Company name |
| | `Close_Mean` | float | Monthly average raw closing price |
| | `Close_Norm_Mean` | float | Monthly average normalised price (base 100) |
| | `Volume_Mean` | float | Monthly average trading volume |
| `company_financials_clean.csv` | `Company` | str | Company name |
| | `Year` | int | Financial year |
| | `Total Revenue_Billions` | float | Annual revenue in billions (local currency) |
| | `Gross Profit_Billions` | float | Annual gross profit in billions (local currency) |
| | `Net Income_Billions` | float | Annual net income in billions (local currency) |
| | `Revenue_Growth_Pct` | float | Year-on-year revenue growth rate (%) |
| `ev_master_clean.csv` | `Country` | str | Country name |
| | `Country_Code` | str | ISO 3-letter country code |
| | `Year` | int | Calendar year |
| | `Electric cars sold` | float | Absolute number of new EVs sold that year |
| | `Electric cars` | float | EV share of new car sales (%) |
| | `Non-electric cars` | float | Non-EV share of new car sales (%) |
| | `Battery-electric` | float | BEV share of new car sales (%) |
| | `Plug-in hybrid` | float | PHEV share of new car sales (%) |
| | `Electric car stocks` | float | Total EVs on the road (cumulative) |
| | `EV_Market_Share_Pct` | float | Calculated EV market share as % of total new car sales |

# All Google Trends values are relative search indices, 100 = peak interest in the UK over the 2018–2024 period. 
# Stock prices are in each company's local trading currency and are not cross-currency comparable in raw form; use `Close_Norm_Mean` for comparisons.
# Financials are similarly in local currency — use `Revenue_Growth_Pct` for cross-company comparisons.


## Outputs

| Figure | File | Description |
|---|---|---|
| 1 | `outputs/figures/fig1_uk_google_trends.png` | UK search interest for electric car, Tesla, EV charging and hybrid car 2018–2024 |
| 2 | `outputs/figures/fig2_stock_prices.png` | Normalised stock prices for Tesla, Volkswagen and BYD indexed to 100 at start of 2018 |
| 3 | `outputs/figures/fig3_ev_market_share.png` | EV share of new car sales by country 2010–2024 for Norway, China, UK, Germany and USA |
| 4 | `outputs/figures/fig4_bev_vs_phev.png` | Global BEV vs PHEV breakdown as share of new car sales 2015–2024 |
| 5 | `outputs/figures/fig5_ev_stocks_global.png` | Total electric vehicles on the road worldwide 2013–2024 |
| 6 | `outputs/figures/fig6_regression.png` | OLS regression scatter plot: lagged search interest vs UK EV market share |
| — | `outputs/tables/regression_results.txt` | Full OLS regression results including coefficients, p-values and R² |

## Key References 
- Acheampong, T. Y. (2026). Who benefits from the EV transition? Electric vehicle adoption, inequality and policy implications. World Electric Vehicle Journal, 17(1), 34.
  https://www.mdpi.com/2032-6653/17/1/34
- Financhill Editor. (2023, March 6). BYD vs Tesla stock: Which is best? Financhill.
  https://financhill.com/blog/investing/byd-vs-tesla-stock 
- Hoium, T. (2020, December 31). Tesla stock surged 695% in 2020. Is it a buy for 2021? Nasdaq.
  https://www.nasdaq.com/articles/tesla-stock-surged-695-in-2020.-is-it-a-buy-for-2021-2020-12-31 
- Jaeger, J. (2025, December 5). These countries are adopting electric vehicles the fastest. World Resources Institute.
  https://www.wri.org/insights/countries-adopting-electric-vehicles-fastest
- Jurevicius, O. (2025, June 16). Tesla SWOT analysis 2025. Strategic Management Insight.
  https://strategicmanagementinsight.com/swot-analyses/tesla-swot-analysis/
- Maglicic, M., et al. (2025). Income inequality in the uptake of environmentally friendly products. World Electric Vehicle Journal.
  https://pubmed.ncbi.nlm.nih.gov/40248117/
- Roytburg, E. (2025, August 26). In China, EVs are now cheaper than gas cars. In the U.S., the Big Three still haven’t closed a premium that’s $14,000 per vehicle. Fortune.
  https://fortune.com/2025/08/26/china-ev-prices-cheaper-gas-us-big-three-gm-ford-stellantis/
- Virta. (2025, July 3). The global electric vehicle market in 2025.
  https://www.virta.global/global-electric-vehicle-market
