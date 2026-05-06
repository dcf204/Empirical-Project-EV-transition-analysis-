# Empirical Project: EV Transition Analysis
**Is the electric vehicle transition driven by real consumer demand, or industry hype?**

Live Blog: https://hackmd.io/@dcf204/SyVvreS0Zx

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
└── requirements.txt      # Python dependencies
```
