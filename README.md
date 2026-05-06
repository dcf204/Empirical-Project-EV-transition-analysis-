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
Empirical-Project-EV-transition-analysis-/
├── data/
│   ├── raw/           # original downloaded data, never modified
│   └── clean/         # processed datasets ready for analysis
├── scripts/
│   ├── 01_scrape.py   # collects data from Google Trends, Yahoo Finance and Our World in Data
│   ├── 02_clean.py    # cleans and processes all raw datasets
│   ├── 03_analysis.py # runs OLS regression and saves results
│   ├── 04_figures.py  # generates all 6 figures
│   └── runAll.py      # runs the full pipeline in one command
├── outputs/
│   ├── figures/       # all 6 generated plots
│   └── tables/        #regression results
├── Makefile           # run entire pipeline with make all
├── requirements.txt   # all package dependencies
└── README.md
