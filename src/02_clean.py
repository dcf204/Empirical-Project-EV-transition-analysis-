# the main tasks:
# - inspecting each dataset to understand what we have
# - handling missing values appropriately
# - standardising column names and data types
# - filtering to relevant time periods and countries
# - reshaping data where needed (wide to long etc)
# - saving cleaned versions to data/clean/

import pandas as pd
import numpy as np
import os

os.makedirs('data/clean', exist_ok=True)

print("starting data cleaning")

# helper function to give a quick overview of any dataframe
# using this throughout to inspect each dataset before cleaning
def inspect(df, name):
    print(f"\nchecking {name}")
    print(f"shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"columns: {list(df.columns)}")
    print(f"missing values: {df.isnull().sum().to_dict()}")
    print(df.head())
    print(df.dtypes)
# PART 1: GOOGLE TRENDS DATA
# loading and inspecting the consumer interest data first
# this is monthly UK search interest from 2018-2024

print("\nloading google trends data...")
trends = pd.read_csv('data/raw/google_trends.csv')
inspect(trends, 'google_trends')

# converting date column to proper datetime format
#needed for time series analysis and plotting 
trends['Date'] = pd.to_datetime(trends['Date'])

# checking for any missing months in the time series
# there should be 84 months (7 years x 12 months) from 2018 to 2024
print(f"\ndate range: {trends['Date'].min()} to {trends['Date'].max()}")
print(f"total months: {len(trends)} (expected 84)")

# the google trends scores are already on a 0-100 scale,they represent relative search interest
# checking the ranges to see if there are any anomalies (e.g. all zeros or all 100s would be suspicious)
print("\nsearch interest score ranges:")
for col in ['electric car', 'hybrid car', 'EV charging', 'Tesla']:
    print(f"  {col}: min={trends[col].min()}, max={trends[col].max()}, mean={trends[col].mean():.1f}")

# saving cleaned trends data
trends.to_csv('data/clean/google_trends_clean.csv', index=False)
print("\nsaved to data/clean/google_trends_clean.csv")

# PART 2: COMPANY STOCK DATA
# loading and cleaning the daily stock prices
# tesla, volkswagen and byd from 2018-2024

print("\nloading company stock data")
stocks = pd.read_csv('data/raw/company_stock_data.csv')
inspect(stocks, 'company_stock_data')

# stripping timezone info from date column
stocks['Date'] = pd.to_datetime(stocks['Date'], utc=True) .dt.tz_localize(None)
stocks['Date'] = stocks['Date'].dt.date
stocks['Date'] = pd.to_datetime(stocks['Date'])

# checking what companies we have
print(f"\ncompanies in dataset: {stocks['Company'].unique()}")
print(f"date range: {stocks['Date'].min()} to {stocks['Date'].max()}")

# checking for missing values in the close price
print(f"\nmissing close prices: {stocks['Close'].isnull().sum()}")

# dropping any rows where close price is missing
stocks = stocks.dropna(subset=['Close'])
print(f"rows after dropping missing prices: {len(stocks)}")

# calculating a normalised stock price (index = 100 at start of 2018)
# this lets us compare the three companies on the same scale
print("\nnormalising stock prices to base 100 at start of 2018")

normalised = []
for company in stocks['Company'].unique():
    company_data = stocks[stocks['Company'] == company].copy()
    company_data = company_data.sort_values('Date')
    
    # getting the first closing price as the base
    base_price = company_data['Close'].iloc[0]
    company_data['Close_Normalised'] = (company_data['Close'] / base_price) * 100
    
    normalised.append(company_data)

stocks = pd.concat(normalised, ignore_index=True)
print("normalisation done - all companies now start at 100")

# also creating a monthly average instead of daily
# monthly averages are cleaner for visualisation
stocks['YearMonth'] = stocks['Date'].dt.to_period('M')
stocks_monthly = stocks.groupby(['Company', 'Ticker', 'YearMonth']).agg(
    Close_Mean=('Close', 'mean'),
    Close_Norm_Mean=('Close_Normalised', 'mean'),
    Volume_Mean=('Volume', 'mean')
).reset_index()

stocks_monthly['YearMonth'] = stocks_monthly['YearMonth'].astype(str)

print(f"\nmonthly stock data shape: {stocks_monthly.shape}")
print(stocks_monthly.head())

stocks.to_csv('data/clean/company_stocks_clean.csv', index=False)
stocks_monthly.to_csv('data/clean/company_stocks_monthly.csv', index=False)
print("saved to data/clean/company_stocks_clean.csv")
print("saved to data/clean/company_stocks_monthly.csv")

# PART 3: COMPANY FINANCIALS
# loading and cleaning the annual income statement data

print("\nloading company financials...")
financials = pd.read_csv('data/raw/company_financials.csv')
inspect(financials, 'company_financials')

# converting date to datetime
financials['Date'] = pd.to_datetime(financials['Date'])

# extracting just the year - we only need annual figures
financials['Year'] = financials['Date'].dt.year

# tesla reports in USD, volkswagen in EUR, BYD in CNY
# comparing absolute revenue figures across currencies is not fair
# so instead, calculate percentage revenue growth over time
# growth rates are comparable regardless of the original currency

for col in ['Total Revenue', 'Gross Profit', 'Net Income']:
    if col in financials.columns:
        # keeping original figures but dividing by billion for readability
        financials[col + '_Billions'] = financials[col] / 1e9

# calculating revenue growth rate year on year per company
# this gives a fair like for like comparison across currencies
financials = financials.sort_values(['Company', 'Year'])
financials['Revenue_Growth_Pct'] = financials.groupby('Company')['Total Revenue'].pct_change() * 100
print("\nrevenue growth rates by company:")
print(financials[['Company', 'Year', 'Total Revenue_Billions', 'Revenue_Growth_Pct']])

# dropping rows where revenue is missing
print(f"\nrows before dropping missing: {len(financials)}")
financials = financials.dropna(subset=['Total Revenue'])
print(f"rows after dropping missing: {len(financials)}")

print("\ncleaned financials:")
print(financials[['Company', 'Year', 'Total Revenue_Billions', 'Net Income_Billions']])

financials.to_csv('data/clean/company_financials_clean.csv', index=False)
print("saved to data/clean/company_financials_clean.csv")

# PART 4: IEA EV ADOPTION DATA
#four separate datasets on EVs vs hybrids, EV sales, car sales by type, and EV stocks on the road

print("\nloading IEA EV adoption datasets")

# loading all five datasets
bev_phev = pd.read_csv('data/raw/ev_bev_vs_phev.csv')
ev_sales = pd.read_csv('data/raw/ev_sales_absolute.csv')
car_sales = pd.read_csv('data/raw/car_sales_by_type.csv')
ev_stocks = pd.read_csv('data/raw/ev_stocks_on_road.csv')

# inspecting each one
inspect(bev_phev, 'ev_bev_vs_phev')
inspect(ev_sales, 'ev_sales_absolute')
inspect(car_sales, 'car_sales_by_type')
inspect(ev_stocks, 'ev_stocks_on_road')

# all four datasets share the same structure - Entity, Code, Year
# so we can clean them together using the same approach

def clean_iea(df, name):
    print(f"\ncleaning {name}...")
    
    # standardising column names
    df = df.rename(columns={'Entity': 'Country', 'Code': 'Country_Code'})
    
    # removing world aggregates and regional groupings
    # we want country level data for fair comparison
    # rows without a country code are regional aggregates
    before = len(df)
    df = df.dropna(subset=['Country_Code'])
    df = df[df['Country_Code'].str.len() == 3]
    print(f"  removed {before - len(df)} regional aggregate rows")
    
    # filtering to 2010 onwards as the data before this is very sparse
    df = df[df['Year'] >= 2010]
    
    # checking missing values after filtering
    print(f"  missing values after cleaning:")
    print(f"  {df.isnull().sum().to_dict()}")
    print(f"  shape: {df.shape}")
    print(f"  countries: {df['Country'].nunique()}")
    print(f"  years: {df['Year'].min()} to {df['Year'].max()}")
    
    return df

bev_phev_clean = clean_iea(bev_phev, 'bev_phev')
ev_sales_clean = clean_iea(ev_sales, 'ev_sales')
car_sales_clean = clean_iea(car_sales, 'car_sales')
ev_stocks_clean = clean_iea(ev_stocks, 'ev_stocks')

# merging all four IEA datasets into one master dataframe
# this keeps all countries even if some metrics are missing for certain years

print("\nmerging all IEA datasets into one master dataframe")

ev_master = ev_sales_clean.merge(
    car_sales_clean,
    on=['Country', 'Country_Code', 'Year'],
    how='left'
)

ev_master = ev_master.merge(
    bev_phev_clean,
    on=['Country', 'Country_Code', 'Year'],
    how='left'
)

ev_master = ev_master.merge(
    ev_stocks_clean,
    on=['Country', 'Country_Code', 'Year'],
    how='left'
)

print(f"merged dataset shape: {ev_master.shape}")
print(f"columns: {list(ev_master.columns)}")

# calculating ev market share as a percentage of total new car sales
# ev sales / total car sales * 100 = percentage of new cars that are EV
ev_master['Total_Cars_Sold'] = ev_master['Electric cars'] + ev_master['Non-electric cars']
ev_master['EV_Market_Share_Pct'] = (
    ev_master['Electric cars sold'] / ev_master['Total_Cars_Sold'] * 100
).round(2)

# checking for countries with lots of missing data
# some countries only have data for a few years
missing_by_country = ev_master.groupby('Country')['EV_Market_Share_Pct'].apply(
    lambda x: x.isnull().sum()
)
print(f"\ncountries with most missing market share data:")
print(missing_by_country[missing_by_country > 5].sort_values(ascending=False).head(10))

# for countries with some missing years we fill forward
# assume the last known value carries forward
ev_master = ev_master.sort_values(['Country', 'Year'])
ev_master['EV_Market_Share_Pct'] = ev_master.groupby('Country')['EV_Market_Share_Pct'].transform(
    lambda x: x.ffill()
)

print(f"\nfinal master dataset:")
print(ev_master.head(10))
print(f"\nshape: {ev_master.shape}")

# saving master IEA dataset
ev_master.to_csv('data/clean/ev_master_clean.csv', index=False)
print("saved to data/clean/ev_master_clean.csv")

# also saving individual cleaned files
bev_phev_clean.to_csv('data/clean/ev_bev_vs_phev_clean.csv', index=False)
ev_sales_clean.to_csv('data/clean/ev_sales_clean.csv', index=False)
car_sales_clean.to_csv('data/clean/car_sales_clean.csv', index=False)
ev_stocks_clean.to_csv('data/clean/ev_stocks_clean.csv', index=False)
print("saved individual cleaned IEA files to data/clean/")

print("\ndata cleaning complete")
print("files saved to data/clean/:")
print("  google_trends_clean.csv")
print("  company_stocks_clean.csv")
print("  company_stocks_monthly.csv")
print("  company_financials_clean.csv")
print("  ev_master_clean.csv")
print("  ev_bev_vs_phev_clean.csv")
print("  ev_sales_clean.csv")
print("  car_sales_clean.csv")
print("  ev_stocks_clean.csv")
