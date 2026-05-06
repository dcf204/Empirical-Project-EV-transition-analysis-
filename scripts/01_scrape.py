# Research Question: Is the EV transition driven by real consumer demand or industry hype?
# this script collects four main datasets:
# 1. google trends - consumer search interest in EVs in the UK over time
# 2. company stock prices - tesla, volkswagen, byd from 2018-2024
# 3. company financial statements - actual revenue and profit data
# 4. IEA EV adoption data - multiple datasets from Our World in Data
#    pulling: EV sales share, BEV vs PHEV breakdown, absolute sales,
#    total car sales by type, and EV stocks on the road
#    source: IEA Global EV Outlook 2025, processed by Our World in Data
#
# separating BEV and PHEV is important for a fair analysis as otherwise would overstate the real transition to clean energy


import pandas as pd
import requests
from pytrends.request import TrendReq
import yfinance as yf
import time
import os

# creating folders to store raw and cleaned data
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/clean', exist_ok=True)

print("starting data collection")

# PART 1: GOOGLE TRENDS
# using google trends as a proxy for consumer interest in EVs
# it gives a score from 0-100 showing how often people search for something


print("\ngoogle trends data for the UK")

try:
    # connecting to google trends
    pytrends = TrendReq(hl='en-GB', tz=0)

    # these keywords capture different types of consumer interest
    # electric car = general interest in fully electric vehicles
    # hybrid car = interest in hybrid/part electric vehicles for comparison
    # EV charging = practical search, suggests someone is actually considering buying
    # Tesla = brand level interest, most recognisable EV company globally
    keywords = ['electric car', 'hybrid car', 'EV charging', 'Tesla']

    # pulling monthly UK data from 2018 to 2024
    # 2018 is the starting point asthats when EV interest started accelerating
    pytrends.build_payload(
        keywords,
        timeframe='2018-01-01 2024-12-31',
        geo='GB'
    )

    google_data = pytrends.interest_over_time()

    # google adds an isPartial column to flag incomplete periods - not useful
    if 'isPartial' in google_data.columns:
        google_data = google_data.drop(columns=['isPartial'])

    # making date a proper column instead of the index
    google_data = google_data.reset_index()
    google_data = google_data.rename(columns={'date': 'Date'})

    print(f"collected {len(google_data)} months of search data")
    print(f"columns: {list(google_data.columns)}")
    print(google_data.head())

    google_data.to_csv('data/raw/google_trends.csv', index=False)
    print("saved to data/raw/google_trends.csv")

except Exception as e:
    print(f"something went wrong with google trends: {e}")

# waiting so google doesnt block the request
time.sleep(2)

# PART 2: COMPANY STOCK PRICES
# stock prices for three companies representing different positions in the EV market
# tesla = pure EV company, the benchmark for the industry
# volkswagen = traditional carmaker trying to transition to electric
# byd = chinese EV giant, important for the global picture
# stock price can be driven by hype and speculation- want to see if investor excitement is connected to real company performance and EV adoption data later on

print("\ncompany stock prices")

companies = {
    'Tesla':      'TSLA',
    'Volkswagen': 'VWAGY',
    'BYD':        'BYDDY'
}

stock_prices = []

for company_name, ticker in companies.items():
    print(f"  fetching {company_name} ({ticker})")

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start='2018-01-01', end='2024-12-31')

        hist = hist.reset_index()

        # keeping date, closing price and volume
        # volume shows how actively the stock is being traded each day
        hist = hist[['Date', 'Close', 'Volume']]
        hist['Company'] = company_name
        hist['Ticker'] = ticker

        stock_prices.append(hist)
        print(f"  collected {len(hist)} rows for {company_name}")

    except Exception as e:
        print(f"  couldnt get data for {company_name}: {e}")

    time.sleep(1)

all_stock_data = pd.concat(stock_prices, ignore_index=True)
print(f"\ntotal stock rows: {len(all_stock_data)}")
print(all_stock_data.head(10))

all_stock_data.to_csv('data/raw/company_stock_data.csv', index=False)
print("saved to data/raw/company_stock_data.csv")

# PART 3: COMPANY FINANCIAL STATEMENTS
# stock price alone doesnt tell the full story
# a rising stock price can be purely driven by investor excitement
# pulling actual income statements to analyse genuine company performance and growth in the EV market


print("\ncompany financial statements")

financials_list = []

for company_name, ticker in companies.items():
    print(f"  fetching financials for {company_name}")

    try:
        stock = yf.Ticker(ticker)

        # annual income statement
        # transposing so years become rows instead of columns
        # makes it much easier to work with in pandas
        income_stmt = stock.financials.T.reset_index()
        income_stmt = income_stmt.rename(columns={'index': 'Date'})

        # only keeping columns that exist and are relevant to analysis
        cols_to_keep = ['Date']
        for col in ['Total Revenue', 'Gross Profit', 'Net Income']:
            if col in income_stmt.columns:
                cols_to_keep.append(col)

        income_stmt = income_stmt[cols_to_keep]
        income_stmt['Company'] = company_name
        income_stmt['Ticker'] = ticker

        financials_list.append(income_stmt)
        print(f"  got {len(income_stmt)} years of financials for {company_name}")
        print(income_stmt)

    except Exception as e:
        print(f"  couldnt get financials for {company_name}: {e}")

    time.sleep(1)

all_financials = pd.concat(financials_list, ignore_index=True)
print(f"\ntotal financial rows: {len(all_financials)}")

all_financials.to_csv('data/raw/company_financials.csv', index=False)
print("saved to data/raw/company_financials.csv")

# PART 4: IEA EV ADOPTION DATA
# this is the real adoption data 
# pulling four separate datasets to build a comprehensive picture:
#
# 
# 1. ev sales share broken down by BEV vs PHEV- separating fully electric from plug-in hybrids
# 2. absolute EV sales - actual number of EVs sold per year per country
# 3. total car sales by type - EVs vs petrol/diesel, shows the full picture
# 4. EV stocks on the road - total EVs in use, not just new sales each year

print("\nIEA EV adoption data")

# helper function to fetch each dataset cleanly
# using a try/except for each one so if one fails the others still run
def fetch_owid(url, filename, description):
    print(f"\n  fetching {description}...")
    try:
        df = pd.read_csv(
            url,
            storage_options={'User-Agent': 'Our World In Data data fetch/1.0'}
        )
        df.to_csv(f'data/raw/{filename}', index=False)
        print(f"  collected {len(df)} rows")
        print(f"  columns: {list(df.columns)}")
        print(df.head())
        print(f"  saved to data/raw/{filename}")
        return df
    except Exception as e:
        print(f"  something went wrong: {e}")
        return None



# 1. breakdown of BEV vs PHEV as share of new car sales
bev_phev = fetch_owid(
    'https://ourworldindata.org/grapher/share-car-sales-battery-plugin.csv?v=1&csvType=full&useColumnShortNames=false',
    'ev_bev_vs_phev.csv',
    'BEV vs PHEV share of new car sales (separated)'
)

# 2. absolute number of new electric cars sold per year
ev_sales = fetch_owid(
    'https://ourworldindata.org/grapher/electric-car-sales.csv?v=1&csvType=full&useColumnShortNames=false',
    'ev_sales_absolute.csv',
    'absolute number of new EVs sold'
)

# 3. total car sales broken down by type - EV vs petrol/diesel
# shows whether non-electric car sales are actually declining
car_sales_type = fetch_owid(
    'https://ourworldindata.org/grapher/car-sales.csv?v=1&csvType=full&useColumnShortNames=false',
    'car_sales_by_type.csv',
    'total car sales by type (EV vs petrol/diesel)'
)

# 4. electric car stocks - total EVs on the road not just new sales
# this is the cumulative picture - how many EVs are actually in use
ev_stocks = fetch_owid(
    'https://ourworldindata.org/grapher/electric-car-stocks.csv?v=1&csvType=full&useColumnShortNames=false',
    'ev_stocks_on_road.csv',
    'total EVs on the road (stocks)'
)

print("\ndone! all files saved to data/raw/:")
print("  google_trends.csv         - UK consumer search interest 2018-2024")
print("  company_stock_data.csv    - daily stock prices Tesla, VW, BYD")
print("  company_financials.csv    - annual revenue and profit Tesla, VW, BYD")
print("  ev_bev_vs_phev.csv        - BEV vs PHEV breakdown separately")
print("  ev_sales_absolute.csv     - absolute EV units sold by country")
print("  car_sales_by_type.csv     - total car sales EV vs petrol/diesel")
print("  ev_stocks_on_road.csv     - total EVs in use on the road")
