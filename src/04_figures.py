# produces all 5 figures for the EV transition blog post
# figures saved to output/figures/:
#   fig1_uk_google_trends.png     - UK search interest 2018-2024
#   fig2_stock_prices.png         - normalised stock prices tesla/vw/byd
#   fig3_ev_market_share.png      - EV market share by country
#   fig4_bev_vs_phev.png          - BEV vs PHEV global breakdown
#   fig5_ev_stocks_global.png     - cumulative EVs on the road worldwide

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

os.makedirs('output/figures', exist_ok=True)

print("loading cleaned data for figures")

trends = pd.read_csv('data/clean/google_trends_clean.csv')
stocks = pd.read_csv('data/clean/company_stocks_monthly.csv')
ev_master = pd.read_csv('data/clean/ev_master_clean.csv')
bev_phev = pd.read_csv('data/clean/ev_bev_vs_phev_clean.csv')
ev_stocks = pd.read_csv('data/clean/ev_stocks_clean.csv')

trends['Date'] = pd.to_datetime(trends['Date'])
stocks['YearMonth'] = pd.to_datetime(stocks['YearMonth'])

print("all data loaded")

# consistent visual style across all figures
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f8f8',
    'axes.grid': True,
    'grid.color': 'white',
    'grid.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'legend.frameon': False,
})

#  colours used throughout all figures
COMPANY_COLOURS = {
    'Tesla': '#E31937',
    'Volkswagen': '#1B5FAA',
    'BYD': '#00A550'
}

COUNTRY_COLOURS = {
    'Norway': '#E31937',
    'China': '#FFB81C',
    'United Kingdom': '#1B5FAA',
    'Germany': '#00A550',
    'United States': '#7B2D8B'
}

print("\ngenerating figures")

# figure 1: UK google trends
# showing how public interest in EVs evolved in the UK from 2018 to 2024
# tesla is included as a proxy for brand level hype vs genuine EV interest
# the gap between tesla searches and electric car searches is a good way to show how much of the hype is focused on one company rather than the technology as a whole

print("  figure 1: UK google trends")

fig, ax = plt.subplots(figsize=(11, 5))

keyword_colours = {
    'electric car': '#E31937',
    'Tesla': '#FFB81C',
    'EV charging': '#1B5FAA',
    'hybrid car': '#888888'
}

keyword_labels = {
    'electric car': 'Electric Car',
    'Tesla': 'Tesla (brand)',
    'EV charging': 'EV Charging',
    'hybrid car': 'Hybrid Car'
}

for keyword, colour in keyword_colours.items():
    ax.plot(
        trends['Date'],
        trends[keyword],
        label=keyword_labels[keyword],
        color=colour,
        linewidth=2,
        alpha=0.9
    )

# shading the covid period to explain the dip in searches during early 2020
ax.axvspan(
    pd.Timestamp('2020-03-01'),
    pd.Timestamp('2020-09-01'),
    alpha=0.08,
    color='grey',
    label='COVID-19 period'
)

ax.set_title('UK Search Interest in Electric Vehicles (2018-2024)', pad=15)
ax.set_xlabel('Date')
ax.set_ylabel('Search Interest (0-100 scale)')
ax.legend(loc='upper left')
ax.text(
    0.01, -0.15,
    'Source: Google Trends. Score of 100 = peak search interest for that term in the UK over the period.',
    transform=ax.transAxes, fontsize=8, color='grey'
)

plt.tight_layout()
plt.savefig('output/figures/fig1_uk_google_trends.png', dpi=150, bbox_inches='tight')
plt.close()
print("  saved fig1_uk_google_trends.png")

# figure 2: normalised stock prices
# all three companies indexed to 100 at the start of 2018 so they are directly comparable

print("  figure 2: normalised stock prices")

fig, ax = plt.subplots(figsize=(11, 5))

for company, colour in COMPANY_COLOURS.items():
    company_data = stocks[stocks['Company'] == company].copy()
    company_data = company_data.sort_values('YearMonth')
    ax.plot(
        company_data['YearMonth'],
        company_data['Close_Norm_Mean'],
        label=company,
        color=colour,
        linewidth=2,
        alpha=0.9
    )

# reference line at 100 so its easy to see who gained and who lost relative to the start
ax.axhline(
    y=100, color='black', linewidth=0.8,
    linestyle='--', alpha=0.4, label='Starting value (Jan 2018)'
)

# annotating tesla's peak since this is the central hype moment in the whole blog
tesla_data = stocks[stocks['Company'] == 'Tesla'].copy()
tesla_peak = tesla_data.loc[tesla_data['Close_Norm_Mean'].idxmax()]
ax.annotate(
    f"Tesla peak\n~{int(tesla_peak['Close_Norm_Mean'])}x starting value",
    xy=(tesla_peak['YearMonth'], tesla_peak['Close_Norm_Mean']),
    xytext=(pd.Timestamp('2021-06-01'), tesla_peak['Close_Norm_Mean'] * 0.95),
    fontsize=8,
    color=COMPANY_COLOURS['Tesla'],
    arrowprops=dict(arrowstyle='->', color=COMPANY_COLOURS['Tesla'], lw=1.2)
)

ax.set_title('EV Company Stock Prices: Indexed to 100 at Start of 2018', pad=15)
ax.set_xlabel('Date')
ax.set_ylabel('Normalised Price (Jan 2018 = 100)')
ax.legend(loc='upper left')
ax.text(
    0.01, -0.15,
    'Source: Yahoo Finance via yfinance. Monthly averages shown. All prices indexed to 100 at first trading day of 2018.',
    transform=ax.transAxes, fontsize=8, color='grey'
)

plt.tight_layout()
plt.savefig('output/figures/fig2_stock_prices.png', dpi=150, bbox_inches='tight')
plt.close()
print("  saved fig2_stock_prices.png")

# figure 3: EV market share by country
# five countries chosen to show global transition patterns and make the blog relevant to a wide audience
# norway is the world leader, china is the biggest market by volume
# uk and germany represent europe, usa shows how far behind the largest economy is

print("  figure 3: EV market share by country")

fig, ax = plt.subplots(figsize=(11, 5))

for country, colour in COUNTRY_COLOURS.items():
    country_data = ev_master[ev_master['Country'] == country].copy()
    country_data = country_data.sort_values('Year')
    country_data = country_data.dropna(subset=['EV_Market_Share_Pct'])

    if len(country_data) > 0:
        ax.plot(
            country_data['Year'],
            country_data['EV_Market_Share_Pct'],
            label=country,
            color=colour,
            linewidth=2.5,
            marker='o',
            markersize=4,
            alpha=0.9
        )

ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax.set_title('EV Share of New Car Sales by Country (2010-2024)', pad=15)
ax.set_xlabel('Year')
ax.set_ylabel('EV Market Share (% of new car sales)')
ax.legend(loc='upper left')
ax.text(
    0.01, -0.15,
    'Source: IEA Global EV Outlook 2025, via Our World in Data. Includes both BEV and PHEV.',
    transform=ax.transAxes, fontsize=8, color='grey'
)

plt.tight_layout()
plt.savefig('output/figures/fig3_ev_market_share.png', dpi=150, bbox_inches='tight')
plt.close()
print("  saved fig3_ev_market_share.png")

# printing key numbers from figure 3 for blog writing
print("\nfigure 3 key numbers (EV market share by country, most recent year):")
for country in ['Norway', 'China', 'United Kingdom', 'Germany', 'United States']:
    d = ev_master[ev_master['Country'] == country].dropna(subset=['EV_Market_Share_Pct'])
    if len(d) > 0:
        latest = d.sort_values('Year').iloc[-1]
        print(f"  {country}: {latest['EV_Market_Share_Pct']:.1f}% ({int(latest['Year'])})")


# figure 4: BEV vs PHEV breakdownn energy transition
# plug-in hybrids still burn petrol so they should not be treated as a full clea
# EV figure is genuinely fully electric vs still partially fossil fuel

print("  figure 4: BEV vs PHEV breakdown")

# averaging across all countries to get a global picture for each year
global_split = bev_phev.groupby('Year')[['Battery-electric', 'Plug-in hybrid']].mean().reset_index()
global_split = global_split[global_split['Year'] >= 2015]

fig, ax = plt.subplots(figsize=(11, 5))

ax.fill_between(
    global_split['Year'],
    global_split['Battery-electric'],
    label='Battery Electric (BEV) - fully electric',
    color='#1B5FAA',
    alpha=0.8
)

ax.fill_between(
    global_split['Year'],
    global_split['Battery-electric'],
    global_split['Battery-electric'] + global_split['Plug-in hybrid'],
    label='Plug-in Hybrid (PHEV) - still burns petrol',
    color='#FFB81C',
    alpha=0.7
)

ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
ax.set_title('Global EV Sales: Battery Electric vs Plug-in Hybrid (2015-2024)', pad=15)
ax.set_xlabel('Year')
ax.set_ylabel('Share of New Car Sales (%)')
ax.legend(loc='upper left')
ax.text(
    0.01, -0.15,
    'Source: IEA Global EV Outlook 2025, via Our World in Data. Global average across all reporting countries.',
    transform=ax.transAxes, fontsize=8, color='grey'
)

plt.tight_layout()
plt.savefig('output/figures/fig4_bev_vs_phev.png', dpi=150, bbox_inches='tight')
plt.close()
print("  saved fig4_bev_vs_phev.png")

# defining variables for growth rate calculations
bev_2020 = global_split[global_split['Year'] == 2020]['Battery-electric'].values[0]
bev_2024 = global_split[global_split['Year'] == 2024]['Battery-electric'].values[0]
phev_2020 = global_split[global_split['Year'] == 2020]['Plug-in hybrid'].values[0]
phev_2023 = global_split[global_split['Year'] == 2023]['Plug-in hybrid'].values[0]
phev_2024 = global_split[global_split['Year'] == 2024]['Plug-in hybrid'].values[0]

# global growth rates 2020 to 2024 for BEVs and PHEVS
print(f"\nfigure 4 key numbers (global average across all reporting countries):")
print(f"  BEV 2020: {bev_2020:.2f}% -> 2024: {bev_2024:.2f}% | growth: {((bev_2024 - bev_2020) / bev_2020) * 100:.1f}%")
print(f"  PHEV 2020: {phev_2020:.2f}% -> 2024: {phev_2024:.2f}% | growth: {((phev_2024 - phev_2020) / phev_2020) * 100:.1f}%")
print(f"  PHEV 2023: {phev_2023:.2f}% -> 2024: {phev_2024:.2f}% | growth: {((phev_2024 - phev_2023) / phev_2023) * 100:.1f}%")

# BEV vs PHEV breakdown for all figure 3 countries
print("\nBEV vs PHEV breakdown for figure 3 countries (2020-2024):")
for country in ['Norway', 'China', 'United Kingdom', 'Germany', 'United States']:
    print(f"\n  {country}:")
    country_bev = bev_phev[bev_phev['Country'] == country].sort_values('Year')
    country_master = ev_master[ev_master['Country'] == country].sort_values('Year')
    country_merged = country_bev.merge(country_master[['Year', 'EV_Market_Share_Pct']], on='Year')
    for _, row in country_merged[country_merged['Year'] >= 2020].iterrows():
        print(f"    {int(row['Year'])}: total EV {row['EV_Market_Share_Pct']:.1f}% = BEV {row['Battery-electric']:.1f}% + PHEV {row['Plug-in hybrid']:.1f}%")

# figure 5: cumulative EV stocks on the road globally
# it shows the total number of EVs actually being driven around the world

print("  figure 5: EV stocks on the road globally")

# summing across all countries each year to get a global total
global_stocks = ev_stocks.groupby('Year')['Electric car stocks'].sum().reset_index()
global_stocks = global_stocks[global_stocks['Year'] >= 2013]

# converting to millions so the axis labels are readable
global_stocks['Stocks_Millions'] = global_stocks['Electric car stocks'] / 1_000_000

fig, ax = plt.subplots(figsize=(11, 5))

ax.fill_between(
    global_stocks['Year'],
    global_stocks['Stocks_Millions'],
    alpha=0.3,
    color='#1B5FAA'
)

ax.plot(
    global_stocks['Year'],
    global_stocks['Stocks_Millions'],
    color='#1B5FAA',
    linewidth=2.5,
    marker='o',
    markersize=5
)

# annotating the most recent figure 
last = global_stocks.iloc[-1]
ax.annotate(
    f"{last['Stocks_Millions']:.1f}M EVs\non the road",
    xy=(last['Year'], last['Stocks_Millions']),
    xytext=(last['Year'] - 2, last['Stocks_Millions'] * 0.8),
    fontsize=9,
    color='#1B5FAA',
    fontweight='bold',
    arrowprops=dict(arrowstyle='->', color='#1B5FAA', lw=1.2)
)

ax.set_title('Total Electric Vehicles on the Road Worldwide (2013-2024)', pad=15)
ax.set_xlabel('Year')
ax.set_ylabel('Total EVs in Use (millions)')
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.text(
    0.01, -0.15,
    'Source: IEA Global EV Outlook 2025, via Our World in Data. Includes BEV and PHEV stock.',
    transform=ax.transAxes, fontsize=8, color='grey'
)

plt.tight_layout()
plt.savefig('output/figures/fig5_ev_stocks_global.png', dpi=150, bbox_inches='tight')
plt.close()
print("  saved fig5_ev_stocks_global.png")

# printing key numbers for writing the blog
print("\nkey numbers for the blog:")

print(f"\ngoogle trends UK peak scores:")
print(f"  electric car peak: {trends['electric car'].max()}")
print(f"  Tesla peak: {trends['Tesla'].max()}")

print(f"\nstock prices peak (all started at 100):")
for company in ['Tesla', 'Volkswagen', 'BYD']:
    peak = stocks[stocks['Company'] == company]['Close_Norm_Mean'].max()
    print(f"  {company}: {peak:.0f}")

print(f"\nEV market share most recent year:")
for country in ['Norway', 'China', 'United Kingdom', 'Germany', 'United States']:
    d = ev_master[ev_master['Country'] == country].dropna(subset=['EV_Market_Share_Pct'])
    if len(d) > 0:
        latest = d.sort_values('Year').iloc[-1]
        print(f"  {country}: {latest['EV_Market_Share_Pct']:.1f}% ({int(latest['Year'])})")

print(f"\nglobal EV stocks:")
last = global_stocks.iloc[-1]
print(f"  {last['Stocks_Millions']:.1f} million EVs on the road in {int(last['Year'])}")

# figure 6: regression scatter plot
# visualises the OLS regression already run in 03_analysis.py

print("  figure 6: regression scatter plot")

# preparing the same data used in 03_analysis.py
uk = ev_master[ev_master['Country'] == 'United Kingdom'].sort_values('Year')
uk_trends = trends.copy()
uk_trends['Year'] = pd.to_datetime(uk_trends['Date']).dt.year
uk_annual = uk_trends.groupby('Year')['electric car'].mean().reset_index()
uk_annual.columns = ['Year', 'Search_Interest']
uk_annual['Search_Interest_Lag1'] = uk_annual['Search_Interest'].shift(1)

reg_data = uk[['Year', 'EV_Market_Share_Pct']].merge(
    uk_annual[['Year', 'Search_Interest_Lag1']], on='Year'
).dropna()

fig, ax = plt.subplots(figsize=(9, 6))

# scatter points
ax.scatter(
    reg_data['Search_Interest_Lag1'],
    reg_data['EV_Market_Share_Pct'],
    color='#1B5FAA',
    s=100,
    zorder=5
)

# year labels 
for _, row in reg_data.iterrows():
    ax.annotate(
        str(int(row['Year'])),
        xy=(row['Search_Interest_Lag1'], row['EV_Market_Share_Pct']),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=9,
        color='#1B5FAA'
    )

# regression line using coefficients from 03_analysis.py
# intercept = -2.2791, coefficient = 0.8707
x_line = np.linspace(reg_data['Search_Interest_Lag1'].min(), reg_data['Search_Interest_Lag1'].max(), 100)
y_line = -2.2791 + 0.8707 * x_line
ax.plot(x_line, y_line, color='#FF8C00', linewidth=2, linestyle='--')

ax.set_title('Does Search Interest Predict EV Adoption?\nLagged Google Search Interest vs UK EV Market Share', pad=15)
ax.set_xlabel('Google Search Interest for "Electric Car" (Previous Year)')
ax.set_ylabel('UK EV Market Share (% of new car sales)')
ax.text(
    0.01, -0.15,
    'Source: Google Trends; IEA Global EV Outlook 2025. Regression coefficients from OLS model in 03_analysis.py. R²=0.791, p=0.018.',
    transform=ax.transAxes, fontsize=8, color='grey'
)

plt.tight_layout()
plt.savefig('output/figures/fig6_regression.png', dpi=150, bbox_inches='tight')
plt.close()
print(" saved fig6_regression.png")

print("\nall figures saved to output/figures/")