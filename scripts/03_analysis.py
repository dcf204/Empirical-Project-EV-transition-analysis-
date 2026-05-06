# script runs an OLS regression to test whether UK google search interest statistically predicts real EV adoption one year later
# if hype leads reality, then search interest this year should predict actual EV sales next year

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

os.makedirs('output/tables', exist_ok=True)

print("loading data for regression")

# loading the two datasets 
# uk google trends as measure of consumer interest and hype
# uk ev market share as measure of real adoption 
ev_master = pd.read_csv('data/clean/ev_master_clean.csv')
trends = pd.read_csv('data/clean/google_trends_clean.csv')

trends['Date'] = pd.to_datetime(trends['Date'])

# get UK EV market share by year
uk_ev = ev_master[ev_master['Country'] == 'United Kingdom'][['Year', 'EV_Market_Share_Pct']].copy()
uk_ev = uk_ev.dropna(subset=['EV_Market_Share_Pct'])
uk_ev = uk_ev.sort_values('Year')

print(f"\nUK EV market share data:")
print(uk_ev)

# aggregate google trends to yearly averages
# the trends data is monthly so must take the annual mean for electric car searches
trends['Year'] = trends['Date'].dt.year
trends_annual = trends.groupby('Year')['electric car'].mean().reset_index()
trends_annual.columns = ['Year', 'Search_Interest']

print(f"\nannual search interest data:")
print(trends_annual)

# merge the two datasets on year so both the dependent variable (EV market share) and independent variable (search interest) are in the same dataframe for regression
reg_data = uk_ev.merge(trends_annual, on='Year', how='inner')

# creating the lagged search interest variable
# shifting search interest forward by one year
# so 2020 search interest gets paired with 2021 EV market share
reg_data = reg_data.sort_values('Year')
reg_data['Search_Interest_Lag1'] = reg_data['Search_Interest'].shift(1)

# dropping the first year since it has no lag value to work with
reg_data = reg_data.dropna(subset=['Search_Interest_Lag1'])

print(f"\nregression dataset: {len(reg_data)} observations")
print(f"years covered: {reg_data['Year'].min()} to {reg_data['Year'].max()}")
print(f"\ndata used in regression:")
print(reg_data[['Year', 'EV_Market_Share_Pct', 'Search_Interest', 'Search_Interest_Lag1']].to_string(index=False))

# run the OLS regression using statsmodels
# dependent variable: UK EV market share
# independent variable: google search interest for 'electric car' lagged one year
model = smf.ols(
    formula='EV_Market_Share_Pct ~ Search_Interest_Lag1',
    data=reg_data
).fit()

print("\nregression results:")
print(model.summary())

# saving results to a text file in output/tables/
with open('output/tables/regression_results.txt', 'w') as f:
    f.write("OLS Regression: Does Search Interest Predict EV Adoption?\n\n")
    f.write("Dependent variable: UK EV Market Share (% of new car sales)\n")
    f.write("Independent variable: Google Search Interest for 'electric car' (lagged 1 year)\n")
    f.write("Sample: UK, 2019-2024\n\n")
    f.write(str(model.summary()))
    f.write("\n\nKey findings:\n")
    f.write(f"  coefficient on lagged search interest: {model.params['Search_Interest_Lag1']:.4f}\n")
    f.write(f"  p-value: {model.pvalues['Search_Interest_Lag1']:.4f}\n")
    f.write(f"  r-squared: {model.rsquared:.4f}\n")

    # interpreting results 
    if model.pvalues['Search_Interest_Lag1'] < 0.05:
        f.write("\n  interpretation: statistically significant at the 5% level.\n")
        f.write(f"  a 1-point increase in search interest is associated with a ")
        f.write(f"{model.params['Search_Interest_Lag1']:.4f} percentage point\n")
        f.write(f"  increase in EV market share the following year.\n")
    else:
        f.write("\n  interpretation: not statistically significant at the 5% level.\n")
        f.write("  search interest alone does not reliably predict adoption one year ahead.\n")

print("\nsaved to output/tables/regression_results.txt")
print(f"\nsummary:")
print(f"  coefficient: {model.params['Search_Interest_Lag1']:.4f}")
print(f"  p-value: {model.pvalues['Search_Interest_Lag1']:.4f}")
print(f"  r-squared: {model.rsquared:.4f}")