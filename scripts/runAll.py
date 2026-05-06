# runs the full project pipeline from start to finish
# execute this file from the project root to replicate all results
# step 1: scrape and collect all raw data
# step 2: clean and process all datasets
# step 3: run the regression analysis
# step 4: generate all figures

import subprocess
import sys

def run(script):
    print(f"\nrunning {script}")
    result = subprocess.run([sys.executable, script], check=True)
    print(f"finished {script}")

run('src/01_scrape.py')
run('src/02_clean.py')
run('src/03_analysis.py')
run('src/04_figures.py')

print("\nall done- check output/figures/ for plots and output/tables/ for regression results")