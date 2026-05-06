.PHONY: all

all:
	python3 scripts/01_scrape.py
	python3 scripts/02_clean.py
	python3 scripts/03_analysis.py
	python3 scripts/04_figures.py