# Bassac River Water Level Forecasting (2020–2026)

Predict tomorrow's daily Bassac River water level (Kandal, Cambodia).

## Folders
- `data/raw/` — original datasets (never edited)

## Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Run (from the repo root, in order)
python src/01_clean_bassac.py
python src/02_clean_kohkhel.py
python src/03_clean_rain.py
python src/04_clean_glofas.py
python src/05_check_clean.py
python src/06_check_calibration.py