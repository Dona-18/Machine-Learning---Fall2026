# Bassac River Water Level Forecasting (2020–2026)

Predict tomorrow's daily Bassac River water level (Kandal, Cambodia).

## Folders
- `data/raw/` — original datasets (never edited)

## Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Run (in order, inside `notebooks/`)
1. `notebooks/00_data_cleaning_and_calibration.ipynb`
2. `notebooks/01_data_prep_and_features.ipynb`
3. `notebooks/02_train_and_evaluate.ipynb`