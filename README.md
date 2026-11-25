# Bassac River Daily Water Level Forecasting (2020–2026)

Predict **tomorrow's daily Bassac River water level (`WL_next`, in metres)** in Kandal Province, Cambodia using past water levels, upstream Mekong river discharge, and local daily rainfall.

---

## 1. Project Purpose & Motivation

At **Phnom Penh (Chaktomuk)**, the Mekong River splits into the Mekong mainstream and the **Bassac River**, which flows south through **Kandal Province** (passing **Koh Khel** and **Koh Thum**) toward Vietnam.

Why this project matters:
1. **Agricultural Timing:** Farmers in Kandal rely on knowing tomorrow's river height to schedule rice planting, irrigation pumping, and harvesting.
2. **Flood Early Warning:** Even 1 day of advance warning during rapid `Sep–Oct` monsoon rises helps communities protect crops, boats, and low-lying roads.
3. **Changing River Regime:** Upstream dams and shifting monsoon patterns make static historical calendar averages unreliable.
4. **Beating `tomorrow = today`:** While `tomorrow = today` (`NaiveBaseline`) looks decent on calm days (`5.82 cm` MAE), it fails badly on fast-changing flood days (`17.68 cm` MAE). Our **Ridge Regression** model cuts overall error by **`36.19%`** (`3.72 cm`) and fast-change error by **`44.6%`** (`9.80 cm`).

---

## 2. Datasets (`data/raw/` → `data/processed/`)

| Dataset | Source | Resolution & Range | Role |
|---|---|---|---|
| `Bassac_01_ori.csv` | ITC / IRD DataSuds (Koh Thum) | 15-min → Daily mean (`2020–2024`) | Primary target (tidal noise removed) |
| `dhrw_water_level_2020_2026.csv` | DHRW / MOWRAM Cambodia (`Koh_Khel`) | Daily 07:00 (`2020–2026`) | Extended target (`2024–2026`, calibrated via `2020–2023` overlap `r = 0.9957`) |
| `glofas_discharge_phnompenh.csv` | Copernicus GloFAS via Open-Meteo | Daily (`2020–2026`) | Upstream Mekong discharge (`m³/s`) ~55 km north |
| `POWER_Point_Daily_...csv` | NASA POWER (`PRECTOTCORR`, `11.05°N, 105.07°E`) | Daily (`2020–2026`) | Local daily rainfall (`mm/day`) |

---

## 3. Features (`X`) & Target (`y`)

All 10 features use **only data from today (`t`) or earlier** to prevent future leakage:

1. **`WL`** (`m`): Today's Bassac water level — strongest base-height anchor.
2. **`WL_lag1`** (`m`): Yesterday's water level (`t - 1`) — establishes short-term direction.
3. **`WL_lag2`** (`m`): Water level 2 days ago (`t - 2`) — captures acceleration or peak flattening.
4. **`WL_lag3`** (`m`): Water level 3 days ago (`t - 3`) — 3-day trend memory & noise smoothing.
5. **`WL_diff`** (`m/day`): `WL(t) - WL(t - 1)` — river momentum (rising vs. draining).
6. **`upstream_discharge_lag1`** (`m³/s`): Yesterday's Mekong flow at Phnom Penh (1–2 day travel time to Kandal).
7. **`upstream_discharge_diff`** (`m³/s/day`): `Q(t) - Q(t - 1)` — early warning of incoming flood waves.
8. **`rain_mm`** (`mm/day`): Today's local rainfall at Koh Thum.
9. **`rain_lag1`** (`mm/day`): Yesterday's rainfall (delayed land runoff).
10. **`rain_3d`** (`mm`): 3-day cumulative rainfall (soil saturation & runoff readiness).
- **Target (`y` = `WL_next`):** Tomorrow's water level (`t + 1`, in metres).

---

## 4. Final Test Set Results (`80/20` Split: `2025-08-09` → `2026-11-08`, `457` Unseen Days)

| Model | MAE (m) | RMSE (m) | R² | Gain vs. Naive | Fast-Change Days MAE (`n=52`) |
|---|---|---|---|---|---|
| **Ridge Regression** | **0.03189** | **0.05219** | **0.99915** | **+38.27%** | **9.53 cm** |
| **NaiveBaseline (`WL_next = WL`)** | 0.05165 | 0.08188 | 0.99790 | 0.00% | 18.18 cm |
| **RandomForest** | 0.05518 | 0.09255 | 0.99732 | -6.83% | 12.16 cm |
| **XGBoost** | 0.05542 | 0.09409 | 0.99723 | -7.29% | 10.59 cm |

---

## 5. Setup & Run Order

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the 3 notebooks in order inside `notebooks/`:
1. `notebooks/01_eda_and_cleaning.ipynb` — raw audit, tidal averaging, gap handling, and daily visualizations.
2. `notebooks/02_calibration_and_features.ipynb` — station calibration (`r = 0.9957`), `2020–2026` merge, 10 features + `WL_next`, and leakage tests.
3. `notebooks/03_train_and_evaluate.ipynb` — chronological `80/20` split (`1,828` Train / `457` Test), `TimeSeriesSplit`, model training, and evaluation plots (`reports/ml/`).
