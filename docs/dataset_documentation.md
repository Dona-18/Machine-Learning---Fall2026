# Dataset Guide (Simple English, Why Each Dataset Is Used & References)

This project predicts **tomorrow's daily water level (`WL_next`, in metres)** of the **Bassac River** in Kandal Province, Cambodia (`2020–2026`). Because one single dataset is not enough, we combine **4 real-world datasets**.

---

## 1. Raw Datasets (`data/raw/`) — Where They Come From & Why We Need Each

### 1.1 `Bassac_01_ori.csv` — Koh Thum Water Level (`2020–2024`)
- **What it is:** Automated sensor readings taken every **15 minutes** (`127,606` rows, `2020-06-19` → `2024-05-26`, `0.410–5.620 m`).
- **Why we need it:** It gives us 4 continuous years of year-round water levels (both dry and flood seasons) and sets our **target metre scale** (height above sea level).
- **Why we clean it:** Twice-daily ocean tides push the river up and down by ~0.7 m every day, so we average all 96 readings(24 hours = 24h  x 4readings) (4readings = 60min / 15min) into **1 daily mean**.

### 1.2 `dhrw_water_level_2020_2026.csv` — DHRW Koh Khel Water Level (`2020–2026`)
- **What it is:** Official government daily readings at **07:00 AM** across 7 stations (`1,511` rows, `2020-06-17` → `2026-11-11`).
- **Why we need it:** `Bassac_01` stopped in May 2024. `Koh_Khel` (~15 km upstream on the Bassac River) keeps recording through **November 2026**, extending our timeline by 2.5 years.
- **Why we calibrate it:** `Koh_Khel` uses a lower local zero mark, so its raw numbers read ~1.48 m higher (`1.230–7.880 m`) and cannot be joined directly without causing a fake 1.48 m jump in July 2024.
- **How we got `r = 0.9957` & why:** We paired the **`643` shared days in `2020–2023`** where both stations recorded the same day and calculated Pearson correlation (`WL` vs. `WL_raw`). Getting **`r = 0.9957` (99.6% match)** proves both stations rise and fall together almost identically, so a straight-line conversion (`0.7942 × WL_raw - 0.4756`, fitted strictly on `2020–2023`) is safe and cuts the gap from `1.482 m` down to **`0.093 m` (9.3 cm)** with zero `2024–2026` leakage.

### 1.3 `glofas_discharge_phnompenh.csv` — Upstream Mekong Flow (`2018–2026`)
- **What it is:** Daily river flow volume (`Q`, in `m³/s`, `3,235` rows, `8.980–100,229.510 m³/s`) at **Phnom Penh** (~55 km upstream).
- **Why we need it:** Most Bassac water comes from the upper Mekong passing Phnom Penh. Because water takes **1–2 days** to travel from Phnom Penh to Kandal, upstream flow gives our model an **early warning** before a flood wave arrives.

### 1.4 `POWER_Point_Daily_20200602_20261122_011d05N_105d07E_UTC.csv` — Daily Rainfall (`2020–2026`)
- **What it is:** Daily satellite + gauge rainfall (`PRECTOTCORR`, `mm/day`, `2,363` rows, `0.000–518.150 mm/day`) at Koh Thum (`11.05°N, 105.07°E`).
- **Why we need it:** Local rain adds surface runoff and saturates the soil, following the exact input design of *Priyasiri et al. (2025)*.

---

## 2. Cleaned Datasets (`data/processed/`)

| File | Rows × Cols | Nulls | What Changed During Cleaning |
|---|---|---|---|
| `bassac_daily.csv` | `1,438 × 3` | `WL`: `3` | Averaged 15-min readings to daily means (`WL`: `0.852–5.609 m`); rejected `<90`-reading partial days |
| `kohkhel_daily.csv` | `1,511 × 2` | `0` | Dropped 6 other stations; kept `Koh_Khel` (`WL_raw`: `1.230–7.880 m`) |
| `glofas_daily.csv` | `2,353 × 2` | `0` | Filtered to `2020-06-01` → `2026-11-09`; renamed `Q` to `upstream_discharge` |
| `nasa_rain_daily.csv` | `2,363 × 2` | `0` | Combined `YEAR, MO, DY` into `YYYY-MM-DD`; kept `rain_mm` |
| `modeling_table.csv` | `2,285 × 13` | `0` | Merged all 4 tables, built 10 past-only features + `WL_next` (`80%` Train: `1,828` rows `2020-06-23` → `2025-08-08`; `20%` Test: `457` rows `2025-08-09` → `2026-11-08`) |

---

## 3. References

1. **Bassac_01 Water Level (2020–2024):** Massuel, S. et al. (2024). *Water Level Data from the Prek System of the Cambodian Mekong Delta (2020–2024)*. DataSuds, IRD. https://doi.org/10.23708/E97HGV
2. **DHRW Koh Khel Water Level (2020–2026):** Department of Hydrology and River Works (DHRW), MOWRAM Cambodia. *Daily River Water Level Bulletin*. https://dhrw-cam.org/bulletintbl.php
3. **GloFAS Upstream Discharge:** Copernicus EMS / Open-Meteo. *Global Flood Awareness System (GloFAS) Flood API*. https://open-meteo.com/en/docs/flood-api
4. **NASA POWER Daily Rainfall:** NASA Langley Research Center. *POWER Project Data Access Viewer (`PRECTOTCORR`)*. https://power.larc.nasa.gov/data-access-viewer/
5. **Reference ML Study:** Priyasiri, W. D. S. M. et al. (2025). *Machine Learning for Water Level Prediction in the Chao Phraya River Basin*. Engineering Journal, 29(8), 147–165. https://doi.org/10.4186/ej.2025.29.8.147
