---
title: Chicago Demand Insights
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
---

# Chicago Demand Insights

Decision-support AI for small Chicago food and beverage businesses. Predicts **how many customers you'll serve each day** — walk-ins, dine-in guests, counter orders — using real weather, events, transit, and calendar signals, then tells you what to do about staffing and inventory.

**Live app:** [huggingface.co/spaces/Berkane-Nexus-Insights/chicago-demand-insights](https://huggingface.co/spaces/Berkane-Nexus-Insights/chicago-demand-insights)

> **Note:** This is a live demo hosted on shared infrastructure at no cost to you. The environment may take **1–3 minutes** to initialize on first load or after a period of inactivity. Once ready, return visits are typically much faster.

**Full pipeline walkthrough:** [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md)

**Customer traffic** (or *foot traffic*) = how many customers you serve in a day — walk-ins, dine-in, counter orders. The app predicts that number and tells you how to staff and prep for it.

---

## What it does

| Capability | Detail |
|------------|--------|
| Demand forecast | Daily prediction of **how many customers to expect** — by region and business type |
| Actionable insights | Staffing, inventory, and day-profile recommendations |
| What-if scenarios | Adjust weather and events to stress-test plans |
| 14-day horizon | Live weather forecasts power predictions up to two weeks ahead |
| Real data foundation | City records, Open-Meteo, CTA, MLB/ESPN schedules — all real inputs |

The training target (`customer_traffic`) is a **synthetic but realistic stand-in for daily customer count** — how many people you serve in a day — built from those same signals (see [target formula](docs/DATA_PIPELINE.md#3-synthetic-target-variable)). **Share your sales or checkout data with us and we retrain the model for your business** — no technical setup on your end.

---

## Model performance

Best model: **XGBoost** (selected by validation adjusted R²)

| Split | MAE | RMSE | R² |
|-------|-----|------|----|
| Validation | 6.76 | 8.49 | 0.816 |
| Test | 6.69 | 8.40 | 0.816 |

---

## Project structure

```
├── app_streamlit/          # Streamlit UI (deployed)
├── src/                    # Python package (deployed)
│   ├── config.py           # Paths, APIs, feature lists
│   ├── prepare.py          # Split + SHAP feature selection
│   ├── inference.py        # Scoring + scenario overrides
│   ├── future_features.py  # Live API fetch for forecasts
│   ├── modeling/           # Train, features, evaluate, persist
│   └── data/
│       └── unified_df.parquet   # Unified ML table (deployed)
├── models/                 # preprocessor.pkl + best_model.pkl (deployed)
├── notebooks/              # Data prep, EDA, modeling (local)
├── DataSourceCSV/          # Raw portal CSVs (local, not in git)
├── docs/                   # Pipeline documentation
├── runner.py               # Train CLI: prepare + fit
├── Dockerfile              # Container config for hosted demo
├── requirements.txt        # Full dev/training deps
└── requirements-hf.txt     # Slim inference deps for Docker
```

---

## Getting started

### 1. Environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Data pipeline (local)

Raw CSVs go in `DataSourceCSV/`. Run the notebooks in order:

1. `notebooks/01_data_loader.ipynb` — load, join, synthetic target → `src/data/unified_df.parquet`
2. `notebooks/02_EDA.ipynb` — exploratory analysis and feature screening
3. `notebooks/03_Modeling.ipynb` — model comparison and evaluation

### 3. Train and save artifacts

```bash
python runner.py
```

Writes `models/preprocessor.pkl`, `models/best_model.pkl`, and split parquets under `src/data/`.

### 4. Run the app locally

```bash
streamlit run app_streamlit/app.py
```

---

## Data sources

| Dataset | Source | Use |
|---------|--------|-----|
| Business licenses | [Chicago Data Portal](https://data.cityofchicago.org/) | Locations, business type |
| Food inspections | Chicago Data Portal | Operating-activity proxy |
| Crimes | Chicago Data Portal | Neighborhood demand signals |
| Weather | [Open-Meteo](https://open-meteo.com/) | Temperature, precip, snow, humidity, wind |
| CTA ridership | Chicago Data Portal (`6iiy-9s97`) | Citywide transit volume |
| Cubs / Bulls games | MLB Stats API / ESPN API | More people near venues on game days |
| Special events | Portal API + curated CSV | Festivals, conventions, street events |
| Pay cycles | Derived (`src/pay_cycle.py`) | Biweekly + semimonthly payday flags |

See [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md) for join keys, feature engineering, and the full synthetic-target formula.

---

## Deployment

The app is a **containerized Streamlit demo** deployed on shared hosting. It bundles the trained model, preprocessor, historical lookup table, and live API clients.

- **Initial load:** as a free demo, the environment may take **1–3 minutes** to spin up on first visit or after idle time; subsequent sessions load much faster
- **Forecast window:** 90 days back through **14 days ahead**
- **14-day cap:** future weather comes from the Open-Meteo forecast API, which provides reliable daily forecasts for the next two weeks only
- **All input signals are real** — weather forecasts, sports schedules, event calendars, CTA profiles, and city records

To redeploy, push to the connected repository. The `Dockerfile` installs `requirements-hf.txt`, copies `app_streamlit/`, `src/`, and `models/`, and starts Streamlit on port 8501.

---

## For business owners

This demo shows that external conditions — weather, holidays, paydays, games, festivals, transit — reliably predict whether a day will be slow or busy. The model captures those relationships with **minimal error** on our synthetic target.

Share your real daily customer counts or sales totals (from your register, Square, Toast, etc.) — **we retrain the model on your data and deliver a custom forecast app** built for your seasonality, weekday mix, and event sensitivity. You run the business; we handle the data science.
