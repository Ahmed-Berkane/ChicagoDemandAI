# ChicagoDemandAI

Decision-making AI system for small Chicago businesses that predicts demand and tells owners what to do next.

## Project structure

```
├── src/                  # Python package (deployed)
├── requirements.txt      # dependencies (deployed)
├── models/               # saved preprocessor + model (deployed)
├── data/                  # raw portal files and curated event calendar
├── app_streamlit/        # Streamlit app (deployed)
├── scripts/              # local CLI — not in git (regenerate data/models)
├── notebooks/            # EDA / training — not in git
├── docs/                 # documentation — not in git
└── data/                 # large CSVs — not in git (see scripts/)
```

**Git** only tracks deployment files. Large datasets and notebooks stay on your machine; use `scripts/fetch_external_data.py` and `scripts/build_unified_dataset.py` to rebuild.

**Full data guide:** [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md) — where each dataset comes from, what it contains, and how it joins into `unified_daily_demand.parquet`.

**Modeling notebook:** `notebooks/02_modeling.ipynb` — feature engineering, EDA, train/val/test, baselines + LSTM/GRU, saves `models/preprocessor.pkl` and `models/best_model.pkl`.

## Getting started

Create and activate a virtual environment before installing dependencies.

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

Then run the initial data load:

```bash
python scripts/load_data.py --save-parquet
```

Options:

- `--coffee-shops-only` — filter business licenses for café/coffee MVP
- `--full-crimes` — load full 2001–present crime file (~470 MB) instead of one-year extract

Open `notebooks/01_load_data.ipynb` for an interactive walkthrough.

## Data loaded (Phase 1)

| Dataset | Rows (May 2026 export) | Use |
|---------|------------------------|-----|
| Business licenses | ~53k active | Locations, business type |
| Food inspections | ~310k | Operating activity proxy |
| Crimes (1 year) | ~233k | Neighborhood demand shocks |
| Special events | ~595 | Event-driven traffic spikes |

**External (fetched via API):**

```bash
python scripts/fetch_external_data.py --all
```

| Source | File | Roadmap fields |
|--------|------|----------------|
| [Open-Meteo archive](https://open-meteo.com/en/docs/historical-weather-api) | `data/external/weather_chicago_daily.csv` | temp, precip, snow, humidity, wind chill |
| [Chicago Building Permits API](https://data.cityofchicago.org/d/ydr8-5enu) | `data/external/building_permits.csv` | seasonality / construction activity |
| Derived | `data/external/calendar_features.csv` | weekends, US (IL) holidays |

**Unified ML table (built):**

```bash
# Default window: 2019-01-01 → today (~2,700 daily rows)
python scripts/build_unified_dataset.py --fetch

# Custom range:
python scripts/build_unified_dataset.py --fetch --start 2017-01-01 --end 2026-05-19
```

Output: **`data/processed/data.csv`** — one row per day, one real licensed café (modeling-ready).

```python
from src.unified_dataset import load_modeling_data
data = load_modeling_data()   # or: df = load_modeling_data()
y = data["customer_traffic"]  # TARGET (synthetic)
```

| Component | Method |
|-----------|--------|
| `customer_traffic` | **Synthetic** from weather, weekends, holidays, Cubs/Bulls, festivals, CTA, payweek |
| Weather | Open-Meteo |
| Cubs home games | MLB Stats API |
| Bulls home games | ESPN API (0 games in May'25–May'26 window until 2025–26 schedule is published) |
| CTA ridership | Chicago Data Portal (`6iiy-9s97`) |
| Pay cycles | Derived (biweekly + semimonthly) |
| Festivals | Keyword match on city `Special_Events` |

**Not implemented (optional):** Yelp API, Eventbrite/Ticketmaster, NYC taxi/Kaggle proxies, real Shopify/Square.

Load portal + external together:

```python
from src.data_loader import load_all
from src.unified_dataset import load_unified_daily_dataset

datasets = load_all(include_external=True)
ml = load_unified_daily_dataset()
```
