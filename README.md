# America250 Economic Impact Intelligence Platform

A **backtest + nowcast + validation pipeline** for the economic impact of the
July 4, 2026 travel window — America's 250th anniversary. Models are trained on
historical holiday-travel patterns (2019–2025), project the June 27 – July 5,
2026 window, and are validated against actuals as TSA publishes them (~1-day
lag). Built entirely from **free, public, cited data** — every proxy and
interpolation is labeled.

![Dashboard](docs/img/dashboard.png)

## Headline result (as of July 3, 2026)

- Best model: **Prophet trained 2019–2023 with a COVID-period indicator** —
  **6.33% MAPE** on a full 2024–2025 holdout, beating a SARIMA baseline (9.22%)
  and a COVID-excluded Prophet variant (9.73%).
- **All six July-window actuals so far (Jun 27 – Jul 2) fall inside the
  forecast's uncertainty interval**, running ~2–6% above the point forecast —
  consistent with AAA's record 72.2M-traveler projection.

## Data sources & caveats

| Source | What | Grain | Caveats |
|---|---|---|---|
| [TSA passenger volumes](https://www.tsa.gov/travel/passenger-volumes) | Daily checkpoint throughput | National, daily | Public data starts **2019** (not earlier); site requires a browser User-Agent |
| [EIA gasoline prices](https://www.eia.gov/petroleum/gasdiesel/) | Weekly retail price, US + PADD 1–5 | Regional, weekly | **Proxy:** forward-filled weekly→daily for modeling; keyless XLS download |
| [Open-Meteo](https://open-meteo.com) | Daily weather, 8 focus cities | City, daily | Archive lags ~5 days; forecast API fills through the window |
| [Nager.Date](https://date.nager.at) | US public holidays | National | Nationwide holidays only |

Deliberately **excluded**: STR hotel occupancy and real-time card spending —
both are paid/proprietary. Nothing here is fabricated; where a free proxy
stands in (gas ffill, future gas held at last observed value), the dashboard
and code say so explicitly.

## Methodology

Daily TSA throughput is forecast with Prophet (native US holiday effects;
extra regressors: national gas price and a 2020-03→2021-12 COVID indicator so
pandemic collapse/recovery doesn't contaminate seasonality). Two Prophet
variants (2019+ with the indicator vs 2022+ excluding COVID) and a SARIMA(1,1,1)
×(1,1,1,7) baseline are compared on a held-out 2024–2025 period (MAE/MAPE in
`data/processed/metrics.csv`); the winner is refit through 2025-12-31 and
projects Jan 1 – Jul 5, 2026, so 2026 actuals can be overlaid as they land.

## Architecture

Precompute-and-commit: ingestion and modeling run locally; the Streamlit app
reads only the committed `data/processed/` CSVs (fast cold start, no scraping
from cloud IPs, no training on the free tier).

```
run_pipeline.py          ingest -> build -> forecast (local)
src/ingest/              tsa.py · eia_gas.py · weather.py · holidays.py · build_dataset.py
src/model/forecast.py    Prophet variants + SARIMA baseline + metrics
data/processed/          committed outputs — the app's only data source
app/                     Streamlit dashboard + Gemini "Ask the Analyst" panel
tests/                   pytest (parsing/transform contracts; no network)
```

## Run it

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python run_pipeline.py        # refresh data + forecast (optional; outputs are committed)
venv/bin/streamlit run app/main.py
venv/bin/pytest                        # tests, no network needed
```

**Ask the Analyst** (optional): put a free Gemini key (aistudio.google.com — no
credit card) in `.env` as `GEMINI_API_KEY=...`. The panel answers questions
strictly from the processed data, citing the numbers it uses
(`gemini-2.5-flash`, falling back to `gemini-2.5-flash-lite` on rate limits).
Without a key the panel shows setup instructions and everything else works.

## Roadmap

City-level layer (BTS T-100, Census ACS/MRTS, America250 event calendar) → 
composite city Economic Impact Index → KMeans destination clusters → 
IsolationForest anomaly flags → predicted-vs-actual post-mortem tab → 
Streamlit Community Cloud deployment.

---
Built by [Chandana Gowda](https://chandanasgowda.com) · July 2026
