# America250 Economic Impact Intelligence Platform — Design Spec

**Date:** 2026-07-02 · **Owner:** Chandana Gowda · **Status:** Approved for planning

## 1. Purpose & framing

Portfolio project for Business Analyst / BI Analyst roles. Because it is built *during* the
July 4, 2026 (America's 250th anniversary) travel window, it is framed as a
**backtest + nowcast + validation pipeline**, not a pure forecast:

1. Train on historical travel/holiday patterns (2019–2025).
2. Project the June 27 – July 5, 2026 window.
3. As actuals land (TSA publishes with ~1-day lag; data through July 1, 2026 already exists),
   overlay predicted vs actual and explain the gaps.

Anchor headline numbers (AAA, cited in the dashboard): 72.2M Americans traveling 50+ miles
June 27 – July 5 (vs 71.8M prior year); 61.4M by car (85%); 5.85M flying; 4.93M bus/train/cruise
(+5.3%); ~$830 avg round-trip domestic airfare; gas near four-year highs; top destinations
Seattle, Anchorage, New York, Chicago, Boston, Orlando.

**Honesty rules:** no fabricated data. Paid sources (STR hotel occupancy, card spend) are out.
Every proxy, interpolation, or synthetic series is labeled as such in code and UI.

## 2. Scope

### MVP (this build)
- 3 real ingestion sources: TSA daily throughput, EIA weekly gas prices, Open-Meteo daily weather.
- Unified national daily dataset (2019–2026).
- Prophet forecast of daily TSA throughput + SARIMA baseline, holdout metrics.
- Streamlit dashboard: KPI tiles, actual-vs-forecast chart with July 4 window highlighted,
  gas price chart, city weather strip, sidebar filters.
- "Ask the Analyst" panel on Gemini free tier (live API — user decision).

### Extensions (in order, after MVP review)
1. BTS T-100 airport traffic, Census ACS population + MRTS retail, curated America250 events CSV.
2. City-level composite Economic Impact Index (configurable weights, documented).
3. KMeans city clustering with plain-English cluster labels.
4. IsolationForest anomaly detection on the national daily series.
5. "Predicted vs Actual" validation tab with written post-mortem; light live-refresh of latest
   TSA rows with graceful fallback.
6. Deploy to Streamlit Community Cloud; link from chandanasgowda.com.

## 3. Architecture (decision: precompute & commit)

Ingestion and modeling run **locally as scripts**; outputs in `data/processed/` are
**committed to git**; the deployed Streamlit app reads only committed CSVs.

Rationale: tsa.gov returns 403 to non-browser clients and may block cloud IPs; Prophet training
is too heavy for Streamlit Cloud's free tier; committed artifacts make the public demo fast and
deterministic. Updating actuals = run the refresh pipeline locally, push.

Rejected: live fetch/train in-app (fragile, slow cold start). Hybrid live-refresh of latest TSA
rows is deferred to extension 5.

```
america250-platform/            (repo root = ~/Desktop/USA250)
├── requirements.txt            pandas, prophet, statsmodels, scikit-learn, plotly,
│                               streamlit, requests, python-dotenv, google-genai,
│                               lxml, xlrd, pytest
├── .gitignore                  .env, venv/, data/raw/, __pycache__ …
├── .env.example                GEMINI_API_KEY=
├── run_pipeline.py             single entry point: ingest → build → forecast
├── src/ingest/
│   ├── tsa.py                  scrape year pages 2019–2026 (browser UA, retry, delay)
│   ├── eia_gas.py              keyless XLS download, national + PADD sheets
│   ├── weather.py              Open-Meteo archive API (keyless), 8 cities
│   ├── holidays.py             Nager.Date public holidays (keyless)
│   └── build_dataset.py        unify → data/processed/national_daily.csv (+ QC report)
├── src/model/
│   └── forecast.py             Prophet + SARIMA, metrics, → forecast.csv, metrics.csv
├── app/main.py                 Streamlit dashboard + Ask the Analyst
├── data/raw/                   git-ignored
├── data/processed/             COMMITTED (app's data source)
├── tests/                      pytest, no network
├── notebooks/
└── docs/superpowers/specs/
```

Cities (weather + later city layer): Seattle, Anchorage, New York, Chicago, Boston, Orlando,
Los Angeles, Denver.

## 4. Data sources (verified 2026-07-02)

| Source | Access (verified) | Grain | Notes |
|---|---|---|---|
| TSA passenger volumes | HTML tables at tsa.gov/travel/passenger-volumes[/YYYY]; needs browser User-Agent; **data starts 2019** (spec's 2015 is not publicly available); current through 2026-07-01 | National, daily | Parse with pandas.read_html (lxml) |
| EIA gasoline prices | Keyless XLS: eia.gov/dnav/pet/hist_xls/EMM_EPM0_PTE_NUS_DPGw.xls (verified 200); national + PADD regions | National + PADD, weekly | No API key or signup needed; xlrd for legacy .xls |
| Open-Meteo | archive-api.open-meteo.com, keyless | City, daily | tmax, tmin, precipitation |
| Nager.Date | date.nager.at API, keyless | National, by date | US public holidays 2019–2026 |
| BTS T-100 (ext.) | transtats.bts.gov | Airport, monthly | Extension only |
| Census ACS/MRTS (ext.) | data.census.gov CSV | Nat./state/county | Extension only |
| America250 events (ext.) | manually curated CSV | City | city, date, event_name, expected_attendance |

Grain honesty: national daily time series and city cross-section are kept as **two separate
grains**; the city index (extension) is modeled from the cross-section and labeled as such.

## 5. Unified dataset

`data/processed/national_daily.csv` — one row per date, 2019-01-01 → latest available:

| column | source / rule |
|---|---|
| date | daily calendar spine |
| tsa_throughput | TSA actuals |
| gas_price | EIA national weekly, forward-filled to daily (documented proxy) |
| is_holiday | Nager.Date |
| day_of_week | derived |
| is_july4_window | date in Jun 27 – Jul 5 of any year |
| is_covid_period | 2020-03-01 – 2021-12-31 indicator (see §6) |

Pipeline prints a QC report: row counts, date coverage, null %, and fails loudly on gaps.
Every assumption/interpolation documented inline.

## 6. Forecasting

- **Primary:** Prophet, trained 2019-01-01 → 2025-12-31. US holidays via Prophet's native
  holiday support; extra regressors: gas_price (future values = last observed, documented
  assumption) and is_covid_period (0 for all future dates).
- **COVID handling:** include 2020–21 with the indicator regressor; report a comparison run
  that excludes 2020–21 so the metrics table shows which choice wins.
- **Baseline:** SARIMA (statsmodels), weekly seasonality.
- **Evaluation:** holdout = 2024-01-01 → 2025-12-31; MAE + MAPE for both models in
  `metrics.csv`; plain-English summary of which won and why printed by the pipeline.
- **Output:** `forecast.csv` with history fit + June 27 – July 5, 2026 projection and
  uncertainty intervals.

## 7. Dashboard (Streamlit + Plotly)

Sections, top to bottom:
1. KPI tile row — AAA headline numbers with source citations.
2. National forecast chart — actual vs predicted TSA throughput, July 4 window shaded,
   actuals through the latest date overlaid (predicted-vs-actual is live from day one).
3. Gas price trend (national + PADD selector).
4. City weather strip for the 8 cities across the window.
5. Ask the Analyst panel (§8).

Sidebar: date-range and city filters. `st.cache_data` on all loads. Styling: corporate-minimal,
neutral palette, one accent color; follow dataviz skill when building charts. Extensions add:
city index map/bar, cluster view, anomaly timeline, validation tab.

## 8. Ask the Analyst (Gemini)

- SDK: **google-genai** (current SDK; spec's `google-generativeai` is deprecated).
- Model: `gemini-2.5-flash`, falling back to `gemini-2.5-flash-lite` on 429, with exponential
  backoff. Free tier, no credit card.
- Context: compact summaries of the processed dataframes (recent actuals, forecast window,
  metrics, gas trend) + the user's question.
- System prompt: answer only from provided data; cite the specific numbers used; say when the
  data is insufficient.
- `GEMINI_API_KEY` from `.env` (never committed). No key → panel renders a friendly setup
  notice instead of crashing, so the public deploy is safe.
- 3 example questions as clickable buttons.

## 9. Testing & verification

- pytest (no network): TSA/EIA parsing on saved fixtures, ffill logic, window/holiday flags,
  dataframe-summary builder for the AI panel.
- Pipeline QC report as runtime verification.
- Manual end-to-end: `python run_pipeline.py` then `streamlit run app/main.py`, verify charts
  and AI panel against known numbers.

## 10. README, guardrails & positioning (spec §5–6)

Guardrails (binding):
- Only real, free, cited data; label every proxy or interpolation in code and UI.
- Ship the MVP end-to-end before clustering, anomaly detection, or the city layer.
- Never commit `.env` or any API key.
- Deploy to Streamlit Community Cloud and link from chandanasgowda.com (extension 6).

README must include: project goal, dashboard screenshot, "Data sources and caveats" section
(incl. the 2019 constraint and all proxies), a one-paragraph methodology note, setup/run
instructions, and link to the live app once deployed.

`docs/positioning.md` captures the employer-specific one-line pitches (retail: demand
forecast → inventory prep; consulting: decision tool from messy public sources with explicit
tradeoffs, validated against reality; travel: multi-mode demand + destination clusters;
government: economic footprint of a federal commemoration from public data only) and the three
interview soundbites (index/weight discipline, the hotel-proxy decision, predicted-vs-actual
learnings). Kept out of the README so the repo stays employer-neutral.

## 11. Risks & assumptions

- TSA may block or reshape its pages → fixtures committed for tests; raw cache in data/raw.
- Weekly→daily gas ffill and last-observed future gas are labeled proxies.
- 2019 training start (not 2015) is a hard public-data constraint, stated in the README.
- Gemini free-tier limits (~1,500 req/day) are fine for a portfolio demo; backoff + fallback
  model handle bursts.
