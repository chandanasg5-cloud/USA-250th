# City Impact Layer — Design (Extensions 1–3)

Date: 2026-07-07 · Status: approved · Parent spec: `2026-07-02-america250-platform-design.md` §2 extensions 1–3

## 1. Purpose & framing

Add a city-grain intelligence layer to the America250 platform: which of the 8 focus
cities were positioned to capture the most America250 economic activity, and what the
observable data says so far.

July 4, 2026 has passed and BTS T-100 publishes with a ~2–3 month lag, so the index is
framed honestly as **exposure** (structural capacity to capture anniversary demand)
paired with an **observed-lift readout** where data exists (YoY air-passenger momentum
through the latest published month). When July 2026 T-100 data publishes (~Oct 2026),
one refresh completes the observed side — mirroring the national validation-tab story.

Extension 4 (IsolationForest anomaly detection on the national series) is explicitly
out of scope; it gets its own design cycle.

## 2. Source-access decisions (verified 2026-07-07)

| Source | Verified state | Decision |
|---|---|---|
| BTS T-100 Domestic Segment | Pre-zipped files 404; legacy `DownLoad_Table.asp` POST returns 500; bts.gov blocks non-browser clients (403); no Socrata mirror | **Documented manual download** from TranStats (browser), raw zip in git-ignored `data/raw/`; scripted parser commits a slim reference CSV |
| Census ACS | Keyless access removed — API now 302s to `missing_key` | **Free API key** (`CENSUS_API_KEY` in `.env`, same pattern as Gemini key); scripted ingestion |
| America250 events | No machine-readable source exists | **Researched, curated CSV** — real events only, every row carries a `source_url` |
| Census MRTS retail | National grain only | **Dropped** from the city layer (grain honesty); may return later as national context |

## 3. Data & ingestion

### 3.1 BTS T-100 (`src/ingest/bts_t100.py`)

- Input: manually downloaded T-100 Domestic Segment (all carriers) zip/CSV for
  2023–latest, placed in `data/raw/`. README documents the exact TranStats click path.
- Parser filters to focus airports, aggregates to city × month passenger totals, and
  writes committed `data/reference/t100_city_monthly.csv`
  (`year, month, airport, city, passengers`).
- City → airport mapping (explicit constant, documented in README):
  Seattle=SEA · Anchorage=ANC · New York=JFK+LGA+EWR · Chicago=ORD+MDW · Boston=BOS ·
  Orlando=MCO · Los Angeles=LAX · Denver=DEN.
- Pipeline reads the committed reference CSV, so `run_pipeline.py` works without a
  fresh raw download. Refresh cadence: monthly, when BTS publishes.

### 3.2 Census ACS (`src/ingest/census_acs.py`)

- Latest ACS 1-year estimates via API for the 8 places: total population
  (`B01003_001E`) and median household income (`B19013_001E`).
- Requires `CENSUS_API_KEY` in `.env` (free, instant signup; `.env.example` updated).
  Missing key → clear pipeline error (pipeline is local-only, so no runtime fallback
  needed in the deployed app).
- Fallback if a place is absent from ACS 1-year: use ACS 5-year for that place and
  flag it in the output.
- Output: committed `data/processed/city_static.csv`.

### 3.3 America250 events (`data/reference/america250_events.csv`)

- Researched real events (e.g., Sail4th 250 in NYC, official America250 programming),
  curated at implementation time via web research.
- Columns: `city, date, event_name, scale_tier, source_url`. Scale tiers
  (1=local, 2=regional, 3=national flagship) avoid fake attendance precision.

## 4. Index & clustering

### 4.1 Exposure index (`src/model/city_index.py`)

- Four components, each min–max scaled to 0–100 across the 8 cities:
  **air capacity** (T-100 passengers, latest 12 observed months), **events intensity**
  (count weighted by scale tier), **population** (ACS), **income** (ACS median HH).
- Composite = weighted sum. Default weights in one config dict:
  air 0.40, events 0.30, population 0.20, income 0.10 — air and events carry the
  anniversary-specific signal; demographics are amplifiers. Rationale documented.
- Output: committed `data/processed/city_index.csv` (component scores, default
  composite, cluster label).

### 4.2 Observed-lift readout

- YoY change in T-100 passengers per city for the latest available months, labeled
  "data through {month} — July 2026 actuals publish ~Oct 2026".
- Computed in `city_index.py` from the reference CSV; written to committed
  `data/processed/city_momentum.csv`
  (`city, month, passengers, passengers_prior_year, yoy_pct`).

### 4.3 Clustering (`src/model/city_cluster.py`)

- KMeans, k=3, fixed seed, on standardized component features; writes the cluster
  label column into `city_index.csv`.
- Plain-English labels derived deterministically from centroid rankings (final wording
  from real centroids). Framed as illustrative segmentation of 8 cities, not inference;
  caption says so.

### 4.4 Determinism & testing

- Pure-function transforms, pytest fixtures, no network in tests (existing pattern).
- Fixed seeds and sorted outputs so pipeline reruns yield identical committed CSVs.

## 5. Dashboard & integration

New **City Impact** tab in the Streamlit app:

1. **Index leaderboard** — horizontal bars of composite with component breakdown;
   four weight sliders recompute the composite in-app from committed component scores
   (default weighting stays canonical, "reset to default" noted).
2. **Cluster view** — scatter of the two dominant components, colored by cluster,
   plain-English labels.
3. **Air momentum** — YoY-change chart per city with the data-through label.
4. **Events table** — curated events with clickable source links (doubles as the
   citations page).
5. **Methodology caption** — grain honesty (city cross-section vs national daily),
   T-100 lag, illustrative-clustering caveat.

Integration:
- `run_pipeline.py` gains steps: census → t100 parse → index → cluster.
- "Ask the Analyst" context gains the city index summary.
- README: City Impact section, refresh instructions (including the manual T-100
  download path), updated data-sources table.

## 6. Risks & assumptions

- TranStats download UI may change — manual step is documented and low-frequency.
- Census key quota 500/day vs our 8 calls — no risk.
- ACS 1-year may omit a place — 5-year fallback, flagged.
- Events list is a judgment call — mitigated by per-row citations and scale tiers.
- 8 observations for KMeans — framed as illustrative segmentation, never "statistically
  distinct groups".
