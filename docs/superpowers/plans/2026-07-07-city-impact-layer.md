# City Impact Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the city-grain intelligence layer (BTS T-100 + Census ACS + curated events → exposure index + clustering + momentum → City Impact dashboard tab) per `docs/superpowers/specs/2026-07-07-city-impact-layer-design.md`.

**Architecture:** Precompute-and-commit, same as MVP: ingestion/modeling run locally via `run_pipeline.py`; outputs in `data/processed/` (and slim reference CSVs in `data/reference/`) are committed; the deployed Streamlit app reads only committed CSVs. T-100 raw data arrives by documented manual browser download (TranStats is script-hostile, verified 2026-07-07); Census ACS is scripted behind a free API key; events are a researched, cited CSV.

**Tech Stack:** pandas, requests, scikit-learn (KMeans, already in requirements.txt), Streamlit + Plotly, pytest (fixtures only, no network in tests).

## Global Constraints

- Tests never touch the network (existing repo rule; all tests use inline fixtures).
- Deterministic pipeline: fixed seeds (`random_state=42`, `n_init=10`), sorted outputs — reruns yield byte-identical committed CSVs.
- Grain honesty: city cross-section is never mixed into the national daily series.
- The deployed app must keep working when city CSVs are absent (graceful tab fallback) — Streamlit Cloud redeploys on every push.
- The 8 focus cities: Anchorage, Boston, Chicago, Denver, Los Angeles, New York, Orlando, Seattle.
- Run tests with `venv/bin/python -m pytest tests/<file> -v` from the repo root.
- Commit after every green task; never commit `data/raw/` (git-ignored) or `.env`.

## File Structure

- `src/ingest/census_acs.py` — ACS place-level population + income → `data/processed/city_static.csv`
- `src/ingest/bts_t100.py` — parse manual T-100 download → `data/reference/t100_city_monthly.csv` (committed)
- `data/reference/america250_events.csv` — researched, cited events (committed)
- `src/model/city_index.py` — component scores, composite, momentum → `city_index.csv`, `city_momentum.csv`
- `src/model/city_cluster.py` — KMeans + plain-English labels, writes cluster columns into `city_index.csv`
- `app/data.py`, `app/charts.py`, `app/analyst.py`, `app/main.py` — City Impact tab
- `run_pipeline.py` — wire new steps
- `tests/test_census_acs.py`, `tests/test_bts_t100.py`, `tests/test_events_csv.py`, `tests/test_city_index.py`, `tests/test_city_cluster.py` — plus additions to `tests/test_app_data.py` / `tests/test_analyst.py`

---

### Task 1: Census ACS ingestion

**Files:**
- Create: `src/ingest/census_acs.py`
- Create: `tests/test_census_acs.py`
- Modify: `.env.example` (add `CENSUS_API_KEY=`)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `data/processed/city_static.csv` with columns `city, population, median_hh_income, acs_source` (8 rows, one per focus city). Functions: `parse_acs_row(rows: list, city: str, source: str) -> dict`, `fetch_place(city, state, place, key) -> dict`, `main() -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_census_acs.py
import pandas as pd
import pytest

from src.ingest.census_acs import PLACES, parse_acs_row


def test_parse_acs_row():
    rows = [
        ["NAME", "B01003_001E", "B19013_001E", "state", "place"],
        ["Seattle city, Washington", "755078", "116340", "53", "63000"],
    ]
    rec = parse_acs_row(rows, "Seattle", "acs1_2024")
    assert rec == {"city": "Seattle", "population": 755078,
                   "median_hh_income": 116340, "acs_source": "acs1_2024"}


def test_parse_acs_row_null_estimate_is_nan():
    rows = [
        ["NAME", "B01003_001E", "B19013_001E", "state", "place"],
        ["Somewhere", "1000", None, "99", "00001"],
    ]
    rec = parse_acs_row(rows, "Somewhere", "acs1_2024")
    assert pd.isna(rec["median_hh_income"])


def test_places_covers_all_8_focus_cities():
    assert sorted(PLACES) == ["Anchorage", "Boston", "Chicago", "Denver",
                              "Los Angeles", "New York", "Orlando", "Seattle"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_census_acs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ingest.census_acs'`

- [ ] **Step 3: Write the implementation**

```python
# src/ingest/census_acs.py
"""Census ACS place-level population + median household income.

The API requires a free key since ~2025 (verified 2026-07-07: keyless
calls 302 to missing_key). CENSUS_API_KEY lives in .env like the Gemini
key. 8 calls per run — far under the 500/day quota. ACS 1-year preferred;
falls back to 5-year per place when 1-year has no usable estimate.
"""
import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv

YEAR = 2024
URL = "https://api.census.gov/data/{year}/acs/{dataset}"
VARS = "B01003_001E,B19013_001E"  # total population, median HH income
PLACES = {  # city -> (state FIPS, place FIPS)
    "Anchorage": ("02", "03000"),
    "Boston": ("25", "07000"),
    "Chicago": ("17", "14000"),
    "Denver": ("08", "20000"),
    "Los Angeles": ("06", "44000"),
    "New York": ("36", "51000"),
    "Orlando": ("12", "53000"),
    "Seattle": ("53", "63000"),
}
OUT_PATH = "data/processed/city_static.csv"


def parse_acs_row(rows: list, city: str, source: str) -> dict:
    rec = dict(zip(rows[0], rows[1]))
    pop = pd.to_numeric(rec["B01003_001E"], errors="coerce")
    income = pd.to_numeric(rec["B19013_001E"], errors="coerce")
    return {"city": city, "population": pop,
            "median_hh_income": income, "acs_source": source}


def fetch_place(city: str, state: str, place: str, key: str) -> dict:
    for dataset in ("acs1", "acs5"):  # 5-year is the per-place fallback
        resp = requests.get(
            URL.format(year=YEAR, dataset=dataset),
            params={"get": f"NAME,{VARS}", "for": f"place:{place}",
                    "in": f"state:{state}", "key": key},
            timeout=60)
        if resp.status_code == 200 and resp.text.strip():
            rec = parse_acs_row(resp.json(), city, f"{dataset}_{YEAR}")
            if pd.notna(rec["population"]) and pd.notna(rec["median_hh_income"]):
                return rec
    raise RuntimeError(f"ACS returned no usable estimate for {city}")


def main() -> None:
    load_dotenv()
    key = os.getenv("CENSUS_API_KEY", "")
    if not key:
        raise RuntimeError(
            "CENSUS_API_KEY missing — free instant signup at "
            "https://api.census.gov/data/key_signup.html, then add to .env")
    records = []
    for city, (state, place) in sorted(PLACES.items()):
        records.append(fetch_place(city, state, place, key))
        time.sleep(0.3)
    df = pd.DataFrame(records)
    df.to_csv(OUT_PATH, index=False)
    print(f"census: {len(df)} cities, sources {sorted(df.acs_source.unique())}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_census_acs.py -v`
Expected: 3 PASS

- [ ] **Step 5: Add `CENSUS_API_KEY=` line to `.env.example`** (below `GEMINI_API_KEY=`).

- [ ] **Step 6: Commit**

```bash
git add src/ingest/census_acs.py tests/test_census_acs.py .env.example
git commit -m "feat: Census ACS ingestion for city population + income"
```

---

### Task 2: BTS T-100 parser

**Files:**
- Create: `src/ingest/bts_t100.py`
- Create: `tests/test_bts_t100.py`

**Interfaces:**
- Consumes: nothing from other tasks. Raw input: a manually downloaded TranStats T-100 Domestic Segment CSV/zip in `data/raw/` matching `*T100*` (columns include `YEAR, MONTH, ORIGIN, PASSENGERS` in any case).
- Produces: committed `data/reference/t100_city_monthly.csv` with columns `year, month, airport, city, passengers` (int). Constant `AIRPORT_TO_CITY: dict[str, str]`; functions `parse_t100(raw: pd.DataFrame) -> pd.DataFrame`, `main() -> None`. `main()` keeps the committed CSV untouched when no raw download exists.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bts_t100.py
import pandas as pd

from src.ingest.bts_t100 import AIRPORT_TO_CITY, parse_t100


def _raw():
    # Shape of a TranStats T-100 Domestic Segment download (uppercase cols)
    return pd.DataFrame({
        "YEAR": [2026, 2026, 2026, 2026, 2026],
        "MONTH": [3, 3, 3, 3, 3],
        "ORIGIN": ["JFK", "LGA", "SEA", "XNA", "ORD"],
        "DEST": ["LAX", "ORD", "ANC", "ORD", "SEA"],
        "PASSENGERS": [10000.0, 5000.0, 8000.0, 900.0, 0.0],
    })


def test_parse_t100_maps_airports_to_cities_and_aggregates():
    out = parse_t100(_raw())
    assert list(out.columns) == ["year", "month", "airport", "city", "passengers"]
    assert "XNA" not in out.airport.values          # non-focus airport dropped
    assert "ORD" not in out.airport.values          # zero-passenger row dropped
    ny = out[out.city == "New York"]
    assert set(ny.airport) == {"JFK", "LGA"}        # multi-airport city kept per airport
    assert out.passengers.dtype.kind == "i"


def test_airport_mapping_covers_all_8_cities():
    assert sorted(set(AIRPORT_TO_CITY.values())) == [
        "Anchorage", "Boston", "Chicago", "Denver",
        "Los Angeles", "New York", "Orlando", "Seattle"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_bts_t100.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ingest.bts_t100'`

- [ ] **Step 3: Write the implementation**

```python
# src/ingest/bts_t100.py
"""BTS T-100 Domestic Segment -> city-month passenger totals.

TranStats has no script-friendly endpoint (verified 2026-07-07: PREZIP
404s, the legacy DownLoad_Table.asp POST 500s, bts.gov 403s non-browser
clients). The raw zip/CSV is therefore downloaded manually in a browser
(README documents the exact click path) into git-ignored data/raw/; this
parser commits the slim city-month reference CSV the pipeline consumes.
T-100 publishes monthly with a ~2-3 month lag.
"""
from pathlib import Path

import pandas as pd

AIRPORT_TO_CITY = {
    "ANC": "Anchorage", "BOS": "Boston",
    "ORD": "Chicago", "MDW": "Chicago",
    "DEN": "Denver", "LAX": "Los Angeles",
    "JFK": "New York", "LGA": "New York", "EWR": "New York",
    "MCO": "Orlando", "SEA": "Seattle",
}
RAW_GLOB = "data/raw/*T100*"
OUT_PATH = "data/reference/t100_city_monthly.csv"


def parse_t100(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=str.lower)
    df = df[df["origin"].isin(AIRPORT_TO_CITY) & (df["passengers"] > 0)]
    out = (df.groupby(["year", "month", "origin"], as_index=False)
             .passengers.sum())
    out["city"] = out["origin"].map(AIRPORT_TO_CITY)
    out = out.rename(columns={"origin": "airport"})
    out["passengers"] = out["passengers"].astype(int)
    return (out[["year", "month", "airport", "city", "passengers"]]
            .sort_values(["year", "month", "airport"]).reset_index(drop=True))


def main() -> None:
    hits = sorted(Path().glob(RAW_GLOB))
    if not hits:
        print(f"(no raw T-100 download in data/raw — keeping committed {OUT_PATH})")
        return
    raw = pd.concat([pd.read_csv(p) for p in hits])  # pandas reads .zip directly
    df = parse_t100(raw)
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"t100: {df.city.nunique()} cities, {df.year.min()} -> {df.year.max()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_bts_t100.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingest/bts_t100.py tests/test_bts_t100.py
git commit -m "feat: T-100 parser committing slim city-month reference CSV"
```

---

### Task 3: America250 events reference CSV

**Files:**
- Create: `data/reference/america250_events.csv`
- Create: `tests/test_events_csv.py`

**Interfaces:**
- Consumes: nothing.
- Produces: committed `data/reference/america250_events.csv` with columns `city, date, event_name, scale_tier, source_url`. `scale_tier`: 1=local, 2=regional, 3=national flagship. Real events only, every row cited.

- [ ] **Step 1: Write the failing schema test**

```python
# tests/test_events_csv.py
import pandas as pd

FOCUS = ["Anchorage", "Boston", "Chicago", "Denver",
         "Los Angeles", "New York", "Orlando", "Seattle"]


def _events():
    return pd.read_csv("data/reference/america250_events.csv",
                       parse_dates=["date"])


def test_events_schema():
    df = _events()
    assert list(df.columns) == ["city", "date", "event_name",
                                "scale_tier", "source_url"]
    assert df.city.isin(FOCUS).all()
    assert df.scale_tier.isin([1, 2, 3]).all()
    assert df.date.dt.year.eq(2026).all()
    assert df.source_url.str.startswith("http").all()
    assert not df.duplicated(["city", "event_name"]).any()


def test_events_have_reasonable_coverage():
    df = _events()
    # the index needs signal: at least half the cities host something,
    # and at least one national flagship (tier 3) exists
    assert df.city.nunique() >= 4
    assert (df.scale_tier == 3).any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_events_csv.py -v`
Expected: FAIL — `FileNotFoundError` for the CSV

- [ ] **Step 3: Research and curate the CSV**

Web-research real 2026 America250 / semiquincentennial events for the 8 focus cities. Known starting points to verify and expand (verify each URL and date before recording; use official/event/press sources, one row per event):

- New York: Sail4th 250 international fleet week around July 4 (sail4th.org)
- Boston: Boston250 programming (boston.gov or boston250 site)
- Chicago: America250 Illinois events (america250.org state pages)
- Denver, Seattle, Los Angeles, Orlando, Anchorage: check america250.org state/city programming, city tourism bureaus, and news coverage

Record each as `city, YYYY-MM-DD, event_name, scale_tier, source_url`. Tier by reach: 3 = national flagship (e.g., Sail4th), 2 = regional/statewide, 1 = local. If a city genuinely has no findable event, it gets no rows (index scores it 0 on events — that is signal, not a gap). Target: every claimed event has a working citation; do not pad with speculative entries.

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_events_csv.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add data/reference/america250_events.csv tests/test_events_csv.py
git commit -m "data: researched America250 events reference CSV with citations"
```

---

### Task 4: Exposure index + momentum (`city_index.py`)

**Files:**
- Create: `src/model/city_index.py`
- Create: `tests/test_city_index.py`

**Interfaces:**
- Consumes: `data/reference/t100_city_monthly.csv` (Task 2 schema), `data/processed/city_static.csv` (Task 1 schema), `data/reference/america250_events.csv` (Task 3 schema).
- Produces: `data/processed/city_index.csv` (`city, air_score, events_score, population_score, income_score, composite` — cluster columns appended by Task 5) and `data/processed/city_momentum.csv` (`city, month, passengers, passengers_prior_year, yoy_pct`). Exports: `DEFAULT_WEIGHTS: dict`, `component_scores(t100, static, events) -> pd.DataFrame`, `composite(scores: pd.DataFrame, weights: dict) -> pd.Series` (weights auto-normalized so slider values need not sum to 1), `momentum(t100) -> pd.DataFrame`, `main()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_city_index.py
import pandas as pd
import pytest

from src.model.city_index import (DEFAULT_WEIGHTS, component_scores,
                                  composite, momentum)


def _t100():
    rows = []
    for year in (2025, 2026):
        for month in (1, 2, 3):
            rows += [
                (year, month, "JFK", "New York", 100000 + year),
                (year, month, "SEA", "Seattle", 50000),
                (year, month, "ANC", "Anchorage", 10000),
            ]
    return pd.DataFrame(rows, columns=["year", "month", "airport", "city",
                                       "passengers"])


def _static():
    return pd.DataFrame({
        "city": ["New York", "Seattle", "Anchorage"],
        "population": [8_500_000, 750_000, 290_000],
        "median_hh_income": [76000, 116000, 98000],
        "acs_source": ["acs1_2024"] * 3,
    })


def _events():
    return pd.DataFrame({
        "city": ["New York", "New York", "Seattle"],
        "date": pd.to_datetime(["2026-07-03", "2026-07-04", "2026-07-04"]),
        "event_name": ["Sail4th", "Macys", "SeattleFest"],
        "scale_tier": [3, 3, 1],
        "source_url": ["http://a", "http://b", "http://c"],
    })


def test_component_scores_scaled_0_100():
    s = component_scores(_t100(), _static(), _events())
    assert list(s.columns) == ["city", "air_score", "events_score",
                               "population_score", "income_score"]
    assert len(s) == 3
    for col in s.columns[1:]:
        assert s[col].min() == 0.0 and s[col].max() == 100.0


def test_city_without_events_scores_zero():
    s = component_scores(_t100(), _static(), _events()).set_index("city")
    assert s.loc["Anchorage", "events_score"] == 0.0


def test_composite_normalizes_weights():
    s = component_scores(_t100(), _static(), _events())
    c1 = composite(s, DEFAULT_WEIGHTS)
    c2 = composite(s, {k: v * 2 for k, v in DEFAULT_WEIGHTS.items()})
    assert (c1 - c2).abs().max() < 1e-9  # scaling all weights changes nothing


def test_momentum_yoy():
    m = momentum(_t100())
    ny = m[(m.city == "New York") & (m.month == "2026-01")].iloc[0]
    assert ny.passengers == 102026 and ny.passengers_prior_year == 102025
    assert ny.yoy_pct == pytest.approx(0.0, abs=0.1)
    assert not (m.month < "2026-01").any()  # only months with a prior-year pair
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_city_index.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.model.city_index'`

- [ ] **Step 3: Write the implementation**

```python
# src/model/city_index.py
"""City exposure index: component scores, composite, air momentum.

Exposure = structural capacity to capture America250 demand (spec §4).
Composite weights are configurable (dashboard sliders re-derive the
composite from committed component scores); DEFAULT_WEIGHTS is the
documented canonical weighting — air + events carry the
anniversary-specific signal, demographics are amplifiers.
"""
import pandas as pd

DEFAULT_WEIGHTS = {"air": 0.40, "events": 0.30,
                   "population": 0.20, "income": 0.10}
T100_PATH = "data/reference/t100_city_monthly.csv"
STATIC_PATH = "data/processed/city_static.csv"
EVENTS_PATH = "data/reference/america250_events.csv"
INDEX_PATH = "data/processed/city_index.csv"
MOMENTUM_PATH = "data/processed/city_momentum.csv"


def _scale(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    if rng == 0:
        return pd.Series(0.0, index=s.index)
    return ((s - s.min()) / rng * 100).round(1)


def _latest_12(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.year * 12 + df.month
    return df[idx > idx.max() - 12]


def component_scores(t100: pd.DataFrame, static: pd.DataFrame,
                     events: pd.DataFrame) -> pd.DataFrame:
    air = _latest_12(t100).groupby("city").passengers.sum()
    ev = events.groupby("city").scale_tier.sum()
    df = static.sort_values("city").set_index("city")
    return pd.DataFrame({
        "air_score": _scale(air.reindex(df.index).fillna(0)),
        "events_score": _scale(ev.reindex(df.index).fillna(0)),
        "population_score": _scale(df.population),
        "income_score": _scale(df.median_hh_income),
    }).reset_index()


def composite(scores: pd.DataFrame, weights: dict) -> pd.Series:
    total = sum(weights.values())
    return sum(scores[f"{k}_score"] * w for k, w in weights.items()) / total


def momentum(t100: pd.DataFrame) -> pd.DataFrame:
    df = t100.groupby(["city", "year", "month"], as_index=False).passengers.sum()
    prior = df.assign(year=df.year + 1).rename(
        columns={"passengers": "passengers_prior_year"})
    m = df.merge(prior, on=["city", "year", "month"])
    m = _latest_12(m)
    m["yoy_pct"] = ((m.passengers - m.passengers_prior_year)
                    / m.passengers_prior_year * 100).round(1)
    m["month"] = (m.year.astype(str) + "-"
                  + m.month.astype(str).str.zfill(2))
    return (m[["city", "month", "passengers", "passengers_prior_year",
               "yoy_pct"]]
            .sort_values(["city", "month"]).reset_index(drop=True))


def main() -> None:
    t100 = pd.read_csv(T100_PATH)
    static = pd.read_csv(STATIC_PATH)
    events = pd.read_csv(EVENTS_PATH, parse_dates=["date"])
    scores = component_scores(t100, static, events)
    scores["composite"] = composite(scores, DEFAULT_WEIGHTS).round(1)
    scores.to_csv(INDEX_PATH, index=False)
    momentum(t100).to_csv(MOMENTUM_PATH, index=False)
    top = scores.loc[scores.composite.idxmax()]
    print(f"city_index: {len(scores)} cities, top {top.city} ({top.composite})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_city_index.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/model/city_index.py tests/test_city_index.py
git commit -m "feat: city exposure index components, composite, air momentum"
```

---

### Task 5: Clustering (`city_cluster.py`)

**Files:**
- Create: `src/model/city_cluster.py`
- Create: `tests/test_city_cluster.py`

**Interfaces:**
- Consumes: `data/processed/city_index.csv` from Task 4 (`city, *_score, composite`); `composite`, `DEFAULT_WEIGHTS`, `INDEX_PATH` from `src.model.city_index`.
- Produces: rewrites `city_index.csv` adding `cluster` (int) and `cluster_label` (str). Exports `cluster_cities(scores: pd.DataFrame, k: int = 3, seed: int = 42) -> pd.DataFrame`, `RANKED_LABELS`, `main()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_city_cluster.py
import pandas as pd

from src.model.city_cluster import RANKED_LABELS, cluster_cities


def _scores():
    # 8 synthetic cities in three obvious groups: two giants, three mids,
    # three smalls
    return pd.DataFrame({
        "city": list("ABCDEFGH"),
        "air_score": [100, 95, 50, 48, 52, 5, 3, 0],
        "events_score": [100, 90, 40, 45, 42, 2, 0, 5],
        "population_score": [100, 92, 55, 50, 45, 4, 6, 0],
        "income_score": [80, 85, 50, 55, 45, 10, 5, 0],
    })


def test_cluster_cities_deterministic_three_groups():
    out1, out2 = cluster_cities(_scores()), cluster_cities(_scores())
    assert out1.cluster_label.tolist() == out2.cluster_label.tolist()
    assert out1.cluster.nunique() == 3
    # the two giants share a cluster and get the top-ranked label
    a, b = out1.set_index("city").loc[["A", "B"], "cluster_label"]
    assert a == b == RANKED_LABELS[0]


def test_labels_are_plain_english_and_exhaustive():
    out = cluster_cities(_scores())
    assert set(out.cluster_label) == set(RANKED_LABELS)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_city_cluster.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.model.city_cluster'`

- [ ] **Step 3: Write the implementation**

```python
# src/model/city_cluster.py
"""KMeans segmentation of the focus cities (k=3, illustrative).

8 observations is a demo of the technique, not inference — the dashboard
caption says so. Labels are plain English, assigned deterministically by
ranking cluster centroids on the default-weight composite.
"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.model.city_index import DEFAULT_WEIGHTS, INDEX_PATH, composite

FEATURES = ["air_score", "events_score", "population_score", "income_score"]
RANKED_LABELS = ["National anniversary magnets", "Strong regional draws",
                 "Focused local hosts"]


def cluster_cities(scores: pd.DataFrame, k: int = 3,
                   seed: int = 42) -> pd.DataFrame:
    X = StandardScaler().fit_transform(scores[FEATURES])
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
    out = scores.copy()
    out["cluster"] = km.labels_
    rank = (out.assign(_c=composite(out, DEFAULT_WEIGHTS))
            .groupby("cluster")._c.mean()
            .sort_values(ascending=False))
    out["cluster_label"] = out.cluster.map(
        {cl: RANKED_LABELS[i] for i, cl in enumerate(rank.index)})
    return out


def main() -> None:
    scores = pd.read_csv(INDEX_PATH)
    out = cluster_cities(scores)
    out.to_csv(INDEX_PATH, index=False)
    print(f"clusters: {out.groupby('cluster_label').city.count().to_dict()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_city_cluster.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add src/model/city_cluster.py tests/test_city_cluster.py
git commit -m "feat: KMeans city clustering with deterministic plain-English labels"
```

---

### Task 6: Pipeline wiring

**Files:**
- Modify: `run_pipeline.py`

**Interfaces:**
- Consumes: `main()` from `census_acs`, `bts_t100`, `city_index`, `city_cluster`.
- Produces: `run_pipeline.py` runs national steps as before, then the city steps; city steps degrade gracefully (skip with a message) when inputs aren't built yet, mirroring the existing forecast ImportError pattern, so a national-only refresh still works mid-build.

- [ ] **Step 1: Modify `run_pipeline.py`**

Replace the whole file with:

```python
"""Run the full local pipeline: ingest -> build -> forecast -> city layer.

Usage: python run_pipeline.py [--skip-ingest]
Outputs land in data/processed/ and are committed (the app's data source).
City-layer inputs: CENSUS_API_KEY in .env; optional fresh T-100 download
in data/raw/ (otherwise the committed reference CSV is reused).
"""
import sys

from src.ingest import build_dataset, eia_gas, holidays, tsa, weather


def run(skip_ingest: bool = False) -> None:
    if not skip_ingest:
        for step in (tsa, eia_gas, weather, holidays):
            print(f"--- {step.__name__} ---")
            step.main()
    print("--- build_dataset ---")
    build_dataset.main()
    from src.model import forecast
    print("--- forecast ---")
    forecast.main()

    # City layer (spec 2026-07-07): each step skips cleanly if its inputs
    # aren't in place yet, so a national-only refresh always works.
    from src.ingest import bts_t100, census_acs
    from src.model import city_cluster, city_index
    for step in (census_acs, bts_t100, city_index, city_cluster):
        print(f"--- {step.__name__} ---")
        try:
            step.main()
        except (RuntimeError, FileNotFoundError) as exc:
            print(f"(skipping city layer from here: {exc})")
            break


if __name__ == "__main__":
    run(skip_ingest="--skip-ingest" in sys.argv)
```

Note: the forecast import is unconditional now (it exists); the old try/ImportError scaffold from the MVP build is gone.

- [ ] **Step 2: Verify the national pipeline still runs and the city layer skips cleanly**

Run: `venv/bin/python run_pipeline.py --skip-ingest`
Expected: build_dataset + forecast output as before, then `--- src.ingest.census_acs ---` followed by `(skipping city layer from here: CENSUS_API_KEY missing …)` if no key is configured yet — exit code 0. If a key IS already configured, census + t100 (reuse message) + index steps run; index may fail on missing `city_static.csv`/events inputs at this point in the build, which the skip message also handles.

- [ ] **Step 3: Run the full test suite**

Run: `venv/bin/python -m pytest -v`
Expected: all tests pass (existing + new)

- [ ] **Step 4: Commit**

```bash
git add run_pipeline.py
git commit -m "feat: wire city layer into pipeline with graceful skips"
```

---

### Task 7: City Impact dashboard tab

**Files:**
- Modify: `app/data.py` (three loaders), `app/charts.py` (three figure builders), `app/analyst.py` (city context), `app/main.py` (new tab)
- Modify: `tests/test_app_data.py`, `tests/test_analyst.py`

**Interfaces:**
- Consumes: committed `city_index.csv`, `city_momentum.csv`, `america250_events.csv` (schemas from Tasks 3–5); `composite`, `DEFAULT_WEIGHTS` from `src.model.city_index`; `CITY_COLORS`, `_LAYOUT` conventions in `app/charts.py`.
- Produces: `load_city_index()`, `load_city_momentum()`, `load_events()` in `app/data.py` (each returns `None` when its CSV is missing, so the deployed app never crashes pre-data); `index_leaderboard(scores, weights)`, `cluster_scatter(scores)`, `momentum_chart(momentum)` in `app/charts.py`; `build_context(daily, forecast, metrics, city_index=None)` gains an optional city arg; a "City impact" tab in `app/main.py`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_app_data.py`:

```python
def test_city_loaders_return_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # empty dir: no data/processed at all
    from app.data import load_city_index, load_city_momentum, load_events
    load_city_index.clear()      # st.cache_data caches across tests
    load_city_momentum.clear()
    load_events.clear()
    assert load_city_index() is None
    assert load_city_momentum() is None
    assert load_events() is None
```

Append to `tests/test_analyst.py` (reuse that file's existing fixture style for `daily`/`forecast`/`metrics` frames):

```python
def test_build_context_includes_city_index_when_given():
    import pandas as pd
    from app.analyst import build_context
    daily = pd.DataFrame({"date": pd.to_datetime(["2026-07-01"]),
                          "tsa_throughput": [2.5e6], "gas_price": [3.65]})
    forecast = pd.DataFrame({"date": pd.to_datetime(["2026-07-04"]),
                             "yhat": [2.0e6], "yhat_lower": [1.8e6],
                             "yhat_upper": [2.2e6], "actual": [1.88e6]})
    metrics = pd.DataFrame({"model": ["prophet_2019_with_covid_flag"],
                            "train_window": ["2019-01-01..2023-12-31"],
                            "mae": [152374.6], "mape": [6.33]})
    city_index = pd.DataFrame({
        "city": ["New York", "Seattle"], "air_score": [100.0, 40.0],
        "events_score": [100.0, 20.0], "population_score": [100.0, 8.0],
        "income_score": [60.0, 100.0], "composite": [93.0, 36.4],
        "cluster": [0, 1],
        "cluster_label": ["National anniversary magnets",
                          "Strong regional draws"]})
    ctx = build_context(daily, forecast, metrics, city_index=city_index)
    assert "New York" in ctx and "93.0" in ctx
    assert "exposure" in ctx.lower()
    # and omitting it still works (backwards compatible)
    assert "New York" not in build_context(daily, forecast, metrics)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_app_data.py tests/test_analyst.py -v`
Expected: new tests FAIL (`ImportError: cannot import name 'load_city_index'`, `TypeError: build_context() got an unexpected keyword argument`), existing tests still pass

- [ ] **Step 3: Implement the loaders** — append to `app/data.py`:

```python
def _optional(path: str, **kw) -> pd.DataFrame | None:
    # City-layer files land after the national MVP; the deployed app must
    # not crash on a push made before the city pipeline has run.
    try:
        return pd.read_csv(path, **kw)
    except FileNotFoundError:
        return None


@st.cache_data
def load_city_index() -> pd.DataFrame | None:
    return _optional("data/processed/city_index.csv")


@st.cache_data
def load_city_momentum() -> pd.DataFrame | None:
    return _optional("data/processed/city_momentum.csv")


@st.cache_data
def load_events() -> pd.DataFrame | None:
    return _optional("data/reference/america250_events.csv",
                     parse_dates=["date"])
```

- [ ] **Step 4: Implement the charts** — append to `app/charts.py` (follow the file's `_LAYOUT`/`CITY_COLORS` conventions; add `from src.model.city_index import composite` at the top of the file):

```python
COMPONENT_COLORS = {  # component -> color, fixed like REGION_COLORS
    "air": "#2a78d6", "events": ACCENT,
    "population": "#1baf7a", "income": "#eda100",
}
CLUSTER_COLORS = {  # ranked label -> color, fixed
    "National anniversary magnets": ACCENT,
    "Strong regional draws": "#2a78d6",
    "Focused local hosts": "#1baf7a",
}


def index_leaderboard(scores: pd.DataFrame, weights: dict) -> go.Figure:
    """Stacked weighted contributions per city; bar length = composite."""
    df = scores.assign(_c=composite(scores, weights)).sort_values("_c")
    total = sum(weights.values())
    fig = go.Figure()
    for comp, color in COMPONENT_COLORS.items():
        fig.add_bar(y=df.city, x=df[f"{comp}_score"] * weights[comp] / total,
                    name=comp.replace("_", " ").title(), orientation="h",
                    marker_color=color)
    fig.update_layout(**_LAYOUT, barmode="stack",
                      title="America250 exposure index (0–100)",
                      xaxis_title="weighted contribution to composite")
    return fig


def cluster_scatter(scores: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for label, grp in scores.groupby("cluster_label"):
        fig.add_scatter(x=grp.air_score, y=grp.events_score, text=grp.city,
                        mode="markers+text", textposition="top center",
                        name=label, marker=dict(
                            size=12, color=CLUSTER_COLORS.get(label, INK)))
    fig.update_layout(**_LAYOUT, title="City segments (KMeans, k=3)",
                      xaxis_title="Air capacity score",
                      yaxis_title="Events intensity score")
    return fig


def momentum_chart(momentum: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for city, grp in momentum.groupby("city"):
        fig.add_scatter(x=grp.month, y=grp.yoy_pct, name=city,
                        mode="lines+markers",
                        line=dict(color=CITY_COLORS.get(city, INK)))
    fig.add_hline(y=0, line_dash="dot", line_color=NEUTRAL)
    fig.update_layout(**_LAYOUT,
                      title="Air passengers, year-over-year change",
                      yaxis_title="YoY %")
    return fig
```

- [ ] **Step 5: Extend the analyst context** — in `app/analyst.py`, change the `build_context` signature to:

```python
def build_context(daily: pd.DataFrame, forecast: pd.DataFrame,
                  metrics: pd.DataFrame,
                  city_index: pd.DataFrame | None = None) -> str:
```

and, just before the final `return "\n".join(lines)`, add:

```python
    if city_index is not None:
        ranked = city_index.sort_values("composite", ascending=False)
        lines += [
            "## City America250 exposure index (0-100; structural exposure "
            "to anniversary demand, NOT observed impact; T-100 air data "
            "lags ~2-3 months)",
            *(f"{r.city}: composite {r.composite} (air {r.air_score}, "
              f"events {r.events_score}), segment: {r.cluster_label}"
              for r in ranked.itertuples()),
        ]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_app_data.py tests/test_analyst.py -v`
Expected: all PASS

- [ ] **Step 7: Add the tab in `app/main.py`**

- Import the new loaders/charts and `DEFAULT_WEIGHTS`, `composite` from `src.model.city_index`.
- Change the tabs line to:

```python
tab_forecast, tab_validate, tab_city = st.tabs(
    ["National forecast", "Predicted vs actual — July 4 window",
     "City impact"])
```

- Add the tab body:

```python
with tab_city:
    city_index, city_momentum = load_city_index(), load_city_momentum()
    events = load_events()
    if city_index is None:
        st.info("City layer data not built yet — run `python run_pipeline.py` "
                "(needs CENSUS_API_KEY and a T-100 download; see README).")
    else:
        st.caption(
            "**Exposure, not impact:** this index scores structural capacity "
            "to capture America250 demand (air capacity, events, "
            "demographics). It is a city cross-section, separate from the "
            "national daily series. BTS T-100 air data lags ~2–3 months; "
            "July 2026 actuals publish ~Oct 2026.")
        with st.expander("Adjust index weights"):
            weights = {
                k: st.slider(k.title(), 0.0, 1.0, v, 0.05,
                             key=f"w_{k}")
                for k, v in DEFAULT_WEIGHTS.items()}
            st.caption("Weights are normalized automatically; defaults "
                       "(0.40/0.30/0.20/0.10) are the documented canonical "
                       "weighting.")
        st.plotly_chart(index_leaderboard(city_index, weights),
                        width='stretch')
        left, right = st.columns(2)
        with left:
            st.plotly_chart(cluster_scatter(city_index), width='stretch')
            st.caption("Illustrative segmentation of 8 cities — a technique "
                       "demo, not statistical inference.")
        with right:
            if city_momentum is not None:
                st.plotly_chart(momentum_chart(city_momentum),
                                width='stretch')
                st.caption(f"Data through {city_momentum.month.max()} "
                           "(latest published T-100 month).")
        if events is not None:
            st.markdown("#### America250 events (every row cited)")
            st.dataframe(
                events.sort_values(["scale_tier", "date"],
                                   ascending=[False, True]),
                hide_index=True,
                column_config={"source_url": st.column_config.LinkColumn()})
```

- Pass the city index into the analyst call: `build_context(daily, forecast, metrics, city_index=load_city_index())`.

- [ ] **Step 8: Verify the app runs**

Run: `venv/bin/streamlit run app/main.py --server.headless true` (then curl `http://localhost:8501` for HTTP 200, Ctrl-C)
Expected: app boots; City impact tab shows the "not built yet" info box (no city data exists yet) and nothing crashes.

- [ ] **Step 9: Run the full suite and commit**

Run: `venv/bin/python -m pytest -v`
Expected: all PASS

```bash
git add app/ tests/test_app_data.py tests/test_analyst.py
git commit -m "feat: City impact dashboard tab with weight sliders, clusters, momentum"
```

---

### Task 8: Real-data integration, README, ship

**Files:**
- Create: `data/processed/city_static.csv`, `data/processed/city_index.csv`, `data/processed/city_momentum.csv`, `data/reference/t100_city_monthly.csv` (all pipeline outputs, committed)
- Modify: `README.md`

**Interfaces:**
- Consumes: everything above.
- Produces: the live app's City Impact tab rendering real data.

- [ ] **Step 1: USER ACTION — Census API key**

Ask the user to sign up at https://api.census.gov/data/key_signup.html (free, instant email) and add `CENSUS_API_KEY=<key>` to `.env`. Blocked until done.

- [ ] **Step 2: USER ACTION — T-100 download**

Ask the user to download, in a browser: transtats.bts.gov → Aviation Data → "Air Carrier Statistics (Form 41 Traffic) — All Carriers" → "T-100 Domestic Segment (All Carriers)" → Download; select fields `YEAR, MONTH, ORIGIN, DEST, PASSENGERS`; filter years 2023–2026 (one download per year if the form requires); save the zip(s) into `data/raw/` keeping "T100" in the filename. Blocked until done. (If the TranStats UI has drifted from this path, note what changed and update the README instructions to match reality.)

- [ ] **Step 3: Run the pipeline end-to-end**

Run: `venv/bin/python run_pipeline.py --skip-ingest`
Expected: census (8 cities) → t100 parse (8 cities) → city_index (top city printed) → clusters (3 labels), exit 0. Sanity-check `data/processed/city_index.csv` by eye: 8 rows, composite between 0 and 100, no NaNs; expect New York or Los Angeles on top under default weights — if something implausible tops the table, investigate before shipping.

- [ ] **Step 4: Verify the dashboard with real data**

Run: `venv/bin/streamlit run app/main.py`
Check: leaderboard renders with 4-segment stacked bars; sliders re-rank live; cluster scatter shows 3 labeled groups; momentum chart has a data-through caption; events table links resolve; Ask the Analyst answers a city question (e.g. "Which city has the highest exposure index and why?") citing index numbers.

- [ ] **Step 5: Update README.md**

- Data sources table: add rows for BTS T-100 (manual download path from Step 2, ~2–3 month lag noted), Census ACS (free key required, `CENSUS_API_KEY` in `.env`), America250 events (curated, cited CSV in `data/reference/`).
- New "City Impact layer" section: exposure-vs-impact framing, component list with default weights and rationale, clustering caveat (8 cities, illustrative), momentum readout, refresh instructions (re-download T-100 monthly → `python run_pipeline.py` → push).
- Note MRTS was dropped for grain honesty (city layer uses no national-grain retail data).

- [ ] **Step 6: Full suite, commit, push**

Run: `venv/bin/python -m pytest -v`
Expected: all PASS

```bash
git add data/processed/ data/reference/ README.md
git commit -m "data: city impact layer live — index, clusters, momentum + README"
git push
```

Then confirm the Streamlit Cloud app redeployed and the City impact tab renders (visit the live URL).
