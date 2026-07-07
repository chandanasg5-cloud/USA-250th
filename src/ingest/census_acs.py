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
