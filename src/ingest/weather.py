"""Open-Meteo daily weather for the 8 focus cities. Keyless.

Archive API for history (lags ~5 days); forecast API (past_days + 16-day
horizon) fills the gap through the July 4 window. Where both cover a date,
archive wins.
"""
import time
from datetime import date

import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DAILY_VARS = "temperature_2m_max,temperature_2m_min,precipitation_sum"
CITIES = {
    "Seattle": (47.6062, -122.3321),
    "Anchorage": (61.2181, -149.9003),
    "New York": (40.7128, -74.0060),
    "Chicago": (41.8781, -87.6298),
    "Boston": (42.3601, -71.0589),
    "Orlando": (28.5383, -81.3792),
    "Los Angeles": (34.0522, -118.2437),
    "Denver": (39.7392, -104.9903),
}
RAW_PATH = "data/raw/weather_daily.csv"


def parse_daily_payload(city: str, payload: dict) -> pd.DataFrame:
    d = payload["daily"]
    return pd.DataFrame({
        "city": city,
        "date": pd.to_datetime(d["time"]),
        "tmax_f": d["temperature_2m_max"],
        "tmin_f": d["temperature_2m_min"],
        "precip_in": d["precipitation_sum"],
    })


def _get(url: str, params: dict, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)


def fetch_weather() -> pd.DataFrame:
    common = {
        "daily": DAILY_VARS,
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": "auto",
    }
    frames = []
    for city, (lat, lon) in CITIES.items():
        loc = {"latitude": lat, "longitude": lon}
        hist = _get(ARCHIVE_URL, {**common, **loc,
                                  "start_date": "2019-01-01",
                                  "end_date": date.today().isoformat()})
        fcst = _get(FORECAST_URL, {**common, **loc,
                                   "past_days": 14, "forecast_days": 16})
        frames += [parse_daily_payload(city, hist), parse_daily_payload(city, fcst)]
        time.sleep(0.5)
    return (
        pd.concat(frames)
        .drop_duplicates(["city", "date"], keep="first")  # archive first -> wins
        .sort_values(["city", "date"])
        .reset_index(drop=True)
    )


def main() -> None:
    df = fetch_weather()
    df.to_csv(RAW_PATH, index=False)
    print(f"weather: {df.city.nunique()} cities, {df.date.min():%Y-%m-%d} -> {df.date.max():%Y-%m-%d}")


if __name__ == "__main__":
    main()
