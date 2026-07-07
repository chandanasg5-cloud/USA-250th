"""Data access for the dashboard. Reads ONLY data/processed (committed)."""
import pandas as pd
import streamlit as st

# Headline numbers are quoted directly from AAA's June 2026 July 4 travel
# forecast press release — cited, not modeled.
AAA_HEADLINES = [
    {"label": "Travelers (50+ mi, Jun 27–Jul 5)", "value": "72.2M",
     "delta": "vs 71.8M in 2025", "source": "AAA"},
    {"label": "Traveling by car", "value": "61.4M",
     "delta": "85% of travelers", "source": "AAA"},
    {"label": "Flying", "value": "5.85M",
     "delta": "~$830 avg domestic round trip", "source": "AAA"},
    {"label": "Bus / train / cruise", "value": "4.93M",
     "delta": "+5.3% — fastest growing", "source": "AAA"},
]


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(f"data/processed/{name}.csv", parse_dates=["date"])


@st.cache_data
def load_national_daily() -> pd.DataFrame:
    return _read("national_daily")


@st.cache_data
def load_forecast() -> pd.DataFrame:
    return _read("forecast")


@st.cache_data
def load_metrics() -> pd.DataFrame:
    return pd.read_csv("data/processed/metrics.csv")


@st.cache_data
def load_gas() -> pd.DataFrame:
    return _read("gas_weekly")


@st.cache_data
def load_weather() -> pd.DataFrame:
    return _read("weather_daily")


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
