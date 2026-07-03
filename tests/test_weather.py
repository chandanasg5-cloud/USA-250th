import pandas as pd

from src.ingest.weather import CITIES, parse_daily_payload

PAYLOAD = {
    "daily": {
        "time": ["2026-07-01", "2026-07-02"],
        "temperature_2m_max": [88.1, 90.3],
        "temperature_2m_min": [71.0, 72.4],
        "precipitation_sum": [0.0, 0.12],
    }
}


def test_parse_daily_payload():
    df = parse_daily_payload("Chicago", PAYLOAD)
    assert list(df.columns) == ["city", "date", "tmax_f", "tmin_f", "precip_in"]
    assert len(df) == 2
    assert df["date"].iloc[0] == pd.Timestamp("2026-07-01")
    assert df["tmax_f"].iloc[1] == 90.3


def test_cities_list():
    assert set(CITIES) == {"Seattle", "Anchorage", "New York", "Chicago",
                           "Boston", "Orlando", "Los Angeles", "Denver"}
