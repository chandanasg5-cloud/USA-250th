import pandas as pd

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
