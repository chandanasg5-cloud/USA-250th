import pandas as pd
import pytest

from src.ingest.bts_t100 import (
    AIRPORT_TO_CITY,
    _check_no_history_loss,
    _check_one_file_per_year,
    parse_t100,
)


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


def test_check_one_file_per_year_raises_on_duplicate_year():
    years_by_file = {
        "data/raw/stale_2023.zip": {2023},
        "data/raw/fresh_2023.zip": {2023, 2024},
    }
    with pytest.raises(RuntimeError, match="2023"):
        _check_one_file_per_year(years_by_file)


def test_check_one_file_per_year_passes_when_years_disjoint():
    years_by_file = {
        "data/raw/a_2023.zip": {2023},
        "data/raw/b_2024.zip": {2024},
        "data/raw/c_2025.zip": {2025},
    }
    _check_one_file_per_year(years_by_file)  # no raise


def test_check_no_history_loss_raises_when_new_drops_existing_months():
    existing = pd.DataFrame({
        "year": [2023, 2023, 2024],
        "month": [4, 5, 1],
    })
    new = pd.DataFrame({
        "year": [2023, 2024],
        "month": [4, 1],
    })
    with pytest.raises(RuntimeError, match=r"2023-05"):
        _check_no_history_loss(new, existing)


def test_check_no_history_loss_passes_when_new_is_a_superset():
    existing = pd.DataFrame({
        "year": [2023, 2023],
        "month": [4, 5],
    })
    new = pd.DataFrame({
        "year": [2023, 2023, 2024, 2026],
        "month": [4, 5, 1, 3],
    })
    _check_no_history_loss(new, existing)  # no raise
