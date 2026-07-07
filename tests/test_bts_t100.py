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
