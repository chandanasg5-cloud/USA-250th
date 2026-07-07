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


def test_composite_all_zero_weights_returns_zeros():
    s = component_scores(_t100(), _static(), _events())
    c = composite(s, {k: 0.0 for k in DEFAULT_WEIGHTS})
    assert (c == 0.0).all()


def test_momentum_yoy():
    m = momentum(_t100())
    ny = m[(m.city == "New York") & (m.month == "2026-01")].iloc[0]
    assert ny.passengers == 102026 and ny.passengers_prior_year == 102025
    assert ny.yoy_pct == pytest.approx(0.0, abs=0.1)
    assert not (m.month < "2026-01").any()  # only months with a prior-year pair
