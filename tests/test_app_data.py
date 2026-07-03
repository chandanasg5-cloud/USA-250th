"""Loader contract tests — run against the real committed processed data."""
from app.data import (AAA_HEADLINES, load_forecast, load_gas, load_metrics,
                      load_national_daily, load_weather)


def test_headlines_have_citations():
    assert len(AAA_HEADLINES) >= 4
    assert all(h["source"] for h in AAA_HEADLINES)


def test_loaders_shapes():
    daily = load_national_daily()
    assert {"date", "tsa_throughput", "gas_price"} <= set(daily.columns)
    assert str(daily["date"].dtype).startswith("datetime64")
    fc = load_forecast()
    assert {"date", "yhat", "yhat_lower", "yhat_upper", "actual"} <= set(fc.columns)
    assert load_metrics()["model"].str.contains("prophet").any()
    assert set(load_gas()["region"]) >= {"US"}
    assert load_weather()["city"].nunique() == 8
