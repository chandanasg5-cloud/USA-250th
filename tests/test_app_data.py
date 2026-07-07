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


def test_city_loaders_return_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # empty dir: no data/processed at all
    from app.data import load_city_index, load_city_momentum, load_events
    load_city_index.clear()      # st.cache_data caches across tests
    load_city_momentum.clear()
    load_events.clear()
    assert load_city_index() is None
    assert load_city_momentum() is None
    assert load_events() is None
