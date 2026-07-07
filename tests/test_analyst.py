import pandas as pd

from app.analyst import EXAMPLE_QUESTIONS, SYSTEM_PROMPT, build_context


def test_example_questions():
    assert len(EXAMPLE_QUESTIONS) == 3


def test_system_prompt_guardrails():
    lowered = SYSTEM_PROMPT.lower()
    assert "only" in lowered and "cite" in lowered and "insufficient" in lowered


def test_build_context_is_compact_and_grounded():
    daily = pd.DataFrame({
        "date": pd.date_range("2026-06-25", periods=7),
        "tsa_throughput": [2.9e6, 2.9e6, 2.58e6, 2.93e6, 2.69e6, 2.48e6, 2654017],
        "gas_price": [3.7] * 7,
    })
    forecast = pd.DataFrame({
        "date": pd.date_range("2026-06-27", periods=9),
        "yhat": [2.8e6] * 9, "yhat_lower": [2.6e6] * 9, "yhat_upper": [3.0e6] * 9,
        "actual": [2.57e6, 2.93e6, 2.69e6, 2.48e6, 2654017, None, None, None, None],
    })
    metrics = pd.DataFrame({"model": ["prophet_2019_with_covid_flag"],
                            "train_window": ["2019..2023"], "mae": [90000.0],
                            "mape": [4.2]})
    ctx = build_context(daily, forecast, metrics)
    assert "2,654,017" in ctx          # grounds the latest actual
    assert "4.2" in ctx                # includes model quality
    assert len(ctx) < 4000             # stays compact for free-tier tokens


def test_build_context_includes_city_index_when_given():
    import pandas as pd
    from app.analyst import build_context
    daily = pd.DataFrame({"date": pd.to_datetime(["2026-07-01"]),
                          "tsa_throughput": [2.5e6], "gas_price": [3.65]})
    forecast = pd.DataFrame({"date": pd.to_datetime(["2026-07-04"]),
                             "yhat": [2.0e6], "yhat_lower": [1.8e6],
                             "yhat_upper": [2.2e6], "actual": [1.88e6]})
    metrics = pd.DataFrame({"model": ["prophet_2019_with_covid_flag"],
                            "train_window": ["2019-01-01..2023-12-31"],
                            "mae": [152374.6], "mape": [6.33]})
    city_index = pd.DataFrame({
        "city": ["New York", "Seattle"], "air_score": [100.0, 40.0],
        "events_score": [100.0, 20.0], "population_score": [100.0, 8.0],
        "income_score": [60.0, 100.0], "composite": [93.0, 36.4],
        "cluster": [0, 1],
        "cluster_label": ["National anniversary magnets",
                          "Strong regional draws"]})
    ctx = build_context(daily, forecast, metrics, city_index=city_index)
    assert "New York" in ctx and "93.0" in ctx
    assert "exposure" in ctx.lower()
    # and omitting it still works (backwards compatible). NOTE: can't assert
    # "New York" is absent here — the base AAA context already names it as a
    # top destination — so check for the city-section-only marker instead.
    assert "composite" not in build_context(daily, forecast, metrics)
