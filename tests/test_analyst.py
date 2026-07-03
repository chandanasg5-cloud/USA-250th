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
