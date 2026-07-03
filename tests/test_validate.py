import pandas as pd

from src.model.validate import post_mortem, window_errors


def _forecast():
    return pd.DataFrame({
        "date": pd.date_range("2026-06-27", periods=9),
        "yhat": [2.5e6, 2.8e6, 2.7e6, 2.4e6, 2.5e6, 2.8e6, 2.8e6, 2.0e6, 2.8e6],
        "yhat_lower": [2.1e6] * 9,
        "yhat_upper": [3.1e6] * 9,
        # first 6 days observed; last 3 pending; day 2 runs hot, day 3 misses low
        "actual": [2575625.0, 2930672.0, 2690919.0, 2477905.0, 2654017.0,
                   3200000.0, None, None, None],
    })


def test_window_errors():
    err = window_errors(_forecast())
    assert len(err) == 6  # only observed days
    assert {"date", "yhat", "actual", "error", "pct_error",
            "inside_interval"} <= set(err.columns)
    first = err.iloc[0]
    assert first["error"] == 2575625.0 - 2.5e6
    assert bool(first["inside_interval"])
    assert not bool(err.iloc[5]["inside_interval"])  # 3.2M > upper 3.1M


def test_post_mortem_mentions_key_facts():
    err = window_errors(_forecast())
    text = post_mortem(err, holdout_mape=6.33)
    assert "6 of 9" in text            # observed days
    assert "5 of 6" in text            # interval coverage
    assert "6.33" in text              # holdout comparison
    assert "above" in text.lower()     # direction of bias (actuals ran hot)


def test_post_mortem_handles_no_actuals():
    fc = _forecast()
    fc["actual"] = None
    text = post_mortem(window_errors(fc), holdout_mape=6.33)
    assert "No actuals" in text
