import numpy as np
import pandas as pd

from src.model.forecast import mae, make_future_regressors, mape


def test_metrics():
    a = np.array([100.0, 200.0])
    p = np.array([110.0, 190.0])
    assert mae(a, p) == 10.0
    assert round(mape(a, p), 2) == 7.5  # (10% + 5%) / 2


def test_make_future_regressors():
    hist = pd.DataFrame({
        "date": pd.to_datetime(["2025-12-30", "2025-12-31"]),
        "gas_price": [3.10, 3.15],
        "is_covid_period": [0, 0],
    })
    future = pd.date_range("2026-06-27", "2026-06-29")
    out = make_future_regressors(hist, future)
    assert list(out.columns) == ["ds", "gas_price", "is_covid_period"]
    assert len(out) == 3
    # documented assumption: future gas = last observed value
    assert (out["gas_price"] == 3.15).all()
    assert (out["is_covid_period"] == 0).all()
