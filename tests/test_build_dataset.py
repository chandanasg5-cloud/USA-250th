import pandas as pd
import pytest

from src.ingest.build_dataset import build_national_daily, quality_report


def _inputs():
    tsa = pd.DataFrame({
        "date": pd.to_datetime(["2020-06-26", "2020-06-27", "2020-06-28",
                                "2020-06-29", "2020-06-30"]),
        "tsa_throughput": [500000, 600000, 620000, 580000, 590000],
    })
    gas = pd.DataFrame({
        "date": pd.to_datetime(["2020-06-22", "2020-06-29"]),
        "region": ["US", "US"],
        "gas_price": [2.10, 2.20],
    })
    holidays = pd.DataFrame({
        "date": pd.to_datetime(["2020-07-03"]),
        "holiday_name": ["Independence Day"],
    })
    return tsa, gas, holidays


def test_build_national_daily():
    df = build_national_daily(*_inputs())
    assert list(df.columns) == ["date", "tsa_throughput", "gas_price",
                                "is_holiday", "day_of_week",
                                "is_july4_window", "is_covid_period"]
    assert len(df) == 5
    # weekly gas forward-filled to daily (documented proxy)
    assert df.loc[df.date == "2020-06-28", "gas_price"].item() == 2.10
    assert df.loc[df.date == "2020-06-29", "gas_price"].item() == 2.20
    assert df.loc[df.date == "2020-06-30", "gas_price"].item() == 2.20
    # window flag: June 27 - July 5
    assert df.loc[df.date == "2020-06-26", "is_july4_window"].item() == 0
    assert df.loc[df.date == "2020-06-27", "is_july4_window"].item() == 1
    # covid indicator covers mid-2020
    assert df["is_covid_period"].all()


def test_quality_report_raises_on_gappy_tsa():
    tsa, gas, holidays = _inputs()
    df = build_national_daily(tsa, gas, holidays)
    df.loc[1:3, "tsa_throughput"] = None  # 60% null
    with pytest.raises(ValueError):
        quality_report(df)


def test_quality_report_ok():
    df = build_national_daily(*_inputs())
    report = quality_report(df)
    assert "rows: 5" in report
