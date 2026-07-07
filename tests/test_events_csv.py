import pandas as pd

FOCUS = ["Anchorage", "Boston", "Chicago", "Denver",
         "Los Angeles", "New York", "Orlando", "Seattle"]


def _events():
    return pd.read_csv("data/reference/america250_events.csv",
                       parse_dates=["date"])


def test_events_schema():
    df = _events()
    assert list(df.columns) == ["city", "date", "event_name",
                                "scale_tier", "source_url"]
    assert df.city.isin(FOCUS).all()
    assert df.scale_tier.isin([1, 2, 3]).all()
    assert df.date.dt.year.eq(2026).all()
    assert df.source_url.str.startswith("http").all()
    assert not df.duplicated(["city", "event_name"]).any()


def test_events_have_reasonable_coverage():
    df = _events()
    # the index needs signal: at least half the cities host something,
    # and at least one national flagship (tier 3) exists
    assert df.city.nunique() >= 4
    assert (df.scale_tier == 3).any()
