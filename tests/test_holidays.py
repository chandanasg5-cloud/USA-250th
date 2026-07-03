import pandas as pd

from src.ingest.holidays import parse_holidays

PAYLOADS = [[
    {"date": "2026-07-03", "name": "Independence Day", "global": True},
    {"date": "2026-07-04", "name": "Independence Day", "global": True},
    {"date": "2026-03-02", "name": "Texas Independence Day", "global": False},
]]


def test_parse_holidays_keeps_only_national():
    df = parse_holidays(PAYLOADS)
    assert list(df.columns) == ["date", "holiday_name"]
    assert len(df) == 2  # state-level row dropped
    assert pd.Timestamp("2026-07-04") in set(df["date"])
