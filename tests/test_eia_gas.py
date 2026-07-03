import pandas as pd

from src.ingest.eia_gas import parse_gas_sheet


def test_parse_gas_sheet():
    # Shape of EIA dnav "Data 1" sheet after skiprows: [Date, price] + junk row
    raw = pd.DataFrame({
        "Date": ["2026-06-22", "2026-06-29", None],
        "Weekly U.S. All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)": [3.65, 3.71, None],
    })
    df = parse_gas_sheet(raw, "US")
    assert list(df.columns) == ["date", "region", "gas_price"]
    assert len(df) == 2  # null row dropped
    assert df["gas_price"].tolist() == [3.65, 3.71]
    assert (df["region"] == "US").all()
