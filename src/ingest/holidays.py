"""US public holidays from Nager.Date. Keyless."""
import time

import pandas as pd
import requests

URL = "https://date.nager.at/api/v3/PublicHolidays/{year}/US"
RAW_PATH = "data/raw/us_holidays.csv"


def parse_holidays(payloads: list) -> pd.DataFrame:
    rows = [
        {"date": h["date"], "holiday_name": h["name"]}
        for payload in payloads
        for h in payload
        if h.get("global", False)  # keep nationwide holidays only
    ]
    df = pd.DataFrame(rows).drop_duplicates()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def fetch_holidays(years: range) -> pd.DataFrame:
    payloads = []
    for year in years:
        resp = requests.get(URL.format(year=year), timeout=30)
        resp.raise_for_status()
        payloads.append(resp.json())
        time.sleep(0.3)
    return parse_holidays(payloads)


def main() -> None:
    df = fetch_holidays(range(2019, 2027))
    df.to_csv(RAW_PATH, index=False)
    print(f"holidays: {len(df)} rows, {df.date.min():%Y-%m-%d} -> {df.date.max():%Y-%m-%d}")


if __name__ == "__main__":
    main()
