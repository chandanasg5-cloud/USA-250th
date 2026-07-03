"""TSA daily checkpoint throughput (national).

Source: https://www.tsa.gov/travel/passenger-volumes — current year on the
base page, one page per prior year back to 2019 (earliest public data).
The site returns 403 to non-browser clients, so we send a browser UA.
"""
import io
import time

import pandas as pd
import requests

BASE_URL = "https://www.tsa.gov/travel/passenger-volumes"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}
FIRST_YEAR = 2019
RAW_PATH = "data/raw/tsa_daily.csv"


def parse_tsa_html(html: str) -> pd.DataFrame:
    df = pd.read_html(io.StringIO(html))[0].iloc[:, :2]
    df.columns = ["date", "tsa_throughput"]
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    df["tsa_throughput"] = (
        df["tsa_throughput"].astype(str).str.replace(",", "", regex=False).astype("int64")
    )
    return df.sort_values("date").reset_index(drop=True)


def fetch_page(url: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)


def fetch_tsa(current_year: int = 2026) -> pd.DataFrame:
    frames = []
    for year in range(FIRST_YEAR, current_year + 1):
        url = BASE_URL if year == current_year else f"{BASE_URL}/{year}"
        frames.append(parse_tsa_html(fetch_page(url)))
        time.sleep(1)  # be polite to tsa.gov
    return (
        pd.concat(frames)
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )


def main() -> None:
    df = fetch_tsa()
    df.to_csv(RAW_PATH, index=False)
    print(f"tsa: {len(df)} rows, {df.date.min():%Y-%m-%d} -> {df.date.max():%Y-%m-%d}")


if __name__ == "__main__":
    main()
