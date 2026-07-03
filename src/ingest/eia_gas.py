"""EIA weekly retail gasoline prices, national + PADD regions.

Source: keyless XLS downloads from eia.gov/dnav (verified 2026-07-02, no
API key or signup required). Each file: sheet "Data 1", data rows after
2 header rows. All grades, all formulations, $/gal.
"""
import io
import time

import pandas as pd
import requests

URL = "https://www.eia.gov/dnav/pet/hist_xls/{sid}.xls"
SERIES = {
    "US": "EMM_EPM0_PTE_NUS_DPGw",
    "PADD1": "EMM_EPM0_PTE_R10_DPGw",
    "PADD2": "EMM_EPM0_PTE_R20_DPGw",
    "PADD3": "EMM_EPM0_PTE_R30_DPGw",
    "PADD4": "EMM_EPM0_PTE_R40_DPGw",
    "PADD5": "EMM_EPM0_PTE_R50_DPGw",
}
RAW_PATH = "data/raw/gas_weekly.csv"


def parse_gas_sheet(raw: pd.DataFrame, region: str) -> pd.DataFrame:
    df = raw.iloc[:, :2].copy()
    df.columns = ["date", "gas_price"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["gas_price"] = pd.to_numeric(df["gas_price"], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    df["region"] = region
    return df[["date", "region", "gas_price"]]


def fetch_series(sid: str, retries: int = 3) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            resp = requests.get(URL.format(sid=sid), timeout=60)
            resp.raise_for_status()
            return pd.read_excel(
                io.BytesIO(resp.content), sheet_name="Data 1", skiprows=2
            )
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)


def fetch_gas() -> pd.DataFrame:
    frames = []
    for region, sid in SERIES.items():
        try:
            frames.append(parse_gas_sheet(fetch_series(sid), region))
        except Exception as exc:  # PADDs are nice-to-have; US is required
            if region == "US":
                raise
            print(f"warn: skipping {region}: {exc}")
        time.sleep(0.5)
    return pd.concat(frames).sort_values(["region", "date"]).reset_index(drop=True)


def main() -> None:
    df = fetch_gas()
    df.to_csv(RAW_PATH, index=False)
    us = df[df.region == "US"]
    print(f"gas: {df.region.nunique()} regions, US {us.date.min():%Y-%m-%d} -> {us.date.max():%Y-%m-%d}")


if __name__ == "__main__":
    main()
