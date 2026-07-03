"""Unify raw sources into data/processed/national_daily.csv.

Documented assumptions/proxies:
- gas_price: EIA is weekly (Mondays); forward-filled to daily. A held price
  between releases is a PROXY, labeled as such in the dashboard.
- is_covid_period: 2020-03-01..2021-12-31 indicator so the model can absorb
  the collapse/recovery instead of contaminating seasonality.
"""
import shutil

import pandas as pd

COVID_START, COVID_END = "2020-03-01", "2021-12-31"
PROCESSED_DIR = "data/processed"


def build_national_daily(tsa: pd.DataFrame, gas_us: pd.DataFrame,
                         holidays: pd.DataFrame) -> pd.DataFrame:
    spine = pd.DataFrame({"date": pd.date_range(tsa.date.min(), tsa.date.max())})
    df = spine.merge(tsa[["date", "tsa_throughput"]], on="date", how="left")
    gas = gas_us[gas_us.region == "US"][["date", "gas_price"]].sort_values("date") \
        if "region" in gas_us.columns else gas_us[["date", "gas_price"]].sort_values("date")
    # merge_asof backward == forward-fill weekly price onto daily spine (proxy)
    df = pd.merge_asof(df.sort_values("date"), gas, on="date")
    df["is_holiday"] = df.date.isin(set(holidays.date)).astype(int)
    df["day_of_week"] = df.date.dt.dayofweek
    m, d = df.date.dt.month, df.date.dt.day
    df["is_july4_window"] = (((m == 6) & (d >= 27)) | ((m == 7) & (d <= 5))).astype(int)
    df["is_covid_period"] = df.date.between(COVID_START, COVID_END).astype(int)
    return df


def quality_report(df: pd.DataFrame) -> str:
    null_pct = df.isna().mean() * 100
    lines = [
        f"rows: {len(df)}",
        f"span: {df.date.min():%Y-%m-%d} -> {df.date.max():%Y-%m-%d}",
        "null % by column:",
        *(f"  {c}: {p:.2f}%" for c, p in null_pct.items()),
    ]
    if null_pct["tsa_throughput"] > 1.0:
        raise ValueError(
            f"tsa_throughput is {null_pct['tsa_throughput']:.1f}% null (>1% threshold)"
        )
    return "\n".join(lines)


def main() -> None:
    tsa = pd.read_csv("data/raw/tsa_daily.csv", parse_dates=["date"])
    gas = pd.read_csv("data/raw/gas_weekly.csv", parse_dates=["date"])
    holidays = pd.read_csv("data/raw/us_holidays.csv", parse_dates=["date"])
    df = build_national_daily(tsa, gas, holidays)
    print(quality_report(df))
    df.to_csv(f"{PROCESSED_DIR}/national_daily.csv", index=False)
    # The app reads only data/processed (data/raw is git-ignored), so copy
    # the two source tables the dashboard shows directly.
    shutil.copy("data/raw/gas_weekly.csv", f"{PROCESSED_DIR}/gas_weekly.csv")
    shutil.copy("data/raw/weather_daily.csv", f"{PROCESSED_DIR}/weather_daily.csv")
    print("wrote national_daily.csv, gas_weekly.csv, weather_daily.csv")


if __name__ == "__main__":
    main()
