"""BTS T-100 Segment (All Carriers), domestic + international -> city-month passenger totals.

TranStats has no script-friendly endpoint (verified 2026-07-07: PREZIP
404s, the legacy DownLoad_Table.asp POST 500s, bts.gov 403s non-browser
clients). The raw zip/CSV is therefore downloaded manually in a browser
(README documents the exact click path) into git-ignored data/raw/; this
parser commits the slim city-month reference CSV the pipeline consumes.
T-100 publishes monthly with a ~2-3 month lag. Filtering by origin airport
gives total departing passengers (domestic + international), a more
complete airport-capacity measure than the domestic-only table.
"""
from pathlib import Path

import pandas as pd

AIRPORT_TO_CITY = {
    "ANC": "Anchorage", "BOS": "Boston",
    "ORD": "Chicago", "MDW": "Chicago",
    "DEN": "Denver", "LAX": "Los Angeles",
    "JFK": "New York", "LGA": "New York", "EWR": "New York",
    "MCO": "Orlando", "SEA": "Seattle",
}
RAW_GLOB = "data/raw/*T100*"
OUT_PATH = "data/reference/t100_city_monthly.csv"


def parse_t100(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=str.lower)
    df = df[df["origin"].isin(AIRPORT_TO_CITY) & (df["passengers"] > 0)]
    out = (df.groupby(["year", "month", "origin"], as_index=False)
             .passengers.sum())
    out["city"] = out["origin"].map(AIRPORT_TO_CITY)
    out = out.rename(columns={"origin": "airport"})
    out["passengers"] = out["passengers"].astype(int)
    return (out[["year", "month", "airport", "city", "passengers"]]
            .sort_values(["year", "month", "airport"]).reset_index(drop=True))


def main() -> None:
    hits = sorted(Path().glob(RAW_GLOB))
    if not hits:
        print(f"(no raw T-100 download in data/raw — keeping committed {OUT_PATH})")
        return
    raw = pd.concat([pd.read_csv(p) for p in hits])  # pandas reads .zip directly
    df = parse_t100(raw)
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"t100: {df.city.nunique()} cities, {df.year.min()} -> {df.year.max()}")


if __name__ == "__main__":
    main()
