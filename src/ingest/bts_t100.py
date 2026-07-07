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


def _check_one_file_per_year(years_by_file: dict) -> None:
    """Raise if any year appears in more than one raw file (double-count risk)."""
    file_by_year: dict = {}
    for path, years in years_by_file.items():
        for year in years:
            file_by_year.setdefault(year, []).append(path)
    dupes = {year: files for year, files in file_by_year.items() if len(files) > 1}
    if dupes:
        detail = "; ".join(
            f"{year}: {', '.join(str(f) for f in files)}"
            for year, files in sorted(dupes.items()))
        raise RuntimeError(
            "Multiple raw files cover the same year — keep exactly one "
            f"TranStats zip per year in data/raw/. Offending year(s): {detail}")


def _check_no_history_loss(new: pd.DataFrame, existing: pd.DataFrame) -> None:
    """Raise if the new parse drops any (year, month) present in the committed CSV."""
    existing_pairs = set(zip(existing["year"], existing["month"]))
    new_pairs = set(zip(new["year"], new["month"]))
    missing = sorted(existing_pairs - new_pairs)
    if missing:
        detail = ", ".join(f"{year}-{month:02d}" for year, month in missing)
        raise RuntimeError(
            "Your download covers less history than the committed CSV — "
            f"download all years 2023 -> present. Missing (year, month): {detail}")


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
    years_by_file = {}
    frames = []
    for p in hits:
        frame = pd.read_csv(p)  # pandas reads .zip directly
        years_by_file[p] = set(frame.rename(columns=str.lower)["year"].unique())
        frames.append(frame)
    _check_one_file_per_year(years_by_file)
    raw = pd.concat(frames)
    df = parse_t100(raw)
    out_path = Path(OUT_PATH)
    if out_path.exists():
        existing = pd.read_csv(out_path)
        _check_no_history_loss(df, existing)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"t100: {df.city.nunique()} cities, {df.year.min()} -> {df.year.max()}")


if __name__ == "__main__":
    main()
