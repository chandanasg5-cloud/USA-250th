"""City exposure index: component scores, composite, air momentum.

Exposure = structural capacity to capture America250 demand (spec §4).
Composite weights are configurable (dashboard sliders re-derive the
composite from committed component scores); DEFAULT_WEIGHTS is the
documented canonical weighting — air + events carry the
anniversary-specific signal, demographics are amplifiers.
"""
import pandas as pd

DEFAULT_WEIGHTS = {"air": 0.40, "events": 0.30,
                   "population": 0.20, "income": 0.10}
T100_PATH = "data/reference/t100_city_monthly.csv"
STATIC_PATH = "data/processed/city_static.csv"
EVENTS_PATH = "data/reference/america250_events.csv"
INDEX_PATH = "data/processed/city_index.csv"
MOMENTUM_PATH = "data/processed/city_momentum.csv"


def _scale(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    if rng == 0:
        return pd.Series(0.0, index=s.index)
    return ((s - s.min()) / rng * 100).round(1)


def _latest_12(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.year * 12 + df.month
    return df[idx > idx.max() - 12]


def component_scores(t100: pd.DataFrame, static: pd.DataFrame,
                     events: pd.DataFrame) -> pd.DataFrame:
    air = _latest_12(t100).groupby("city").passengers.sum()
    ev = events.groupby("city").scale_tier.sum()
    df = static.sort_values("city").set_index("city")
    return pd.DataFrame({
        "air_score": _scale(air.reindex(df.index).fillna(0)),
        "events_score": _scale(ev.reindex(df.index).fillna(0)),
        "population_score": _scale(df.population),
        "income_score": _scale(df.median_hh_income),
    }).reset_index()


def composite(scores: pd.DataFrame, weights: dict) -> pd.Series:
    total = sum(weights.values())
    if total == 0:
        # All-zero weights (e.g. every dashboard slider dragged to 0)
        # honestly means "no index" — not a divide-by-zero inf/NaN.
        return pd.Series(0.0, index=scores.index)
    return sum(scores[f"{k}_score"] * w for k, w in weights.items()) / total


def momentum(t100: pd.DataFrame) -> pd.DataFrame:
    df = t100.groupby(["city", "year", "month"], as_index=False).passengers.sum()
    prior = df.assign(year=df.year + 1).rename(
        columns={"passengers": "passengers_prior_year"})
    m = df.merge(prior, on=["city", "year", "month"])
    m = _latest_12(m)
    m["yoy_pct"] = ((m.passengers - m.passengers_prior_year)
                    / m.passengers_prior_year * 100).round(1)
    m["month"] = (m.year.astype(str) + "-"
                  + m.month.astype(str).str.zfill(2))
    return (m[["city", "month", "passengers", "passengers_prior_year",
               "yoy_pct"]]
            .sort_values(["city", "month"]).reset_index(drop=True))


def main() -> None:
    t100 = pd.read_csv(T100_PATH)
    static = pd.read_csv(STATIC_PATH)
    events = pd.read_csv(EVENTS_PATH, parse_dates=["date"])
    scores = component_scores(t100, static, events)
    scores["composite"] = composite(scores, DEFAULT_WEIGHTS).round(1)
    scores.to_csv(INDEX_PATH, index=False)
    momentum(t100).to_csv(MOMENTUM_PATH, index=False)
    top = scores.loc[scores.composite.idxmax()]
    print(f"city_index: {len(scores)} cities, top {top.city} ({top.composite})")


if __name__ == "__main__":
    main()
