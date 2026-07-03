"""Forecast daily TSA throughput for the July 4, 2026 window.

Approach (per spec §6):
- Prophet with native US holidays + regressors gas_price and is_covid_period.
- Two Prophet variants compared on a 2024-2025 holdout:
  A) trained 2019-2023 with the COVID indicator,
  B) trained 2022-2023 (COVID years excluded).
- SARIMA baseline (weekly seasonality) trained 2022-2023.
- Winning Prophet variant is refit through 2025-12-31 and projects
  2026-01-01..2026-07-05 so the dashboard can overlay 2026 actuals to date.

Documented assumptions: future gas_price = last observed weekly value;
is_covid_period = 0 for all future dates.
"""
import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX

HOLDOUT_START, HOLDOUT_END = "2024-01-01", "2025-12-31"
TRAIN_END = "2025-12-31"
FORECAST_START, FORECAST_END = "2026-01-01", "2026-07-05"
REGRESSORS = ["gas_price", "is_covid_period"]


def mae(actual, pred) -> float:
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(pred))))


def mape(actual, pred) -> float:
    a, p = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    return float(np.mean(np.abs((a - p) / a)) * 100)


def make_future_regressors(history: pd.DataFrame,
                           future_dates: pd.DatetimeIndex) -> pd.DataFrame:
    last_gas = history.sort_values("date")["gas_price"].iloc[-1]
    return pd.DataFrame({
        "ds": future_dates,
        "gas_price": last_gas,          # assumption: hold last observed price
        "is_covid_period": 0,
    })


def _to_prophet(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={"date": "ds", "tsa_throughput": "y"})
    return out[["ds", "y", *REGRESSORS]].dropna(subset=["y"])


def fit_prophet(train: pd.DataFrame) -> Prophet:
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                daily_seasonality=False)
    m.add_country_holidays("US")
    for reg in REGRESSORS:
        m.add_regressor(reg)
    m.fit(train)
    return m


def main() -> None:
    daily = pd.read_csv("data/processed/national_daily.csv", parse_dates=["date"])
    obs = daily.dropna(subset=["tsa_throughput"])
    holdout = _to_prophet(obs[obs.date.between(HOLDOUT_START, HOLDOUT_END)])

    variants = {
        "prophet_2019_with_covid_flag": obs[obs.date < HOLDOUT_START],
        "prophet_2022_excl_covid": obs[(obs.date >= "2022-01-01") & (obs.date < HOLDOUT_START)],
    }
    rows = []
    for name, train_df in variants.items():
        m = fit_prophet(_to_prophet(train_df))
        fc = m.predict(holdout.drop(columns="y"))
        rows.append({"model": name,
                     "train_window": f"{train_df.date.min():%Y-%m-%d}..{train_df.date.max():%Y-%m-%d}",
                     "mae": mae(holdout.y, fc.yhat), "mape": mape(holdout.y, fc.yhat)})

    sarima_train = obs[obs.date.between("2022-01-01", "2023-12-31")]
    sm = SARIMAX(sarima_train.tsa_throughput.to_numpy(),
                 order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)).fit(disp=False)
    sarima_pred = sm.forecast(len(holdout))
    rows.append({"model": "sarima_baseline", "train_window": "2022-01-01..2023-12-31",
                 "mae": mae(holdout.y, sarima_pred), "mape": mape(holdout.y, sarima_pred)})

    metrics = pd.DataFrame(rows).sort_values("mape")
    metrics.to_csv("data/processed/metrics.csv", index=False)

    best = metrics[metrics.model.str.startswith("prophet")].iloc[0]
    train_start = "2019-01-01" if "2019" in best.model else "2022-01-01"
    final_train = obs[obs.date.between(train_start, TRAIN_END)]
    m = fit_prophet(_to_prophet(final_train))
    future = make_future_regressors(final_train,
                                    pd.date_range(FORECAST_START, FORECAST_END))
    fc = m.predict(future)

    out = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"ds": "date"})
    actual_2026 = obs[obs.date >= FORECAST_START][["date", "tsa_throughput"]]
    out = out.merge(actual_2026.rename(columns={"tsa_throughput": "actual"}),
                    on="date", how="left")
    out.to_csv("data/processed/forecast.csv", index=False)

    print(metrics.to_string(index=False))
    print(f"\nWinner: {best.model} (MAPE {best.mape:.2f}% vs "
          f"SARIMA {metrics[metrics.model == 'sarima_baseline'].mape.item():.2f}%). "
          f"Refit through {TRAIN_END}; projected {FORECAST_START}..{FORECAST_END}.")


if __name__ == "__main__":
    main()
