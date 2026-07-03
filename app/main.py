"""America250 Economic Impact Intelligence Platform — dashboard."""
import pandas as pd
import streamlit as st

from app.charts import forecast_chart, gas_chart, weather_chart
from app.data import (AAA_HEADLINES, load_forecast, load_gas, load_metrics,
                      load_national_daily, load_weather)

st.set_page_config(page_title="America250 Economic Impact", page_icon="🇺🇸",
                   layout="wide")

daily, forecast = load_national_daily(), load_forecast()
gas, weather, metrics = load_gas(), load_weather(), load_metrics()

st.title("America250 Economic Impact Intelligence Platform")
st.caption(
    "Backtest + nowcast of the July 4, 2026 (250th anniversary) travel window. "
    "Free public data only (TSA, EIA, Open-Meteo); every proxy labeled. "
    f"TSA actuals through {daily.dropna(subset=['tsa_throughput']).date.max():%b %d, %Y}."
)

cols = st.columns(len(AAA_HEADLINES))
for col, h in zip(cols, AAA_HEADLINES):
    col.metric(h["label"], h["value"], h["delta"], delta_color="off",
               help=f"Source: {h['source']}")

st.sidebar.header("Filters")
all_cities = sorted(weather.city.unique())
cities = st.sidebar.multiselect(
    "Cities", all_cities, default=["Chicago", "New York", "Orlando", "Seattle"])
start, end = st.sidebar.date_input(
    "Weather date range",
    value=(pd.Timestamp("2026-06-20").date(), weather.date.max().date()),
    min_value=weather.date.min().date(), max_value=weather.date.max().date())

st.plotly_chart(forecast_chart(daily, forecast), width='stretch')
with st.expander("Model performance (2024–2025 holdout)"):
    st.dataframe(metrics, hide_index=True)
    st.caption("Gas price is forward-filled weekly→daily (proxy). Future gas "
               "held at last observed value. TSA public data starts 2019.")

left, right = st.columns(2)
with left:
    regions = st.multiselect("Gas regions", sorted(gas.region.unique()),
                             default=["US"])
    st.plotly_chart(gas_chart(gas, regions or ["US"]), width='stretch')
with right:
    st.plotly_chart(weather_chart(weather, cities or all_cities,
                                  str(start), str(end)),
                    width='stretch')
