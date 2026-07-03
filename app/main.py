"""America250 Economic Impact Intelligence Platform — dashboard."""
import os
import sys
from pathlib import Path

# `streamlit run app/main.py` puts app/ (not the repo root) on sys.path;
# anchor the root so `from app.X import ...` works in every runner.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from app.analyst import EXAMPLE_QUESTIONS, ask_analyst, build_context
from app.charts import (forecast_chart, gas_chart, validation_chart,
                        weather_chart)
from src.model.validate import post_mortem, window_errors
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

tab_forecast, tab_validate = st.tabs(
    ["National forecast", "Predicted vs actual — July 4 window"])

with tab_validate:
    if st.button("Check TSA for newer actuals",
                 help="Fetches tsa.gov live; falls back to committed data if "
                      "unreachable. Committed data updates via run_pipeline.py."):
        try:
            from src.ingest.tsa import BASE_URL, fetch_page, parse_tsa_html
            latest = parse_tsa_html(fetch_page(BASE_URL))
            merged = forecast.merge(
                latest.rename(columns={"tsa_throughput": "live"}),
                on="date", how="left")
            newer = merged["actual"].isna() & merged["live"].notna()
            if newer.any():
                forecast = forecast.assign(
                    actual=merged["actual"].fillna(merged["live"]))
                st.success(f"Pulled {int(newer.sum())} newer day(s) from "
                           "tsa.gov (this session only).")
            else:
                st.info("No newer days published yet — committed data is "
                        "already current.")
        except Exception:
            st.warning("Could not reach tsa.gov — showing committed data.")

    errors = window_errors(forecast)
    if len(errors):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Window days observed", f"{len(errors)} / 9")
        m2.metric("Inside forecast interval",
                  f"{int(errors.inside_interval.sum())} / {len(errors)}")
        m3.metric("Live MAPE", f"{errors.pct_error.abs().mean():.1f}%")
        bias = errors.pct_error.mean()
        m4.metric("Bias vs forecast", f"{bias:+.1f}%",
                  "actuals running hot" if bias > 0 else "actuals running cool",
                  delta_color="off")
    st.plotly_chart(validation_chart(forecast), width='stretch')
    if len(errors):
        show = errors.assign(date=errors.date.dt.strftime("%a %b %d"))[
            ["date", "yhat", "actual", "error", "pct_error", "inside_interval"]]
        show.columns = ["Date", "Predicted", "Actual", "Error", "% error",
                        "In interval"]
        st.dataframe(show.round(0), hide_index=True)
    st.markdown("#### Post-mortem")
    best_mape = metrics[metrics.model.str.startswith("prophet")].mape.min()
    st.markdown(post_mortem(errors, holdout_mape=round(best_mape, 2)))

with tab_forecast:
    st.plotly_chart(forecast_chart(daily, forecast), width='stretch')
    with st.expander("Model performance (2024–2025 holdout)"):
        st.markdown(
            "**How to read this:** each model was trained on history, then asked "
            "to predict all of 2024–2025 — data it never saw. Scoring those blind "
            "predictions against what actually happened tells us which model to "
            "trust for 2026.\n"
            "- **MAE** — average daily miss, in passengers (vs ~2.5M/day typical).\n"
            "- **MAPE** — the same miss as a percentage; lower is better.\n")
        st.dataframe(metrics, hide_index=True)
        st.markdown(
            "**Key finding:** the two Prophet rows differ only in COVID handling — "
            "keeping 2020–21 *with an indicator flag* (6.3% MAPE) beat deleting "
            "those years (9.7%). The messy years still carry seasonal signal; the "
            "flag stops the crash from reading as a recurring pattern. The SARIMA "
            "baseline (9.2%) confirms a simpler model wouldn't have done as well. "
            "The winner was refit through 2025 to produce the forecast above.")
        st.caption("Gas price is forward-filled weekly→daily (proxy). Future gas "
                   "held at last observed value. TSA public data starts 2019.")

left, right = st.columns(2)
with left:
    regions = st.multiselect(
        "Gas regions", sorted(gas.region.unique()), default=["US"],
        help="PADD = Petroleum Administration for Defense Districts, the EIA's "
             "five reporting regions: PADD1 East Coast · PADD2 Midwest · "
             "PADD3 Gulf Coast · PADD4 Rocky Mountain · PADD5 West Coast "
             "(incl. AK & HI).")
    st.plotly_chart(gas_chart(gas, regions or ["US"]), width='stretch')
    st.caption("PADD regions (EIA): 1 East Coast · 2 Midwest · 3 Gulf Coast · "
               "4 Rocky Mountain · 5 West Coast incl. AK/HI. Regional prices "
               "matter because 85% of July 4 travelers drive (AAA).")
with right:
    st.plotly_chart(weather_chart(weather, cities or all_cities,
                                  str(start), str(end)),
                    width='stretch')

st.divider()
st.subheader("Ask the Analyst")
st.caption("Gemini answers from this dashboard's processed data only, with numbers cited.")
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY", "")
if not api_key:
    st.info("Set GEMINI_API_KEY in a local .env (free key: aistudio.google.com) "
            "to enable live Q&A. The rest of the dashboard works without it.")
else:
    qcols = st.columns(3)
    for col, q in zip(qcols, EXAMPLE_QUESTIONS):
        if col.button(q, width='stretch'):
            st.session_state["analyst_q"] = q
    question = st.text_input("Your question",
                             value=st.session_state.get("analyst_q", ""))
    if question:
        with st.spinner("Analyzing..."):
            try:
                answer = ask_analyst(question,
                                     build_context(daily, forecast, metrics),
                                     api_key)
                st.markdown(answer)
            except Exception as exc:
                st.error(f"Analyst unavailable: {exc}")
