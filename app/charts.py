"""Plotly figure builders.

Color discipline (dataviz method): color follows the entity via fixed maps —
filters never repaint surviving series. Region/city palettes validated for
CVD separation and lightness band (validate_palette.js, light surface).
Slots below 3:1 surface contrast rely on the legend + unified hover labels.
"""
import pandas as pd
import plotly.graph_objects as go

ACCENT = "#B31942"      # brand accent: forecast + US series
INK = "#0b0b0b"
NEUTRAL = "#898781"     # muted ink for historical actuals
GRID = "#e1e0d9"
WINDOW_START, WINDOW_END = "2026-06-27", "2026-07-05"

# Fixed entity -> color assignments (never cycled, never rank-based).
REGION_COLORS = {
    "US": ACCENT, "PADD1": "#2a78d6", "PADD2": "#1baf7a",
    "PADD3": "#eda100", "PADD4": "#4a3aa7", "PADD5": "#eb6834",
}
CITY_COLORS = {
    "Anchorage": "#2a78d6", "Boston": "#1baf7a", "Chicago": "#eda100",
    "Denver": "#008300", "Los Angeles": "#4a3aa7", "New York": "#e34948",
    "Orlando": "#e87ba4", "Seattle": "#eb6834",
}

_LAYOUT = dict(
    template="plotly_white", margin=dict(l=40, r=20, t=48, b=40),
    hovermode="x unified", font=dict(size=13, color=INK),
    xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID),
    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
)


def forecast_chart(daily: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    hist = daily.dropna(subset=["tsa_throughput"])
    hist = hist[hist.date >= "2025-01-01"]  # readable default zoom
    fig.add_scatter(x=hist.date, y=hist.tsa_throughput, name="Actual",
                    line=dict(color=NEUTRAL, width=1.5))
    fig.add_scatter(x=forecast.date, y=forecast.yhat_upper, line=dict(width=0),
                    showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=forecast.date, y=forecast.yhat_lower, fill="tonexty",
                    fillcolor="rgba(179,25,66,0.12)", line=dict(width=0),
                    name="Uncertainty", hoverinfo="skip")
    fig.add_scatter(x=forecast.date, y=forecast.yhat, name="Forecast",
                    line=dict(color=ACCENT, width=2))
    actual26 = forecast.dropna(subset=["actual"])
    fig.add_scatter(x=actual26.date, y=actual26.actual, name="2026 actual",
                    mode="markers", marker=dict(color=INK, size=6))
    fig.add_vrect(x0=WINDOW_START, x1=WINDOW_END, fillcolor="rgba(179,25,66,0.07)",
                  line_width=0, annotation_text="July 4 window",
                  annotation_position="top left")
    fig.update_layout(title="Daily TSA throughput — actual vs forecast",
                      yaxis_title="Passengers/day", **_LAYOUT)
    return fig


def gas_chart(gas: pd.DataFrame, regions: list[str]) -> go.Figure:
    fig = go.Figure()
    for region in sorted(regions, key=lambda r: (r != "US", r)):
        sub = gas[(gas.region == region) & (gas.date >= "2019-01-01")]
        fig.add_scatter(x=sub.date, y=sub.gas_price, name=region,
                        line=dict(width=2 if region == "US" else 1.2,
                                  color=REGION_COLORS.get(region, NEUTRAL)))
    fig.update_layout(title="Weekly retail gasoline price (EIA, $/gal)",
                      yaxis_title="$/gal", **_LAYOUT)
    return fig


def weather_chart(weather: pd.DataFrame, cities: list[str],
                  start, end) -> go.Figure:
    fig = go.Figure()
    sub = weather[weather.city.isin(cities) & weather.date.between(start, end)]
    for city, grp in sub.groupby("city"):
        fig.add_scatter(x=grp.date, y=grp.tmax_f, name=city,
                        line=dict(width=1.5, color=CITY_COLORS.get(city, NEUTRAL)))
    fig.update_layout(title="Daily high temperature by destination (Open-Meteo)",
                      yaxis_title="°F", **_LAYOUT)
    return fig
