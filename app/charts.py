"""Plotly figure builders.

Color discipline (dataviz method): color follows the entity via fixed maps —
filters never repaint surviving series. Region/city palettes validated for
CVD separation and lightness band (validate_palette.js, light surface).
Slots below 3:1 surface contrast rely on the legend + unified hover labels.
"""
import pandas as pd
import plotly.graph_objects as go

from src.model.city_index import composite

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
    fig.add_scatter(x=forecast.date, y=forecast.yhat_upper, mode="lines",
                    line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=forecast.date, y=forecast.yhat_lower, mode="lines",
                    fill="tonexty", fillcolor="rgba(179,25,66,0.12)",
                    line=dict(width=0), name="Uncertainty", hoverinfo="skip")
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


def validation_chart(forecast: pd.DataFrame) -> go.Figure:
    """Window-zoomed predicted-vs-actual view."""
    fig = go.Figure()
    zoom = forecast[forecast.date.between("2026-06-20", "2026-07-05")]
    fig.add_scatter(x=zoom.date, y=zoom.yhat_upper, mode="lines",
                    line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=zoom.date, y=zoom.yhat_lower, mode="lines",
                    fill="tonexty", fillcolor="rgba(179,25,66,0.12)",
                    line=dict(width=0), name="Uncertainty", hoverinfo="skip")
    fig.add_scatter(x=zoom.date, y=zoom.yhat, name="Forecast",
                    line=dict(color=ACCENT, width=2))
    obs = zoom.dropna(subset=["actual"])
    fig.add_scatter(x=obs.date, y=obs.actual, name="Actual",
                    mode="lines+markers",
                    line=dict(color=INK, width=1.5, dash="dot"),
                    marker=dict(color=INK, size=8))
    fig.add_vrect(x0=WINDOW_START, x1=WINDOW_END, fillcolor="rgba(179,25,66,0.05)",
                  line_width=0, annotation_text="July 4 window",
                  annotation_position="top left")
    fig.update_layout(title="July 4 window — predicted vs actual",
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


COMPONENT_COLORS = {  # component -> color, fixed like REGION_COLORS
    "air": "#2a78d6", "events": ACCENT,
    "population": "#1baf7a", "income": "#eda100",
}
CLUSTER_COLORS = {  # ranked label -> color, fixed
    "National anniversary magnets": ACCENT,
    "Strong regional draws": "#2a78d6",
    "Focused local hosts": "#1baf7a",
}


def index_leaderboard(scores: pd.DataFrame, weights: dict) -> go.Figure:
    """Stacked weighted contributions per city; bar length = composite."""
    df = scores.assign(_c=composite(scores, weights)).sort_values("_c")
    total = sum(weights.values())
    fig = go.Figure()
    for comp, color in COMPONENT_COLORS.items():
        fig.add_bar(y=df.city, x=df[f"{comp}_score"] * weights[comp] / total,
                    name=comp.replace("_", " ").title(), orientation="h",
                    marker_color=color)
    fig.update_layout(**_LAYOUT, barmode="stack",
                      title="America250 exposure index (0–100)",
                      xaxis_title="weighted contribution to composite")
    return fig


def cluster_scatter(scores: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for label, grp in scores.groupby("cluster_label"):
        fig.add_scatter(x=grp.air_score, y=grp.events_score, text=grp.city,
                        mode="markers+text", textposition="top center",
                        name=label, marker=dict(
                            size=12, color=CLUSTER_COLORS.get(label, INK)))
    fig.update_layout(**_LAYOUT, title="City segments (KMeans, k=3)",
                      xaxis_title="Air capacity score",
                      yaxis_title="Events intensity score")
    return fig


def momentum_chart(momentum: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for city, grp in momentum.groupby("city"):
        fig.add_scatter(x=grp.month, y=grp.yoy_pct, name=city,
                        mode="lines+markers",
                        line=dict(color=CITY_COLORS.get(city, INK)))
    fig.add_hline(y=0, line_dash="dot", line_color=NEUTRAL)
    fig.update_layout(**_LAYOUT,
                      title="Air passengers, year-over-year change",
                      yaxis_title="YoY %")
    return fig
