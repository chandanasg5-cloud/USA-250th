"""Predicted-vs-actual validation for the July 4, 2026 window.

The post-mortem is generated deterministically from the numbers (no LLM),
so the closing story works offline and is reproducible.
"""
import pandas as pd

WINDOW_START, WINDOW_END = "2026-06-27", "2026-07-05"


def window_errors(forecast: pd.DataFrame) -> pd.DataFrame:
    """Error table for window days that have an observed actual."""
    w = forecast[forecast.date.between(WINDOW_START, WINDOW_END)].copy()
    w = w.dropna(subset=["actual"]).reset_index(drop=True)
    w["error"] = w["actual"] - w["yhat"]
    w["pct_error"] = (w["actual"] - w["yhat"]) / w["actual"] * 100
    w["inside_interval"] = w["actual"].between(w["yhat_lower"], w["yhat_upper"])
    return w


def post_mortem(errors: pd.DataFrame, holdout_mape: float) -> str:
    """Plain-English write-up of how the forecast is holding up."""
    window_days = pd.date_range(WINDOW_START, WINDOW_END)
    n_total, n_obs = len(window_days), len(errors)
    if n_obs == 0:
        return (
            "No actuals for the July 4 window yet — TSA publishes with about "
            "a one-day lag. Check back after June 27, 2026."
        )

    coverage = int(errors["inside_interval"].sum())
    live_mape = errors["pct_error"].abs().mean()
    bias = errors["pct_error"].mean()
    direction = "above" if bias > 0 else "below"
    best = errors.loc[errors["pct_error"].abs().idxmin()]
    worst = errors.loc[errors["pct_error"].abs().idxmax()]

    parts = [
        f"**{n_obs} of {n_total}** window days observed so far. "
        f"**{coverage} of {n_obs}** actuals landed inside the forecast's "
        "uncertainty interval.",
        f"Actuals are running **{abs(bias):.1f}% {direction}** the point "
        f"forecast on average (live MAPE {live_mape:.1f}%, vs {holdout_mape} "
        "on the 2024–25 holdout). "
        + ("A positive bias is consistent with AAA's record 72.2M-traveler "
           "projection: demand is outrunning a model trained on ordinary "
           "years, exactly what a once-in-a-generation anniversary should do."
           if bias > 0 else
           "Actuals coming in under forecast suggests the anniversary surge "
           "is materializing more on roads than in air travel."),
        f"Closest call: {best.date:%b %d} (off by {best.pct_error:+.1f}%). "
        f"Largest miss: {worst.date:%b %d} (off by {worst.pct_error:+.1f}%"
        + (", outside the interval" if not worst.inside_interval else "")
        + ").",
        "Known model limitations feeding these gaps: gas price is held at the "
        "last observed weekly value for future days, and the model cannot see "
        "one-off 250th-anniversary events — both documented assumptions.",
    ]
    if n_obs < n_total:
        parts.append(
            f"{n_total - n_obs} day(s) still pending — TSA publishes with "
            "about a one-day lag; rerun the pipeline to pull them in."
        )
    return "\n\n".join(parts)
