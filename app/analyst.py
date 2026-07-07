"""'Ask the Analyst' — Gemini free tier over the processed data only."""
import time

import pandas as pd

MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
EXAMPLE_QUESTIONS = [
    "How does July 4 week travel in 2026 compare to the model's expectation?",
    "How should a retailer near a top destination prepare inventory for the window?",
    "What do gas prices suggest about road-trip demand this week?",
]
SYSTEM_PROMPT = (
    "You are a careful travel-economics analyst. Answer ONLY from the data "
    "summary provided in the user message. Cite the specific numbers you use. "
    "If the data is insufficient to answer, say so plainly instead of guessing. "
    "Keep answers under 200 words. Note that hotel occupancy and card-spend "
    "data are not available in this dataset."
)


def build_context(daily: pd.DataFrame, forecast: pd.DataFrame,
                  metrics: pd.DataFrame,
                  city_index: pd.DataFrame | None = None) -> str:
    recent = daily.dropna(subset=["tsa_throughput"]).tail(14)
    window = forecast[forecast.date.between("2026-06-27", "2026-07-05")]
    lines = [
        "## Recent daily TSA throughput (passengers/day)",
        *(f"{r.date:%Y-%m-%d} ({r.date:%a}): {int(r.tsa_throughput):,}"
          for r in recent.itertuples()),
        f"Latest national avg gas price: ${recent.gas_price.iloc[-1]:.2f}/gal "
        "(EIA weekly, forward-filled — proxy)",
        "## Forecast for July 4 window (Jun 27 - Jul 5, 2026)",
        *(f"{r.date:%Y-%m-%d}: predicted {int(r.yhat):,} "
          f"[{int(r.yhat_lower):,}..{int(r.yhat_upper):,}]"
          + (f", actual {int(r.actual):,}" if pd.notna(r.actual) else ", actual pending")
          for r in window.itertuples()),
        "## Model quality (2024-2025 holdout)",
        *(f"{r.model} (train {r.train_window}): MAE {r.mae:,.0f}, MAPE {r.mape}%"
          for r in metrics.itertuples()),
        "## Context (AAA, cited): 72.2M travelers Jun 27-Jul 5 2026 (vs 71.8M "
        "2025); 61.4M by car (85%); 5.85M flying; top destinations Seattle, "
        "Anchorage, New York, Chicago, Boston, Orlando.",
    ]
    if city_index is not None:
        ranked = city_index.sort_values("composite", ascending=False)
        lines += [
            "## City America250 exposure index (0-100; structural exposure "
            "to anniversary demand, NOT observed impact; T-100 air data "
            "lags ~2-3 months)",
            *(f"{r.city}: composite {r.composite} (air {r.air_score}, "
              f"events {r.events_score}), segment: {r.cluster_label}"
              for r in ranked.itertuples()),
        ]
    return "\n".join(lines)


def ask_analyst(question: str, context: str, api_key: str) -> str:
    from google import genai
    from google.genai import errors, types

    client = genai.Client(api_key=api_key)
    last_exc: Exception | None = None
    for model in MODELS:
        for attempt in range(3):
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=f"DATA SUMMARY:\n{context}\n\nQUESTION: {question}",
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT),
                )
                return resp.text
            except errors.APIError as exc:
                last_exc = exc
                if getattr(exc, "code", None) == 429:
                    time.sleep(2 * 2**attempt)  # exponential backoff
                    continue
                raise
    raise RuntimeError(f"Gemini rate-limited on all models: {last_exc}")
