# Positioning notes (not part of the public README)

One-line pitches per employer type, and the interview soundbites to prepare.

## One-line pitches

- **Retail:** "I forecast demand around a national event and translated it into
  inventory-prep recommendations by market." Lean on the AI panel's inventory
  answers and the city index (extension).
- **Consulting:** "I built a decision tool from messy public sources, made the
  tradeoffs explicit, and validated the model against reality." Lean on the
  honesty about proxies and the predicted-vs-actual loop.
- **Travel / hospitality:** "I modeled travel demand across air, road, and
  cruise and clustered destination markets." Lean on TSA, BTS (extension), and
  the destination rankings.
- **Government / public sector:** "I quantified the economic footprint of a
  federal commemoration using only public data." Lean on Census (extension),
  the transparency, and the reproducible pipeline.

## Interview soundbites

1. **Metric-definition discipline:** how the composite index is defined and why
   the weights are defensible (extension; document when built).
2. **The data-gap decision:** why hotel occupancy uses a proxy instead of faking
   STR data — handling imperfect data cleanly is the point of the project.
3. **The validation:** predicted vs actual for the July 4, 2026 window, and what
   the model got right and missed. Current state: Prophet (2019–2023 train,
   COVID indicator) hit 6.33% MAPE on the 2024–25 holdout; all six window
   actuals through July 2 landed inside the forecast interval, running ~2–6%
   hot — consistent with AAA's record-travel call.
