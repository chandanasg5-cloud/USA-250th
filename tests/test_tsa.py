import pandas as pd

from src.ingest.tsa import parse_tsa_html

# Mirrors real tsa.gov markup (verified 2026-07-02)
FIXTURE = """
<table>
<thead><tr><th style="text-align:center;">Date</th>
<th style="text-align:center;">Numbers</th></tr></thead>
<tbody>
<tr><td class="text-align-center">7/1/2026</td><td class="text-align-center">2,654,017</td></tr>
<tr><td class="text-align-center">6/30/2026</td><td class="text-align-center">2,477,905</td></tr>
<tr><td class="text-align-center">6/29/2026</td><td class="text-align-center">2,690,919</td></tr>
</tbody></table>
"""


def test_parse_tsa_html():
    df = parse_tsa_html(FIXTURE)
    assert list(df.columns) == ["date", "tsa_throughput"]
    assert len(df) == 3
    assert df["date"].is_monotonic_increasing
    assert df.loc[df["date"] == pd.Timestamp("2026-07-01"), "tsa_throughput"].item() == 2654017
    assert df["tsa_throughput"].dtype == "int64"
