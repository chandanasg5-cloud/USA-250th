"""Run the full local pipeline: ingest -> build -> forecast -> city layer.

Usage: python run_pipeline.py [--skip-ingest]
Outputs land in data/processed/ and are committed (the app's data source).
City-layer inputs: CENSUS_API_KEY in .env; optional fresh T-100 download
in data/raw/ (otherwise the committed reference CSV is reused).
"""
import sys

from src.ingest import build_dataset, eia_gas, holidays, tsa, weather


def run(skip_ingest: bool = False) -> None:
    if not skip_ingest:
        for step in (tsa, eia_gas, weather, holidays):
            print(f"--- {step.__name__} ---")
            step.main()
    print("--- build_dataset ---")
    build_dataset.main()
    from src.model import forecast
    print("--- forecast ---")
    forecast.main()

    # City layer (spec 2026-07-07): each step skips cleanly if its inputs
    # aren't in place yet, so a national-only refresh always works.
    from src.ingest import bts_t100, census_acs
    from src.model import city_cluster, city_index
    for step in (census_acs, bts_t100, city_index, city_cluster):
        print(f"--- {step.__name__} ---")
        try:
            step.main()
        except (RuntimeError, FileNotFoundError) as exc:
            print(f"(skipping city layer from here: {exc})")
            break


if __name__ == "__main__":
    run(skip_ingest="--skip-ingest" in sys.argv)
