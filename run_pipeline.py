"""Run the full local pipeline: ingest -> build -> forecast.

Usage: python run_pipeline.py [--skip-ingest]
Outputs land in data/processed/ and are committed (the app's data source).
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
    try:
        from src.model import forecast  # created in Task 8
        print("--- forecast ---")
        forecast.main()
    except ImportError:
        print("(forecast model not built yet — skipping)")


if __name__ == "__main__":
    run(skip_ingest="--skip-ingest" in sys.argv)
