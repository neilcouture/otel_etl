"""
OTel Metrics ETL - Transform high-cardinality OpenTelemetry metrics into ML-ready features.

Public API:
    - run_profiler(): Generate schema config from Prometheus metrics
    - denormalize_metrics(): Transform raw metrics into wide-format DataFrame
    - fetch_and_denormalize(): Fetch from Prometheus and transform in one call
"""

from otel_etl.main import run_profiler, run_profiler_from_dataframe, denormalize_metrics

__all__ = ["run_profiler", "run_profiler_from_dataframe", "denormalize_metrics"]
__version__ = "0.1.0"
