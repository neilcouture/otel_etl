"""Aggregator modules for metric computation."""

from otel_etl.aggregator.counter_agg import aggregate_counter
from otel_etl.aggregator.histogram_agg import aggregate_histogram
from otel_etl.aggregator.gauge_agg import aggregate_gauge
from otel_etl.aggregator.derived_agg import compute_derived_metrics

__all__ = [
    "aggregate_counter",
    "aggregate_histogram",
    "aggregate_gauge",
    "compute_derived_metrics",
]
