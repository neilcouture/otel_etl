"""Profiler modules for schema discovery and cardinality analysis."""

from otel_etl.profiler.metric_discovery import discover_metrics
from otel_etl.profiler.label_discovery import discover_labels
from otel_etl.profiler.cardinality_analyzer import analyze_cardinality
from otel_etl.profiler.semantic_classifier import classify_label
from otel_etl.profiler.schema_generator import generate_schema

__all__ = [
    "discover_metrics",
    "discover_labels",
    "analyze_cardinality",
    "classify_label",
    "generate_schema",
]
