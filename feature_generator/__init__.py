"""Feature generation modules for ML-ready output."""

from otel_etl.feature_generator.entity_grouper import compute_entity_key
from otel_etl.feature_generator.feature_namer import generate_feature_name
from otel_etl.feature_generator.wide_formatter import pivot_to_wide
from otel_etl.feature_generator.delta_features import compute_delta_features
from otel_etl.feature_generator.schema_registry import SchemaRegistry

__all__ = [
    "compute_entity_key",
    "generate_feature_name",
    "pivot_to_wide",
    "compute_delta_features",
    "SchemaRegistry",
]
