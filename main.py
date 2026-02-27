"""Main entry point and orchestration for the OTel ETL pipeline."""

from typing import Any
from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging
import os

import pandas as pd
import numpy as np
import yaml

from otel_etl.utils.prometheus_client import PrometheusClient
from otel_etl.utils.name_sanitizer import extract_metric_family, classify_metric_type

from otel_etl.profiler.metric_discovery import discover_metrics, MetricFamily
from otel_etl.profiler.label_discovery import discover_labels
from otel_etl.profiler.cardinality_analyzer import analyze_cardinality
from otel_etl.profiler.schema_generator import (
    generate_schema,
    save_schema,
    load_schema,
    SchemaConfig,
)
from otel_etl.profiler.semantic_classifier import (
    classify_label,
    LabelCategory,
)

from otel_etl.transformer.status_bucketer import bucket_status_code, StatusBucket
from otel_etl.transformer.method_bucketer import bucket_http_method
from otel_etl.transformer.operation_bucketer import bucket_operation
from otel_etl.transformer.route_parameterizer import parameterize_route
from otel_etl.transformer.top_n_filter import TopNFilter

from otel_etl.aggregator.counter_agg import aggregate_counter
from otel_etl.aggregator.histogram_agg import aggregate_histogram
from otel_etl.aggregator.gauge_agg import aggregate_gauge
from otel_etl.aggregator.derived_agg import compute_derived_metrics

from otel_etl.feature_generator.entity_grouper import (
    EntityGrouper,
    compute_entity_key,
    add_entity_key_column,
)
from otel_etl.feature_generator.feature_namer import generate_feature_name
from otel_etl.feature_generator.wide_formatter import WideFormatter, pivot_to_wide
from otel_etl.feature_generator.delta_features import DeltaFeatureGenerator
from otel_etl.feature_generator.schema_registry import SchemaRegistry

from otel_etl.config.defaults import (
    DEFAULT_CARDINALITY_THRESHOLDS,
    DEFAULT_PROFILING_WINDOW_HOURS,
    DEFAULT_AGGREGATION_WINDOW_SECONDS,
    DEFAULT_TOP_N,
    CardinalityThresholds,
)

logger = logging.getLogger(__name__)


def run_profiler(
    prometheus_url: str = "http://localhost:9090",
    output_path: str = "schema_config.yaml",
    profiling_window_hours: float = DEFAULT_PROFILING_WINDOW_HOURS,
    cardinality_thresholds: dict[str, int] | None = None,
    top_n: int = DEFAULT_TOP_N,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> SchemaConfig:
    """Run the profiler to generate schema configuration.

    Queries Prometheus to discover metrics, labels, and cardinality,
    then generates a schema configuration file.

    Args:
        prometheus_url: Prometheus server URL
        output_path: Path to write schema config YAML
        profiling_window_hours: Time window for profiling
        cardinality_thresholds: Custom thresholds (keys: tier1_max, tier2_max, tier3_max)
        top_n: Number of top values to capture for high-cardinality labels
        include_patterns: Regex patterns to include metrics
        exclude_patterns: Regex patterns to exclude metrics

    Returns:
        Generated SchemaConfig
    """
    logger.info(f"Starting profiler against {prometheus_url}")

    thresholds: CardinalityThresholds = DEFAULT_CARDINALITY_THRESHOLDS.copy()
    if cardinality_thresholds:
        thresholds.update(cardinality_thresholds)

    client = PrometheusClient(prometheus_url)

    logger.info("Discovering metrics...")
    families = discover_metrics(client)

    if include_patterns or exclude_patterns:
        from otel_etl.profiler.metric_discovery import filter_otel_metrics
        families = filter_otel_metrics(families, include_patterns, exclude_patterns)
        logger.info(f"Filtered to {len(families)} metric families")

    logger.info("Discovering labels...")
    labels_by_family = discover_labels(client, families)

    logger.info("Analyzing cardinality...")
    cardinality_results = analyze_cardinality(
        client,
        labels_by_family,
        thresholds,
        top_n,
        profiling_window_hours,
    )

    logger.info("Generating schema...")
    schema = generate_schema(
        families,
        cardinality_results,
        thresholds,
        profiling_window_hours,
    )

    save_schema(schema, output_path)
    logger.info(f"Schema saved to {output_path}")

    return schema


def denormalize_metrics(
    raw_df: pd.DataFrame,
    schema_config: str | SchemaConfig | None = None,
    column_registry: str | SchemaRegistry | None = None,
    layers: list[int] | None = None,
    window_seconds: float = DEFAULT_AGGREGATION_WINDOW_SECONDS,
    include_deltas: bool = True,
    entity_labels: list[str] | None = None,
    feature_labels: list[str] | None = None,
    overrides_path: str | None = None,
    unique_timestamps: bool = False,
) -> pd.DataFrame:
    """Transform raw metrics into ML-ready wide-format DataFrame.

    Args:
        raw_df: DataFrame with columns: timestamp, metric, labels (dict), value
        schema_config: Schema config (path or object) or None to use defaults
        column_registry: Column registry (path or object) for schema stability
        layers: Feature layers to include (1, 2, 3)
        window_seconds: Aggregation window in seconds
        include_deltas: Whether to compute delta features
        entity_labels: Labels to use for entity key (default: ['service_name'])
        feature_labels: Labels to include in feature names (default: [] = none)
        overrides_path: Path to overrides YAML
        unique_timestamps: If True, pivot only by timestamp (entity embedded in column names)

    Returns:
        Wide-format DataFrame with features as columns
    """
    if raw_df.empty:
        logger.warning("Empty input DataFrame")
        return pd.DataFrame()

    layers = layers or [1, 2, 3]

    schema = _load_schema_config(schema_config)
    registry = _load_column_registry(column_registry)
    overrides = _load_overrides(overrides_path)

    logger.info(f"Processing {len(raw_df)} raw metric rows")

    transformed_df = _apply_transformations(raw_df, schema, overrides)

    transformed_df = add_entity_key_column(transformed_df, "labels", entity_labels)

    aggregated_df = _aggregate_metrics(transformed_df, schema, window_seconds)

    feature_df = _generate_features(aggregated_df, schema, layers, unique_timestamps)

    wide_df = _pivot_to_wide_format(feature_df, registry, unique_timestamps)

    # Ensure all status bucket columns exist for schema stability
    wide_df = _ensure_status_columns(wide_df)

    if include_deltas:
        delta_gen = DeltaFeatureGenerator(
            entity_col="entity_key" if not unique_timestamps else None
        )
        wide_df = delta_gen.generate(wide_df)

    if registry is not None:
        wide_df = registry.align_dataframe(wide_df, register_new=True)

    logger.info(
        f"Output: {len(wide_df)} rows, {len(wide_df.columns)} columns"
    )

    return wide_df


def _load_schema_config(
    config: str | SchemaConfig | None,
) -> SchemaConfig | None:
    """Load schema config from path or return as-is."""
    if config is None:
        return None
    if isinstance(config, str):
        if os.path.exists(config):
            return load_schema(config)
        logger.warning(f"Schema config not found: {config}")
        return None
    return config


def _load_column_registry(
    registry: str | SchemaRegistry | None,
) -> SchemaRegistry | None:
    """Load column registry from path or return as-is."""
    if registry is None:
        return None
    if isinstance(registry, str):
        if os.path.exists(registry):
            return SchemaRegistry.load(registry)
        return SchemaRegistry()
    return registry


def _load_overrides(path: str | None) -> dict[str, Any]:
    """Load overrides from YAML file."""
    if path is None:
        default_path = Path(__file__).parent / "config" / "overrides.yaml"
        if default_path.exists():
            path = str(default_path)
        else:
            return {}

    if not os.path.exists(path):
        return {}

    with open(path) as f:
        return yaml.safe_load(f) or {}


# Core status buckets that should always have columns (for schema stability)
CORE_STATUS_BUCKETS = ["success", "client_error", "server_error"]


def _ensure_status_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all status bucket variants exist for signal-split metrics.

    For each metric that has status-split columns (e.g., metric__p50__success),
    ensure all core status buckets have columns, filling with NaN if missing.
    This provides schema stability for ML pipelines.
    """
    if df.empty:
        return df

    existing_cols = set(df.columns)
    new_cols = {}

    # Find columns that have status suffixes
    for col in existing_cols:
        for bucket in CORE_STATUS_BUCKETS:
            if col.endswith(f"__{bucket}"):
                # Extract base name (without status suffix)
                base = col.rsplit("__", 1)[0]
                # Ensure all status variants exist
                for other_bucket in CORE_STATUS_BUCKETS:
                    variant = f"{base}__{other_bucket}"
                    if variant not in existing_cols and variant not in new_cols:
                        new_cols[variant] = np.nan
                break

    if new_cols:
        for col_name, fill_value in new_cols.items():
            df[col_name] = fill_value

    return df


def _extract_signal_key(labels: dict[str, Any]) -> str:
    """Extract signal labels (status codes) from labels dict into a key.

    Signal labels are identified by semantic classification (category=SIGNAL).
    Returns empty string if no signal labels found.
    """
    signal_parts = []
    for label_name, value in sorted(labels.items()):
        classification = classify_label(label_name)
        if classification["category"] == LabelCategory.SIGNAL and classification["bucket_type"]:
            signal_parts.append(f"{value}")
    return "__".join(signal_parts) if signal_parts else ""


def _apply_transformations(
    df: pd.DataFrame,
    schema: SchemaConfig | None,
    overrides: dict[str, Any],
) -> pd.DataFrame:
    """Apply label transformations based on schema."""
    result = df.copy()

    def transform_labels(row):
        labels = row["labels"].copy()
        metric = row["metric"]
        metric_family = extract_metric_family(metric)

        metric_schema = None
        if schema and metric_family in schema.get("metrics", {}):
            metric_schema = schema["metrics"][metric_family]

        force_drop = overrides.get("force_drop_labels", [])
        for label in force_drop:
            labels.pop(label, None)

        transformed = {}
        for label, value in labels.items():
            if label in force_drop:
                continue

            action = "keep"
            bucket_type = None

            # First check schema, then fall back to semantic classification
            if metric_schema and label in metric_schema.get("labels", {}):
                label_schema = metric_schema["labels"][label]
                action = label_schema.get("action", "keep")
                bucket_type = label_schema.get("bucket_type")
            else:
                # Use semantic classifier for labels not in schema
                classification = classify_label(label)
                if classification["category"] == LabelCategory.SIGNAL:
                    bucket_type = classification["bucket_type"]

            if action == "drop":
                continue

            if bucket_type == "status_code":
                value = bucket_status_code(value, label)
            elif bucket_type == "http_method":
                value = bucket_http_method(value)
            elif bucket_type == "operation":
                value = bucket_operation(value)
            elif bucket_type == "route":
                value = parameterize_route(value)

            if action == "top_n" and metric_schema:
                top_values = metric_schema["labels"][label].get("top_values", [])
                if top_values:
                    filter_instance = TopNFilter(top_values)
                    value = filter_instance.filter(str(value))

            transformed[label] = value

        return transformed

    result["labels"] = result.apply(transform_labels, axis=1)
    # Extract signal key after transformation (so status is already bucketed)
    result["signal_key"] = result["labels"].apply(_extract_signal_key)
    return result


def _aggregate_metrics(
    df: pd.DataFrame,
    schema: SchemaConfig | None,
    window_seconds: float,
) -> pd.DataFrame:
    """Aggregate metrics by type, splitting by signal labels (status codes)."""
    results = []

    df["metric_family"] = df["metric"].apply(extract_metric_family)
    df["metric_type"] = df["metric"].apply(classify_metric_type)

    # Group by signal_key in addition to timestamp, entity_key, metric_family
    # This splits aggregations by status bucket (success, client_error, server_error)
    for (ts, entity_key, family, signal_key), group in df.groupby(
        ["timestamp", "entity_key", "metric_family", "signal_key"], sort=False
    ):
        metric_types = group["metric_type"].unique()

        if "histogram" in metric_types or "histogram_component" in metric_types:
            agg_result = _aggregate_histogram_group(group)
        elif "counter" in metric_types:
            agg_result = _aggregate_counter_group(group, window_seconds)
        else:
            agg_result = _aggregate_gauge_group(group)

        for agg_name, agg_value in agg_result.items():
            label_values = {}
            if not group.empty:
                first_labels = group["labels"].iloc[0]
                label_values = {
                    k: v for k, v in first_labels.items()
                    if k not in ["le", "quantile"]
                }

            results.append({
                "timestamp": ts,
                "entity_key": entity_key,
                "metric_family": family,
                "signal_key": signal_key,
                "aggregation": agg_name,
                "value": agg_value,
                "labels": label_values,
            })

    return pd.DataFrame(results)


def _aggregate_histogram_group(group: pd.DataFrame) -> dict[str, float]:
    """Aggregate histogram metric group."""
    bucket_df = group[group["metric"].str.endswith("_bucket")].copy()

    if bucket_df.empty:
        return {}

    bucket_df["le"] = bucket_df["labels"].apply(lambda x: x.get("le", "+Inf"))

    sum_val = None
    count_val = None

    sum_rows = group[group["metric"].str.endswith("_sum")]
    if not sum_rows.empty:
        sum_val = sum_rows["value"].iloc[-1]

    count_rows = group[group["metric"].str.endswith("_count")]
    if not count_rows.empty:
        count_val = count_rows["value"].iloc[-1]

    from otel_etl.aggregator.histogram_agg import aggregate_histogram as agg_hist
    result = agg_hist(bucket_df, sum_val, count_val)

    return {
        "p50": result["p50"],
        "p75": result["p75"],
        "p90": result["p90"],
        "p95": result["p95"],
        "p99": result["p99"],
        "mean": result["mean"],
        "count": result["count"],
        "sum": result["sum"],
    }


def _aggregate_counter_group(
    group: pd.DataFrame,
    window_seconds: float,
) -> dict[str, float]:
    """Aggregate counter metric group."""
    total_rows = group[group["metric"].str.endswith("_total")]

    if total_rows.empty:
        return {}

    from otel_etl.aggregator.counter_agg import aggregate_counter as agg_counter
    result = agg_counter(
        total_rows["value"],
        total_rows["timestamp"],
        window_seconds,
    )

    return {
        "rate": result["rate_per_sec"],
        "count": result["count"],
    }


def _aggregate_gauge_group(group: pd.DataFrame) -> dict[str, float]:
    """Aggregate gauge metric group."""
    if group.empty:
        return {}

    from otel_etl.aggregator.gauge_agg import aggregate_gauge as agg_gauge
    result = agg_gauge(group["value"], group["timestamp"])

    return {
        "last": result["last"],
        "mean": result["mean"],
        "min": result["min"],
        "max": result["max"],
        "stddev": result["stddev"],
    }


def _generate_features(
    aggregated_df: pd.DataFrame,
    schema: SchemaConfig | None,
    layers: list[int],
    unique_timestamps: bool = False,
) -> pd.DataFrame:
    """Generate feature names from aggregated data."""
    if aggregated_df.empty:
        return aggregated_df

    def make_feature_name(row):
        # Base: metric_family__aggregation
        base_name = f"{row['metric_family']}__{row['aggregation']}"

        # Append signal_key if present (e.g., success, client_error, server_error)
        signal_key = row.get("signal_key", "")
        if signal_key:
            base_name = f"{base_name}__{signal_key}"

        # For unique_timestamps, prefix with service name
        if unique_timestamps:
            labels = row.get("labels", {})
            service = labels.get("service_name") or labels.get("service") or ""
            if service:
                return f"{service}__{base_name}"
        return base_name

    result = aggregated_df.copy()
    result["feature"] = result.apply(make_feature_name, axis=1)

    return result


def _pivot_to_wide_format(
    feature_df: pd.DataFrame,
    registry: SchemaRegistry | None,
    unique_timestamps: bool = False,
) -> pd.DataFrame:
    """Pivot to wide format."""
    if feature_df.empty:
        return pd.DataFrame()

    if unique_timestamps:
        index_cols = ["timestamp"]
    else:
        index_cols = ["timestamp", "entity_key"]

    wide_df = pivot_to_wide(
        feature_df,
        index_cols=index_cols,
        feature_col="feature",
        value_col="value",
    )

    return wide_df

