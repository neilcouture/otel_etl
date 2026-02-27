"""Schema generator - outputs YAML schema configuration."""

from typing import TypedDict, Any
from datetime import datetime, timezone
import yaml
import logging

from otel_etl.profiler.metric_discovery import MetricFamily
from otel_etl.profiler.cardinality_analyzer import CardinalityResult
from otel_etl.profiler.semantic_classifier import (
    classify_label,
    should_keep_label,
    get_bucket_type,
)
from otel_etl.config.defaults import CardinalityThresholds, DEFAULT_CARDINALITY_THRESHOLDS

logger = logging.getLogger(__name__)


class LabelSchema(TypedDict):
    """Schema for a single label."""
    tier: int
    cardinality: int
    action: str
    bucket_type: str | None
    top_values: list[str] | None
    semantic_category: str


class MetricSchema(TypedDict):
    """Schema for a metric family."""
    type: str
    labels: dict[str, LabelSchema]


class SchemaConfig(TypedDict):
    """Complete schema configuration."""
    profiled_at: str
    profiling_window_hours: float
    cardinality_thresholds: CardinalityThresholds
    metrics: dict[str, MetricSchema]


def generate_schema(
    families: dict[str, MetricFamily],
    cardinality_results: dict[str, dict[str, CardinalityResult]],
    thresholds: CardinalityThresholds | None = None,
    profiling_window_hours: float = 1.0,
) -> SchemaConfig:
    """Generate a schema configuration from profiling results.

    Args:
        families: Metric families from discover_metrics()
        cardinality_results: Results from analyze_cardinality()
        thresholds: Cardinality thresholds used
        profiling_window_hours: Window used for profiling

    Returns:
        Complete SchemaConfig
    """
    thresholds = thresholds or DEFAULT_CARDINALITY_THRESHOLDS

    metrics: dict[str, MetricSchema] = {}

    for family_name, family in families.items():
        labels_schema: dict[str, LabelSchema] = {}

        family_cardinality = cardinality_results.get(family_name, {})

        for label_name, cardinality_result in family_cardinality.items():
            classification = classify_label(label_name)
            tier = cardinality_result["tier"]

            if not should_keep_label(classification, tier):
                action = "drop"
            else:
                bucket_type = get_bucket_type(classification)
                if bucket_type:
                    if tier >= 2:
                        action = "bucket"
                    else:
                        action = "keep"
                elif classification["handling"] == "parameterize_or_top_n":
                    if tier >= 2:
                        action = "top_n"
                    else:
                        action = "keep"
                elif tier == 3:
                    action = "top_n"
                else:
                    action = cardinality_result["action"]

            labels_schema[label_name] = LabelSchema(
                tier=tier,
                cardinality=cardinality_result["cardinality"],
                action=action,
                bucket_type=get_bucket_type(classification),
                top_values=cardinality_result["top_values"] if action == "top_n" else None,
                semantic_category=classification["category"].value,
            )

        metrics[family_name] = MetricSchema(
            type=family["type"],
            labels=labels_schema,
        )

    schema = SchemaConfig(
        profiled_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        profiling_window_hours=profiling_window_hours,
        cardinality_thresholds=thresholds,
        metrics=metrics,
    )

    kept_labels = sum(
        1 for m in metrics.values()
        for l in m["labels"].values()
        if l["action"] != "drop"
    )
    dropped_labels = sum(
        1 for m in metrics.values()
        for l in m["labels"].values()
        if l["action"] == "drop"
    )

    logger.info(
        f"Generated schema: {len(metrics)} metrics, "
        f"{kept_labels} labels kept, {dropped_labels} labels dropped"
    )

    return schema


def save_schema(schema: SchemaConfig, path: str) -> None:
    """Save schema configuration to YAML file.

    Args:
        schema: Schema configuration to save
        path: Output file path
    """
    def convert_for_yaml(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: convert_for_yaml(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_for_yaml(v) for v in obj]
        return obj

    with open(path, "w") as f:
        yaml.dump(convert_for_yaml(schema), f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved schema to {path}")


def load_schema(path: str) -> SchemaConfig:
    """Load schema configuration from YAML file.

    Args:
        path: Input file path

    Returns:
        SchemaConfig
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return SchemaConfig(
        profiled_at=data["profiled_at"],
        profiling_window_hours=data["profiling_window_hours"],
        cardinality_thresholds=CardinalityThresholds(**data["cardinality_thresholds"]),
        metrics={
            name: MetricSchema(
                type=m["type"],
                labels={
                    label_name: LabelSchema(**label_data)
                    for label_name, label_data in m["labels"].items()
                },
            )
            for name, m in data["metrics"].items()
        },
    )


def diff_schemas(old: SchemaConfig, new: SchemaConfig) -> dict[str, Any]:
    """Compare two schema configurations.

    Args:
        old: Previous schema
        new: New schema

    Returns:
        Dictionary with added, removed, and changed metrics/labels
    """
    diff: dict[str, Any] = {
        "added_metrics": [],
        "removed_metrics": [],
        "added_labels": [],
        "removed_labels": [],
        "tier_changes": [],
    }

    old_metrics = set(old["metrics"].keys())
    new_metrics = set(new["metrics"].keys())

    diff["added_metrics"] = list(new_metrics - old_metrics)
    diff["removed_metrics"] = list(old_metrics - new_metrics)

    for metric_name in old_metrics & new_metrics:
        old_labels = set(old["metrics"][metric_name]["labels"].keys())
        new_labels = set(new["metrics"][metric_name]["labels"].keys())

        for label in new_labels - old_labels:
            diff["added_labels"].append(f"{metric_name}.{label}")

        for label in old_labels - new_labels:
            diff["removed_labels"].append(f"{metric_name}.{label}")

        for label in old_labels & new_labels:
            old_tier = old["metrics"][metric_name]["labels"][label]["tier"]
            new_tier = new["metrics"][metric_name]["labels"][label]["tier"]
            if old_tier != new_tier:
                diff["tier_changes"].append({
                    "metric": metric_name,
                    "label": label,
                    "old_tier": old_tier,
                    "new_tier": new_tier,
                })

    return diff
