"""Feature namer - generates stable feature names from metric components."""

from otel_etl.utils.name_sanitizer import (
    sanitize_name,
    sanitize_label_value,
    build_feature_name,
    MAX_FEATURE_NAME_LENGTH,
)


def generate_feature_name(
    metric_family: str,
    aggregation: str,
    label_values: dict[str, str] | None = None,
    include_labels: list[str] | None = None,
) -> str:
    """Generate a stable feature name.

    Format: {metric_family}__{agg}__{label1}_{value1}__{label2}_{value2}__...

    Args:
        metric_family: Base metric name
        aggregation: Aggregation type (p99, rate, count, etc.)
        label_values: Dictionary of label -> value to include
        include_labels: Optional list of labels to include (in order)

    Returns:
        Feature name string
    """
    if label_values is None:
        label_values = {}

    if include_labels is None:
        include_labels = sorted(label_values.keys())

    label_parts = [
        (label, label_values[label])
        for label in include_labels
        if label in label_values
    ]

    return build_feature_name(metric_family, aggregation, label_parts)


def generate_feature_names_for_metric(
    metric_family: str,
    metric_type: str,
    label_combinations: list[dict[str, str]],
    include_labels: list[str] | None = None,
) -> list[str]:
    """Generate all feature names for a metric.

    Args:
        metric_family: Base metric name
        metric_type: Type of metric (counter, histogram, gauge)
        label_combinations: List of label value dictionaries
        include_labels: Labels to include in names

    Returns:
        List of feature names
    """
    aggregations = _get_aggregations_for_type(metric_type)

    feature_names = []

    for agg in aggregations:
        for labels in label_combinations:
            name = generate_feature_name(metric_family, agg, labels, include_labels)
            feature_names.append(name)

    return feature_names


def _get_aggregations_for_type(metric_type: str) -> list[str]:
    """Get standard aggregations for a metric type.

    Args:
        metric_type: Type of metric

    Returns:
        List of aggregation names
    """
    if metric_type == "counter":
        return ["rate", "count"]

    if metric_type == "histogram":
        return ["p50", "p75", "p90", "p95", "p99", "mean", "count", "sum"]

    if metric_type == "gauge":
        return ["last", "mean", "min", "max", "stddev"]

    return ["value"]


def generate_derived_feature_name(
    base_name: str,
    modifier: str,
) -> str:
    """Generate derived feature name from base name.

    Args:
        base_name: Base feature name
        modifier: Modifier to append (delta_5m, pct_change_1h, etc.)

    Returns:
        Derived feature name
    """
    suffix = f"__{sanitize_name(modifier)}"

    if len(base_name) + len(suffix) > MAX_FEATURE_NAME_LENGTH:
        truncated = base_name[: MAX_FEATURE_NAME_LENGTH - len(suffix)]
        last_sep = truncated.rfind("__")
        if last_sep > 0:
            truncated = truncated[:last_sep]
        return truncated + suffix

    return base_name + suffix


def parse_feature_name(feature_name: str) -> dict[str, str]:
    """Parse a feature name into components.

    Args:
        feature_name: Feature name string

    Returns:
        Dictionary with metric_family, aggregation, and label_values
    """
    parts = feature_name.split("__")

    if len(parts) < 2:
        return {
            "metric_family": feature_name,
            "aggregation": "value",
            "label_values": {},
        }

    metric_family = parts[0]
    aggregation = parts[1]

    label_values = {}
    for part in parts[2:]:
        if "_" in part:
            idx = part.index("_")
            label_name = part[:idx]
            label_value = part[idx + 1:]
            label_values[label_name] = label_value

    return {
        "metric_family": metric_family,
        "aggregation": aggregation,
        "label_values": label_values,
    }


class FeatureNamer:
    """Generates and tracks feature names."""

    def __init__(
        self,
        metric_family: str,
        include_labels: list[str] | None = None,
    ):
        """Initialize feature namer.

        Args:
            metric_family: Base metric name
            include_labels: Labels to include in names
        """
        self.metric_family = metric_family
        self.include_labels = include_labels
        self._generated_names: set[str] = set()

    def generate(
        self,
        aggregation: str,
        label_values: dict[str, str] | None = None,
    ) -> str:
        """Generate and track a feature name.

        Args:
            aggregation: Aggregation type
            label_values: Label values to include

        Returns:
            Feature name
        """
        name = generate_feature_name(
            self.metric_family,
            aggregation,
            label_values,
            self.include_labels,
        )
        self._generated_names.add(name)
        return name

    def get_all_generated(self) -> list[str]:
        """Get all generated feature names.

        Returns:
            Sorted list of feature names
        """
        return sorted(self._generated_names)
