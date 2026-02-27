"""Name sanitization utilities for metrics, labels, and feature names."""

import re
from typing import Optional

# Max length for feature names
MAX_FEATURE_NAME_LENGTH = 128


def sanitize_name(name: str, max_length: Optional[int] = None) -> str:
    """Sanitize a metric or label name for use in feature names.

    Rules:
    - Replace '.' with '_'
    - Replace '-' with '_'
    - Lowercase everything
    - Remove special characters (keep alphanumeric and underscore)
    - Collapse multiple underscores
    - Strip leading/trailing underscores
    - Truncate to max_length if specified

    Args:
        name: Original name
        max_length: Optional maximum length for output

    Returns:
        Sanitized name
    """
    result = name.lower()
    result = result.replace(".", "_")
    result = result.replace("-", "_")
    result = re.sub(r"[^a-z0-9_]", "", result)
    result = re.sub(r"_+", "_", result)
    result = result.strip("_")

    if max_length and len(result) > max_length:
        result = result[:max_length].rstrip("_")

    return result


def sanitize_label_value(value: str, max_length: int = 50) -> str:
    """Sanitize a label value for use in feature names.

    Similar to sanitize_name but preserves more structure for values
    that may contain meaningful patterns.

    Args:
        value: Original label value
        max_length: Maximum length for output

    Returns:
        Sanitized value
    """
    result = value.lower()
    result = result.replace(".", "_")
    result = result.replace("-", "_")
    result = result.replace("/", "_")
    result = result.replace(":", "_")
    result = re.sub(r"[^a-z0-9_]", "", result)
    result = re.sub(r"_+", "_", result)
    result = result.strip("_")

    if len(result) > max_length:
        result = result[:max_length].rstrip("_")

    if not result:
        result = "empty"

    return result


def build_feature_name(
    metric_family: str,
    aggregation: str,
    label_parts: list[tuple[str, str]],
) -> str:
    """Build a feature name from components.

    Format: {metric_family}__{agg}__{label1}_{value1}__{label2}_{value2}__...

    Args:
        metric_family: Base metric name (already sanitized)
        aggregation: Aggregation type (e.g., 'p99', 'rate', 'count')
        label_parts: List of (label_name, label_value) tuples

    Returns:
        Complete feature name, truncated if necessary
    """
    parts = [sanitize_name(metric_family), sanitize_name(aggregation)]

    for label_name, label_value in sorted(label_parts):
        sanitized_label = sanitize_name(label_name)
        sanitized_value = sanitize_label_value(label_value)
        parts.append(f"{sanitized_label}_{sanitized_value}")

    feature_name = "__".join(parts)

    if len(feature_name) > MAX_FEATURE_NAME_LENGTH:
        feature_name = feature_name[:MAX_FEATURE_NAME_LENGTH]
        last_sep = feature_name.rfind("__")
        if last_sep > 0:
            feature_name = feature_name[:last_sep]

    return feature_name


def extract_metric_family(metric_name: str) -> str:
    """Extract the base metric family from a metric name.

    Strips suffixes like _total, _bucket, _sum, _count, _info, _created.

    Args:
        metric_name: Full metric name

    Returns:
        Base metric family name
    """
    suffixes = ["_total", "_bucket", "_sum", "_count", "_info", "_created"]

    result = metric_name
    for suffix in suffixes:
        if result.endswith(suffix):
            result = result[: -len(suffix)]
            break

    return result


def classify_metric_type(metric_name: str) -> str:
    """Classify a metric by its type based on naming convention.

    Args:
        metric_name: Full metric name

    Returns:
        One of: 'counter', 'histogram', 'histogram_component', 'info', 'timestamp', 'gauge'
    """
    if metric_name.endswith("_total"):
        return "counter"
    if metric_name.endswith("_bucket"):
        return "histogram"
    if metric_name.endswith("_sum") or metric_name.endswith("_count"):
        return "histogram_component"
    if metric_name.endswith("_info"):
        return "info"
    if metric_name.endswith("_created"):
        return "timestamp"
    return "gauge"
