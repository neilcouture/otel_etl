"""Semantic classifier - classifies labels by name patterns."""

import re
from enum import Enum
from typing import TypedDict


class LabelCategory(str, Enum):
    """Categories for label semantic classification."""
    RESOURCE = "resource"           # Service, instance, pod, etc.
    SIGNAL = "signal"               # Status codes, error types
    DIMENSION = "dimension"         # Method, route, operation
    CORRELATION = "correlation"     # trace_id, request_id, etc.
    HISTOGRAM_INTERNAL = "internal" # le, quantile
    METADATA = "metadata"           # version, sdk info


class LabelClassification(TypedDict):
    """Classification result for a label."""
    category: LabelCategory
    handling: str
    bucket_type: str | None  # If applicable


# Label name patterns and their classifications
LABEL_PATTERNS: list[tuple[list[str], LabelCategory, str, str | None]] = [
    # Resource labels - always keep
    (
        ["service", "service_name", "job", "app", "application"],
        LabelCategory.RESOURCE,
        "keep",
        None,
    ),
    (
        ["instance", "host", "pod", "node", "container", "namespace", "pod_name", "container_name"],
        LabelCategory.RESOURCE,
        "keep_or_aggregate",
        None,
    ),
    (
        ["env", "environment", "deployment", "cluster", "region", "zone", "dc", "datacenter"],
        LabelCategory.RESOURCE,
        "keep",
        None,
    ),

    # Signal labels - bucket by value
    (
        ["status", "status_code", "http_status_code", "code", "grpc_code", "grpc_status", "response_code"],
        LabelCategory.SIGNAL,
        "bucket",
        "status_code",
    ),
    (
        ["error", "exception", "error_type", "exception_type", "exception_class"],
        LabelCategory.SIGNAL,
        "keep_type",
        None,
    ),
    (
        ["error_message", "exception_message", "error_msg"],
        LabelCategory.SIGNAL,
        "drop",
        None,
    ),

    # Dimension labels - various handling
    (
        ["method", "http_method", "request_method", "http_request_method"],
        LabelCategory.DIMENSION,
        "bucket",
        "http_method",
    ),
    (
        ["route", "http_route", "uri", "path", "url", "endpoint", "target", "http_target", "url_path"],
        LabelCategory.DIMENSION,
        "parameterize_or_top_n",
        "route",
    ),
    (
        ["operation", "db_operation", "db_statement", "command", "db_system"],
        LabelCategory.DIMENSION,
        "bucket",
        "operation",
    ),
    (
        ["rpc_method", "rpc_service", "grpc_method", "grpc_service"],
        LabelCategory.DIMENSION,
        "bucket",
        "rpc_operation",
    ),
    (
        ["messaging_operation", "messaging_destination", "messaging_system"],
        LabelCategory.DIMENSION,
        "bucket",
        "messaging",
    ),

    # Correlation labels - drop for aggregation
    (
        ["trace_id", "span_id", "traceid", "spanid"],
        LabelCategory.CORRELATION,
        "drop",
        None,
    ),
    (
        ["user_id", "userid", "customer_id", "customerid", "account_id"],
        LabelCategory.CORRELATION,
        "drop",
        None,
    ),
    (
        ["request_id", "requestid", "correlation_id", "correlationid", "session_id", "sessionid"],
        LabelCategory.CORRELATION,
        "drop",
        None,
    ),

    # Histogram internals - special handling
    (
        ["le", "quantile"],
        LabelCategory.HISTOGRAM_INTERNAL,
        "special",
        None,
    ),

    # Metadata labels - usually drop
    (
        ["version", "sdk_version", "library_version", "otel_scope_version"],
        LabelCategory.METADATA,
        "drop_or_resource",
        None,
    ),
    (
        ["otel_scope_name", "telemetry_sdk_name", "telemetry_sdk_language", "telemetry_sdk_version"],
        LabelCategory.METADATA,
        "drop",
        None,
    ),
]

# Regex patterns for suffix matching
SUFFIX_PATTERNS: list[tuple[str, LabelCategory, str, str | None]] = [
    (r"_id$", LabelCategory.CORRELATION, "drop", None),
    (r"_uuid$", LabelCategory.CORRELATION, "drop", None),
    (r"_key$", LabelCategory.CORRELATION, "drop", None),
]


def classify_label(label_name: str) -> LabelClassification:
    """Classify a label by its name pattern.

    Args:
        label_name: The label name to classify

    Returns:
        LabelClassification with category, handling, and optional bucket_type
    """
    normalized = label_name.lower().replace("-", "_")

    # First try exact matching
    for patterns, category, handling, bucket_type in LABEL_PATTERNS:
        if normalized in patterns:
            return LabelClassification(
                category=category,
                handling=handling,
                bucket_type=bucket_type,
            )

    # Then try substring matching for signal labels (status codes)
    # This catches labels like "rpc_grpc_status_code", "http_response_status_code", etc.
    status_substrings = ["status_code", "status", "grpc_status", "grpc_code", "response_code"]
    for substring in status_substrings:
        if substring in normalized:
            return LabelClassification(
                category=LabelCategory.SIGNAL,
                handling="bucket",
                bucket_type="status_code",
            )

    for pattern, category, handling, bucket_type in SUFFIX_PATTERNS:
        if re.search(pattern, normalized):
            return LabelClassification(
                category=category,
                handling=handling,
                bucket_type=bucket_type,
            )

    return LabelClassification(
        category=LabelCategory.DIMENSION,
        handling="auto",
        bucket_type=None,
    )


def should_keep_label(classification: LabelClassification, tier: int) -> bool:
    """Determine if a label should be kept based on classification and tier.

    Args:
        classification: The label's classification
        tier: The cardinality tier (1-4)

    Returns:
        True if the label should be kept
    """
    if classification["handling"] == "drop":
        return False

    if classification["category"] == LabelCategory.CORRELATION:
        return False

    if classification["category"] == LabelCategory.HISTOGRAM_INTERNAL:
        return False

    if classification["handling"] == "keep":
        return True

    if classification["handling"] == "keep_or_aggregate":
        return tier <= 3

    if tier == 4 and classification["handling"] != "keep":
        return False

    return True


def get_bucket_type(classification: LabelClassification) -> str | None:
    """Get the bucket type for a label if bucketing is needed.

    Args:
        classification: The label's classification

    Returns:
        Bucket type string or None
    """
    if classification["handling"] in ("bucket", "parameterize_or_top_n"):
        return classification["bucket_type"]
    return None


def is_entity_label(label_name: str) -> bool:
    """Check if a label should be used as part of the entity key.

    Entity labels are typically resource identifiers like service, instance, etc.

    Args:
        label_name: The label name

    Returns:
        True if this is an entity-defining label
    """
    classification = classify_label(label_name)

    if classification["category"] != LabelCategory.RESOURCE:
        return False

    if classification["handling"] in ("drop", "drop_or_resource"):
        return False

    normalized = label_name.lower().replace("-", "_")
    entity_labels = [
        "service", "service_name", "job", "app", "application",
        "instance", "host", "pod", "node", "namespace",
    ]

    return normalized in entity_labels
