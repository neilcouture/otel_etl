"""Metric discovery module - queries Prometheus for available metrics."""

from typing import TypedDict
import logging

from otel_etl.utils.prometheus_client import PrometheusClient
from otel_etl.utils.name_sanitizer import extract_metric_family, classify_metric_type

logger = logging.getLogger(__name__)


class MetricInfo(TypedDict):
    """Information about a discovered metric."""
    name: str
    family: str
    type: str


class MetricFamily(TypedDict):
    """A group of related metrics (e.g., histogram with _bucket, _sum, _count)."""
    name: str
    type: str
    metrics: list[str]


def discover_metrics(client: PrometheusClient) -> dict[str, MetricFamily]:
    """Discover all metrics and group them into families.

    Queries Prometheus for all metric names and groups them by:
    - *_total → counter
    - *_bucket, *_sum, *_count → histogram
    - *_info → info gauge
    - *_created → timestamp
    - everything else → gauge

    Args:
        client: PrometheusClient instance

    Returns:
        Dictionary mapping family name to MetricFamily info
    """
    metric_names = client.get_metric_names()
    logger.info(f"Discovered {len(metric_names)} metrics")

    families: dict[str, MetricFamily] = {}

    for name in metric_names:
        family_name = extract_metric_family(name)
        metric_type = classify_metric_type(name)

        if family_name not in families:
            if metric_type == "histogram" or metric_type == "histogram_component":
                family_type = "histogram"
            elif metric_type == "counter":
                family_type = "counter"
            elif metric_type == "info":
                family_type = "info"
            elif metric_type == "timestamp":
                family_type = "timestamp"
            else:
                family_type = "gauge"

            families[family_name] = MetricFamily(
                name=family_name,
                type=family_type,
                metrics=[],
            )

        families[family_name]["metrics"].append(name)

        if metric_type in ("histogram", "histogram_component"):
            families[family_name]["type"] = "histogram"

    logger.info(f"Grouped into {len(families)} metric families")

    type_counts = {}
    for family in families.values():
        t = family["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    for t, count in sorted(type_counts.items()):
        logger.info(f"  {t}: {count} families")

    return families


def filter_otel_metrics(
    families: dict[str, MetricFamily],
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> dict[str, MetricFamily]:
    """Filter metric families by name patterns.

    Args:
        families: Dictionary of metric families
        include_patterns: If provided, only include families matching any pattern
        exclude_patterns: Exclude families matching any pattern

    Returns:
        Filtered dictionary of metric families
    """
    import re

    result = {}

    for name, family in families.items():
        if include_patterns:
            if not any(re.search(p, name) for p in include_patterns):
                continue

        if exclude_patterns:
            if any(re.search(p, name) for p in exclude_patterns):
                continue

        result[name] = family

    return result


def get_histogram_families(families: dict[str, MetricFamily]) -> list[str]:
    """Get names of histogram metric families.

    Args:
        families: Dictionary of metric families

    Returns:
        List of histogram family names
    """
    return [name for name, family in families.items() if family["type"] == "histogram"]


def get_counter_families(families: dict[str, MetricFamily]) -> list[str]:
    """Get names of counter metric families.

    Args:
        families: Dictionary of metric families

    Returns:
        List of counter family names
    """
    return [name for name, family in families.items() if family["type"] == "counter"]


def get_gauge_families(families: dict[str, MetricFamily]) -> list[str]:
    """Get names of gauge metric families.

    Args:
        families: Dictionary of metric families

    Returns:
        List of gauge family names
    """
    return [name for name, family in families.items() if family["type"] == "gauge"]
