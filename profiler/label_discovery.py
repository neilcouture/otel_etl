"""Label discovery module - queries labels for each metric."""

from typing import TypedDict
import logging

from otel_etl.utils.prometheus_client import PrometheusClient
from otel_etl.profiler.metric_discovery import MetricFamily

logger = logging.getLogger(__name__)


class LabelInfo(TypedDict):
    """Information about a discovered label."""
    name: str
    metrics: list[str]  # Metrics that have this label


def discover_labels(
    client: PrometheusClient,
    families: dict[str, MetricFamily],
) -> dict[str, dict[str, LabelInfo]]:
    """Discover all labels for each metric family.

    Args:
        client: PrometheusClient instance
        families: Dictionary of metric families from discover_metrics()

    Returns:
        Nested dict: family_name -> label_name -> LabelInfo
    """
    result: dict[str, dict[str, LabelInfo]] = {}

    for family_name, family in families.items():
        family_labels: dict[str, LabelInfo] = {}

        for metric_name in family["metrics"]:
            try:
                labels = client.get_labels_for_metric(metric_name)

                for label in labels:
                    if label == "le" or label == "quantile":
                        continue

                    if label not in family_labels:
                        family_labels[label] = LabelInfo(
                            name=label,
                            metrics=[],
                        )
                    family_labels[label]["metrics"].append(metric_name)

            except RuntimeError as e:
                logger.warning(f"Failed to get labels for {metric_name}: {e}")
                continue

        result[family_name] = family_labels

        if family_labels:
            logger.debug(
                f"Family {family_name}: {len(family_labels)} labels - "
                f"{list(family_labels.keys())}"
            )

    total_labels = sum(len(labels) for labels in result.values())
    logger.info(f"Discovered {total_labels} labels across {len(result)} families")

    return result


def get_common_labels(
    labels_by_family: dict[str, dict[str, LabelInfo]],
    min_family_ratio: float = 0.5,
) -> list[str]:
    """Find labels that appear across many metric families.

    Args:
        labels_by_family: Output from discover_labels()
        min_family_ratio: Minimum ratio of families that must have the label

    Returns:
        List of common label names
    """
    label_counts: dict[str, int] = {}
    total_families = len(labels_by_family)

    for family_labels in labels_by_family.values():
        for label_name in family_labels.keys():
            label_counts[label_name] = label_counts.get(label_name, 0) + 1

    min_count = int(total_families * min_family_ratio)

    return [
        name for name, count in label_counts.items()
        if count >= min_count
    ]


def get_unique_labels(
    labels_by_family: dict[str, dict[str, LabelInfo]],
) -> dict[str, list[str]]:
    """Find labels unique to specific metric families.

    Args:
        labels_by_family: Output from discover_labels()

    Returns:
        Dict mapping family name to list of unique labels
    """
    all_labels: dict[str, list[str]] = {}

    for family_name, family_labels in labels_by_family.items():
        for label_name in family_labels.keys():
            if label_name not in all_labels:
                all_labels[label_name] = []
            all_labels[label_name].append(family_name)

    result: dict[str, list[str]] = {}

    for label_name, families in all_labels.items():
        if len(families) == 1:
            family = families[0]
            if family not in result:
                result[family] = []
            result[family].append(label_name)

    return result
