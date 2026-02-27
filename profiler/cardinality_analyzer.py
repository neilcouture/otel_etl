"""Cardinality analyzer - counts distinct values per label."""

from typing import TypedDict
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from otel_etl.utils.prometheus_client import PrometheusClient
from otel_etl.profiler.label_discovery import LabelInfo
from otel_etl.config.defaults import (
    CardinalityThresholds,
    DEFAULT_CARDINALITY_THRESHOLDS,
    get_tier,
    get_action,
)

logger = logging.getLogger(__name__)


class CardinalityResult(TypedDict):
    """Result of cardinality analysis for a label."""
    label: str
    cardinality: int
    tier: int
    action: str
    top_values: list[str] | None  # Top N values if applicable


def analyze_cardinality(
    client: PrometheusClient,
    labels_by_family: dict[str, dict[str, LabelInfo]],
    thresholds: CardinalityThresholds | None = None,
    top_n: int = 20,
    window_hours: float = 1.0,
    max_workers: int = 4,
) -> dict[str, dict[str, CardinalityResult]]:
    """Analyze cardinality for all labels.

    Args:
        client: PrometheusClient instance
        labels_by_family: Output from discover_labels()
        thresholds: Cardinality thresholds for tier classification
        top_n: Number of top values to capture for tier 3 labels
        window_hours: Time window for analysis
        max_workers: Number of parallel workers

    Returns:
        Nested dict: family_name -> label_name -> CardinalityResult
    """
    thresholds = thresholds or DEFAULT_CARDINALITY_THRESHOLDS
    result: dict[str, dict[str, CardinalityResult]] = {}

    tasks = []
    for family_name, family_labels in labels_by_family.items():
        result[family_name] = {}
        for label_name, label_info in family_labels.items():
            if label_info["metrics"]:
                metric_name = label_info["metrics"][0]
                tasks.append((family_name, label_name, metric_name))

    def analyze_single(task: tuple[str, str, str]) -> tuple[str, str, CardinalityResult]:
        family_name, label_name, metric_name = task
        try:
            cardinality = client.count_label_cardinality(
                metric_name, label_name, window_hours
            )

            tier = get_tier(cardinality, thresholds)
            action = get_action(tier)

            top_values = None
            if tier >= 2 and cardinality > 0:
                top_values = client.get_top_n_values(
                    metric_name, label_name, top_n, window_hours
                )

            return (
                family_name,
                label_name,
                CardinalityResult(
                    label=label_name,
                    cardinality=cardinality,
                    tier=tier,
                    action=action,
                    top_values=top_values,
                ),
            )
        except RuntimeError as e:
            logger.warning(f"Failed to analyze {family_name}.{label_name}: {e}")
            return (
                family_name,
                label_name,
                CardinalityResult(
                    label=label_name,
                    cardinality=0,
                    tier=4,
                    action="drop",
                    top_values=None,
                ),
            )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(analyze_single, task): task for task in tasks}

        for future in as_completed(futures):
            family_name, label_name, cardinality_result = future.result()
            result[family_name][label_name] = cardinality_result

    tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for family_results in result.values():
        for cr in family_results.values():
            tier_counts[cr["tier"]] += 1

    logger.info(
        f"Cardinality analysis complete: "
        f"T1={tier_counts[1]}, T2={tier_counts[2]}, "
        f"T3={tier_counts[3]}, T4={tier_counts[4]}"
    )

    return result


def get_high_cardinality_labels(
    cardinality_results: dict[str, dict[str, CardinalityResult]],
    min_tier: int = 3,
) -> list[tuple[str, str, int]]:
    """Get labels with high cardinality.

    Args:
        cardinality_results: Output from analyze_cardinality()
        min_tier: Minimum tier to include

    Returns:
        List of (family_name, label_name, cardinality) tuples
    """
    high_cardinality = []

    for family_name, family_results in cardinality_results.items():
        for label_name, result in family_results.items():
            if result["tier"] >= min_tier:
                high_cardinality.append((
                    family_name,
                    label_name,
                    result["cardinality"],
                ))

    return sorted(high_cardinality, key=lambda x: x[2], reverse=True)


def get_labels_by_tier(
    cardinality_results: dict[str, dict[str, CardinalityResult]],
    tier: int,
) -> list[tuple[str, str]]:
    """Get all labels in a specific tier.

    Args:
        cardinality_results: Output from analyze_cardinality()
        tier: Tier number (1-4)

    Returns:
        List of (family_name, label_name) tuples
    """
    labels = []

    for family_name, family_results in cardinality_results.items():
        for label_name, result in family_results.items():
            if result["tier"] == tier:
                labels.append((family_name, label_name))

    return labels
