"""Top-N filter - keeps top N values, buckets rest into __other__."""

from typing import Any
from collections import Counter
import logging

logger = logging.getLogger(__name__)


OTHER_BUCKET = "__other__"


class TopNFilter:
    """Filter that keeps top N values and buckets the rest."""

    def __init__(
        self,
        top_values: list[str],
        other_bucket: str = OTHER_BUCKET,
    ):
        """Initialize with known top values.

        Args:
            top_values: List of values to keep (in priority order)
            other_bucket: Value to use for non-top values
        """
        self.top_values = set(top_values)
        self.top_values_list = list(top_values)
        self.other_bucket = other_bucket

    def filter(self, value: str) -> str:
        """Filter a value, returning it or the other bucket.

        Args:
            value: Value to filter

        Returns:
            Original value if in top-N, otherwise other_bucket
        """
        if value in self.top_values:
            return value
        return self.other_bucket

    def filter_series(self, values: list[str]) -> list[str]:
        """Filter a series of values.

        Args:
            values: List of values to filter

        Returns:
            Filtered list
        """
        return [self.filter(v) for v in values]

    def get_value_counts(self, values: list[str]) -> dict[str, int]:
        """Get counts of filtered values.

        Args:
            values: List of values

        Returns:
            Dict mapping filtered value to count
        """
        filtered = self.filter_series(values)
        return dict(Counter(filtered))


def build_top_n_filter_from_data(
    values: list[str],
    n: int = 20,
    vip_values: list[str] | None = None,
) -> TopNFilter:
    """Build a TopNFilter from observed data.

    Args:
        values: List of observed values
        n: Number of top values to keep
        vip_values: Values that must always be included

    Returns:
        TopNFilter configured with top N values
    """
    counts = Counter(values)

    top_by_count = [v for v, _ in counts.most_common(n)]

    if vip_values:
        vip_set = set(vip_values)
        top_values = list(vip_set)
        for v in top_by_count:
            if v not in vip_set and len(top_values) < n:
                top_values.append(v)
    else:
        top_values = top_by_count

    return TopNFilter(top_values)


def apply_top_n_to_dataframe(
    df: Any,  # pandas DataFrame
    column: str,
    top_values: list[str],
    other_bucket: str = OTHER_BUCKET,
) -> Any:
    """Apply top-N filtering to a DataFrame column.

    Args:
        df: pandas DataFrame
        column: Column name to filter
        top_values: List of values to keep
        other_bucket: Value for non-top values

    Returns:
        DataFrame with filtered column
    """
    filter_instance = TopNFilter(top_values, other_bucket)

    result = df.copy()
    result[column] = result[column].apply(filter_instance.filter)

    return result


def suggest_top_n(
    values: list[str],
    n: int = 20,
    min_coverage: float = 0.8,
) -> tuple[list[str], float]:
    """Suggest top N values to achieve minimum coverage.

    Args:
        values: List of observed values
        n: Maximum N
        min_coverage: Minimum coverage ratio to achieve

    Returns:
        Tuple of (suggested values, actual coverage)
    """
    if not values:
        return [], 0.0

    counts = Counter(values)
    total = len(values)

    sorted_values = [v for v, _ in counts.most_common()]

    cumulative = 0
    top_values = []

    for v in sorted_values[:n]:
        top_values.append(v)
        cumulative += counts[v]
        coverage = cumulative / total

        if coverage >= min_coverage:
            break

    actual_coverage = cumulative / total
    return top_values, actual_coverage
