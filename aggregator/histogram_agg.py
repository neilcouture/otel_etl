"""Histogram aggregation - percentile computation from bucket data."""

from typing import TypedDict
import pandas as pd
import numpy as np

from otel_etl.config.defaults import DEFAULT_PERCENTILES


class HistogramAggResult(TypedDict):
    """Result of histogram aggregation."""
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float
    mean: float
    count: float
    sum: float


def estimate_percentile_from_buckets(
    bucket_boundaries: list[float],
    bucket_counts: list[float],
    percentile: float,
) -> float:
    """Estimate a percentile from histogram bucket data.

    Uses linear interpolation within buckets.

    Args:
        bucket_boundaries: Upper bounds of buckets (including +Inf)
        bucket_counts: Cumulative counts at each boundary
        percentile: Percentile to estimate (0-1)

    Returns:
        Estimated percentile value
    """
    if not bucket_counts or all(c == 0 for c in bucket_counts):
        return 0.0

    total_count = bucket_counts[-1]
    if total_count == 0:
        return 0.0

    target_count = percentile * total_count

    prev_boundary = 0.0
    prev_count = 0.0

    for i, (boundary, count) in enumerate(zip(bucket_boundaries, bucket_counts)):
        if count >= target_count:
            if count == prev_count:
                return boundary

            if np.isinf(boundary):
                return prev_boundary

            fraction = (target_count - prev_count) / (count - prev_count)
            return prev_boundary + fraction * (boundary - prev_boundary)

        prev_boundary = boundary
        prev_count = count

    return bucket_boundaries[-2] if len(bucket_boundaries) > 1 else 0.0


def aggregate_histogram(
    bucket_df: pd.DataFrame,
    sum_value: float | None = None,
    count_value: float | None = None,
    percentiles: list[float] | None = None,
) -> HistogramAggResult:
    """Aggregate histogram data from bucket DataFrame.

    Args:
        bucket_df: DataFrame with columns 'le' (bucket boundary) and 'value' (cumulative count)
        sum_value: Sum from _sum metric (optional)
        count_value: Count from _count metric (optional)
        percentiles: Percentiles to compute (default: [0.5, 0.75, 0.9, 0.95, 0.99])

    Returns:
        HistogramAggResult with percentiles and summary stats
    """
    percentiles = percentiles or DEFAULT_PERCENTILES

    if bucket_df.empty:
        return HistogramAggResult(
            p50=0.0, p75=0.0, p90=0.0, p95=0.0, p99=0.0,
            mean=0.0, count=0.0, sum=0.0,
        )

    # Convert le to float for proper numerical sorting
    df = bucket_df.copy()
    df["le_float"] = df["le"].apply(
        lambda x: float("inf") if x in ("+Inf", "Inf") else float(x)
    )
    sorted_df = df.sort_values("le_float")

    boundaries = sorted_df["le"].tolist()
    counts = sorted_df["value"].tolist()

    boundaries_float = []
    for b in boundaries:
        if b == "+Inf" or b == "Inf":
            boundaries_float.append(float("inf"))
        else:
            boundaries_float.append(float(b))

    counts_float = [float(c) for c in counts]

    p50 = estimate_percentile_from_buckets(boundaries_float, counts_float, 0.5)
    p75 = estimate_percentile_from_buckets(boundaries_float, counts_float, 0.75)
    p90 = estimate_percentile_from_buckets(boundaries_float, counts_float, 0.9)
    p95 = estimate_percentile_from_buckets(boundaries_float, counts_float, 0.95)
    p99 = estimate_percentile_from_buckets(boundaries_float, counts_float, 0.99)

    total_count = counts_float[-1] if counts_float else 0.0

    if count_value is not None:
        total_count = count_value

    total_sum = sum_value if sum_value is not None else 0.0
    mean = total_sum / total_count if total_count > 0 else 0.0

    return HistogramAggResult(
        p50=p50,
        p75=p75,
        p90=p90,
        p95=p95,
        p99=p99,
        mean=mean,
        count=total_count,
        sum=total_sum,
    )


def aggregate_histogram_from_raw(
    df: pd.DataFrame,
    metric_family: str,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate histogram data from raw metric DataFrame.

    Expects DataFrame with columns: metric, labels, value
    where metric includes _bucket, _sum, _count suffixes.

    Args:
        df: Raw metric DataFrame
        metric_family: Base metric name (without suffixes)
        group_cols: Additional columns to group by

    Returns:
        DataFrame with histogram aggregations
    """
    bucket_metric = f"{metric_family}_bucket"
    sum_metric = f"{metric_family}_sum"
    count_metric = f"{metric_family}_count"

    bucket_df = df[df["metric"] == bucket_metric].copy()
    sum_df = df[df["metric"] == sum_metric]
    count_df = df[df["metric"] == count_metric]

    if bucket_df.empty:
        return pd.DataFrame()

    bucket_df["le"] = bucket_df["labels"].apply(lambda x: x.get("le", "+Inf"))

    def get_non_le_labels(labels: dict) -> tuple:
        return tuple(sorted((k, v) for k, v in labels.items() if k != "le"))

    bucket_df["label_key"] = bucket_df["labels"].apply(get_non_le_labels)

    results = []

    for label_key, group in bucket_df.groupby("label_key", sort=False):
        labels_dict = dict(label_key)

        sum_val = None
        count_val = None

        matching_sum = sum_df[sum_df["labels"].apply(
            lambda x: all(x.get(k) == v for k, v in labels_dict.items())
        )]
        if not matching_sum.empty:
            sum_val = matching_sum["value"].iloc[-1]

        matching_count = count_df[count_df["labels"].apply(
            lambda x: all(x.get(k) == v for k, v in labels_dict.items())
        )]
        if not matching_count.empty:
            count_val = matching_count["value"].iloc[-1]

        agg_result = aggregate_histogram(group, sum_val, count_val)

        row = dict(labels_dict)
        row.update(agg_result)
        results.append(row)

    return pd.DataFrame(results)


def compute_histogram_delta(
    old_result: HistogramAggResult,
    new_result: HistogramAggResult,
) -> dict[str, float]:
    """Compute delta between two histogram aggregations.

    Args:
        old_result: Previous histogram result
        new_result: Current histogram result

    Returns:
        Dictionary with delta values
    """
    return {
        "count_delta": new_result["count"] - old_result["count"],
        "sum_delta": new_result["sum"] - old_result["sum"],
        "p50_delta": new_result["p50"] - old_result["p50"],
        "p99_delta": new_result["p99"] - old_result["p99"],
    }
