"""Counter aggregation - rate and delta computation for counter metrics."""

from typing import TypedDict
import pandas as pd
import numpy as np


class CounterAggResult(TypedDict):
    """Result of counter aggregation."""
    rate_per_sec: float
    count: float


def aggregate_counter(
    values: pd.Series,
    timestamps: pd.Series,
    window_seconds: float = 60.0,
) -> CounterAggResult:
    """Aggregate counter values over a time window.

    Counters are monotonically increasing, so we compute deltas.

    Args:
        values: Series of counter values (cumulative)
        timestamps: Series of corresponding timestamps
        window_seconds: Expected window size in seconds

    Returns:
        CounterAggResult with rate and delta
    """
    if len(values) == 0:
        return CounterAggResult(rate_per_sec=0.0, count=0.0)

    if len(values) == 1:
        return CounterAggResult(rate_per_sec=0.0, count=0.0)

    sorted_idx = timestamps.argsort()
    sorted_values = values.iloc[sorted_idx]
    sorted_timestamps = timestamps.iloc[sorted_idx]

    first_val = sorted_values.iloc[0]
    last_val = sorted_values.iloc[-1]

    delta = last_val - first_val

    if delta < 0:
        delta = last_val

    if isinstance(sorted_timestamps.iloc[0], pd.Timestamp):
        time_delta = (sorted_timestamps.iloc[-1] - sorted_timestamps.iloc[0]).total_seconds()
    else:
        time_delta = float(sorted_timestamps.iloc[-1] - sorted_timestamps.iloc[0])

    if time_delta <= 0:
        time_delta = window_seconds

    rate = delta / time_delta if time_delta > 0 else 0.0

    return CounterAggResult(
        rate_per_sec=float(rate),
        count=float(delta),
    )


def compute_rate(
    df: pd.DataFrame,
    value_col: str = "value",
    timestamp_col: str = "timestamp",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute rate for counter metrics in a DataFrame.

    Args:
        df: DataFrame with counter data
        value_col: Name of value column
        timestamp_col: Name of timestamp column
        group_cols: Columns to group by (e.g., entity_key, metric)

    Returns:
        DataFrame with rate_per_sec column
    """
    if group_cols is None:
        result = aggregate_counter(df[value_col], df[timestamp_col])
        return pd.DataFrame([{
            "rate_per_sec": result["rate_per_sec"],
            "count": result["count"],
        }])

    results = []

    for group_key, group_df in df.groupby(group_cols, sort=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        agg_result = aggregate_counter(
            group_df[value_col],
            group_df[timestamp_col],
        )

        row = dict(zip(group_cols, group_key))
        row["rate_per_sec"] = agg_result["rate_per_sec"]
        row["count"] = agg_result["count"]
        results.append(row)

    return pd.DataFrame(results)


def detect_counter_reset(
    values: pd.Series,
    threshold_ratio: float = 0.5,
) -> list[int]:
    """Detect counter reset points.

    Args:
        values: Series of counter values
        threshold_ratio: A decrease greater than this ratio indicates reset

    Returns:
        List of indices where resets occurred
    """
    if len(values) < 2:
        return []

    resets = []
    prev_val = values.iloc[0]

    for i in range(1, len(values)):
        curr_val = values.iloc[i]
        if curr_val < prev_val * threshold_ratio:
            resets.append(i)
        prev_val = curr_val

    return resets


def aggregate_counter_with_resets(
    values: pd.Series,
    timestamps: pd.Series,
    window_seconds: float = 60.0,
) -> CounterAggResult:
    """Aggregate counter handling potential resets.

    Args:
        values: Series of counter values
        timestamps: Series of timestamps
        window_seconds: Expected window size

    Returns:
        CounterAggResult accounting for resets
    """
    if len(values) < 2:
        return CounterAggResult(rate_per_sec=0.0, count=0.0)

    sorted_idx = timestamps.argsort()
    sorted_values = values.iloc[sorted_idx].reset_index(drop=True)
    sorted_timestamps = timestamps.iloc[sorted_idx].reset_index(drop=True)

    resets = detect_counter_reset(sorted_values)

    total_delta = 0.0
    prev_val = sorted_values.iloc[0]
    prev_idx = 0

    for reset_idx in resets:
        segment_delta = sorted_values.iloc[reset_idx - 1] - prev_val
        total_delta += max(0, segment_delta)
        prev_val = sorted_values.iloc[reset_idx]
        prev_idx = reset_idx

    final_delta = sorted_values.iloc[-1] - prev_val
    total_delta += max(0, final_delta)

    if isinstance(sorted_timestamps.iloc[0], pd.Timestamp):
        time_delta = (sorted_timestamps.iloc[-1] - sorted_timestamps.iloc[0]).total_seconds()
    else:
        time_delta = float(sorted_timestamps.iloc[-1] - sorted_timestamps.iloc[0])

    if time_delta <= 0:
        time_delta = window_seconds

    rate = total_delta / time_delta

    return CounterAggResult(
        rate_per_sec=float(rate),
        count=float(total_delta),
    )
